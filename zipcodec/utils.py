# ==============================================================================
# Copyright 2026 Luca Della Libera.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Dependency-free audio utilities."""

import math
import struct
import wave
from array import array
from pathlib import Path
from sys import byteorder
from typing import Tuple, Union

import torch
from torch import Tensor, nn


__all__ = ["load_audio", "resample_audio", "save_audio"]


def load_audio(path: "Union[str, Path]") -> "Tuple[Tensor, int]":
    """Load a PCM WAV file as a float tensor shaped (channels, time)."""
    try:
        with wave.open(str(path), "rb") as file:
            channels = file.getnchannels()
            sample_rate = file.getframerate()
            sample_width = file.getsampwidth()
            frames = file.readframes(file.getnframes())
        audio_format = 1
    except wave.Error as error:
        contents = Path(path).read_bytes()
        if contents[:4] != b"RIFF" or contents[8:12] != b"WAVE":
            raise ValueError(f"Expected a RIFF/WAVE file: {path}") from error

        chunks = {}
        offset = 12
        while offset + 8 <= len(contents):
            chunk_name = contents[offset : offset + 4]
            chunk_size = struct.unpack_from("<I", contents, offset + 4)[0]
            chunk_start = offset + 8
            chunks[chunk_name] = contents[chunk_start : chunk_start + chunk_size]
            offset = chunk_start + chunk_size + (chunk_size % 2)

        if b"fmt " not in chunks or b"data" not in chunks:
            raise ValueError(f"Missing fmt or data chunk in WAV file: {path}")
        audio_format, channels, sample_rate, _, _, bits_per_sample = struct.unpack_from(
            "<HHIIHH", chunks[b"fmt "]
        )
        sample_width = bits_per_sample // 8
        frames = chunks[b"data"]

    if audio_format == 3 and sample_width == 4:
        waveform = torch.frombuffer(bytearray(frames), dtype=torch.float32)
    elif audio_format != 1:
        raise ValueError(
            f"Unsupported WAV format: format={audio_format}, "
            f"sample_width={sample_width} bytes"
        )
    elif sample_width == 1:
        waveform = torch.frombuffer(bytearray(frames), dtype=torch.uint8)
        waveform = (waveform.to(torch.float32) - 128.0) / 128.0
    elif sample_width == 2:
        waveform = torch.frombuffer(bytearray(frames), dtype=torch.int16)
        waveform = waveform.to(torch.float32) / 32768.0
    elif sample_width == 3:
        values = torch.frombuffer(bytearray(frames), dtype=torch.uint8)
        values = values.reshape(-1, 3).to(torch.int32)
        waveform = values[:, 0] | values[:, 1].bitwise_left_shift(8)
        waveform |= values[:, 2].bitwise_left_shift(16)
        waveform = waveform - waveform.ge(1 << 23).to(torch.int32) * (1 << 24)
        waveform = waveform.to(torch.float32) / float(1 << 23)
    elif sample_width == 4:
        waveform = torch.frombuffer(bytearray(frames), dtype=torch.int32)
        waveform = waveform.to(torch.float32) / float(1 << 31)
    else:
        raise ValueError(f"Unsupported PCM sample width: {sample_width} bytes")

    if waveform.numel() % channels:
        raise ValueError(f"Invalid WAV data in {path}")
    return waveform.reshape(-1, channels).t().contiguous(), sample_rate


def save_audio(
    path: "Union[str, Path]",
    waveform: "Tensor",
    sample_rate: "int",
    encoding: "str" = "float32",
) -> "None":
    """Save a waveform shaped (channels, time) as float32 or 16-bit PCM WAV."""
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.ndim != 2:
        raise ValueError("`waveform` must have shape (channels, time) or (time,)")
    if sample_rate <= 0:
        raise ValueError("`sample_rate` must be positive")
    if encoding not in ("float32", "pcm16"):
        raise ValueError("`encoding` must be 'float32' or 'pcm16'")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    channels = waveform.shape[0]
    samples = waveform.detach().to(device="cpu", dtype=torch.float32)
    samples = samples.t().contiguous().view(-1)

    if encoding == "pcm16":
        samples = samples.clamp(-1.0, 1.0).mul(32767.0).round().to(torch.int16)
        pcm = array("h", samples.tolist())
        if byteorder == "big":
            pcm.byteswap()
        with wave.open(str(path), "wb") as file:
            file.setnchannels(channels)
            file.setsampwidth(2)
            file.setframerate(sample_rate)
            file.writeframes(pcm.tobytes())
        return

    floating_point = array("f", samples.tolist())
    if byteorder == "big":
        floating_point.byteswap()
    data = floating_point.tobytes()
    byte_rate = sample_rate * channels * 4
    block_align = channels * 4
    fmt = struct.pack("<HHIIHH", 3, channels, sample_rate, byte_rate, block_align, 32)
    riff_size = 4 + 8 + len(fmt) + 8 + len(data)
    with path.open("wb") as file:
        file.write(b"RIFF")
        file.write(struct.pack("<I", riff_size))
        file.write(b"WAVEfmt ")
        file.write(struct.pack("<I", len(fmt)))
        file.write(fmt)
        file.write(b"data")
        file.write(struct.pack("<I", len(data)))
        file.write(data)


def resample_audio(
    waveform: "Tensor",
    orig_freq: "int",
    new_freq: "int",
) -> "Tensor":
    """Resample the last axis using Hann-windowed sinc interpolation."""
    if not isinstance(orig_freq, int) or not isinstance(new_freq, int):
        raise TypeError("Sample rates must be integers")
    if orig_freq <= 0 or new_freq <= 0:
        raise ValueError("Sample rates must be positive")
    if not waveform.is_floating_point():
        raise TypeError("`waveform` must have a floating-point dtype")
    if orig_freq == new_freq or waveform.shape[-1] == 0:
        return waveform

    common = math.gcd(orig_freq, new_freq)
    source_rate = orig_freq // common
    target_rate = new_freq // common
    lowpass_filter_width = 6
    base_rate = min(source_rate, target_rate) * 0.99
    width = math.ceil(lowpass_filter_width * source_rate / base_rate)

    output_dtype = waveform.dtype
    work_dtype = (
        torch.float32
        if output_dtype in (torch.float16, torch.bfloat16)
        else output_dtype
    )
    work = waveform.to(work_dtype)
    idx = torch.arange(
        -width,
        width + source_rate,
        dtype=work_dtype,
        device=waveform.device,
    )[None, None]
    idx = idx / source_rate
    offsets = torch.arange(
        0,
        -target_rate,
        -1,
        dtype=work_dtype,
        device=waveform.device,
    )[:, None, None]
    t = (offsets / target_rate + idx) * base_rate
    t = t.clamp(-lowpass_filter_width, lowpass_filter_width)
    window = torch.cos(t * math.pi / lowpass_filter_width / 2).square()
    t = t * math.pi
    sinc = torch.where(t == 0, torch.ones_like(t), t.sin() / t)
    kernel = sinc * window * (base_rate / source_rate)

    shape = work.shape
    length = shape[-1]
    packed = work.reshape(-1, length)
    packed = nn.functional.pad(packed, (width, width + source_rate))
    output = nn.functional.conv1d(packed[:, None], kernel, stride=source_rate)
    output = output.transpose(1, 2).reshape(packed.shape[0], -1)
    target_length = math.ceil(target_rate * length / source_rate)
    output = output[..., :target_length]
    output = output.reshape(shape[:-1] + (target_length,))
    return output.to(output_dtype)
