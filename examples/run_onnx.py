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

"""Run an exported ZipCodec endpoint using only NumPy and ONNX Runtime."""

import argparse
import hashlib
import hmac
import json
import struct
import wave
from pathlib import Path
from typing import Dict

import numpy as np
import onnxruntime as ort

SAMPLE_RATE = 16000
REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = REPOSITORY_ROOT / "outputs/onnx/zipcodec_step.onnx"
DEFAULT_AUDIO_PATH = REPOSITORY_ROOT / "audios/english/251-118436-0003.wav"
DEFAULT_OUTPUT_PATH = REPOSITORY_ROOT / "outputs/onnx/reconstruction.wav"


def _sha256(path: "Path") -> "str":
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(16 * 1024**2):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_checksums(
    metadata: "dict",
    model_path: "Path",
    inputs_path: "Path",
) -> "None":
    try:
        checksums = metadata["artifacts"]["sha256"]
    except KeyError as error:
        raise ValueError("Metadata does not contain artifact checksums") from error
    expected = {
        model_path: checksums["model"],
        inputs_path: checksums["example_inputs"],
    }
    expected.update(
        {
            model_path.parent / path: checksum
            for path, checksum in checksums["external_data"].items()
        }
    )
    for path, checksum in expected.items():
        actual = _sha256(path)
        if not hmac.compare_digest(actual, checksum):
            raise ValueError(
                f"SHA-256 mismatch for {path}: expected {checksum}, got {actual}"
            )
    print(f"Verified SHA-256 checksums for {len(expected)} artifacts")


def _load_audio(path: "Path") -> "np.ndarray":
    contents = path.read_bytes()
    if contents[:4] != b"RIFF" or contents[8:12] != b"WAVE":
        raise ValueError(f"Expected a RIFF/WAVE file: {path}")

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
    if sample_rate <= 0:
        raise ValueError(f"Expected a positive sample rate, got {sample_rate} Hz")
    if channels != 1:
        raise ValueError(f"Expected mono audio, got {channels} channels")
    if audio_format == 1 and bits_per_sample == 16:
        wav = np.frombuffer(chunks[b"data"], dtype="<i2").astype(np.float32)
        wav /= 32768.0
    elif audio_format == 3 and bits_per_sample == 32:
        wav = np.frombuffer(chunks[b"data"], dtype="<f4").astype(np.float32)
    else:
        raise ValueError(
            "Expected 16-bit PCM or 32-bit float WAV, got "
            f"format={audio_format}, bits={bits_per_sample}"
        )

    if sample_rate != SAMPLE_RATE:
        output_length = round(wav.shape[0] * SAMPLE_RATE / sample_rate)
        source_positions = np.arange(wav.shape[0], dtype=np.float64)
        target_positions = (
            np.arange(output_length, dtype=np.float64) * sample_rate / SAMPLE_RATE
        )
        wav = np.interp(target_positions, source_positions, wav).astype(np.float32)
    return wav[None]


def _save_audio(path: "Path", wav: "np.ndarray") -> "None":
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = np.rint(np.clip(wav.squeeze(0), -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as file:
        file.setnchannels(1)
        file.setsampwidth(2)
        file.setframerate(SAMPLE_RATE)
        file.writeframes(samples.tobytes())


def main(
    model_path: "Path",
    audio_path: "Path",
    inputs_path: "Path",
    output_path: "Path",
    metadata_path: "Path | None" = None,
    providers: "tuple[str, ...] | None" = None,
    verify_checksums: "bool" = False,
) -> "None":
    metadata_path = metadata_path or model_path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if verify_checksums:
        _verify_checksums(metadata, model_path, inputs_path)
    input_names = [item["name"] for item in metadata["inputs"]]
    input_shapes = {item["name"]: tuple(item["shape"]) for item in metadata["inputs"]}
    output_names = [item["name"] for item in metadata["outputs"]]
    state_inputs = metadata["state"]["inputs"]
    state_outputs = metadata["state"]["outputs"]
    if len(state_inputs) != len(state_outputs):
        raise ValueError("Metadata state input/output counts do not match")

    with np.load(inputs_path) as archive:
        inputs: Dict[str, np.ndarray] = {
            name: np.ascontiguousarray(archive[name]) for name in input_names
        }
    if "wav" not in inputs or "wav_rec" not in output_names:
        raise ValueError("The exported endpoint must have wav input and wav_rec output")
    if len(input_shapes["wav"]) != 2 or input_shapes["wav"][0] != 1:
        raise ValueError("Audio-file reconstruction requires ONNX batch size 1")
    wav = _load_audio(audio_path).astype(inputs["wav"].dtype, copy=False)
    chunk_size = input_shapes["wav"][1]

    session = ort.InferenceSession(
        str(model_path),
        providers=None if providers is None else list(providers),
    )
    actual_inputs = [item.name for item in session.get_inputs()]
    actual_outputs = [item.name for item in session.get_outputs()]
    if actual_inputs != input_names or actual_outputs != output_names:
        raise ValueError(
            "ONNX model and metadata names do not match: "
            f"inputs={actual_inputs}, outputs={actual_outputs}"
        )

    reconstructed = []
    for offset in range(0, wav.shape[1], chunk_size):
        chunk = wav[:, offset : offset + chunk_size]
        valid_length = chunk.shape[1]
        if valid_length < chunk_size:
            chunk = np.pad(chunk, ((0, 0), (0, chunk_size - valid_length)))
        inputs["wav"] = np.ascontiguousarray(chunk)
        values = session.run(output_names, inputs)
        named_outputs = dict(zip(output_names, values, strict=True))
        reconstructed.append(named_outputs["wav_rec"][:, :valid_length])
        for input_name, output_name in zip(
            state_inputs,
            state_outputs,
            strict=True,
        ):
            value = named_outputs[output_name]
            expected_shape = input_shapes[input_name]
            if value.shape != expected_shape:
                expected_size = int(np.prod(expected_shape, dtype=np.int64))
                if value.size != expected_size:
                    raise ValueError(
                        f"State output {output_name!r} has shape {value.shape}; "
                        f"cannot feed input {input_name!r} with shape "
                        f"{expected_shape}"
                    )
                value = value.reshape(expected_shape)
            inputs[input_name] = value

    wav_rec = np.concatenate(reconstructed, axis=1)
    _save_audio(output_path, wav_rec)
    print(f"Processed {wav.shape[1] / SAMPLE_RATE:.3f} s of audio")
    print(f"Providers: {session.get_providers()}")
    print(f"Saved reconstruction: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Run a persistent ZipCodec ONNX artifact without importing "
            "PyTorch or ZipCodec."
        )
    )
    parser.add_argument(
        "model",
        type=Path,
        nargs="?",
        default=DEFAULT_MODEL_PATH,
        help=f"Exported ONNX model path (default: {DEFAULT_MODEL_PATH}).",
    )
    parser.add_argument(
        "audio",
        type=Path,
        nargs="?",
        default=DEFAULT_AUDIO_PATH,
        help=f"Input mono WAV path (default: {DEFAULT_AUDIO_PATH}).",
    )
    parser.add_argument(
        "--inputs",
        type=Path,
        default=None,
        help="NPZ input archive (default: <model stem>_inputs.npz).",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="JSON metadata path (default: model path with .json suffix).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Reconstructed WAV path (default: outputs/onnx/reconstruction.wav).",
    )
    parser.add_argument(
        "--providers",
        nargs="+",
        default=None,
        help="ONNX Runtime execution providers in priority order.",
    )
    parser.add_argument(
        "--verify-checksums",
        action="store_true",
        help="Verify artifact SHA-256 checksums before loading the model.",
    )
    args = parser.parse_args()
    inputs_path = args.inputs or args.model.with_name(f"{args.model.stem}_inputs.npz")
    main(
        model_path=args.model,
        audio_path=args.audio,
        inputs_path=inputs_path,
        output_path=args.output,
        metadata_path=args.metadata,
        providers=None if args.providers is None else tuple(args.providers),
        verify_checksums=args.verify_checksums,
    )
