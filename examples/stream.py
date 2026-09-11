# ==============================================================================
# Copyright 2026 Luca Della Libera.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Stream ZipCodec resynthesis on CPU with eager PyTorch or OpenVINO."""

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor


torch.backends.cudnn.benchmark = True

State = Tuple[Tensor, ...]
REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIRECTORY = REPOSITORY_ROOT / "outputs/stream"
DEFAULT_CHECKPOINT = "lucadellalib/zipcodec"
DEFAULT_FOCALCODEC_CHECKPOINT = "lucadellalib/focalcodec_50hz"
SAMPLE_RATE = 16000
PROVIDERS = ("OpenVINOExecutionProvider", "CPUExecutionProvider")
BACKENDS = ("eager", "compile", "jit", "openvino")
DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
}


def _clone_state(state: "State") -> "State":
    return tuple(value.clone() for value in state)


def _target_audio_files(directory: "Path") -> "Tuple[Path, ...]":
    if not directory.is_dir():
        raise NotADirectoryError(f"Target voice folder not found: {directory}")
    paths = tuple(
        sorted(path for path in directory.rglob("*") if path.suffix.lower() == ".wav")
    )
    if not paths:
        raise FileNotFoundError(
            f"No WAV files found under target voice folder: {directory}"
        )
    return paths


def _build_matching_set(
    codec: "Any",
    directory: "Path",
    focalcodec_checkpoint: "str",
    dtype: "torch.dtype",
) -> "Tensor":
    start = time.perf_counter()
    print(f"Loading FocalCodec WavLM6 encoder: {focalcodec_checkpoint}")
    focal_codec = (
        torch.hub.load(
            "lucadellalib/focalcodec",
            "focalcodec",
            config=focalcodec_checkpoint,
            trust_repo=True,
        )
        .eval()
        .to(dtype=dtype)
    )
    focal_codec.requires_grad_(False)

    features = []
    paths = _target_audio_files(directory)
    with torch.no_grad():
        for index, path in enumerate(paths, start=1):
            print(f"Extracting target features [{index}/{len(paths)}]: {path}")
            wav, sample_rate = codec.load_audio(path)
            wav = codec.resample_audio(wav, sample_rate, codec.sample_rate_input)
            wav = wav.to(dtype=dtype)
            file_features = focal_codec.sig_to_feats(wav)
            features.append(file_features.squeeze(0).to(dtype=dtype).cpu())

    matching_set = torch.cat(features, dim=0).contiguous()
    elapsed = time.perf_counter() - start
    size_mib = matching_set.numel() * matching_set.element_size() / 1024**2
    print(
        f"Target pool: {matching_set.shape[0]} frames x "
        f"{matching_set.shape[1]} features, {size_mib:.1f} MiB, "
        f"built in {elapsed:.3f} s"
    )
    return matching_set


def _create_endpoint(
    checkpoint: "str",
    temporary_directory: "Optional[Path]",
    backend: "str",
    target_voice: "Optional[Path]",
    focalcodec_checkpoint: "str",
    dtype: "torch.dtype",
) -> "Tuple[Any, Any, State, int, Optional[Tensor]]":
    device = torch.device("cpu")
    codec = (
        torch.hub.load(
            "lucadellalib/zipcodec",
            "zipcodec",
            config=checkpoint,
            trust_repo=True,
        )
        .eval()
        .to(device=device, dtype=dtype)
    )
    matching_set = None
    endpoint_name = "step"
    if target_voice is not None:
        matching_set = _build_matching_set(
            codec,
            target_voice,
            focalcodec_checkpoint,
            dtype,
        )
        expected_dim = codec.backend.output_dim
        if matching_set.shape[1] != expected_dim:
            raise ValueError(
                f"Target features have dimension {matching_set.shape[1]}; "
                f"ZipCodec expects {expected_dim}"
            )
        endpoint_name = "wav_to_wav_vc_step"

    chunk_size = codec.chunk_size
    state = codec.init_endpoint_state(
        endpoint_name,
        batch_size=1,
        device=device,
        dtype=dtype,
    )
    example_inputs: Dict[str, Any] = {
        "wav": torch.zeros(1, chunk_size, device=device, dtype=dtype),
        "state": _clone_state(state),
    }
    if matching_set is not None:
        example_inputs["matching_set"] = matching_set

    if backend == "eager":
        endpoint = codec.eager(endpoint_name)
        print("Backend: eager PyTorch on CPU")
    elif backend == "compile":
        endpoint = codec.compile(endpoint_name)
        print("Backend: torch.compile on CPU")
    elif backend == "jit":
        endpoint = codec.jit(endpoint_name, example_inputs)
        print("Backend: traced TorchScript on CPU")
    else:
        try:
            import onnxruntime as ort
        except ImportError as error:
            raise ImportError(
                "Install onnxruntime-openvino to run OpenVINO streaming"
            ) from error

        available_providers = tuple(ort.get_available_providers())
        if "OpenVINOExecutionProvider" not in available_providers:
            raise RuntimeError(
                "ONNX Runtime does not expose OpenVINOExecutionProvider. Install "
                "onnxruntime-openvino in an environment without another "
                f"onnxruntime distribution. Available providers: {available_providers}"
            )

        endpoint = codec.onnx(
            endpoint_name,
            example_inputs,
            providers=PROVIDERS,
            temporary_directory=temporary_directory,
        )
        active_providers = tuple(endpoint.session.get_providers())
        if "OpenVINOExecutionProvider" not in active_providers:
            raise RuntimeError(
                "OpenVINOExecutionProvider did not activate. "
                f"Active providers: {active_providers}"
            )
        print(f"ONNX Runtime providers: {active_providers}")

    print(f"Streaming chunk: {chunk_size} samples ({chunk_size / SAMPLE_RATE:.3f} s)")
    return codec, endpoint, state, chunk_size, matching_set


@torch.no_grad()
def _step(
    endpoint: "Any",
    wav: "Tensor",
    state: "State",
    matching_set: "Optional[Tensor]",
) -> "Tuple[Tensor, State]":
    state = tuple(
        value.to(dtype=wav.dtype) if value.is_floating_point() else value
        for value in state
    )
    inputs: Dict[str, Any] = {"wav": wav, "state": state}
    output_name = "wav_rec"
    if matching_set is not None:
        inputs["matching_set"] = matching_set.to(dtype=wav.dtype)
        output_name = "wav_vc"
    outputs = endpoint(inputs)
    return outputs[output_name], outputs["next_state"]


def _print_stream_metrics(
    latencies: "Sequence[float]",
    processed_samples: "int",
    wall_elapsed: "float",
) -> "None":
    if not latencies:
        print("No audio chunks were processed.")
        return

    values = np.asarray(latencies, dtype=np.float64)
    audio_duration = processed_samples / SAMPLE_RATE
    compute_elapsed = float(values.sum())
    print(
        f"Processed {audio_duration:.3f} s across {len(latencies)} chunks; "
        f"compute {compute_elapsed:.3f} s, wall {wall_elapsed:.3f} s"
    )
    print(
        "Chunk latency: "
        f"mean={values.mean() * 1000:.2f} ms, "
        f"p50={np.percentile(values, 50) * 1000:.2f} ms, "
        f"p95={np.percentile(values, 95) * 1000:.2f} ms, "
        f"p99={np.percentile(values, 99) * 1000:.2f} ms, "
        f"max={values.max() * 1000:.2f} ms"
    )
    print(
        f"Steady-state RTF: {audio_duration / compute_elapsed:.2f}x; "
        f"end-to-end RTF: {audio_duration / wall_elapsed:.2f}x"
    )


def resynthesize_file(
    codec: "Any",
    endpoint: "Any",
    initial_state: "State",
    chunk_size: "int",
    input_path: "Path",
    output_path: "Path",
    matching_set: "Optional[Tensor]",
    dtype: "torch.dtype",
) -> "None":
    wav, sample_rate = codec.load_audio(input_path)
    wav = codec.resample_audio(wav, sample_rate, codec.sample_rate_input)
    wav = wav.to(dtype=dtype)
    state = _clone_state(initial_state)
    outputs = []
    latencies = []

    start = time.perf_counter()
    for offset in range(0, wav.shape[1], chunk_size):
        chunk = wav[:, offset : offset + chunk_size]
        if chunk.shape[1] < chunk_size:
            chunk = torch.nn.functional.pad(
                chunk,
                (0, chunk_size - chunk.shape[1]),
            )
        chunk_start = time.perf_counter()
        wav_rec, state = _step(endpoint, chunk, state, matching_set)
        latencies.append(time.perf_counter() - chunk_start)
        outputs.append(wav_rec)
    elapsed = time.perf_counter() - start

    wav_rec = torch.cat(outputs, dim=1)[:, : wav.shape[1]]
    codec.save_audio(output_path, wav_rec, codec.sample_rate_output)
    _print_stream_metrics(latencies, wav.shape[1], elapsed)


def _sounddevice_id(value: "Optional[str]") -> "Any":
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return value


def resynthesize_microphone(
    codec: "Any",
    endpoint: "Any",
    initial_state: "State",
    chunk_size: "int",
    duration: "float",
    output_path: "Optional[Path]",
    input_device: "Optional[str]",
    output_device: "Optional[str]",
    matching_set: "Optional[Tensor]",
    dtype: "torch.dtype",
) -> "None":
    try:
        import sounddevice as sd
    except ImportError as error:
        raise ImportError("Install sounddevice to stream from a microphone") from error

    if duration < 0:
        raise ValueError("duration must be non-negative")
    num_chunks = None
    if duration > 0:
        num_chunks = math.ceil(duration * SAMPLE_RATE / chunk_size)

    state = _clone_state(initial_state)
    recorded = []
    latencies = []
    input_overflows = 0
    output_underflows = 0
    stream = sd.Stream(
        samplerate=SAMPLE_RATE,
        blocksize=chunk_size,
        channels=1,
        dtype="float32",
        device=(_sounddevice_id(input_device), _sounddevice_id(output_device)),
    )

    start_message = "Microphone resynthesis started; press Ctrl+C to stop."
    if sys.stdout.isatty():
        start_message = f"\033[1;4;34m{start_message}\033[0m"
    print(start_message)
    chunk_index = 0
    start = time.perf_counter()
    try:
        with stream:
            while num_chunks is None or chunk_index < num_chunks:
                input_frames, overflowed = stream.read(chunk_size)
                if overflowed:
                    input_overflows += 1
                    print("Warning: microphone input overflow")
                wav = (
                    torch.from_numpy(input_frames[:, 0].copy())
                    .unsqueeze(0)
                    .to(dtype=dtype)
                )
                chunk_start = time.perf_counter()
                wav_rec, state = _step(endpoint, wav, state, matching_set)
                latencies.append(time.perf_counter() - chunk_start)
                output_frames = (
                    wav_rec.squeeze(0)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)[:, None]
                )
                underflowed = stream.write(output_frames)
                if underflowed:
                    output_underflows += 1
                    print("Warning: speaker output underflow")
                if output_path is not None:
                    recorded.append(wav_rec)
                chunk_index += 1
    except KeyboardInterrupt:
        print("Microphone resynthesis stopped.")
    elapsed = time.perf_counter() - start

    if output_path is not None and recorded:
        codec.save_audio(
            output_path,
            torch.cat(recorded, dim=1),
            codec.sample_rate_output,
        )
    _print_stream_metrics(latencies, chunk_index * chunk_size, elapsed)
    print(
        f"Stream status events: input overflows={input_overflows}, "
        f"output underflows={output_underflows}"
    )


def main(
    checkpoint: "str" = DEFAULT_CHECKPOINT,
    audio_path: "Optional[Path]" = None,
    microphone: "bool" = False,
    output_path: "Optional[Path]" = None,
    duration: "float" = 10.0,
    input_device: "Optional[str]" = None,
    output_device: "Optional[str]" = None,
    onnx_temporary_directory: "Optional[Path]" = Path("/tmp"),
    backend: "str" = "eager",
    target_voice: "Optional[Path]" = None,
    focalcodec_checkpoint: "str" = DEFAULT_FOCALCODEC_CHECKPOINT,
    dtype_name: "str" = "fp32",
    threads: "int" = 4,
) -> "None":
    if (audio_path is None) == (not microphone):
        raise ValueError("Specify exactly one of audio_path or microphone")
    if backend not in BACKENDS:
        raise ValueError(f"Unknown backend: {backend!r}")
    if dtype_name not in DTYPES:
        raise ValueError(f"Unknown dtype: {dtype_name!r}")
    if threads <= 0:
        raise ValueError("threads must be greater than zero")
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    print(f"PyTorch CPU threads: intra-op={threads}, inter-op=1")
    dtype = DTYPES[dtype_name]
    display_dtype = "bf16" if dtype == torch.bfloat16 else dtype_name
    print(f"Inference dtype: {display_dtype}")

    setup_start = time.perf_counter()
    codec, endpoint, state, chunk_size, matching_set = _create_endpoint(
        checkpoint,
        onnx_temporary_directory,
        backend,
        target_voice,
        focalcodec_checkpoint,
        dtype,
    )
    setup_elapsed = time.perf_counter() - setup_start
    warmup_start = time.perf_counter()
    _step(
        endpoint,
        torch.zeros(1, chunk_size, dtype=dtype),
        _clone_state(state),
        matching_set,
    )
    warmup_elapsed = time.perf_counter() - warmup_start
    print(
        f"Startup: setup/export {setup_elapsed:.3f} s, "
        f"first-call warmup {warmup_elapsed:.3f} s, "
        f"total {setup_elapsed + warmup_elapsed:.3f} s"
    )
    mode = "vc_" if matching_set is not None else ""
    if audio_path is not None:
        output_path = output_path or DEFAULT_OUTPUT_DIRECTORY / (
            f"{backend}_{mode}file.wav"
        )
        resynthesize_file(
            codec,
            endpoint,
            state,
            chunk_size,
            audio_path,
            output_path,
            matching_set,
            dtype,
        )
    else:
        output_path = output_path or DEFAULT_OUTPUT_DIRECTORY / (
            f"{backend}_{mode}microphone.wav"
        )
        resynthesize_microphone(
            codec,
            endpoint,
            state,
            chunk_size,
            duration,
            output_path,
            input_device,
            output_device,
            matching_set,
            dtype,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stream ZipCodec resynthesis on CPU with PyTorch or OpenVINO."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--audio",
        type=Path,
        help="Input mono WAV file (resampled to 16 kHz when necessary).",
    )
    source.add_argument(
        "--microphone",
        action="store_true",
        help="Resynthesize live microphone input to the default speaker.",
    )
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help=(
            "Local config path or Hugging Face repository "
            f"(default: {DEFAULT_CHECKPOINT})."
        ),
    )
    parser.add_argument(
        "--backend",
        choices=BACKENDS,
        default="eager",
        help="CPU inference backend (default: eager).",
    )
    parser.add_argument(
        "--target-voice",
        type=Path,
        default=None,
        help=(
            "Folder of mono WAV files used to build the target WavLM6 feature "
            "pool (resampled to 16 kHz when necessary)."
        ),
    )
    parser.add_argument(
        "--focalcodec-checkpoint",
        default=DEFAULT_FOCALCODEC_CHECKPOINT,
        help=(
            "FocalCodec checkpoint used for target WavLM6 features "
            f"(default: {DEFAULT_FOCALCODEC_CHECKPOINT})."
        ),
    )
    parser.add_argument(
        "--dtype",
        choices=tuple(DTYPES),
        default="fp32",
        help="Inference dtype.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help=(
            "PyTorch intra-op CPU threads; also sets inter-op threads to 1 "
            "(default: 4)."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output WAV path; microphone audio recorded before Ctrl+C is saved "
            "here (default: outputs/stream/<backend>_microphone.wav)."
        ),
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Microphone duration in seconds; 0 runs until Ctrl+C (default: 0).",
    )
    parser.add_argument("--input-device", default=None)
    parser.add_argument("--output-device", default=None)
    parser.add_argument(
        "--onnx-temp-directory",
        type=Path,
        default=Path("/tmp"),
        help="Directory for transient ONNX exports (default: /tmp).",
    )
    args = parser.parse_args()
    main(
        checkpoint=args.checkpoint,
        audio_path=args.audio,
        microphone=args.microphone,
        output_path=args.output,
        duration=args.duration,
        input_device=args.input_device,
        output_device=args.output_device,
        onnx_temporary_directory=args.onnx_temp_directory,
        backend=args.backend,
        target_voice=args.target_voice,
        focalcodec_checkpoint=args.focalcodec_checkpoint,
        dtype_name=args.dtype,
        threads=args.threads,
    )
