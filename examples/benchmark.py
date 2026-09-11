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

"""Benchmark streaming ZipCodec round-trip reconstruction backends."""

import argparse
import gc
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
from torch import Tensor


torch.backends.cudnn.benchmark = True

State = Tuple[Tensor, ...]
REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_PATH = REPOSITORY_ROOT / "outputs/benchmark.txt"
DEFAULT_CHECKPOINT = "lucadellalib/zipcodec"
BACKENDS = (
    "eager",
    "compile",
    "jit",
    "cudagraph",
    "onnx",
    "onnx-iobinding",
    "openvino",
)
DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "bfp16": torch.bfloat16,
}


@dataclass(frozen=True)
class BenchmarkResult:
    """Measurements for one backend configuration."""

    backend: str
    device: str
    dtype: str
    model_mib: float
    peak_mib: Optional[float]
    median: float
    mean: float
    std: float
    rtf: float


def _load_codec(
    checkpoint: "str",
    device: "torch.device",
    dtype: "torch.dtype",
) -> "Any":
    return (
        torch.hub.load(
            "lucadellalib/zipcodec",
            "zipcodec",
            config=checkpoint,
            trust_repo=True,
        )
        .eval()
        .to(device=device, dtype=dtype)
    )


def _model_size_mib(codec: "Any") -> "float":
    size = sum(
        value.numel() * value.element_size()
        for value in tuple(codec.parameters()) + tuple(codec.buffers())
    )
    return size / 1024**2


def _clone_state(state: "State") -> "State":
    return tuple(value.clone() for value in state)


def _initial_state(codec: "Any", wav: "Tensor") -> "State":
    return codec.init_endpoint_state(
        "step",
        batch_size=wav.shape[0],
        device=wav.device,
        dtype=wav.dtype,
    )


def _model_device(codec: "Any") -> "torch.device":
    try:
        return next(codec.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _example_inputs(
    codec: "Any",
    wav: "Tensor",
    state: "State",
) -> "Dict[str, Any]":
    chunk_size = codec.chunk_size
    chunk = torch.nn.functional.pad(
        wav[:, :chunk_size],
        (0, max(0, chunk_size - wav.shape[1])),
    )
    return {
        "wav": chunk,
        "state": _clone_state(state),
    }


def _run_stream(
    endpoint: "Any",
    codec: "Any",
    wav: "Tensor",
    initial_state: "State",
) -> "None":
    chunk_size = codec.chunk_size
    state = _clone_state(initial_state)

    for offset in range(0, wav.shape[1], chunk_size):
        chunk = wav[:, offset : offset + chunk_size]
        if chunk.shape[1] != chunk_size:
            chunk = torch.nn.functional.pad(
                chunk,
                (0, chunk_size - chunk.shape[1]),
            )
        result = endpoint({"wav": chunk, "state": state})
        state = result["next_state"]


def _synchronize(device: "torch.device") -> "None":
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _benchmark_endpoint(
    endpoint: "Any",
    codec: "Any",
    wav: "Tensor",
    initial_state: "State",
    warmup_runs: "int",
    runs: "int",
) -> "Tuple[float, float, float, Optional[float]]":
    with torch.no_grad():
        for _ in range(warmup_runs):
            _run_stream(endpoint, codec, wav, initial_state)
            _synchronize(wav.device)

        if wav.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(wav.device)

        timings = []
        for _ in range(runs):
            start = time.perf_counter()
            _run_stream(endpoint, codec, wav, initial_state)
            _synchronize(wav.device)
            timings.append(time.perf_counter() - start)

    peak_mib = None
    if wav.device.type == "cuda":
        peak_mib = torch.cuda.max_memory_allocated(wav.device) / 1024**2

    return (
        statistics.median(timings),
        statistics.mean(timings),
        statistics.stdev(timings) if len(timings) > 1 else 0.0,
        peak_mib,
    )


def _release(*values: "Any") -> "None":
    del values
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _format_results(
    results: "Sequence[BenchmarkResult]",
) -> "str":
    lines = [
        f"{'Backend':<16} {'Device':<8} {'DType':<6} {'Model MiB':>10} "
        f"{'Peak MiB':>10} {'Median (s)':>12} {'Mean (s)':>12} "
        f"{'Std (s)':>10} {'RTF (x)':>10}",
        "-" * 107,
    ]
    for result in results:
        peak = "-" if result.peak_mib is None else f"{result.peak_mib:.1f}"
        lines.append(
            f"{result.backend:<16} {result.device:<8} {result.dtype:<6} "
            f"{result.model_mib:>10.1f} {peak:>10} {result.median:>12.4f} "
            f"{result.mean:>12.4f} "
            f"{result.std:>10.4f} {result.rtf:>10.4f}"
        )
    return "\n".join(lines)


def _resolve_device(name: "str") -> "torch.device":
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return device


def main(
    duration: "float" = 10.0,
    runs: "int" = 3,
    warmup_runs: "int" = 1,
    device_name: "str" = "auto",
    dtype_name: "str" = "fp32",
    backends: "Sequence[str]" = BACKENDS,
    checkpoint: "str" = DEFAULT_CHECKPOINT,
    onnx_temporary_directory: "Optional[Path]" = Path("/tmp"),
    output_path: "Path" = DEFAULT_OUTPUT_PATH,
) -> "None":
    if duration <= 0:
        raise ValueError("duration must be greater than zero")
    if runs <= 0:
        raise ValueError("runs must be greater than zero")
    if warmup_runs < 0:
        raise ValueError("warmup_runs must be non-negative")

    unknown = set(backends) - set(BACKENDS)
    if unknown:
        raise ValueError(f"Unknown backends: {sorted(unknown)}")
    if dtype_name not in DTYPES:
        raise ValueError(f"Unknown dtype: {dtype_name!r}")

    device = _resolve_device(device_name)
    dtype = DTYPES[dtype_name]
    display_dtype = "bf16" if dtype == torch.bfloat16 else dtype_name
    results = []
    skipped = []

    print(f"Audio duration: {duration:.3f} s")
    print(f"Timed runs: {runs}; warmup runs: {warmup_runs}")
    print(f"PyTorch device: {device}")
    print(f"Inference dtype: {display_dtype}")
    print(f"Checkpoint: {checkpoint}")

    torch_backends = tuple(
        backend
        for backend in ("eager", "compile", "jit", "cudagraph")
        if backend in backends
    )
    if torch_backends:
        codec = _load_codec(
            checkpoint,
            device,
            dtype,
        )
        model_mib = _model_size_mib(codec)
        sample_rate = codec.sample_rate_input
        num_samples = max(1, round(duration * sample_rate))
        wav = torch.zeros(1, num_samples, device=device, dtype=dtype)
        initial_state = _initial_state(codec, wav)
        example_inputs = _example_inputs(codec, wav, initial_state)
        model_device = _model_device(codec)
        if model_device != device or wav.device != device:
            raise RuntimeError(
                f"Device mismatch: requested={device}, "
                f"model={model_device}, input={wav.device}"
            )
        print(f"Model device: {model_device}; input device: {wav.device}")

        for backend in torch_backends:
            if backend == "cudagraph" and device.type != "cuda":
                skipped.append((backend, "requires CUDA"))
                continue
            print(f"Preparing {backend}...")
            endpoint = None
            try:
                if backend == "eager":
                    endpoint = codec.eager("step")
                elif backend == "compile":
                    endpoint = codec.compile("step")
                elif backend == "jit":
                    endpoint = codec.jit("step", example_inputs)
                else:
                    endpoint = codec.cudagraph(
                        "step",
                        example_inputs,
                    )

                print(f"Benchmarking {backend} on {wav.device}")
                median, mean, std, peak_mib = _benchmark_endpoint(
                    endpoint,
                    codec,
                    wav,
                    initial_state,
                    warmup_runs,
                    runs,
                )
                results.append(
                    BenchmarkResult(
                        backend=backend,
                        device=str(device),
                        dtype=display_dtype,
                        model_mib=model_mib,
                        peak_mib=peak_mib,
                        median=median,
                        mean=mean,
                        std=std,
                        rtf=duration / median,
                    )
                )
            except Exception as error:
                skipped.append((backend, f"{type(error).__name__}: {error}"))
            finally:
                del endpoint
                _release()

        del codec, wav, initial_state, example_inputs
        _release()

    onnx_backends = tuple(
        backend
        for backend in ("onnx", "onnx-iobinding", "openvino")
        if backend in backends
    )
    for backend in onnx_backends:
        if backend != "openvino" and dtype != torch.float32:
            skipped.append((backend, "supports only fp32 in this benchmark"))
            continue
        if backend == "openvino" and dtype not in (torch.float32, torch.float16):
            skipped.append((backend, "supports fp32 and fp16 in this benchmark"))
            continue
        io_binding = backend == "onnx-iobinding"
        if backend == "openvino":
            onnx_device = torch.device("cpu")
            providers = ("OpenVINOExecutionProvider", "CPUExecutionProvider")
        else:
            onnx_device = device
            providers = (
                ("CUDAExecutionProvider", "CPUExecutionProvider")
                if onnx_device.type == "cuda"
                else ("CPUExecutionProvider",)
            )
        print(f"Preparing {backend}...")
        try:
            codec = _load_codec(checkpoint, onnx_device, dtype)
            model_mib = _model_size_mib(codec)
            sample_rate = codec.sample_rate_input
            num_samples = max(1, round(duration * sample_rate))
            wav = torch.zeros(
                1,
                num_samples,
                device=onnx_device,
                dtype=dtype,
            )
            initial_state = _initial_state(codec, wav)
            endpoint = codec.onnx(
                "step",
                _example_inputs(codec, wav, initial_state),
                output_device=onnx_device if io_binding else None,
                providers=providers,
                temporary_directory=onnx_temporary_directory,
                io_binding=io_binding,
            )
            if backend == "openvino":
                active_providers = tuple(endpoint.session.get_providers())
                if "OpenVINOExecutionProvider" not in active_providers:
                    raise RuntimeError(
                        "OpenVINOExecutionProvider did not activate; "
                        f"active providers: {active_providers}"
                    )
            print(f"Benchmarking {backend} on {onnx_device}")
            median, mean, std, peak_mib = _benchmark_endpoint(
                endpoint,
                codec,
                wav,
                initial_state,
                warmup_runs,
                runs,
            )
            results.append(
                BenchmarkResult(
                    backend=backend,
                    device=str(onnx_device),
                    dtype=display_dtype,
                    model_mib=model_mib,
                    peak_mib=peak_mib,
                    median=median,
                    mean=mean,
                    std=std,
                    rtf=duration / median,
                )
            )
        except Exception as error:
            skipped.append((backend, f"{type(error).__name__}: {error}"))
        finally:
            _release()

    report = _format_results(results)
    if skipped:
        report += "\n\n" + "\n".join(
            f"Skipped {backend}: {reason}" for backend, reason in skipped
        )

    print()
    print(report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report + "\n", encoding="utf-8")
    print(f"\nSaved benchmark report: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark streaming ZipCodec step reconstruction speed "
            "(RTF = audio time / processing time)."
        ),
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Synthetic audio duration in seconds (default: 10).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of timed full-audio runs (default: 3).",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=1,
        help="Number of untimed full-audio runs (default: 1).",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="PyTorch backend device (default: auto).",
    )
    parser.add_argument(
        "--dtype",
        choices=tuple(DTYPES),
        default="fp32",
        help="Inference dtype; bfp16 is accepted as an alias for bf16.",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=BACKENDS,
        default=list(BACKENDS),
        help="Backends to benchmark.",
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
        "--onnx-temp-directory",
        type=Path,
        default=Path("/tmp"),
        help="Directory for transient ONNX exports (default: /tmp).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Benchmark report path (default: outputs/benchmark.txt).",
    )
    args = parser.parse_args()
    main(
        duration=args.duration,
        runs=args.runs,
        warmup_runs=args.warmup_runs,
        device_name=args.device,
        dtype_name=args.dtype,
        backends=args.backends,
        checkpoint=args.checkpoint,
        onnx_temporary_directory=args.onnx_temp_directory,
        output_path=args.output,
    )
