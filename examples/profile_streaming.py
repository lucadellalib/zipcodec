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

"""Profile CPU/GPU streaming-step performance for ZipCodec."""

import argparse
import gc
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

import torch
from torch import Tensor

State = Tuple[Tensor, ...]
REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_PATH = REPOSITORY_ROOT / "outputs/profile_streaming.txt"
DEFAULT_CHECKPOINT = "lucadellalib/zipcodec"
GPU_BATCH_SIZES = (1, 2, 4, 8, 16)
DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


@dataclass(frozen=True)
class Result:
    """Measurements for one device and batch size."""

    device: "str"
    batch_size: "int"
    rtf: "float"
    latency_ms: "float"
    p99_latency_ms: "float"
    throughput: "float"
    gpu_memory_mib: "Optional[float]"


def _synchronize(device: "torch.device") -> "None":
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _benchmark(
    codec: "Any",
    endpoint: "Any",
    device: "torch.device",
    dtype: "torch.dtype",
    batch_size: "int",
    warmup_steps: "int",
    steps: "int",
    runs: "int",
) -> "Result":
    wav = torch.zeros(
        batch_size,
        codec.chunk_size,
        device=device,
        dtype=dtype,
    )
    state = codec.init_endpoint_state(
        "step",
        batch_size=batch_size,
        device=device,
        dtype=dtype,
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    with torch.inference_mode():
        for _ in range(warmup_steps):
            state = endpoint({"wav": wav, "state": state})["next_state"]
        _synchronize(device)

        elapsed = 0.0
        for _ in range(runs):
            start = time.perf_counter()
            for _ in range(steps):
                state = endpoint({"wav": wav, "state": state})["next_state"]
            _synchronize(device)
            elapsed += time.perf_counter() - start

        step_latencies = []
        for _ in range(runs):
            for _ in range(steps):
                _synchronize(device)
                start = time.perf_counter()
                state = endpoint({"wav": wav, "state": state})["next_state"]
                _synchronize(device)
                step_latencies.append(time.perf_counter() - start)

    aggregate_latency = elapsed / (runs * steps)
    mean_latency = sum(step_latencies) / len(step_latencies)
    p99_latency = sorted(step_latencies)[math.ceil(0.99 * len(step_latencies)) - 1]
    chunk_duration = codec.chunk_size / codec.sample_rate_input
    gpu_memory_mib = (
        torch.cuda.max_memory_reserved(device) / 1024**2
        if device.type == "cuda"
        else None
    )
    return Result(
        device="GPU" if device.type == "cuda" else "CPU",
        batch_size=batch_size,
        rtf=chunk_duration / aggregate_latency,
        latency_ms=mean_latency * 1000,
        p99_latency_ms=p99_latency * 1000,
        throughput=batch_size / aggregate_latency,
        gpu_memory_mib=gpu_memory_mib,
    )


def _format_table(results: "Sequence[Result]") -> "str":
    lines = [
        "Device\tBatch size\tRTF ↑\tStep latency (ms) ↓\t"
        "p99 step latency (ms) ↓\t"
        "Throughput (streams/s) ↑\tPeak GPU VRAM",
    ]
    lines.extend(
        f"{result.device}\t{result.batch_size}\t{result.rtf:.4f}\t"
        f"{result.latency_ms:.2f}\t{result.p99_latency_ms:.2f}\t"
        f"{result.throughput:.2f}\t"
        f"{'-' if result.gpu_memory_mib is None else f'{result.gpu_memory_mib:.1f} MiB'}"
        for result in results
    )
    return "\n".join(lines)


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


def main(
    checkpoint: "str" = DEFAULT_CHECKPOINT,
    dtype_name: "str" = "fp32",
    gpu_batch_sizes: "Sequence[int]" = GPU_BATCH_SIZES,
    duration: "float" = 40.96,
    runs: "int" = 5,
    warmup_steps: "int" = 10,
    threads: "int" = 4,
    output_path: "Path" = DEFAULT_OUTPUT_PATH,
) -> "None":
    if dtype_name not in DTYPES:
        raise ValueError(f"Unknown dtype: {dtype_name!r}")
    if duration <= 0:
        raise ValueError("duration must be positive")
    if runs <= 0:
        raise ValueError("runs must be positive")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if threads <= 0:
        raise ValueError("threads must be positive")
    if any(batch_size <= 0 for batch_size in gpu_batch_sizes):
        raise ValueError("GPU batch sizes must be positive")

    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    dtype = DTYPES[dtype_name]
    results = []

    cpu = torch.device("cpu")
    cpu_codec = _load_codec(checkpoint, cpu, dtype)
    chunk_duration = cpu_codec.chunk_size / cpu_codec.sample_rate_input
    steps = round(duration / chunk_duration)
    if not math.isclose(steps * chunk_duration, duration):
        raise ValueError(
            f"duration must be a multiple of the {chunk_duration:.6f}-second "
            "streaming chunk duration"
        )
    print(
        f"Audio duration per stream: {duration:.2f} s "
        f"({steps} streaming steps per run; {runs} runs)"
    )
    print("Benchmarking CPU, batch 1...")
    results.append(
        _benchmark(
            cpu_codec,
            cpu_codec.eager("step"),
            cpu,
            dtype,
            batch_size=1,
            warmup_steps=warmup_steps,
            steps=steps,
            runs=runs,
        )
    )
    del cpu_codec
    gc.collect()

    if torch.cuda.is_available():
        cuda = torch.device("cuda", torch.cuda.current_device())
        cuda_codec = _load_codec(checkpoint, cuda, dtype)
        endpoint = cuda_codec.eager("step")
        for batch_size in gpu_batch_sizes:
            gc.collect()
            torch.cuda.empty_cache()
            print(f"Benchmarking GPU, batch {batch_size}...")
            results.append(
                _benchmark(
                    cuda_codec,
                    endpoint,
                    cuda,
                    dtype,
                    batch_size=batch_size,
                    warmup_steps=warmup_steps,
                    steps=steps,
                    runs=runs,
                )
            )
    else:
        print("CUDA is unavailable; emitting the CPU row only.")

    table = _format_table(results)
    print(table)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(table + "\n", encoding="utf-8")
    print(f"\nSaved table: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dtype", choices=tuple(DTYPES), default="fp32")
    parser.add_argument(
        "--gpu-batch-sizes",
        type=int,
        nargs="+",
        default=list(GPU_BATCH_SIZES),
    )
    parser.add_argument("--duration", type=float, default=40.96)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()
    main(
        checkpoint=args.checkpoint,
        dtype_name=args.dtype,
        gpu_batch_sizes=args.gpu_batch_sizes,
        duration=args.duration,
        runs=args.runs,
        warmup_steps=args.warmup_steps,
        threads=args.threads,
        output_path=args.output,
    )
