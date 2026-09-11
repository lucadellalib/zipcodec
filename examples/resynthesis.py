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

"""Offline and streaming resynthesis examples across supported backends."""

import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor


DEFAULT_CHECKPOINT = "lucadellalib/zipcodec"
REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_AUDIO_PATH = REPOSITORY_ROOT / "audios/english/251-118436-0003.wav"


def _rmse(input: "Tensor", target: "Tensor") -> "float":
    difference = input.to(torch.float64) - target.to(torch.float64)
    return float((difference**2).mean().sqrt())


def _compare_outputs(
    name: "str",
    actual: "Dict[str, Any]",
    expected: "Dict[str, Any]",
    output_name: "str",
) -> "None":
    output_rmse = _rmse(actual[output_name], expected[output_name])
    state_rmses = [
        _rmse(actual_state, expected_state)
        for actual_state, expected_state in zip(
            actual["next_state"],
            expected["next_state"],
            strict=True,
        )
    ]
    state_rmse = max(state_rmses)
    print(f"{name} output RMSE: {output_rmse:.3e}")
    print(f"{name} state RMSE: {state_rmse:.3e}")
    if state_rmse >= 1e-3:
        print(
            f"{name} state RMSEs:",
            ", ".join(f"{value:.3e}" for value in state_rmses),
        )


def _compare_offline_outputs(
    name: "str",
    actual: "Dict[str, Any]",
    expected: "Dict[str, Any]",
) -> "None":
    output_rmse = _rmse(actual["wav_rec"], expected["wav_rec"])
    print(f"{name} output RMSE: {output_rmse:.3e}")


def _load_codec(
    checkpoint: "str",
    device: "torch.device",
    dtype: "torch.dtype" = torch.float32,
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


def _clone_state(state: "Tuple[Tensor, ...]") -> "Tuple[Tensor, ...]":
    return tuple(value.clone() for value in state)


def _initial_state(
    codec: "Any",
    wav: "Tensor",
    endpoint_name: "str" = "step",
) -> "Tuple[Tensor, ...]":
    return codec.init_endpoint_state(
        endpoint_name,
        batch_size=wav.shape[0],
        device=wav.device,
        dtype=wav.dtype,
    )


def _example_inputs(
    codec: "Any",
    wav: "Tensor",
    state: "Tuple[Tensor, ...]",
) -> "Dict[str, Any]":
    chunk_size = codec.chunk_size
    chunk = wav[:, :chunk_size]
    chunk = torch.nn.functional.pad(chunk, (0, chunk_size - chunk.shape[1]))
    return {
        "wav": chunk,
        "state": _clone_state(state),
    }


def _run_stream(
    endpoint: "Any",
    codec: "Any",
    wav: "Tensor",
    state: "Tuple[Tensor, ...]",
) -> "Dict[str, Any]":
    chunk_size = codec.chunk_size
    outputs = []

    for offset in range(0, wav.shape[1], chunk_size):
        chunk = wav[:, offset : offset + chunk_size]
        chunk = torch.nn.functional.pad(chunk, (0, chunk_size - chunk.shape[1]))
        inputs = {
            "wav": chunk,
            "state": state,
        }
        result = endpoint(inputs)
        outputs.append(result["wav_rec"])
        state = result["next_state"]

    return {
        "wav_rec": torch.cat(outputs, dim=1)[:, : wav.shape[1]],
        "next_state": state,
    }


def _run_direct(
    codec: "Any",
    wav: "Tensor",
    state: "Tuple[Tensor, ...]",
) -> "Dict[str, Any]":
    def endpoint(inputs: "Dict[str, Any]") -> "Dict[str, Any]":
        wav_rec, next_state = codec.step(inputs["wav"], inputs["state"])
        return {
            "wav_rec": wav_rec,
            "next_state": next_state,
        }

    with torch.no_grad():
        return _run_stream(
            endpoint,
            codec,
            wav,
            state,
        )


def main(
    checkpoint: "str" = DEFAULT_CHECKPOINT,
    audio_path: "Optional[Path]" = None,
    run_compile: "bool" = True,
    run_jit: "bool" = True,
    run_cudagraph: "bool" = True,
    run_onnx: "bool" = True,
    dtype: "torch.dtype" = torch.float32,
) -> "None":
    audio_path = audio_path or DEFAULT_AUDIO_PATH
    output_directory = REPOSITORY_ROOT / "outputs/resynthesis"
    streaming_output_directory = output_directory / "streaming"
    offline_output_directory = output_directory / "offline"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    codec = _load_codec(checkpoint, device, dtype=dtype)
    wav, sample_rate = codec.load_audio(audio_path)
    wav = codec.resample_audio(wav, sample_rate, codec.sample_rate_input)
    wav = wav.to(device=device, dtype=dtype)

    offline_inputs = {"wav": wav}
    with torch.no_grad():
        offline_direct_outputs = {"wav_rec": codec(wav)}
        offline_eager_outputs = codec.eager("forward")(offline_inputs)

    valid_samples = min(wav.shape[-1], offline_direct_outputs["wav_rec"].shape[-1])
    offline_round_trip_rmse = _rmse(
        offline_direct_outputs["wav_rec"][..., :valid_samples],
        wav[..., :valid_samples],
    )
    print("offline input shape:", wav.shape)
    print("offline round-trip shape:", offline_direct_outputs["wav_rec"].shape)
    print("offline valid samples:", valid_samples)
    print("offline audio RMSE:", f"{offline_round_trip_rmse:.3e}")
    codec.save_audio(
        offline_output_directory / "original.wav", wav, codec.sample_rate_input
    )
    codec.save_audio(
        offline_output_directory / "direct.wav",
        offline_direct_outputs["wav_rec"],
        codec.sample_rate_output,
    )
    codec.save_audio(
        offline_output_directory / "eager.wav",
        offline_eager_outputs["wav_rec"],
        codec.sample_rate_output,
    )
    _compare_offline_outputs(
        "offline eager endpoint",
        offline_eager_outputs,
        offline_direct_outputs,
    )

    if run_compile:
        offline_compiled_endpoint = codec.compile("forward")
        with torch.no_grad():
            offline_compiled_outputs = offline_compiled_endpoint(offline_inputs)
        codec.save_audio(
            offline_output_directory / "compile.wav",
            offline_compiled_outputs["wav_rec"],
            codec.sample_rate_output,
        )
        _compare_offline_outputs(
            "offline torch.compile",
            offline_compiled_outputs,
            offline_eager_outputs,
        )

    # The remaining paths use fixed-size step inputs, unlike forward above
    torch.backends.cudnn.benchmark = True
    initial_state = _initial_state(codec, wav)
    direct_outputs = _run_direct(
        codec,
        wav,
        _clone_state(initial_state),
    )
    with torch.no_grad():
        eager_outputs = _run_stream(
            codec.eager("step"),
            codec,
            wav,
            _clone_state(initial_state),
        )

    print("streaming input shape:", wav.shape)
    print("inference dtype:", dtype)
    print("streaming round-trip shape:", direct_outputs["wav_rec"].shape)
    print(
        "streaming audio RMSE:",
        f"{_rmse(direct_outputs['wav_rec'], wav):.3e}",
    )
    print("state tensors:", len(direct_outputs["next_state"]))
    codec.save_audio(
        streaming_output_directory / "original.wav", wav, codec.sample_rate_input
    )
    codec.save_audio(
        streaming_output_directory / "direct.wav",
        direct_outputs["wav_rec"],
        codec.sample_rate_output,
    )
    codec.save_audio(
        streaming_output_directory / "eager.wav",
        eager_outputs["wav_rec"],
        codec.sample_rate_output,
    )
    _compare_outputs(
        "eager endpoint",
        eager_outputs,
        direct_outputs,
        "wav_rec",
    )

    if run_compile:
        compiled_endpoint = codec.compile("step")
        with torch.no_grad():
            compiled_outputs = _run_stream(
                compiled_endpoint,
                codec,
                wav,
                _clone_state(initial_state),
            )
        codec.save_audio(
            streaming_output_directory / "compile.wav",
            compiled_outputs["wav_rec"],
            codec.sample_rate_output,
        )
        _compare_outputs(
            "torch.compile",
            compiled_outputs,
            eager_outputs,
            "wav_rec",
        )

    if run_jit:
        jit_endpoint = codec.jit(
            "step",
            _example_inputs(codec, wav, initial_state),
        )
        with torch.no_grad():
            jit_outputs = _run_stream(
                jit_endpoint,
                codec,
                wav,
                _clone_state(initial_state),
            )
        codec.save_audio(
            streaming_output_directory / "jit.wav",
            jit_outputs["wav_rec"],
            codec.sample_rate_output,
        )
        _compare_outputs("JIT", jit_outputs, eager_outputs, "wav_rec")

    if run_cudagraph:
        if torch.cuda.is_available():
            cuda = torch.device("cuda")
            cuda_codec = codec
            cuda_wav = wav.to(cuda)
            cuda_initial_state = _initial_state(cuda_codec, cuda_wav)
            with torch.no_grad():
                cuda_eager_outputs = _run_stream(
                    cuda_codec.eager("step"),
                    cuda_codec,
                    cuda_wav,
                    _clone_state(cuda_initial_state),
                )
            cudagraph_endpoint = cuda_codec.cudagraph(
                "step",
                _example_inputs(cuda_codec, cuda_wav, cuda_initial_state),
            )
            cudagraph_outputs = _run_stream(
                cudagraph_endpoint,
                cuda_codec,
                cuda_wav,
                _clone_state(cuda_initial_state),
            )
            codec.save_audio(
                streaming_output_directory / "cudagraph.wav",
                cudagraph_outputs["wav_rec"],
                codec.sample_rate_output,
            )
            _compare_outputs(
                "CUDA Graph",
                cudagraph_outputs,
                cuda_eager_outputs,
                "wav_rec",
            )

        else:
            print("CUDA Graph skipped: CUDA is not available")

    if run_onnx:
        if dtype != torch.float32:
            print("ONNX skipped: the CPU example supports only fp32")
            return
        cpu = torch.device("cpu")
        onnx_wav = wav.to(cpu)
        onnx_codec = _load_codec(
            checkpoint,
            cpu,
        )
        onnx_initial_state = _initial_state(onnx_codec, onnx_wav)
        with torch.no_grad():
            onnx_eager_outputs = _run_stream(
                onnx_codec.eager("step"),
                onnx_codec,
                onnx_wav,
                _clone_state(onnx_initial_state),
            )
        onnx_endpoint = onnx_codec.onnx(
            "step",
            _example_inputs(onnx_codec, onnx_wav, onnx_initial_state),
            providers=("CPUExecutionProvider",),
        )
        onnx_outputs = _run_stream(
            onnx_endpoint,
            onnx_codec,
            onnx_wav,
            _clone_state(onnx_initial_state),
        )
        codec.save_audio(
            streaming_output_directory / "onnx.wav",
            onnx_outputs["wav_rec"],
            codec.sample_rate_output,
        )
        _compare_outputs(
            "ONNX",
            onnx_outputs,
            onnx_eager_outputs,
            "wav_rec",
        )


if __name__ == "__main__":
    dtypes = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "bfp16": torch.bfloat16,
    }
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help=(
            "Local config path or Hugging Face repository "
            f"(default: {DEFAULT_CHECKPOINT})."
        ),
    )
    parser.add_argument(
        "--audio",
        type=Path,
        default=None,
        help=(
            "Input mono WAV file, resampled to 16 kHz when necessary "
            f"(default: {DEFAULT_AUDIO_PATH})."
        ),
    )
    parser.add_argument("--skip-compile", action="store_true")
    parser.add_argument("--skip-jit", action="store_true")
    parser.add_argument("--skip-cudagraph", action="store_true")
    parser.add_argument("--skip-onnx", action="store_true")
    parser.add_argument(
        "--dtype",
        choices=tuple(dtypes),
        default="fp32",
        help="Inference dtype; bfp16 is accepted as an alias for bf16",
    )
    args = parser.parse_args()
    main(
        checkpoint=args.checkpoint,
        audio_path=args.audio,
        run_compile=not args.skip_compile,
        run_jit=not args.skip_jit,
        run_cudagraph=not args.skip_cudagraph,
        run_onnx=not args.skip_onnx,
        dtype=dtypes[args.dtype],
    )
