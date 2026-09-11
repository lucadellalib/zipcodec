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

"""Export a standalone ZipCodec ONNX streaming checkpoint."""

import argparse
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = REPOSITORY_ROOT / "outputs/onnx/zipcodec_step.onnx"
DEFAULT_CHECKPOINT = "lucadellalib/zipcodec"
DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
}


def main(
    checkpoint: "str" = DEFAULT_CHECKPOINT,
    output_path: "Path" = DEFAULT_MODEL_PATH,
    batch_size: "int" = 1,
    dtype_name: "str" = "fp32",
) -> "None":
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    if dtype_name not in DTYPES:
        raise ValueError(f"Unknown dtype: {dtype_name!r}")

    device = torch.device("cpu")
    dtype = DTYPES[dtype_name]
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
    state = codec.init_endpoint_state(
        "step",
        batch_size=batch_size,
        device=device,
        dtype=dtype,
    )
    example_inputs = {
        "wav": torch.zeros(
            batch_size,
            codec.chunk_size,
            device=device,
            dtype=dtype,
        ),
        "state": state,
    }
    model_path, metadata_path, inputs_path = codec.export_onnx(
        "step",
        example_inputs,
        output_path,
    )

    print(f"Exported ONNX model: {model_path}")
    print(f"Exported metadata: {metadata_path}")
    print(f"Exported initial inputs: {inputs_path}")
    print(f"External weights: {model_path.with_suffix(f'{model_path.suffix}.data')}")
    print(f"Inference dtype: {dtype_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export a standalone ZipCodec ONNX streaming checkpoint."
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
        "--output",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="ONNX model path (default: outputs/onnx/zipcodec_step.onnx).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Fixed ONNX batch size (default: 1).",
    )
    parser.add_argument(
        "--dtype",
        choices=tuple(DTYPES),
        default="fp32",
        help="ONNX model and input dtype (default: fp32).",
    )
    args = parser.parse_args()
    main(
        checkpoint=args.checkpoint,
        output_path=args.output,
        batch_size=args.batch_size,
        dtype_name=args.dtype,
    )
