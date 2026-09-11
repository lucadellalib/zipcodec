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

"""Patching layers."""

import torch
from torch import Tensor, nn


__all__ = ["Patch1d", "Unpatch1d"]


class Patch1d(nn.Module):
    """1D patching layer.

    Parameters
    ----------
    input_dim:
        Dimension of input features.
    output_dim:
        Dimension of output features.
    patch_size:
        Number of consecutive frames grouped into one patch.

    """

    def __init__(
        self,
        input_dim: "int" = 80,
        output_dim: "int" = 2048,
        patch_size: "int" = 16,
    ) -> "None":
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.patch_size = patch_size

        # Modules
        self.proj = nn.Linear(input_dim * patch_size, output_dim)

    def forward(self, input: "Tensor") -> "Tensor":
        """Forward pass.

        Parameters
        ----------
        input:
            Input of shape (batch_size, seq_length, input_dim).

        Returns
        -------
            Output of shape
            (batch_size, ceil(seq_length / patch_size), output_dim).

        """
        B, T, C = input.shape
        P = self.patch_size

        pad_len = (P - T % P) % P
        input = nn.functional.pad(
            input,
            (0, 0, 0, pad_len),
            value=-13.815510557964274,  # log(1e-6)
        )

        output = input.reshape(
            B,
            (T + pad_len) // P,
            P * C,
        )

        output = self.proj(output)
        return output


class Unpatch1d(nn.Module):
    """1D unpatching layer.

    Parameters
    ----------
    input_dim:
        Dimension of input features.
    output_dim:
        Dimension of output features.
    patch_size:
        Number of consecutive frames reconstructed from one patch.

    """

    def __init__(
        self,
        input_dim: "int" = 2048,
        output_dim: "int" = 1024,
        patch_size: "int" = 8,
    ) -> "None":
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.patch_size = patch_size

        # Modules
        self.proj = nn.Linear(input_dim, output_dim * patch_size)

    def forward(self, input: "Tensor") -> "Tensor":
        """Forward pass.

        Parameters
        ----------
        input:
            Input of shape (batch_size, seq_length, input_dim).

        Returns
        -------
            Output of shape
            (batch_size, seq_length * patch_size, output_dim).

        """
        B, T, _ = input.shape
        P = self.patch_size

        output = self.proj(input)
        output = output.reshape(B, T * P, self.output_dim)
        return output


def _assert_allclose(x: "Tensor", y: "Tensor") -> "None":
    assert torch.allclose(x, y, atol=1e-5), (((x - y) ** 2).mean().sqrt(),)


def _test_batch_invariance(model: "nn.Module", input: "Tensor") -> "None":
    model.eval()

    batch_output = model(input)

    single_output = torch.cat(
        [model(input[i : i + 1]) for i in range(input.shape[0])],
        dim=0,
    )

    _assert_allclose(batch_output, single_output)


def _test_jit(model: "nn.Module", input: "Tensor") -> "None":
    model.eval()

    model_jit = torch.jit.script(model)

    output = model(input)
    output_jit = model_jit(input)

    _assert_allclose(output, output_jit)


@torch.no_grad()
def _test_cuda_graph(model: "nn.Module", input: "Tensor") -> "None":
    if not torch.cuda.is_available():
        print("CUDA graph test skipped")
        return

    model.eval()

    static_input = input.clone()

    for _ in range(3):
        model(static_input)

    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()

    with torch.cuda.graph(graph):
        static_output = model(static_input)

    new_input = torch.randn_like(static_input)
    static_input.copy_(new_input)

    graph.replay()
    graph_output = static_output.clone()

    eager_output = model(new_input)

    _assert_allclose(graph_output, eager_output)


@torch.no_grad()
def _test_onnx(model: "nn.Module", input: "Tensor") -> "None":
    import io
    import warnings

    try:
        import onnxruntime as ort
    except ImportError:
        warnings.warn("`pip install onnxruntime` to test ONNX", stacklevel=1)
        return

    model.eval()

    output = model(input)

    f = io.BytesIO()
    torch.onnx.export(
        model,
        input,
        f,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch", 1: "time"},
            "output": {0: "batch", 1: "time"},
        },
    )

    session = ort.InferenceSession(f.getvalue())
    output_ort = session.run(
        None,
        {"input": input.cpu().numpy()},
    )[0]

    output_ort = torch.tensor(
        output_ort,
        device=input.device,
        dtype=output.dtype,
    )

    _assert_allclose(output, output_ort)


def test_patch1d() -> "None":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B = 3
    T = 101

    model = Patch1d().to(device)

    input = torch.randn(B, T, model.input_dim, device=device)
    output = model(input)

    expected_T = (T + model.patch_size - 1) // model.patch_size

    assert output.shape == (B, expected_T, model.output_dim)

    output.sum().backward()
    for k, v in model.named_parameters():
        assert v.grad is not None, k

    _test_batch_invariance(model, input)
    _test_jit(model, input)
    _test_cuda_graph(model, input)
    _test_onnx(model, input)

    print("Patch1d test passed")


def test_unpatch1d() -> "None":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B = 3
    T = 17

    model = Unpatch1d().to(device)

    input = torch.randn(B, T, model.input_dim, device=device)
    output = model(input)

    assert output.shape == (
        B,
        T * model.patch_size,
        model.output_dim,
    )

    output.sum().backward()
    for k, v in model.named_parameters():
        assert v.grad is not None, k

    _test_batch_invariance(model, input)
    _test_jit(model, input)
    _test_cuda_graph(model, input)
    _test_onnx(model, input)

    print("Unpatch1d test passed")


if __name__ == "__main__":
    test_patch1d()
    test_unpatch1d()
