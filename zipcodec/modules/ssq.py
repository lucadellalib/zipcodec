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

"""Scalar spherical quantizer (see https://arxiv.org/abs/2601.23174)."""

import math
from typing import List, Tuple, Union

import torch
from torch import Size, Tensor, nn


__all__ = ["ScalarSphericalQuantizer"]


class DynamicErf(nn.Module):
    """Dynamic erf activation function.

    See https://arxiv.org/abs/2512.10938.

    Parameters
    ----------
    normalized_shape:
        Input shape for normalization.
    erfscale_init:
        Initial value for erf scaling parameter.
    shift_init:
        Initial value for erf shift parameter.

    """

    def __init__(
        self,
        normalized_shape: "Union[int, List[int], Size]",
        erfscale_init: "float" = 0.5,
        shift_init: "float" = 0.0,
    ) -> "None":
        super().__init__()
        self.normalized_shape = normalized_shape
        self.erfscale_init = erfscale_init
        self.shift_init = shift_init

        # Parameters
        self.alpha = nn.Parameter(torch.full((1,), erfscale_init))
        self.shift = nn.Parameter(torch.full((1,), shift_init))
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, input: "Tensor") -> "Tensor":
        """Forward pass.

        Parameters
        ----------
        input:
            Input of shape (..., *normalized_shape).

        Returns
        -------
            Output of shape (..., *normalized_shape).

        """
        output = (self.alpha * input + self.shift).erf()
        output = output * self.weight + self.bias
        return output

    def __repr__(self) -> "str":
        return (
            f"{self.__class__.__name__}("
            f"normalized_shape={self.normalized_shape}, "
            f"erfscale_init={self.erfscale_init}, "
            f"shift_init={self.shift_init})"
        )


class ScalarSphericalQuantizer(nn.Module):
    """Scalar spherical quantizer.

    Parameters
    ----------
    input_dim:
        Dimension of input/output features.
    latent_dim:
        Dimension of the scalar latent code.
    bits_per_dim:
        Number of quantization bits per latent dimension.
    erfscale_init:
        Initial value for erf scaling parameter.
    shift_init:
        Initial value for erf shift parameter.

    """

    def __init__(
        self,
        input_dim: "int" = 2048,
        latent_dim: "int" = 64,
        bits_per_dim: "int" = 2,
        erfscale_init: "float" = 0.5,
        shift_init: "float" = 0.0,
    ) -> "None":
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.bits_per_dim = bits_per_dim
        self.erfscale_init = erfscale_init
        self.shift_init = shift_init

        # Modules
        self.norm = DynamicErf(input_dim, erfscale_init, shift_init)
        self.in_proj = nn.Linear(input_dim, latent_dim)
        self.out_proj = nn.Linear(latent_dim, input_dim)

        # Buffers
        self.n_levels = n_levels = 2**bits_per_dim
        inv_sqrtD = torch.as_tensor(1.0 / math.sqrt(latent_dim))
        step = torch.as_tensor(2.0 / (n_levels - 1))
        scale = torch.as_tensor((n_levels - 1) / (2.0 * inv_sqrtD))
        bias = torch.as_tensor((n_levels - 1) / 2.0)
        self.register_buffer("inv_sqrtD", inv_sqrtD, persistent=False)
        self.register_buffer("scale", scale, persistent=False)
        self.register_buffer("bias", bias, persistent=False)
        self.register_buffer("step", step, persistent=False)

    @property
    def codebook(self) -> "Tensor":
        """Return the scalar spherical codebook.

        Returns
        -------
            Codebook of shape (latent_dim, 2 ** bits_per_dim), where each column
            corresponds to one scalar quantization level replicated across dimensions.

        """
        # Scalar levels in [-1/sqrt(D), +1/sqrt(D)]
        levels = (
            torch.arange(self.n_levels, device=self.inv_sqrtD.device) * self.step - 1.0
        ) * self.inv_sqrtD

        # Expand to full dimensional codes
        return levels[None].expand(self.latent_dim, self.n_levels)

    def forward(self, input: "Tensor") -> "Tuple[Tensor, Tensor]":
        """Forward pass.

        Parameters
        ----------
        input:
            Input of shape (..., input_dim).

        Returns
        -------
            - Output tokens of shape (..., latent_dim);
            - output codes of shape (..., input_dim).

        """
        tokens = self.encode(input)
        codes = self.decode(tokens)
        return tokens, codes

    @torch.jit.export
    def encode(self, input: "Tensor") -> "Tensor":
        """Encode input to tokens.

        Parameters
        ----------
        input:
            Input of shape (..., input_dim).

        Returns
        -------
            Output of shape (..., latent_dim).

        """
        z = self.norm(input)
        z = self.in_proj(z)
        z = nn.functional.normalize(z, dim=-1)

        toks = (z * self.scale + self.bias).round()
        toks = toks.clamp(0, self.n_levels - 1).to(torch.long)

        return toks

    @torch.jit.export
    def decode(self, input: "Tensor") -> "Tensor":
        """Decode codes from tokens.

        Parameters
        ----------
        input:
            Input of shape (..., latent_dim).

        Returns
        -------
            Output of shape (..., input_dim).

        """
        codes = (input.to(dtype=self.step.dtype) * self.step - 1.0) * self.inv_sqrtD
        codes = nn.functional.normalize(codes, dim=-1)

        return self.out_proj(codes)


def test_model() -> "None":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B = 3
    T = 17

    model = ScalarSphericalQuantizer().to(device)

    input = torch.randn(B, T, model.input_dim, device=device)

    tokens, codes = model(input)

    assert tokens.shape == (B, T, model.latent_dim)
    assert codes.shape == input.shape
    assert tokens.dtype == torch.long
    assert tokens.min() >= 0
    assert tokens.max() < model.n_levels

    print("Model test passed")


@torch.no_grad()
def test_batch_invariance() -> "None":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B = 10
    T = 17

    model = ScalarSphericalQuantizer().eval().to(device)

    input = torch.randn(B, T, model.input_dim, device=device)

    batch_tokens, batch_codes = model(input)

    single_tokens = []
    single_codes = []
    for i in range(B):
        tokens_i, codes_i = model(input[i : i + 1])
        single_tokens.append(tokens_i)
        single_codes.append(codes_i)

    single_tokens = torch.cat(single_tokens, dim=0)
    single_codes = torch.cat(single_codes, dim=0)

    assert torch.equal(batch_tokens, single_tokens)
    assert torch.allclose(batch_codes, single_codes, atol=1e-5), (
        ((batch_codes - single_codes) ** 2).mean().sqrt(),
    )

    print("Batch invariance test passed")


def test_jit() -> "None":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B = 3
    T = 17

    model = ScalarSphericalQuantizer().eval().to(device)
    model_jit = torch.jit.script(model)

    input = torch.randn(B, T, model.input_dim, device=device)

    tokens, codes = model(input)
    tokens_jit, codes_jit = model_jit(input)

    assert torch.equal(tokens, tokens_jit)
    assert torch.allclose(codes, codes_jit, atol=1e-5), (
        ((codes - codes_jit) ** 2).mean().sqrt(),
    )

    print("JIT test passed")


@torch.no_grad()
def test_cuda_graph() -> "None":
    if not torch.cuda.is_available():
        print("CUDA graph test skipped")
        return

    torch.manual_seed(0)
    device = torch.device("cuda")

    B = 3
    T = 17

    model = ScalarSphericalQuantizer().eval().to(device)

    static_input = torch.randn(B, T, model.input_dim, device=device)

    for _ in range(3):
        model(static_input)

    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()

    with torch.cuda.graph(graph):
        static_tokens, static_codes = model(static_input)

    new_input = torch.randn_like(static_input)
    static_input.copy_(new_input)

    graph.replay()

    graph_tokens = static_tokens.clone()
    graph_codes = static_codes.clone()

    eager_tokens, eager_codes = model(new_input)

    assert torch.equal(graph_tokens, eager_tokens)
    assert torch.allclose(graph_codes, eager_codes, atol=1e-5), (
        ((graph_codes - eager_codes) ** 2).mean().sqrt(),
    )

    print("CUDA graph test passed")


@torch.no_grad()
def test_onnx() -> "None":
    import io
    import warnings

    try:
        import onnxruntime as ort
    except ImportError:
        warnings.warn("`pip install onnxruntime` to test ONNX", stacklevel=1)
        return

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B = 3
    T = 17

    model = ScalarSphericalQuantizer().eval().to(device)

    input = torch.randn(B, T, model.input_dim, device=device)

    tokens, codes = model(input)

    f = io.BytesIO()
    torch.onnx.export(
        model,
        input,
        f,
        input_names=["input"],
        output_names=["tokens", "codes"],
        dynamic_axes={
            "input": {0: "batch", 1: "time"},
            "tokens": {0: "batch", 1: "time"},
            "codes": {0: "batch", 1: "time"},
        },
    )

    session = ort.InferenceSession(f.getvalue())
    tokens_ort, codes_ort = session.run(
        None,
        {"input": input.cpu().numpy()},
    )

    tokens_ort = torch.tensor(tokens_ort, device=device, dtype=tokens.dtype)
    codes_ort = torch.tensor(codes_ort, device=device, dtype=codes.dtype)

    assert torch.equal(tokens, tokens_ort)
    assert torch.allclose(codes, codes_ort, atol=1e-5), (
        ((codes - codes_ort) ** 2).mean().sqrt(),
    )

    print("ONNX test passed")


if __name__ == "__main__":
    test_model()
    test_batch_invariance()
    test_jit()
    test_cuda_graph()
    test_onnx()
