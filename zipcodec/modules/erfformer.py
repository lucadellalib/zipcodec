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

"""ErfFormer: a causal transformer with DynamicErf activations (see https://arxiv.org/abs/2512.10938)
and no positional encodings.

"""

from typing import List, Optional, Tuple, Union

import torch
from torch import Size, Tensor, nn


__all__ = ["ErfFormer"]


try:
    nn.functional.scaled_dot_product_attention(
        *torch.empty(3, 1, 1, 1), enable_gqa=True
    )
    HAS_ENABLE_GQA = True
except Exception:
    HAS_ENABLE_GQA = False


class FeedForward(nn.Module):
    """Feed-forward layer.

    Parameters
    ----------
    dim:
        Dimension of input/output features.
    ffn_dim:
        Dimension of feed-forward features.
    dropout:
        Dropout probability.

    """

    def __init__(
        self,
        dim: "int" = 2048,
        ffn_dim: "int" = 2048 * 4,
        dropout: "float" = 0.0,
    ) -> "None":
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.dropout_ = dropout

        # Modules
        self.in_proj = nn.Linear(dim, 2 * ffn_dim, bias=False)
        self.activation = nn.SiLU()
        self.out_proj = nn.Linear(ffn_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input: "Tensor") -> "Tensor":
        """Forward pass.

        Parameters
        ----------
        input:
            Input of shape (..., dim).

        Returns
        -------
            Output of shape (..., dim).

        """
        gate, value = self.in_proj(input).chunk(2, dim=-1)
        output = self.activation(gate) * value
        output = self.out_proj(output)
        output = self.dropout(output)
        return output


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


class GroupedQueryAttention(nn.Module):
    """Grouped-query attention layer.

    Parameters
    ----------
    dim:
        Dimension of input/output features.
    num_heads:
        Number of attention heads for the queries.
    num_kv_heads:
        Number of attention heads for the keys and values.
    dropout:
        Dropout probability.

    """

    def __init__(
        self,
        dim: "int" = 2048,
        num_heads: "int" = 16,
        num_kv_heads: "int" = 4,
        dropout: "float" = 0.0,
    ) -> "None":
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.dropout = dropout

        self.head_dim = dim // num_heads
        self.num_kv_head_reps = num_heads // num_kv_heads

        # Modules
        self.q_proj = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.kv_proj = nn.Linear(dim, num_kv_heads * self.head_dim * 2, bias=False)
        self.out_proj = nn.Linear(num_heads * self.head_dim, dim, bias=False)

    def forward(self, input: "Tensor") -> "Tensor":
        """Forward pass.

        Parameters
        ----------
        input:
            Input of shape (batch_size, seq_length, dim).

        Returns
        -------
            Output of shape (batch_size, seq_length, dim).

        """
        B, T, _ = input.shape

        qs = (
            self.q_proj(input)
            .reshape(B, T, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        kvs = (
            self.kv_proj(input)
            .reshape(B, T, self.num_kv_heads, self.head_dim, 2)
            .transpose(1, 2)
        )

        ks = kvs[..., 0]
        vs = kvs[..., 1]

        output = self._grouped_query_attention(
            qs,
            ks,
            vs,
            attn_mask=None,
            is_causal=True,
        )

        output = (
            output.transpose(1, 2)
            .contiguous()
            .reshape(B, T, self.num_heads * self.head_dim)
        )
        output = self.out_proj(output)
        return output

    @torch.jit.export
    def step(
        self,
        input: "Tensor",
        kv_cache: "Tensor",
        write_idx: "Tensor",
        attn_mask: "Tensor",
    ) -> "Tuple[Tensor, Tensor]":
        """Streaming forward pass.

        Parameters
        ----------
        input:
            Input of shape (batch_size, seq_length, dim).
        kv_cache:
            Per-layer fixed-size KV cache of shape
            (batch_size, num_kv_heads, window_size, head_dim, 2).
        write_idx:
            Write indices of shape (seq_length,).
        attn_mask:
            Attention mask of shape (..., seq_length, window_size).

        Returns
        -------
            - Output of shape (batch_size, seq_length, dim);
            - updated per-layer KV cache.

        """
        B, T, _ = input.shape

        qs = (
            self.q_proj(input)
            .reshape(B, T, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        kvs = (
            self.kv_proj(input)
            .reshape(B, T, self.num_kv_heads, self.head_dim, 2)
            .transpose(1, 2)
            .contiguous()
            .to(dtype=kv_cache.dtype)
        )

        kv_cache = kv_cache.index_copy(2, write_idx, kvs)

        ks = kv_cache[..., 0]
        vs = kv_cache[..., 1]

        output = self._grouped_query_attention(
            qs,
            ks,
            vs,
            attn_mask=attn_mask,
            is_causal=False,
        )

        output = output.transpose(1, 2).contiguous().reshape(B, T, -1)
        output = self.out_proj(output)
        return output, kv_cache

    @torch.jit.export
    def step_desync(
        self,
        input: "Tensor",
        kv_cache: "Tensor",
        write_idx: "Tensor",
        attn_mask: "Tensor",
        exec_mask: "Tensor",
    ) -> "Tuple[Tensor, Tensor]":
        """Streaming forward pass for desynchronized streams.

        Parameters
        ----------
        input:
            Input of shape (batch_size, seq_length, dim).
        kv_cache:
            Per-layer fixed-size KV cache of shape
            (batch_size, num_kv_heads, window_size, head_dim, 2).
        write_idx:
            Per-stream write indices of shape (batch_size, seq_length).
        attn_mask:
            Attention mask of shape (batch_size, ..., seq_length, window_size).
        exec_mask:
            Execution mask of shape (batch_size,). True means the stream
            advances; False means the KV cache is kept unchanged.

        Returns
        -------
            - Output of shape (batch_size, seq_length, dim);
            - updated per-layer KV cache.

        """
        B, T, _ = input.shape

        qs = (
            self.q_proj(input)
            .reshape(B, T, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        kvs = (
            self.kv_proj(input)
            .reshape(B, T, self.num_kv_heads, self.head_dim, 2)
            .transpose(1, 2)
            .contiguous()
            .to(dtype=kv_cache.dtype)
        )

        scatter_idx = write_idx[:, None, :, None, None]
        scatter_idx = scatter_idx.expand(-1, self.num_kv_heads, -1, self.head_dim, 2)

        old_kvs = kv_cache.gather(2, scatter_idx)
        kvs = torch.where(
            exec_mask[:, None, None, None, None],
            kvs,
            old_kvs,
        )

        kv_cache = kv_cache.scatter(2, scatter_idx, kvs)

        ks = kv_cache[..., 0]
        vs = kv_cache[..., 1]

        output = self._grouped_query_attention(
            qs,
            ks,
            vs,
            attn_mask=attn_mask,
            is_causal=False,
        )

        output = output.transpose(1, 2).contiguous().reshape(B, T, -1)
        output = self.out_proj(output)
        return output, kv_cache

    @torch.jit.export
    def _grouped_query_attention(
        self,
        query: "Tensor",
        key: "Tensor",
        value: "Tensor",
        attn_mask: "Optional[Tensor]" = None,
        is_causal: "bool" = False,
    ) -> "Tensor":
        return nn.functional.scaled_dot_product_attention(
            query,
            key.repeat_interleave(self.num_kv_head_reps, dim=1),
            value.repeat_interleave(self.num_kv_head_reps, dim=1),
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )


class GroupedQueryAttentionNative(GroupedQueryAttention):
    """See documentation of `GroupedQueryAttention`."""

    @torch.jit.export
    def _grouped_query_attention(
        self,
        query: "Tensor",
        key: "Tensor",
        value: "Tensor",
        attn_mask: "Optional[Tensor]" = None,
        is_causal: "bool" = False,
    ) -> "Tensor":
        return nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
            enable_gqa=True,
        )


class ErfFormerLayer(nn.Module):
    """ErfFormer layer.

    Parameters
    ----------
    dim:
        Dimension of input/output features.
    num_heads:
        Number of attention heads for the queries.
    num_kv_heads:
        Number of attention heads for the keys and values.
    dropout:
        Dropout probability.
    erfscale_init:
        Initial value for erf scaling parameter.
    shift_init:
        Initial value for erf shift parameter.
    native_gqa:
        Whether to use native grouped-query attention when available.

    """

    def __init__(
        self,
        dim: "int" = 2048,
        ffn_dim: "int" = 2048 * 4,
        num_heads: "int" = 16,
        num_kv_heads: "int" = 4,
        dropout: "float" = 0.0,
        erfscale_init: "float" = 0.5,
        shift_init: "float" = 0.0,
        native_gqa: "bool" = True,
    ) -> "None":
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.dropout = dropout
        self.erfscale_init = erfscale_init
        self.shift_init = shift_init
        self.native_gqa = native_gqa

        # Modules
        self.attention = (
            GroupedQueryAttentionNative
            if native_gqa and HAS_ENABLE_GQA
            else GroupedQueryAttention
        )(dim, num_heads, num_kv_heads, dropout)
        self.attention_norm = DynamicErf(dim, erfscale_init, shift_init)
        self.feed_forward = FeedForward(dim, ffn_dim, dropout)
        self.feed_forward_norm = DynamicErf(dim, erfscale_init, shift_init)

    def forward(self, input: "Tensor") -> "Tensor":
        """Forward pass.

        Parameters
        ----------
        input:
            Input of shape (batch_size, seq_length, dim).

        Returns
        -------
            Output of shape (batch_size, seq_length, dim).

        """
        hidden = self.attention(self.attention_norm(input))
        hidden += input
        output = self.feed_forward_norm(hidden)
        output = self.feed_forward(output)
        output += hidden
        return output

    @torch.jit.export
    def step(
        self,
        input: "Tensor",
        kv_cache: "Tensor",
        write_idx: "Tensor",
        attn_mask: "Tensor",
    ) -> "Tuple[Tensor, Tensor]":
        """Streaming forward pass.

        Parameters
        ----------
        input:
            Input of shape (batch_size, seq_length, dim).
        kv_cache:
            Per-layer fixed-size KV cache of shape
            (batch_size, num_kv_heads, window_size, head_dim, 2).
        write_idx:
            Write indices of shape (seq_length,).
        attn_mask:
            Attention mask of shape (..., seq_length, window_size).

        Returns
        -------
            - Output of shape (batch_size, seq_length, dim);
            - updated per-layer KV cache.

        """
        hidden, kv_cache = self.attention.step(
            self.attention_norm(input),
            kv_cache,
            write_idx,
            attn_mask,
        )
        hidden += input
        output = self.feed_forward_norm(hidden)
        output = self.feed_forward(output)
        output += hidden
        return output, kv_cache

    @torch.jit.export
    def step_desync(
        self,
        input: "Tensor",
        kv_cache: "Tensor",
        write_idx: "Tensor",
        attn_mask: "Tensor",
        exec_mask: "Tensor",
    ) -> "Tuple[Tensor, Tensor]":
        """Streaming forward pass for desynchronized streams.

        Parameters
        ----------
        input:
            Input of shape (batch_size, seq_length, dim).
        kv_cache:
            Per-layer fixed-size KV cache of shape
            (batch_size, num_kv_heads, window_size, head_dim, 2).
        write_idx:
            Per-stream write indices of shape (batch_size, seq_length).
        attn_mask:
            Attention mask of shape (batch_size, ..., seq_length, window_size).
        exec_mask:
            Execution mask of shape (batch_size,). True means the stream
            advances; False means the KV cache is kept unchanged.

        Returns
        -------
            - Output of shape (batch_size, seq_length, dim);
            - updated per-layer KV cache.

        """
        hidden, kv_cache = self.attention.step_desync(
            self.attention_norm(input),
            kv_cache,
            write_idx,
            attn_mask,
            exec_mask,
        )
        hidden += input
        output = self.feed_forward_norm(hidden)
        output = self.feed_forward(output)
        output += hidden
        return output, kv_cache


class ErfFormer(nn.Module):
    """ErfFormer model.

    Parameters
    ----------
    dim:
        Dimension of input/output features.
    ffn_dim:
        Dimension of feed-forward features.
    num_layers:
        Number of layers.
    num_heads:
        Number of attention heads for the queries.
    num_kv_heads:
        Number of attention heads for the keys and values.
    dropout:
        Dropout probability.
    window_size:
        Window size.
    erfscale_init:
        Initial value for erf scaling parameter.
    shift_init:
        Initial value for erf shift parameter.
    native_gqa:
        Whether to use native grouped-query attention when available.

    """

    def __init__(
        self,
        dim: "int" = 2048,
        ffn_dim: "int" = 2048 * 4,
        num_layers: "int" = 6,
        num_heads: "int" = 16,
        num_kv_heads: "int" = 4,
        dropout: "float" = 0.0,
        window_size: "int" = 256,
        erfscale_init: "float" = 0.5,
        shift_init: "float" = 0.0,
        native_gqa: "bool" = True,
    ) -> "None":
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.dropout = dropout
        self.window_size = window_size
        self.erfscale_init = erfscale_init
        self.shift_init = shift_init
        self.native_gqa = native_gqa

        self.head_dim = dim // num_heads

        self.layers = nn.ModuleList(
            [
                ErfFormerLayer(
                    dim=dim,
                    ffn_dim=ffn_dim,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    dropout=dropout,
                    erfscale_init=erfscale_init,
                    shift_init=shift_init,
                    native_gqa=native_gqa,
                )
                for _ in range(num_layers)
            ]
        )
        self.register_buffer(
            "window_idx",
            torch.arange(window_size, dtype=torch.long),
            persistent=False,
        )

    @torch.jit.export
    def init_state(
        self,
        batch_size: "int",
        device: "torch.device",
        dtype: "torch.dtype",
    ) -> "Tuple[Tensor, Tensor]":
        """Initialize streaming state.

        Parameters
        ----------
        batch_size:
            Batch size for the streaming state.
        device:
            Device on which to allocate the state.
        dtype:
            Floating-point dtype used for the KV cache.

        Returns
        -------
            - Current logical stream position, initialized to 0, as a scalar;
            - global ring KV cache of shape
              (num_layers, batch_size, num_kv_heads, window_size, head_dim, 2).

        """
        offset = torch.tensor(0, device=device, dtype=torch.long)

        kv_cache = torch.zeros(
            self.num_layers,
            batch_size,
            self.num_kv_heads,
            self.window_size,
            self.head_dim,
            2,
            device=device,
            dtype=dtype,
        )

        return offset, kv_cache

    @torch.jit.export
    def init_state_desync(
        self,
        batch_size: "int",
        device: "torch.device",
        dtype: "torch.dtype",
    ) -> "Tuple[Tensor, Tensor]":
        """Initialize desynchronized streaming state.

        Parameters
        ----------
        batch_size:
            Batch size for the streaming state.
        device:
            Device on which to allocate the state.
        dtype:
            Floating-point dtype used for the KV cache.

        Returns
        -------
            - Current logical stream position, initialized to 0,
              of shape (batch_size,);
            - global ring KV cache of shape
              (num_layers, batch_size, num_kv_heads, window_size, head_dim, 2).

        """
        offset = torch.zeros(batch_size, device=device, dtype=torch.long)

        kv_cache = torch.zeros(
            self.num_layers,
            batch_size,
            self.num_kv_heads,
            self.window_size,
            self.head_dim,
            2,
            device=device,
            dtype=dtype,
        )

        return offset, kv_cache

    def forward(self, input: "Tensor") -> "Tensor":
        """Forward pass.

        Parameters
        ----------
        input:
            Input of shape (batch_size, seq_length, dim).

        Returns
        -------
            Output of shape (batch_size, seq_length, dim).

        """
        B, T, D = input.shape
        W = self.window_size

        # if T <= W:
        #    output = input
        #    for layer in self.layers:
        #        output = layer(output)
        #    return output

        # Pad to a multiple of W
        pad = (-T) % W
        input = nn.functional.pad(input, (0, 0, 0, pad))

        T_pad = input.shape[1]
        N = T_pad // W

        # [B, N, W, D] -> [B * N, W, D]
        output = input.reshape(B, N, W, D)
        output = output.reshape(B * N, W, D)

        # Run each W-sized block independently
        for layer in self.layers:
            output = layer(output)

        # [B * N, W, D] -> [B, T_pad, D]
        output = output.reshape(B, N, W, D)
        output = output.reshape(B, T_pad, D)

        # Remove padding
        output = output[:, :T]

        return output

    @torch.jit.export
    def step(
        self,
        input: "Tensor",
        offset: "Tensor",
        kv_cache: "Tensor",
    ) -> "Tuple[Tensor, Tensor, Tensor]":
        """Streaming forward pass.

        Parameters
        ----------
        input:
            Input of shape (batch_size, 1, dim).
        offset:
            Current logical stream position.
        kv_cache:
            Global fixed-size KV cache of shape
            (num_layers, batch_size, num_kv_heads, window_size, head_dim, 2).

        Returns
        -------
            - Output of shape (batch_size, 1, dim);
            - updated logical stream position;
            - updated global KV cache.

        """
        # B, T, _ = input.shape
        W = self.window_size

        # if not torch.jit.is_tracing() and not torch.onnx.is_in_onnx_export():
        #     if T != 1:
        #         raise ValueError(f"Expected T=1, got T={T}")

        kv_cache = kv_cache.to(device=input.device, dtype=input.dtype)

        write_pos = offset.remainder(W)
        write_idx = write_pos.reshape(1)

        src = self.window_idx
        src_len = write_pos + 1
        attn_mask = src < src_len
        attn_mask = attn_mask[None, None, None, :]

        output = input
        next_kv_cache = []

        for i, layer in enumerate(self.layers):
            output, layer_kv_cache = layer.step(
                output,
                kv_cache[i],
                write_idx,
                attn_mask,
            )
            next_kv_cache.append(layer_kv_cache)

        return output, offset + 1, torch.stack(next_kv_cache)

    @torch.jit.export
    def step_desync(
        self,
        input: "Tensor",
        offset: "Tensor",
        kv_cache: "Tensor",
        exec_mask: "Tensor",
    ) -> "Tuple[Tensor, Tensor, Tensor]":
        """Streaming forward pass for desynchronized streams.

        Parameters
        ----------
        input:
            Input of shape (batch_size, 1, dim).
        offset:
            Current logical stream position of shape (batch_size,).
        kv_cache:
            Global fixed-size KV cache of shape
            (num_layers, batch_size, num_kv_heads, window_size, head_dim, 2).
        exec_mask:
            Execution mask of shape (batch_size,). True means the stream
            advances; False means the KV cache is kept unchanged.

        Returns
        -------
            - Output of shape (batch_size, 1, dim);
            - updated logical stream position;
            - updated global KV cache.

        """
        B, _, _ = input.shape
        W = self.window_size

        # if not torch.jit.is_tracing() and not torch.onnx.is_in_onnx_export():
        #     if T != 1:
        #         raise ValueError(f"Expected T=1, got T={T}")

        kv_cache = kv_cache.to(device=input.device, dtype=input.dtype)

        write_pos = offset.remainder(W)
        write_idx = write_pos.reshape(B, 1)

        # Per-row valid prefix: [0, ..., write_pos[b]]
        src = self.window_idx[None, :]
        src_len = write_pos + 1
        attn_mask = src < src_len[:, None]
        attn_mask = attn_mask[:, None, None, :]

        output = input
        next_kv_cache = []

        for i, layer in enumerate(self.layers):
            output, layer_kv_cache = layer.step_desync(
                output,
                kv_cache[i],
                write_idx,
                attn_mask,
                exec_mask,
            )
            next_kv_cache.append(layer_kv_cache)

        next_offset = torch.where(exec_mask, offset + 1, offset)

        return output, next_offset, torch.stack(next_kv_cache)


def test_model() -> "None":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B = 3
    T = 6

    model = ErfFormer().to(device)

    print(
        f"Model size: {sum(x.numel() for x in model.state_dict().values()) / 1e6:.2f}M"
    )

    input = torch.randn(B, T, model.dim, device=device)
    output = model(input)

    model_jit = torch.jit.script(model)
    output_jit = model_jit(input)

    print(f"Input shape: {input.shape}")
    print(f"Output shape: {output.shape}")

    output.sum().backward()
    for k, v in model.named_parameters():
        assert v.grad is not None, k

    assert torch.allclose(output, output_jit, atol=1e-5), (
        ((output - output_jit) ** 2).mean().sqrt(),
    )

    print("Model test passed")


@torch.no_grad()
def test_batch_invariance() -> "None":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B = 10
    T = 6

    model = ErfFormer().eval().to(device)

    input = torch.randn(B, T, model.dim, device=device)
    batch_output = model(input)

    single_output = []
    for i in range(B):
        output_i = model(input[i : i + 1])
        single_output.append(output_i)

    single_output = torch.cat(single_output, dim=0)

    assert torch.allclose(batch_output, single_output, atol=1e-5), (
        ((batch_output - single_output) ** 2).mean().sqrt(),
    )

    print("Batch invariance test passed")


@torch.no_grad()
def test_streaming() -> "None":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B = 3
    chunk_size = 1
    window_size = 16
    num_chunks = 64
    T = num_chunks * chunk_size

    model = ErfFormer(window_size=window_size).eval().to(device)
    input = torch.randn(B, T, model.dim, device=device)

    # Reference: standard batched forward
    ref_output = model(input)

    # Streaming path: one token at a time from an empty cache
    offset, kv_cache = model.init_state(B, input.device, input.dtype)
    stream_outputs = []

    for i in range(T):
        output_i, offset, kv_cache = model.step(
            input[:, i : i + 1],
            offset,
            kv_cache,
        )
        stream_outputs.append(output_i)

    stream_output = torch.cat(stream_outputs, dim=1)

    assert torch.allclose(ref_output, stream_output, atol=1e-5), (
        ((ref_output - stream_output) ** 2).mean().sqrt(),
    )

    print("Streaming test passed")


@torch.no_grad()
def test_desync() -> "None":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B = 3
    chunk_size = 1
    window_size = 16
    offsets = [window_size, window_size + 3, window_size + 7]
    num_chunks = max(offsets) + 2
    T = num_chunks * chunk_size

    model = ErfFormer(window_size=window_size).eval().to(device)
    input = torch.randn(B, T, model.dim, device=device)

    # Reference: run each stream independently with the synchronized step path
    pre_offsets = []
    pre_kv_caches = []

    single_outputs = []
    single_offsets = []
    single_kv_caches = []

    for b, offset_b in enumerate(offsets):
        offset_b_tensor, kv_cache_b = model.init_state(
            1,
            input.device,
            input.dtype,
        )

        for i in range(offset_b):
            x_i = input[b : b + 1, i : i + 1]
            _, offset_b_tensor, kv_cache_b = model.step(
                x_i,
                offset_b_tensor,
                kv_cache_b,
            )

        pre_offsets.append(offset_b_tensor.clone())
        pre_kv_caches.append(kv_cache_b.clone())

        x_i = input[b : b + 1, offset_b : offset_b + 1]
        output_b, offset_b_tensor, kv_cache_b = model.step(
            x_i,
            offset_b_tensor,
            kv_cache_b,
        )

        single_outputs.append(output_b)
        single_offsets.append(offset_b_tensor)
        single_kv_caches.append(kv_cache_b)

    pre_offset = torch.stack(pre_offsets, dim=0)
    pre_kv_cache = torch.cat(pre_kv_caches, dim=1)

    single_output = torch.cat(single_outputs, dim=0)
    single_offset = torch.stack(single_offsets, dim=0)
    single_kv_cache = torch.cat(single_kv_caches, dim=1)

    x = torch.cat(
        [input[b : b + 1, offsets[b] : offsets[b] + 1] for b in range(B)],
        dim=0,
    )

    # An all-true exec_mask advances every stream
    output, offset, kv_cache = model.step_desync(
        x,
        pre_offset.clone(),
        pre_kv_cache.clone(),
        torch.ones(B, device=device, dtype=torch.bool),
    )

    assert torch.allclose(output, single_output, atol=1e-5), (
        ((output - single_output) ** 2).mean().sqrt(),
    )
    assert torch.equal(offset, single_offset)
    assert torch.allclose(kv_cache, single_kv_cache, atol=1e-5), (
        ((kv_cache - single_kv_cache) ** 2).mean().sqrt(),
    )

    # Desynchronized stream with exec_mask:
    # inactive streams should keep their previous state
    exec_mask = torch.tensor([True, False, True], device=device)

    output_masked, offset_masked, kv_cache_masked = model.step_desync(
        x,
        pre_offset.clone(),
        pre_kv_cache.clone(),
        exec_mask,
    )

    expected_offset = torch.where(exec_mask, pre_offset + 1, pre_offset)

    assert torch.equal(offset_masked, expected_offset)
    assert torch.allclose(
        kv_cache_masked[:, ~exec_mask],
        pre_kv_cache[:, ~exec_mask],
        atol=1e-5,
    )
    assert torch.allclose(
        kv_cache_masked[:, exec_mask],
        single_kv_cache[:, exec_mask],
        atol=1e-5,
    )

    # Outputs for inactive streams are intentionally ignored, not zeroed
    assert output_masked.shape == output.shape

    print("Desync test passed")


def test_jit() -> "None":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B = 3
    chunk_size = 1
    window_size = 16
    num_chunks = 32
    T = num_chunks * chunk_size

    model = ErfFormer(window_size=window_size).eval().to(device)
    model_jit = torch.jit.script(model)

    input = torch.randn(B, T, model.dim, device=device)
    input_forward = input[:, :window_size]

    # Normal forward
    output = model(input_forward)
    output_jit = model_jit(input_forward)

    assert torch.allclose(output, output_jit, atol=1e-5), (
        ((output - output_jit) ** 2).mean().sqrt(),
    )

    # Synchronized streaming from an empty cache
    offset, kv_cache = model.init_state(B, input.device, input.dtype)
    offset_jit, kv_cache_jit = model_jit.init_state(
        B,
        input.device,
        input.dtype,
    )

    for i in range(window_size + 1):
        x_i = input[:, i : i + 1]

        output, next_offset, next_kv_cache = model.step(
            x_i,
            offset,
            kv_cache,
        )
        output_jit, next_offset_jit, next_kv_cache_jit = model_jit.step(
            x_i,
            offset_jit,
            kv_cache_jit,
        )

        assert torch.allclose(output, output_jit, atol=1e-5), (
            ((output - output_jit) ** 2).mean().sqrt(),
        )
        assert torch.equal(next_offset, next_offset_jit)
        assert torch.allclose(
            next_kv_cache,
            next_kv_cache_jit,
            atol=1e-5,
        ), (((next_kv_cache - next_kv_cache_jit) ** 2).mean().sqrt(),)

        offset, kv_cache = next_offset, next_kv_cache
        offset_jit, kv_cache_jit = next_offset_jit, next_kv_cache_jit

    # Desynchronized streaming. Advance each stream to a different offset
    offsets = [window_size, window_size + 3, window_size + 7]

    pre_offsets = []
    pre_kv_caches = []

    pre_offsets_jit = []
    pre_kv_caches_jit = []

    for b, offset_b in enumerate(offsets):
        offset_b_tensor, kv_cache_b = model.init_state(
            1,
            input.device,
            input.dtype,
        )
        offset_b_tensor_jit, kv_cache_b_jit = model_jit.init_state(
            1,
            input.device,
            input.dtype,
        )

        for i in range(offset_b):
            x_i = input[b : b + 1, i : i + 1]

            _, offset_b_tensor, kv_cache_b = model.step(
                x_i,
                offset_b_tensor,
                kv_cache_b,
            )
            _, offset_b_tensor_jit, kv_cache_b_jit = model_jit.step(
                x_i,
                offset_b_tensor_jit,
                kv_cache_b_jit,
            )

        pre_offsets.append(offset_b_tensor)
        pre_kv_caches.append(kv_cache_b)

        pre_offsets_jit.append(offset_b_tensor_jit)
        pre_kv_caches_jit.append(kv_cache_b_jit)

    offset = torch.stack(pre_offsets, dim=0)
    kv_cache = torch.cat(pre_kv_caches, dim=1)

    offset_jit = torch.stack(pre_offsets_jit, dim=0)
    kv_cache_jit = torch.cat(pre_kv_caches_jit, dim=1)

    x = torch.cat(
        [input[b : b + 1, offsets[b] : offsets[b] + 1] for b in range(B)],
        dim=0,
    )

    # An all-true exec_mask advances every stream
    output, next_offset, next_kv_cache = model.step_desync(
        x,
        offset,
        kv_cache,
        torch.ones(B, device=device, dtype=torch.bool),
    )
    output_jit, next_offset_jit, next_kv_cache_jit = model_jit.step_desync(
        x,
        offset_jit,
        kv_cache_jit,
        torch.ones(B, device=device, dtype=torch.bool),
    )

    assert torch.allclose(output, output_jit, atol=1e-5), (
        ((output - output_jit) ** 2).mean().sqrt(),
    )
    assert torch.equal(next_offset, next_offset_jit)
    assert torch.allclose(next_kv_cache, next_kv_cache_jit, atol=1e-5), (
        ((next_kv_cache - next_kv_cache_jit) ** 2).mean().sqrt(),
    )

    # Desync with exec_mask
    exec_mask = torch.tensor([True, False, True], device=device)

    output, next_offset, next_kv_cache = model.step_desync(
        x,
        offset,
        kv_cache,
        exec_mask,
    )
    output_jit, next_offset_jit, next_kv_cache_jit = model_jit.step_desync(
        x,
        offset_jit,
        kv_cache_jit,
        exec_mask,
    )

    assert torch.allclose(output, output_jit, atol=1e-5), (
        ((output - output_jit) ** 2).mean().sqrt(),
    )
    assert torch.equal(next_offset, next_offset_jit)
    assert torch.allclose(next_kv_cache, next_kv_cache_jit, atol=1e-5), (
        ((next_kv_cache - next_kv_cache_jit) ** 2).mean().sqrt(),
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
    chunk_size = 1
    window_size = 16

    model = ErfFormer(window_size=window_size).eval().to(device)

    offset, kv_cache = model.init_state(B, device, torch.float32)

    # Warm up the cache and kernels through the same step endpoint
    for _ in range(window_size):
        x = torch.randn(B, chunk_size, model.dim, device=device)
        _, offset, kv_cache = model.step(x, offset, kv_cache)

    static_input = torch.randn(B, chunk_size, model.dim, device=device)

    for _ in range(3):
        model.step(static_input, offset, kv_cache)

    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()

    with torch.cuda.graph(graph):
        static_output, static_next_offset, static_kv_cache = model.step(
            static_input,
            offset,
            kv_cache,
        )

    new_input = torch.randn(B, chunk_size, model.dim, device=device)
    static_input.copy_(new_input)

    # Compute eager reference before replay if step updates kv_cache in place
    eager_output, eager_offset, eager_kv_cache = model.step(
        new_input,
        offset.clone(),
        kv_cache.clone(),
    )

    graph.replay()

    graph_output = static_output.clone()
    graph_offset = static_next_offset.clone()
    graph_kv_cache = static_kv_cache.clone()

    assert torch.allclose(graph_output, eager_output, atol=1e-5), (
        ((graph_output - eager_output) ** 2).mean().sqrt(),
    )
    assert torch.equal(graph_offset, eager_offset)
    assert torch.allclose(graph_kv_cache, eager_kv_cache, atol=1e-5), (
        ((graph_kv_cache - eager_kv_cache) ** 2).mean().sqrt(),
    )

    # Desync CUDA graph with exec_mask
    offset, kv_cache = model.init_state(B, device, torch.float32)

    for _ in range(window_size):
        x = torch.randn(B, chunk_size, model.dim, device=device)
        _, offset, kv_cache = model.step(x, offset, kv_cache)

    offset = torch.tensor(
        [window_size, window_size + 3, window_size + 7],
        device=device,
        dtype=torch.long,
    )

    static_input = torch.randn(B, chunk_size, model.dim, device=device)
    static_exec_mask = torch.tensor([True, False, True], device=device)

    for _ in range(3):
        model.step_desync(
            static_input,
            offset,
            kv_cache,
            static_exec_mask,
        )

    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()

    with torch.cuda.graph(graph):
        (
            static_output,
            static_next_offset,
            static_kv_cache,
        ) = model.step_desync(
            static_input,
            offset,
            kv_cache,
            static_exec_mask,
        )

    new_input = torch.randn(B, chunk_size, model.dim, device=device)
    static_input.copy_(new_input)

    eager_output, eager_offset, eager_kv_cache = model.step_desync(
        new_input,
        offset.clone(),
        kv_cache.clone(),
        static_exec_mask.clone(),
    )

    graph.replay()

    graph_output = static_output.clone()
    graph_offset = static_next_offset.clone()
    graph_kv_cache = static_kv_cache.clone()

    assert torch.allclose(graph_output, eager_output, atol=1e-5), (
        ((graph_output - eager_output) ** 2).mean().sqrt(),
    )
    assert torch.equal(graph_offset, eager_offset)
    assert torch.allclose(graph_kv_cache, eager_kv_cache, atol=1e-5), (
        ((graph_kv_cache - eager_kv_cache) ** 2).mean().sqrt(),
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

    class ErfFormerStreamWrapper(nn.Module):
        def __init__(self, model: "ErfFormer") -> "None":
            super().__init__()
            self.model = model

        def forward(
            self,
            input: "Tensor",
            offset: "Tensor",
            kv_cache: "Tensor",
        ) -> "Tuple[Tensor, Tensor, Tensor]":
            return self.model.step(input, offset, kv_cache)

    class ErfFormerStreamDesyncWrapper(nn.Module):
        def __init__(self, model: "ErfFormer") -> "None":
            super().__init__()
            self.model = model

        def forward(
            self,
            input: "Tensor",
            offset: "Tensor",
            kv_cache: "Tensor",
            exec_mask: "Tensor",
        ) -> "Tuple[Tensor, Tensor, Tensor]":
            return self.model.step_desync(
                input,
                offset,
                kv_cache,
                exec_mask,
            )

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B = 3
    chunk_size = 1
    window_size = 16
    num_chunks = 32
    T = num_chunks * chunk_size

    model = (
        ErfFormer(
            num_layers=8,  # Make smaller to not hit the 2 GiB limit
            window_size=window_size,
            native_gqa=False,  # ONNX does not support it
        )
        .eval()
        .to(device)
    )

    input = torch.randn(B, T, model.dim, device=device)
    input_forward = input[:, :window_size]

    # Normal forward ONNX
    output = model(input_forward)

    f = io.BytesIO()
    torch.onnx.export(
        model,
        input_forward,
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
        {"input": input_forward.cpu().numpy()},
    )[0]
    output_ort = torch.tensor(output_ort, device=device, dtype=output.dtype)

    assert torch.allclose(output, output_ort, atol=1e-5), (
        ((output - output_ort) ** 2).mean().sqrt(),
    )

    # Stream ONNX. Build cache only through step
    offset, kv_cache = model.init_state(B, input.device, input.dtype)

    for i in range(window_size):
        x_i = input[:, i : i + 1]
        _, offset, kv_cache = model.step(x_i, offset, kv_cache)

    x_i = input[:, window_size : window_size + 1]
    output, next_offset, next_kv_cache = model.step(x_i, offset, kv_cache)

    wrapper = ErfFormerStreamWrapper(model).eval().to(device)

    f = io.BytesIO()
    torch.onnx.export(
        wrapper,
        (x_i, offset, kv_cache),
        f,
        input_names=["input", "offset", "kv_cache"],
        output_names=["output", "next_offset", "next_kv_cache"],
        dynamic_axes={
            "input": {0: "batch"},
            "output": {0: "batch"},
            "kv_cache": {1: "batch"},
            "next_kv_cache": {1: "batch"},
        },
    )

    session = ort.InferenceSession(f.getvalue())
    outputs_ort = session.run(
        None,
        {
            "input": x_i.cpu().numpy(),
            "offset": offset.cpu().numpy(),
            "kv_cache": kv_cache.cpu().numpy(),
        },
    )

    output_ort = torch.tensor(outputs_ort[0], device=device, dtype=output.dtype)
    next_offset_ort = torch.tensor(
        outputs_ort[1],
        device=device,
        dtype=next_offset.dtype,
    )
    next_kv_cache_ort = torch.tensor(
        outputs_ort[2],
        device=device,
        dtype=next_kv_cache.dtype,
    )

    assert torch.allclose(output, output_ort, atol=1e-5), (
        ((output - output_ort) ** 2).mean().sqrt(),
    )
    assert torch.equal(next_offset, next_offset_ort)
    assert torch.allclose(next_kv_cache, next_kv_cache_ort, atol=1e-5), (
        ((next_kv_cache - next_kv_cache_ort) ** 2).mean().sqrt(),
    )

    # Desync stream ONNX with exec_mask
    offsets = [window_size, window_size + 3, window_size + 7]
    pre_offsets = []
    pre_kv_caches = []

    for b, offset_b in enumerate(offsets):
        offset_b_tensor, kv_cache_b = model.init_state(
            1,
            input.device,
            input.dtype,
        )

        for i in range(offset_b):
            x_b = input[b : b + 1, i : i + 1]
            _, offset_b_tensor, kv_cache_b = model.step(
                x_b,
                offset_b_tensor,
                kv_cache_b,
            )

        pre_offsets.append(offset_b_tensor)
        pre_kv_caches.append(kv_cache_b)

    offset = torch.stack(pre_offsets, dim=0)
    kv_cache = torch.cat(pre_kv_caches, dim=1)

    exec_mask = torch.tensor([True, False, True], device=device)

    x_i = torch.cat(
        [input[b : b + 1, offsets[b] : offsets[b] + 1] for b in range(B)],
        dim=0,
    )

    output, next_offset, next_kv_cache = model.step_desync(
        x_i,
        offset,
        kv_cache,
        exec_mask,
    )

    wrapper = ErfFormerStreamDesyncWrapper(model).eval().to(device)

    f = io.BytesIO()
    torch.onnx.export(
        wrapper,
        (x_i, offset, kv_cache, exec_mask),
        f,
        input_names=["input", "offset", "kv_cache", "exec_mask"],
        output_names=["output", "next_offset", "next_kv_cache"],
        dynamic_axes={
            "input": {0: "batch"},
            "output": {0: "batch"},
            "offset": {0: "batch"},
            "next_offset": {0: "batch"},
            "kv_cache": {1: "batch"},
            "next_kv_cache": {1: "batch"},
            "exec_mask": {0: "batch"},
        },
    )

    session = ort.InferenceSession(f.getvalue())
    outputs_ort = session.run(
        None,
        {
            "input": x_i.cpu().numpy(),
            "offset": offset.cpu().numpy(),
            "kv_cache": kv_cache.cpu().numpy(),
            "exec_mask": exec_mask.cpu().numpy(),
        },
    )

    output_ort = torch.tensor(outputs_ort[0], device=device, dtype=output.dtype)
    next_offset_ort = torch.tensor(
        outputs_ort[1],
        device=device,
        dtype=next_offset.dtype,
    )
    next_kv_cache_ort = torch.tensor(
        outputs_ort[2],
        device=device,
        dtype=next_kv_cache.dtype,
    )

    assert torch.allclose(output, output_ort, atol=1e-5), (
        ((output - output_ort) ** 2).mean().sqrt(),
    )
    assert torch.equal(next_offset, next_offset_ort)
    assert torch.allclose(next_kv_cache, next_kv_cache_ort, atol=1e-5), (
        ((next_kv_cache - next_kv_cache_ort) ** 2).mean().sqrt(),
    )

    print("ONNX test passed")


if __name__ == "__main__":
    test_model()
    test_batch_invariance()
    test_streaming()
    test_desync()
    test_jit()
    test_cuda_graph()
    test_onnx()
