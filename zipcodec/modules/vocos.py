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

"""Vocos (see https://arxiv.org/abs/2306.00814)."""

from typing import Tuple

import torch
from torch import Tensor, nn


__all__ = ["Vocos"]


class ConvNeXtBlock(nn.Module):
    """ConvNeXt block.

    Parameters
    ----------
    dim:
        Number of input/output channels.
    ffn_dim:
        Number of channels in the pointwise convolution.
    kernel_size:
        Size of the depthwise convolution kernel.
    patch_size:
        Number of consecutive frames grouped into one patch.
    layerscale_init:
        Initial value for layer scaling parameter.
    eps:
        Small constant for numerical stability.

    """

    def __init__(
        self,
        dim: "int" = 1024,
        ffn_dim: "int" = 2048,
        kernel_size: "int" = 7,
        patch_size: "int" = 8,
        layerscale_init: "float" = 0.05,
        eps: "float" = 1e-6,
    ) -> "None":
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.kernel_size = kernel_size
        self.patch_size = patch_size
        self.layerscale_init = layerscale_init
        self.eps = eps

        self.pad = kernel_size // 2
        self.buffer_frames = kernel_size - 1

        # Modules
        self.dwconv = nn.Conv1d(dim, dim, kernel_size, groups=dim)
        self.norm = nn.LayerNorm(dim, eps)
        self.pwconv1 = nn.Linear(dim, ffn_dim)
        self.activation = nn.GELU()
        self.pwconv2 = nn.Linear(ffn_dim, dim)
        self.gamma = nn.Parameter(
            torch.full((dim,), layerscale_init),
        )

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
        B, _, D = input.shape
        buffer = input.new_zeros(B, self.buffer_frames, D)
        return self.prefill(input, buffer)

    @torch.jit.export
    def prefill(
        self,
        input: "Tensor",
        buffer: "Tensor",
    ) -> "Tensor":
        """Streaming forward pass for arbitrary-length input.

        Parameters
        ----------
        input:
            Input of shape (batch_size, seq_length, dim).
        buffer:
            Buffer of shape (batch_size, buffer_frames, dim).

        Returns
        -------
            Output of shape (batch_size, seq_length, dim).

        """
        B, T, D = input.shape

        pad_len = (self.patch_size - T % self.patch_size) % self.patch_size
        input_pad = nn.functional.pad(input, (0, 0, 0, pad_len))

        T_pad = input_pad.shape[1]

        context = torch.cat([buffer, input_pad], dim=1)

        P = self.patch_size
        M = T_pad // P
        W = self.buffer_frames + P

        windows = context.unfold(1, W, P)
        windows = windows.transpose(-1, -2)
        windows = windows.reshape(B * M, W, D)

        y = self._forward_context(windows, P)

        y = y.reshape(B, T_pad, D)
        y = input_pad + y

        return y[:, :T]

    @torch.jit.export
    def step(
        self,
        input: "Tensor",
        buffer: "Tensor",
    ) -> "Tensor":
        """Streaming forward pass.

        Parameters
        ----------
        input:
            Input of shape (batch_size, patch_size, dim).
        buffer:
            Buffer of shape (batch_size, buffer_frames, dim).

        Returns
        -------
            Output of shape (batch_size, patch_size, dim).

        """
        _, T, _ = input.shape

        # if not torch.jit.is_tracing() and not torch.onnx.is_in_onnx_export():
        #     if T != self.patch_size:
        #         raise ValueError(f"Expected seq_length={self.patch_size}, got {T}")

        context = torch.cat([buffer, input], dim=1)
        y = self._forward_context(context, T)

        return input + y

    def _forward_context(
        self,
        context: "Tensor",
        T_out: "int",
    ) -> "Tensor":
        """Apply ConvNeXt block to a context window."""
        y = context.permute(0, 2, 1)
        y = nn.functional.pad(y, [self.pad, self.pad], mode="replicate")
        y = self.dwconv(y)
        y = y.permute(0, 2, 1)

        y = y[:, -T_out:]

        y = self.norm(y)
        y = self.pwconv1(y)
        y = self.activation(y)
        y = self.pwconv2(y)
        y = self.gamma * y

        return y


class VocosBackbone(nn.Module):
    """Vocos backbone.

    Parameters
    ----------
    input_dim:
        Number of input channels.
    num_layers:
        Number of ConvNeXt blocks.
    dim:
        Number of hidden channels.
    ffn_dim:
        Number of channels in the pointwise convolution.
    kernel_size:
        Size of each convolution kernel.
    patch_size:
        Number of consecutive frames grouped into one patch.
    eps:
        Small constant for numerical stability.

    """

    def __init__(
        self,
        input_dim: "int" = 1024,
        num_layers: "int" = 20,
        dim: "int" = 1024,
        ffn_dim: "int" = 2048,
        kernel_size: "int" = 7,
        patch_size: "int" = 8,
        eps: "float" = 1e-6,
    ) -> "None":
        super().__init__()
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.kernel_size = kernel_size
        self.patch_size = patch_size
        self.eps = eps

        self.pad = kernel_size // 2
        self.buffer_frames = kernel_size - 1
        self.embedding_buffer_size = self.buffer_frames * input_dim

        # Modules
        self.embedding = nn.Conv1d(input_dim, dim, kernel_size)
        self.input_norm = nn.LayerNorm(dim, eps)
        self.layers = nn.ModuleList(
            ConvNeXtBlock(
                dim=dim,
                ffn_dim=ffn_dim,
                kernel_size=kernel_size,
                patch_size=patch_size,
                layerscale_init=1 / num_layers,
                eps=eps,
            )
            for _ in range(num_layers)
        )
        self.output_norm = nn.LayerNorm(dim, eps)

        self.layer_buffer_size = self.buffer_frames * dim
        self.buffer_size = (
            self.embedding_buffer_size + num_layers * self.layer_buffer_size
        )

    @torch.jit.export
    def init_state(
        self,
        batch_size: "int",
        device: "torch.device",
        dtype: "torch.dtype",
    ) -> "Tensor":
        """Initialize streaming state.

        Parameters
        ----------
        batch_size:
            Batch size for the streaming state.
        device:
            Device on which to allocate the state.
        dtype:
            Floating-point dtype used for the buffer.

        Returns
        -------
            Buffer of shape
            (batch_size, embedding_buffer_size + num_layers * layer_buffer_size).

        """
        return torch.zeros(batch_size, self.buffer_size, device=device, dtype=dtype)

    @torch.jit.export
    def init_state_desync(
        self,
        batch_size: "int",
        device: "torch.device",
        dtype: "torch.dtype",
    ) -> "Tensor":
        """Initialize desynchronized streaming state.

        Parameters
        ----------
        batch_size:
            Batch size for the streaming state.
        device:
            Device on which to allocate the state.
        dtype:
            Floating-point dtype used for the buffer.

        Returns
        -------
            Buffer of shape
            (batch_size, embedding_buffer_size + num_layers * layer_buffer_size).

        """
        return self.init_state(batch_size, device, dtype)

    def forward(self, input: "Tensor") -> "Tensor":
        """Forward pass.

        Parameters
        ----------
        input:
            Input of shape (batch_size, seq_length, input_dim).

        Returns
        -------
            Output of shape (batch_size, seq_length, dim).

        """
        B, T, _ = input.shape
        buffer = self.init_state(B, input.device, input.dtype)

        pad_len = (self.patch_size - T % self.patch_size) % self.patch_size
        input_pad = nn.functional.pad(input, (0, 0, 0, pad_len))

        output = self._forward_context(input_pad, buffer)

        return output[:, :T]

    @torch.jit.export
    def step(
        self,
        input: "Tensor",
        buffer: "Tensor",
    ) -> "Tuple[Tensor, Tensor]":
        """Streaming forward pass.

        Parameters
        ----------
        input:
            Input of shape (batch_size, patch_size, input_dim).
        buffer:
            Buffer of shape (batch_size, buffer_size).

        Returns
        -------
            - Output of shape (batch_size, patch_size, dim);
            - updated buffer.

        """
        # _, T, _ = input.shape

        # if not torch.jit.is_tracing() and not torch.onnx.is_in_onnx_export():
        #     if T != self.patch_size:
        #        raise ValueError(f"Expected seq_length={self.patch_size}, got {T}")

        return self._forward_context_stream(input, buffer)

    @torch.jit.export
    def step_desync(
        self,
        input: "Tensor",
        buffer: "Tensor",
        exec_mask: "Tensor",
    ) -> "Tuple[Tensor, Tensor]":
        """Streaming forward pass for desynchronized streams.

        Parameters
        ----------
        input:
            Input of shape (batch_size, patch_size, input_dim).
        buffer:
            Buffer of shape (batch_size, buffer_size).
        exec_mask:
            Execution mask of shape (batch_size,). True means the stream
            advances; False means the buffer is kept unchanged.

        Returns
        -------
            - Output of shape (batch_size, patch_size, dim);
            - updated buffer.

        """
        output, next_buffer = self.step(input, buffer)

        next_buffer = torch.where(exec_mask[:, None], next_buffer, buffer)

        return output, next_buffer

    def _forward_context(
        self,
        input: "Tensor",
        buffer: "Tensor",
    ) -> "Tensor":
        """Apply ConvNeXt backbone to an arbitrary-length input sequence."""
        B, T, D = input.shape

        read_offset = 0

        emb_buffer = buffer[
            :, read_offset : read_offset + self.embedding_buffer_size
        ].reshape(B, self.buffer_frames, D)

        read_offset += self.embedding_buffer_size

        context = torch.cat([emb_buffer, input], dim=1)

        P = self.patch_size
        M = T // P
        W = self.buffer_frames + P

        windows = context.unfold(dimension=1, size=W, step=P)
        windows = windows.transpose(-1, -2)
        windows = windows.reshape(B * M, W, D)

        x = windows.permute(0, 2, 1)
        x = nn.functional.pad(x, (self.pad, self.pad), mode="replicate")
        x = self.embedding(x)
        x = x.permute(0, 2, 1)

        x = x[:, -P:]
        x = x.reshape(B, T, self.dim)
        x = x.to(dtype=self.input_norm.weight.dtype)
        x = self.input_norm(x)

        S = self.layer_buffer_size

        for layer in self.layers:
            F = layer.buffer_frames

            layer_buffer = buffer[:, read_offset : read_offset + S].reshape(
                B, F, self.dim
            )

            read_offset += S

            x_in = x
            x = layer.prefill(x_in, layer_buffer)

        x = self.output_norm(x)

        return x

    def _forward_context_stream(
        self,
        input: "Tensor",
        buffer: "Tensor",
    ) -> "Tuple[Tensor, Tensor]":
        """Apply ConvNeXt backbone to a single streaming chunk."""
        B, T, D = input.shape

        read_offset = 0
        next_buffer_parts = []

        emb_buffer = buffer[
            :, read_offset : read_offset + self.embedding_buffer_size
        ].reshape(B, self.buffer_frames, D)

        read_offset += self.embedding_buffer_size

        context = torch.cat([emb_buffer, input], dim=1)
        next_buffer_parts.append(context[:, -self.buffer_frames :].reshape(B, -1))

        x = context.permute(0, 2, 1)
        x = nn.functional.pad(x, (self.pad, self.pad), mode="replicate")
        x = self.embedding(x)
        x = x.permute(0, 2, 1)

        x = x[:, -T:]
        x = x.to(dtype=self.input_norm.weight.dtype)
        x = self.input_norm(x)

        S = self.layer_buffer_size

        for layer in self.layers:
            F = layer.buffer_frames

            layer_buffer = buffer[:, read_offset : read_offset + S].reshape(
                B, F, self.dim
            )

            read_offset += S

            x_in = x
            x = layer.step(x_in, layer_buffer)
            next_buffer_parts.append(x_in[:, -F:].reshape(B, -1))

        x = self.output_norm(x)
        next_buffer = torch.cat(next_buffer_parts, dim=1)

        return x, next_buffer


class ISTFTHead(nn.Module):
    """Inverse STFT head.

    Parameters
    ----------
    dim:
        Number of input channels.
    n_fft:
        FFT size.
    win_length:
        Window size in samples.
    hop_length:
        Hop size between consecutive frames in samples.
    istft_backend:
        iSTFT backend. Options:
        - "fft": faster training/inference backend based on FFT;
        - "conv": ONNX/TensorRT-friendly convolutional backend.

    """

    def __init__(
        self,
        dim: "int" = 1024,
        n_fft: "int" = 1024,
        win_length: "int" = 1024,
        hop_length: "int" = 320,
        istft_backend: "str" = "fft",
    ) -> "None":
        super().__init__()
        self.dim = dim
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.istft_backend = istft_backend
        self.freq_bins = n_fft // 2 + 1
        self.buffer_size = 2 * win_length

        if istft_backend not in ("fft", "conv"):
            raise ValueError(f"istft_backend={istft_backend} must be 'fft' or 'conv'")

        # Modules
        self.proj = nn.Linear(dim, n_fft + 2)

        # Buffers
        window = torch.hann_window(win_length)
        basis_real, basis_imag = self._build_irfft_basis()
        self.register_buffer("window", window, persistent=False)
        self.register_buffer("window_sq", window.square(), persistent=False)
        self.register_buffer("basis_real", basis_real, persistent=False)
        self.register_buffer("basis_imag", basis_imag, persistent=False)

    @torch.jit.export
    def init_state(
        self,
        batch_size: "int",
        device: "torch.device",
        dtype: "torch.dtype",
    ) -> "Tensor":
        """Initialize streaming state.

        Parameters
        ----------
        batch_size:
            Batch size for the streaming state.
        device:
            Device on which to allocate the state.
        dtype:
            Floating-point dtype used for the buffer.

        Returns
        -------
            Buffer of shape (batch_size, 2 * win_length).

        """
        return torch.zeros(
            batch_size,
            self.buffer_size,
            device=device,
            dtype=dtype,
        )

    @torch.jit.export
    def init_state_desync(
        self,
        batch_size: "int",
        device: "torch.device",
        dtype: "torch.dtype",
    ) -> "Tensor":
        """Initialize desynchronized streaming state.

        Parameters
        ----------
        batch_size:
            Batch size for the streaming state.
        device:
            Device on which to allocate the state.
        dtype:
            Floating-point dtype used for the buffer.

        Returns
        -------
            Buffer of shape (batch_size, 2 * win_length).

        """
        return self.init_state(batch_size, device, dtype)

    def forward(self, input: "Tensor") -> "Tensor":
        """Forward pass.

        Parameters
        ----------
        input:
            Input of shape
            (batch_size, seq_length, dim).

        Returns
        -------
            Output of shape (batch_size, seq_length * hop_length).

        """
        input = self.proj(input)

        B, T, _ = input.shape
        H = self.hop_length
        W = self.win_length

        frames = self._irfft(input)
        frames = frames * self.window.to(dtype=frames.dtype)[None, None, :]

        frames = frames.transpose(1, 2)  # [B, W, T]

        output = nn.functional.fold(
            frames,
            output_size=(1, (T - 1) * H + W),
            kernel_size=(1, W),
            stride=(1, H),
        ).squeeze(2)

        norm_frames = self.window_sq.to(dtype=frames.dtype)[None, :, None]
        norm_frames = norm_frames.expand(B, W, T)

        norm = nn.functional.fold(
            norm_frames,
            output_size=(1, (T - 1) * H + W),
            kernel_size=(1, W),
            stride=(1, H),
        ).squeeze(2)

        output = output / norm.clamp_min(1e-8)
        return output[:, 0, : T * H]

    @torch.jit.export
    def step(
        self,
        input: "Tensor",
        buffer: "Tensor",
    ) -> "Tuple[Tensor, Tensor]":
        """Streaming forward pass.

        Parameters
        ----------
        input:
            Input of shape
            (batch_size, seq_length, dim).
        buffer:
            Buffer of shape (batch_size, 2 * win_length).

        Returns
        -------
            - Output of shape (batch_size, seq_length * hop_length);
            - updated buffer.

        """
        input = self.proj(input)

        B, T, _ = input.shape
        H = self.hop_length
        W = self.win_length

        ola = buffer[:, :W]
        norm = buffer[:, W:]

        frames = self._irfft(input)
        frames = frames * self.window.to(dtype=frames.dtype)[None, None, :]

        window_sq = self.window_sq.to(dtype=frames.dtype)[None, :]
        output = input.new_empty(B, T * H)

        for t in range(T):
            ola = ola + frames[:, t]
            norm = norm + window_sq

            start = t * H
            end = start + H
            output[:, start:end] = ola[:, :H] / norm[:, :H].clamp_min(1e-8)

            ola = torch.cat(
                [ola[:, H:], ola.new_zeros(B, H)],
                dim=1,
            )
            norm = torch.cat(
                [norm[:, H:], norm.new_zeros(B, H)],
                dim=1,
            )

        buffer = torch.cat([ola, norm], dim=1)
        return output, buffer

    @torch.jit.export
    def step_desync(
        self,
        input: "Tensor",
        buffer: "Tensor",
        exec_mask: "Tensor",
    ) -> "Tuple[Tensor, Tensor]":
        """Streaming forward pass for desynchronized streams.

        Parameters
        ----------
        input:
            Input of shape
            (batch_size, seq_length, dim).
        buffer:
            Buffer of shape (batch_size, 2 * win_length).
        exec_mask:
            Execution mask of shape (batch_size,). True means the stream
            advances; False means the buffer is kept unchanged.

        Returns
        -------
            - Output of shape (batch_size, seq_length * hop_length);
            - updated buffer.

        """
        output, next_buffer = self.step(input, buffer)

        next_buffer = torch.where(
            exec_mask[:, None],
            next_buffer,
            buffer,
        )

        return output, next_buffer

    def _irfft(self, input: "Tensor") -> "Tensor":
        """Compute inverse FFT frames."""
        mag = input[..., : self.freq_bins].exp().clamp(max=1e2)
        phase = input[..., self.freq_bins :]

        real = mag * phase.cos()
        imag = mag * phase.sin()

        if self.istft_backend == "fft":
            spectrum = torch.complex(real, imag)
            frames = torch.fft.irfft(spectrum, n=self.n_fft, dim=-1)
        else:
            frames = real @ self.basis_real + imag @ self.basis_imag

        return frames[:, :, : self.win_length]

    def _build_irfft_basis(self) -> "Tuple[Tensor, Tensor]":
        """Build real-valued inverse FFT basis."""
        eye = torch.eye(self.freq_bins)

        basis_real = torch.fft.irfft(
            torch.complex(eye, torch.zeros_like(eye)),
            n=self.n_fft,
            dim=-1,
        )
        basis_imag = torch.fft.irfft(
            torch.complex(torch.zeros_like(eye), eye),
            n=self.n_fft,
            dim=-1,
        )

        return basis_real[:, : self.win_length], basis_imag[:, : self.win_length]


class Vocos(nn.Module):
    """Vocos.

    Parameters
    ----------
    input_dim:
        Number of input channels.
    num_layers:
        Number of ConvNeXt blocks.
    dim:
        Number of hidden channels.
    ffn_dim:
        Number of channels in the pointwise convolution.
    kernel_size:
        Size of each convolution kernel.
    patch_size:
        Number of consecutive frames grouped into one patch.
    n_fft:
        FFT size.
    win_length:
        Window size in samples.
    hop_length:
        Hop size between consecutive frames in samples.
    istft_backend:
        iSTFT backend. Options:
        - "fft": faster training/inference backend based on FFT;
        - "conv": ONNX/TensorRT-friendly convolutional backend.
    eps:
        Small constant for numerical stability.

    """

    def __init__(
        self,
        input_dim: "int" = 1024,
        num_layers: "int" = 20,
        dim: "int" = 1024,
        ffn_dim: "int" = 2048,
        kernel_size: "int" = 7,
        patch_size: "int" = 8,
        n_fft: "int" = 1024,
        win_length: "int" = 1024,
        hop_length: "int" = 320,
        istft_backend: "str" = "fft",
        eps: "float" = 1e-6,
    ) -> "None":
        super().__init__()
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.kernel_size = kernel_size
        self.patch_size = patch_size
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.istft_backend = istft_backend
        self.eps = eps

        # Modules
        self.backbone = VocosBackbone(
            input_dim=input_dim,
            num_layers=num_layers,
            dim=dim,
            ffn_dim=ffn_dim,
            kernel_size=kernel_size,
            patch_size=patch_size,
            eps=eps,
        )
        self.head = ISTFTHead(
            dim=dim,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            istft_backend=istft_backend,
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
            Floating-point dtype used for the buffer.

        Returns
        -------
            - Backbone buffer of shape
              (batch_size, embedding_buffer_size + num_layers * layer_buffer_size);
            - head buffer of shape (batch_size, 2 * win_length).

        """
        backbone_buffer = self.backbone.init_state(batch_size, device, dtype)
        head_buffer = self.head.init_state(batch_size, device, dtype)
        return backbone_buffer, head_buffer

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
            Floating-point dtype used for the buffer.

        Returns
        -------
            - Backbone buffer of shape
              (batch_size, embedding_buffer_size + num_layers * layer_buffer_size);
            - head buffer of shape (batch_size, 2 * win_length).

        """
        backbone_buffer = self.backbone.init_state_desync(batch_size, device, dtype)
        head_buffer = self.head.init_state_desync(batch_size, device, dtype)
        return backbone_buffer, head_buffer

    def forward(self, input: "Tensor") -> "Tensor":
        """Forward pass.

        Parameters
        ----------
        input:
            Input of shape (batch_size, seq_length, input_dim).

        Returns
        -------
            Output of shape (batch_size, seq_length * hop_length).

        """
        hidden = self.backbone(input)
        output = self.head(hidden)
        return output

    @torch.jit.export
    def step(
        self,
        input: "Tensor",
        backbone_buffer: "Tensor",
        head_buffer: "Tensor",
    ) -> "Tuple[Tensor, Tensor, Tensor]":
        """Streaming forward pass.

        Parameters
        ----------
        input:
            Input of shape (batch_size, patch_size, input_dim).
        backbone_buffer:
            Backbone buffer of shape
            (batch_size, embedding_buffer_size + num_layers * layer_buffer_size).
        head_buffer:
            Head buffer of shape (batch_size, 2 * win_length).

        Returns
        -------
            - Output of shape (batch_size, patch_size * hop_length);
            - updated backbone buffer;
            - updated head buffer.

        """
        hidden, backbone_buffer = self.backbone.step(
            input,
            backbone_buffer,
        )

        output, head_buffer = self.head.step(
            hidden,
            head_buffer,
        )

        return output, backbone_buffer, head_buffer

    @torch.jit.export
    def step_desync(
        self,
        input: "Tensor",
        backbone_buffer: "Tensor",
        head_buffer: "Tensor",
        exec_mask: "Tensor",
    ) -> "Tuple[Tensor, Tensor, Tensor]":
        """Streaming forward pass for desynchronized streams.

        Parameters
        ----------
        input:
            Input of shape (batch_size, patch_size, input_dim).
        backbone_buffer:
            Backbone buffer of shape
            (batch_size, embedding_buffer_size + num_layers * layer_buffer_size).
        head_buffer:
            Head buffer of shape (batch_size, 2 * win_length).
        exec_mask:
            Execution mask of shape (batch_size,). True means the stream
            advances; False means the buffer is kept unchanged.

        Returns
        -------
            - Output of shape (batch_size, patch_size * hop_length);
            - updated backbone buffer;
            - updated head buffer.

        """
        hidden, backbone_buffer = self.backbone.step_desync(
            input,
            backbone_buffer,
            exec_mask,
        )

        output, head_buffer = self.head.step_desync(
            hidden,
            head_buffer,
            exec_mask,
        )

        return output, backbone_buffer, head_buffer


def test_model() -> "None":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B = 3
    T = 16

    model = Vocos(
        num_layers=2,
        dim=128,
        ffn_dim=256,
        n_fft=256,
        win_length=256,
        hop_length=80,
    ).to(device)

    print(
        f"Model size: {sum(x.numel() for x in model.state_dict().values()) / 1e6:.2f}M"
    )

    input = torch.randn(B, T, model.input_dim, device=device)
    output = model(input)

    model_jit = torch.jit.script(model)
    output_jit = model_jit(input)

    print(f"Input shape: {input.shape}")
    print(f"Output shape: {output.shape}")

    output.sum().backward()
    for k, v in model.named_parameters():
        assert v.grad is not None, k

    assert output.shape == (B, T * model.hop_length)
    assert torch.allclose(output, output_jit, atol=1e-5), (
        ((output - output_jit) ** 2).mean().sqrt(),
    )

    print("Model test passed")


@torch.no_grad()
def test_batch_invariance() -> "None":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B = 10
    T = 16

    model = (
        Vocos(
            num_layers=2,
            dim=128,
            ffn_dim=256,
            n_fft=256,
            win_length=256,
            hop_length=80,
        )
        .eval()
        .to(device)
    )

    input = torch.randn(B, T, model.input_dim, device=device)
    batch_output = model(input)

    single_output = []
    for i in range(B):
        output_i = model(input[i : i + 1])
        single_output.append(output_i)

    single_output = torch.cat(single_output, dim=0)

    assert torch.allclose(batch_output, single_output, atol=1e-2), (
        ((batch_output - single_output) ** 2).mean().sqrt(),
    )

    print("Batch invariance test passed")


@torch.no_grad()
def test_streaming() -> "None":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B = 3
    chunk_size = 8
    num_chunks = 16
    T = num_chunks * chunk_size

    model = (
        Vocos(
            num_layers=2,
            dim=128,
            ffn_dim=256,
            patch_size=chunk_size,
            n_fft=256,
            win_length=256,
            hop_length=80,
        )
        .eval()
        .to(device)
    )

    input = torch.randn(B, T, model.input_dim, device=device)

    # Reference: standard batched forward
    ref_output = model(input)

    # Stream path
    backbone_buffer, head_buffer = model.init_state(B, input.device, input.dtype)
    stream_outputs = []
    for i in range(0, T, chunk_size):
        output_i, backbone_buffer, head_buffer = model.step(
            input[:, i : i + chunk_size],
            backbone_buffer,
            head_buffer,
        )
        stream_outputs.append(output_i)

    stream_output = torch.cat(stream_outputs, dim=1)

    assert torch.allclose(ref_output, stream_output, atol=1e-3), (
        ((ref_output - stream_output) ** 2).mean().sqrt(),
    )

    print("Streaming test passed")


@torch.no_grad()
def test_desync() -> "None":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B = 3
    chunk_size = 8
    offsets = [4, 7, 11]
    num_chunks = max(offsets) + 2
    T = num_chunks * chunk_size

    model = (
        Vocos(
            num_layers=2,
            dim=128,
            ffn_dim=256,
            patch_size=chunk_size,
            n_fft=256,
            win_length=256,
            hop_length=80,
        )
        .eval()
        .to(device)
    )

    input = torch.randn(B, T, model.input_dim, device=device)

    pre_backbone_buffers = []
    pre_head_buffers = []

    single_outputs = []
    single_backbone_buffers = []
    single_head_buffers = []

    for b, offset_b in enumerate(offsets):
        backbone_buffer_b, head_buffer_b = model.init_state(
            1,
            input.device,
            input.dtype,
        )

        for i in range(offset_b):
            x_i = input[b : b + 1, i * chunk_size : (i + 1) * chunk_size]
            _, backbone_buffer_b, head_buffer_b = model.step(
                x_i,
                backbone_buffer_b,
                head_buffer_b,
            )

        pre_backbone_buffers.append(backbone_buffer_b.clone())
        pre_head_buffers.append(head_buffer_b.clone())

        x_i = input[
            b : b + 1,
            offset_b * chunk_size : (offset_b + 1) * chunk_size,
        ]
        output_b, backbone_buffer_b, head_buffer_b = model.step(
            x_i,
            backbone_buffer_b,
            head_buffer_b,
        )

        single_outputs.append(output_b)
        single_backbone_buffers.append(backbone_buffer_b)
        single_head_buffers.append(head_buffer_b)

    pre_backbone_buffer = torch.cat(pre_backbone_buffers, dim=0)
    pre_head_buffer = torch.cat(pre_head_buffers, dim=0)

    single_output = torch.cat(single_outputs, dim=0)
    single_backbone_buffer = torch.cat(single_backbone_buffers, dim=0)
    single_head_buffer = torch.cat(single_head_buffers, dim=0)

    x = torch.cat(
        [
            input[
                b : b + 1,
                offsets[b] * chunk_size : (offsets[b] + 1) * chunk_size,
            ]
            for b in range(B)
        ],
        dim=0,
    )

    exec_mask_all = torch.ones(B, device=device, dtype=torch.bool)

    output, backbone_buffer, head_buffer = model.step_desync(
        x,
        pre_backbone_buffer.clone(),
        pre_head_buffer.clone(),
        exec_mask_all,
    )

    assert torch.allclose(output, single_output, atol=1e-5), (
        ((output - single_output) ** 2).mean().sqrt(),
    )
    assert torch.allclose(backbone_buffer, single_backbone_buffer, atol=1e-5), (
        ((backbone_buffer - single_backbone_buffer) ** 2).mean().sqrt(),
    )
    assert torch.allclose(head_buffer, single_head_buffer, atol=1e-5), (
        ((head_buffer - single_head_buffer) ** 2).mean().sqrt(),
    )

    exec_mask = torch.tensor([True, False, True], device=device)

    output_masked, backbone_buffer_masked, head_buffer_masked = model.step_desync(
        x,
        pre_backbone_buffer.clone(),
        pre_head_buffer.clone(),
        exec_mask,
    )

    assert torch.allclose(
        backbone_buffer_masked[~exec_mask],
        pre_backbone_buffer[~exec_mask],
        atol=1e-5,
    )
    assert torch.allclose(
        head_buffer_masked[~exec_mask],
        pre_head_buffer[~exec_mask],
        atol=1e-5,
    )
    assert torch.allclose(
        backbone_buffer_masked[exec_mask],
        single_backbone_buffer[exec_mask],
        atol=1e-5,
    )
    assert torch.allclose(
        head_buffer_masked[exec_mask],
        single_head_buffer[exec_mask],
        atol=1e-5,
    )

    assert output_masked.shape == output.shape

    print("Desync test passed")


def test_jit() -> "None":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B = 3
    chunk_size = 8
    num_chunks = 8
    T = num_chunks * chunk_size

    model = (
        Vocos(
            num_layers=2,
            dim=128,
            ffn_dim=256,
            patch_size=chunk_size,
            n_fft=256,
            win_length=256,
            hop_length=80,
        )
        .eval()
        .to(device)
    )

    model_jit = torch.jit.script(model)

    input = torch.randn(B, T, model.input_dim, device=device)

    # Normal forward
    output = model(input)
    output_jit = model_jit(input)

    assert torch.allclose(output, output_jit, atol=1e-5), (
        ((output - output_jit) ** 2).mean().sqrt(),
    )

    # Stream
    backbone_buffer, head_buffer = model.init_state(B, input.device, input.dtype)
    backbone_buffer_jit, head_buffer_jit = model_jit.init_state(
        B,
        input.device,
        input.dtype,
    )

    for i in range(num_chunks):
        x_i = input[:, i * chunk_size : (i + 1) * chunk_size]

        output, next_backbone_buffer, next_head_buffer = model.step(
            x_i,
            backbone_buffer,
            head_buffer,
        )
        output_jit, next_backbone_buffer_jit, next_head_buffer_jit = model_jit.step(
            x_i,
            backbone_buffer_jit,
            head_buffer_jit,
        )

        assert torch.allclose(output, output_jit, atol=1e-5), (
            ((output - output_jit) ** 2).mean().sqrt(),
        )
        assert torch.allclose(
            next_backbone_buffer,
            next_backbone_buffer_jit,
            atol=1e-5,
        )
        assert torch.allclose(
            next_head_buffer,
            next_head_buffer_jit,
            atol=1e-5,
        )

        backbone_buffer, head_buffer = next_backbone_buffer, next_head_buffer
        backbone_buffer_jit = next_backbone_buffer_jit
        head_buffer_jit = next_head_buffer_jit

    # Desync stream
    exec_mask = torch.tensor([True, False, True], device=device)
    x_i = input[:, :chunk_size]

    output, next_backbone_buffer, next_head_buffer = model.step_desync(
        x_i,
        backbone_buffer,
        head_buffer,
        exec_mask,
    )
    output_jit, next_backbone_buffer_jit, next_head_buffer_jit = model_jit.step_desync(
        x_i,
        backbone_buffer_jit,
        head_buffer_jit,
        exec_mask,
    )

    assert torch.allclose(output, output_jit, atol=1e-5), (
        ((output - output_jit) ** 2).mean().sqrt(),
    )
    assert torch.allclose(next_backbone_buffer, next_backbone_buffer_jit, atol=1e-5)
    assert torch.allclose(next_head_buffer, next_head_buffer_jit, atol=1e-5)

    print("JIT test passed")


@torch.no_grad()
def test_cuda_graph() -> "None":
    if not torch.cuda.is_available():
        print("CUDA graph test skipped")
        return

    torch.manual_seed(0)
    device = torch.device("cuda")

    B = 3
    chunk_size = 8

    model = (
        Vocos(
            num_layers=2,
            dim=128,
            ffn_dim=256,
            patch_size=chunk_size,
            n_fft=256,
            win_length=256,
            hop_length=80,
        )
        .eval()
        .to(device)
    )

    backbone_buffer, head_buffer = model.init_state(B, device, torch.float32)

    static_input = torch.randn(B, chunk_size, model.input_dim, device=device)

    for _ in range(3):
        model.step(
            static_input,
            backbone_buffer,
            head_buffer,
        )

    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()

    with torch.cuda.graph(graph):
        (
            static_output,
            static_backbone_buffer,
            static_head_buffer,
        ) = model.step(
            static_input,
            backbone_buffer,
            head_buffer,
        )

    new_input = torch.randn(B, chunk_size, model.input_dim, device=device)
    static_input.copy_(new_input)

    eager_output, eager_backbone_buffer, eager_head_buffer = model.step(
        new_input,
        backbone_buffer.clone(),
        head_buffer.clone(),
    )

    graph.replay()
    graph_output = static_output.clone()
    graph_backbone_buffer = static_backbone_buffer.clone()
    graph_head_buffer = static_head_buffer.clone()

    assert torch.allclose(graph_output, eager_output, atol=1e-5), (
        ((graph_output - eager_output) ** 2).mean().sqrt(),
    )
    assert torch.allclose(graph_backbone_buffer, eager_backbone_buffer, atol=1e-5)
    assert torch.allclose(graph_head_buffer, eager_head_buffer, atol=1e-5)

    # Desync CUDA graph
    exec_mask = torch.tensor([True, False, True], device=device)

    for _ in range(3):
        model.step_desync(
            static_input,
            backbone_buffer,
            head_buffer,
            exec_mask,
        )

    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()

    with torch.cuda.graph(graph):
        (
            static_output,
            static_backbone_buffer,
            static_head_buffer,
        ) = model.step_desync(
            static_input,
            backbone_buffer,
            head_buffer,
            exec_mask,
        )

    new_input = torch.randn(B, chunk_size, model.input_dim, device=device)
    static_input.copy_(new_input)

    eager_output, eager_backbone_buffer, eager_head_buffer = model.step_desync(
        new_input,
        backbone_buffer.clone(),
        head_buffer.clone(),
        exec_mask.clone(),
    )

    graph.replay()
    graph_output = static_output.clone()
    graph_backbone_buffer = static_backbone_buffer.clone()
    graph_head_buffer = static_head_buffer.clone()

    assert torch.allclose(graph_output, eager_output, atol=1e-5), (
        ((graph_output - eager_output) ** 2).mean().sqrt(),
    )
    assert torch.allclose(graph_backbone_buffer, eager_backbone_buffer, atol=1e-5)
    assert torch.allclose(graph_head_buffer, eager_head_buffer, atol=1e-5)

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

    class VocosStreamWrapper(nn.Module):
        def __init__(self, model: "Vocos") -> "None":
            super().__init__()
            self.model = model

        def forward(
            self,
            input: "Tensor",
            backbone_buffer: "Tensor",
            head_buffer: "Tensor",
        ) -> "Tuple[Tensor, Tensor, Tensor]":
            return self.model.step(
                input,
                backbone_buffer,
                head_buffer,
            )

    class VocosStreamDesyncWrapper(nn.Module):
        def __init__(self, model: "Vocos") -> "None":
            super().__init__()
            self.model = model

        def forward(
            self,
            input: "Tensor",
            backbone_buffer: "Tensor",
            head_buffer: "Tensor",
            exec_mask: "Tensor",
        ) -> "Tuple[Tensor, Tensor, Tensor]":
            return self.model.step_desync(
                input,
                backbone_buffer,
                head_buffer,
                exec_mask,
            )

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B = 3
    chunk_size = 8

    model = (
        Vocos(
            num_layers=2,
            dim=128,
            ffn_dim=256,
            patch_size=chunk_size,
            n_fft=256,
            win_length=256,
            hop_length=80,
            istft_backend="conv",
        )
        .eval()
        .to(device)
    )

    input = torch.randn(B, chunk_size, model.input_dim, device=device)

    backbone_buffer, head_buffer = model.init_state(
        B,
        input.device,
        input.dtype,
    )

    # Stream ONNX
    output, next_backbone_buffer, next_head_buffer = model.step(
        input,
        backbone_buffer,
        head_buffer,
    )

    wrapper = VocosStreamWrapper(model).eval().to(device)

    f = io.BytesIO()
    torch.onnx.export(
        wrapper,
        (input, backbone_buffer, head_buffer),
        f,
        input_names=["input", "backbone_buffer", "head_buffer"],
        output_names=["output", "next_backbone_buffer", "next_head_buffer"],
        dynamic_axes={
            "input": {0: "batch"},
            "output": {0: "batch"},
            "backbone_buffer": {0: "batch"},
            "head_buffer": {0: "batch"},
            "next_backbone_buffer": {0: "batch"},
            "next_head_buffer": {0: "batch"},
        },
    )

    session = ort.InferenceSession(f.getvalue())
    outputs_ort = session.run(
        None,
        {
            "input": input.cpu().numpy(),
            "backbone_buffer": backbone_buffer.cpu().numpy(),
            "head_buffer": head_buffer.cpu().numpy(),
        },
    )

    output_ort = torch.tensor(outputs_ort[0], device=device, dtype=output.dtype)
    next_backbone_buffer_ort = torch.tensor(
        outputs_ort[1],
        device=device,
        dtype=next_backbone_buffer.dtype,
    )
    next_head_buffer_ort = torch.tensor(
        outputs_ort[2],
        device=device,
        dtype=next_head_buffer.dtype,
    )

    assert torch.allclose(output, output_ort, atol=5e-1), (
        ((output - output_ort) ** 2).mean().sqrt(),
    )
    assert torch.allclose(next_backbone_buffer, next_backbone_buffer_ort, atol=1e-2), (
        ((next_backbone_buffer - next_backbone_buffer_ort) ** 2).mean().sqrt(),
    )
    assert torch.allclose(next_head_buffer, next_head_buffer_ort, atol=1e-2), (
        ((next_head_buffer - next_head_buffer_ort) ** 2).mean().sqrt(),
    )

    # Desync stream ONNX
    exec_mask = torch.tensor([True, False, True], device=device)

    output, next_backbone_buffer, next_head_buffer = model.step_desync(
        input,
        backbone_buffer,
        head_buffer,
        exec_mask,
    )

    wrapper = VocosStreamDesyncWrapper(model).eval().to(device)

    f = io.BytesIO()
    torch.onnx.export(
        wrapper,
        (input, backbone_buffer, head_buffer, exec_mask),
        f,
        input_names=[
            "input",
            "backbone_buffer",
            "head_buffer",
            "exec_mask",
        ],
        output_names=[
            "output",
            "next_backbone_buffer",
            "next_head_buffer",
        ],
        dynamic_axes={
            "input": {0: "batch"},
            "output": {0: "batch"},
            "backbone_buffer": {0: "batch"},
            "head_buffer": {0: "batch"},
            "exec_mask": {0: "batch"},
            "next_backbone_buffer": {0: "batch"},
            "next_head_buffer": {0: "batch"},
        },
    )

    session = ort.InferenceSession(f.getvalue())
    outputs_ort = session.run(
        None,
        {
            "input": input.cpu().numpy(),
            "backbone_buffer": backbone_buffer.cpu().numpy(),
            "head_buffer": head_buffer.cpu().numpy(),
            "exec_mask": exec_mask.cpu().numpy(),
        },
    )

    output_ort = torch.tensor(outputs_ort[0], device=device, dtype=output.dtype)
    next_backbone_buffer_ort = torch.tensor(
        outputs_ort[1],
        device=device,
        dtype=next_backbone_buffer.dtype,
    )
    next_head_buffer_ort = torch.tensor(
        outputs_ort[2],
        device=device,
        dtype=next_head_buffer.dtype,
    )

    assert torch.allclose(output, output_ort, atol=5e-1), (
        ((output - output_ort) ** 2).mean().sqrt(),
    )
    assert torch.allclose(next_backbone_buffer, next_backbone_buffer_ort, atol=1e-2), (
        ((next_backbone_buffer - next_backbone_buffer_ort) ** 2).mean().sqrt(),
    )
    assert torch.allclose(next_head_buffer, next_head_buffer_ort, atol=1e-2), (
        ((next_head_buffer - next_head_buffer_ort) ** 2).mean().sqrt(),
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
