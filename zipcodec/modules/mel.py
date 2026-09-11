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

"""Mel-spectrogram frontend."""

import math
from typing import Optional, Tuple

import torch
from torch import Tensor, nn


__all__ = ["LogMelSpectrogram"]


class LogMelSpectrogram(nn.Module):
    """Log-mel spectrogram extractor.

    Parameters
    ----------
    sample_rate:
        Audio sample rate in Hz.
    n_fft:
        FFT size.
    win_length:
        Window size in samples.
    hop_length:
        Hop size between consecutive frames in samples.
    n_mels:
        Number of mel frequency bins.
    f_min:
        Minimum mel frequency in Hz.
    f_max:
        Maximum mel frequency in Hz.
        Defaults to `sample_rate // 2`.
    eps:
        Small constant for numerical stability.
    stft_backend:
        STFT backend. Options:
        - "fft": faster training/inference backend based on FFT;
        - "conv": ONNX/TensorRT-friendly convolutional backend.

    """

    def __init__(
        self,
        sample_rate: "int" = 16000,
        n_fft: "int" = 400,
        win_length: "int" = 400,
        hop_length: "int" = 160,
        n_mels: "int" = 80,
        f_min: "float" = 0.0,
        f_max: "Optional[float]" = None,
        eps: "float" = 1e-6,
        stft_backend: "str" = "fft",
    ) -> "None":
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = float(sample_rate // 2) if f_max is None else f_max
        self.eps = eps
        self.stft_backend = stft_backend

        if stft_backend not in ("fft", "conv"):
            raise ValueError(f"stft_backend={stft_backend} must be 'fft' or 'conv'")

        # Buffers
        window = torch.hann_window(win_length)
        mel_fb = self._build_mel_filterbank()

        self.register_buffer("window", window, persistent=False)
        self.register_buffer("mel_fb", mel_fb, persistent=False)

        freqs = torch.arange(n_fft // 2 + 1, dtype=torch.float32)[:, None]
        time = torch.arange(win_length, dtype=torch.float32)[None, :]
        angle = 2.0 * math.pi * freqs * time / n_fft

        real_basis = angle.cos() * window[None, :]
        imag_basis = -angle.sin() * window[None, :]

        basis = torch.cat([real_basis, imag_basis], dim=0)
        self.register_buffer("basis", basis[:, None, :], persistent=False)

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
            Buffer of shape (batch_size, win_length - hop_length).

        """
        return torch.zeros(
            batch_size,
            self.win_length - self.hop_length,
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
            Buffer of shape (batch_size, win_length - hop_length).

        """
        return self.init_state(batch_size, device, dtype)

    def forward(self, input: "Tensor") -> "Tensor":
        """Forward pass.

        Parameters
        ----------
        input:
            Input of shape (batch_size, seq_length).

        Returns
        -------
            Output of shape
            (batch_size, ceil(seq_length / hop_length), n_mels).

        """
        buffer_size = self.win_length - self.hop_length
        input = nn.functional.pad(input, (buffer_size, 0))
        if self.stft_backend == "fft":
            return self._compute_logmel_fft(input)
        return self._compute_logmel_conv(input)

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
            Input of shape (batch_size, seq_length).
        buffer:
            Buffer of shape (batch_size, win_length - hop_length).

        Returns
        -------
            - Output of shape
              (batch_size, ceil(seq_length / hop_length), n_mels);
            - updated buffer.

        """
        H = self.hop_length

        pad_len = (H - input.shape[1] % H) % H
        input = nn.functional.pad(input, (0, pad_len))

        context = torch.cat([buffer, input], dim=1)

        if self.stft_backend == "fft":
            output = self._compute_logmel_fft(context)
        else:
            output = self._compute_logmel_conv(context)

        buffer_size = self.win_length - self.hop_length
        start = context.shape[1] - buffer_size
        buffer = context[:, start:]

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
            Input of shape (batch_size, seq_length).
        buffer:
            Buffer of shape (batch_size, win_length - hop_length).
        exec_mask:
            Execution mask of shape (batch_size,). True means the stream
            advances; False means the buffer is kept unchanged.

        Returns
        -------
            - Output of shape
              (batch_size, ceil(seq_length / hop_length), n_mels);
            - updated buffer.

        """
        output, next_buffer = self.step(input, buffer)

        next_buffer = torch.where(
            exec_mask[:, None],
            next_buffer,
            buffer,
        )

        return output, next_buffer

    def _compute_logmel_fft(self, input: "Tensor") -> "Tensor":
        """Compute center=False log-mel features via FFT."""
        frames = input.unfold(
            dimension=1,
            size=self.win_length,
            step=self.hop_length,
        )

        frames = frames * self.window.to(dtype=input.dtype)

        spectrum = torch.fft.rfft(
            frames,
            n=self.n_fft,
            dim=-1,
        )

        power = spectrum.real.square() + spectrum.imag.square()
        mel = power @ self.mel_fb.to(dtype=input.dtype)
        return mel.clamp_min(self.eps).log()

    def _compute_logmel_conv(self, input: "Tensor") -> "Tensor":
        """Compute center=False log-mel features via convolution."""
        input = input[:, None]

        spectrum = nn.functional.conv1d(
            input,
            self.basis.to(dtype=input.dtype),
            stride=self.hop_length,
        )

        real, imag = spectrum.chunk(2, dim=1)

        power = real.square() + imag.square()
        power = power.transpose(1, 2)

        mel = power @ self.mel_fb.to(dtype=input.dtype)
        return mel.clamp_min(self.eps).log()

    def _build_mel_filterbank(self) -> "Tensor":
        """Build triangular HTK-style mel filterbank."""
        n_freqs = self.n_fft // 2 + 1

        f_min_mel = self._hz_to_mel(self.f_min)
        f_max_mel = self._hz_to_mel(self.f_max)

        mels = torch.linspace(f_min_mel, f_max_mel, self.n_mels + 2)
        freqs = self._mel_to_hz(mels)

        fft_freqs = torch.linspace(0.0, self.sample_rate / 2, n_freqs)

        lower = freqs[:-2]
        center = freqs[1:-1]
        upper = freqs[2:]

        slopes = fft_freqs[:, None]

        left = (slopes - lower[None, :]) / (center - lower)[None, :]
        right = (upper[None, :] - slopes) / (upper - center)[None, :]

        return left.minimum(right).clamp_min(0.0)

    def _hz_to_mel(self, freq: "float") -> "float":
        return 2595.0 * math.log10(1.0 + freq / 700.0)

    def _mel_to_hz(self, mels: "Tensor") -> "Tensor":
        return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)


def test_model() -> "None":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B = 3
    T = 16000

    model = LogMelSpectrogram().to(device)

    input = torch.randn(B, T, device=device)
    output = model(input)

    model_jit = torch.jit.script(model)
    output_jit = model_jit(input)

    print(f"Input shape: {input.shape}")
    print(f"Output shape: {output.shape}")

    assert output.shape[0] == B
    assert output.shape[2] == model.n_mels

    assert torch.allclose(output, output_jit, atol=1e-5), (
        ((output - output_jit) ** 2).mean().sqrt(),
    )

    print("Model test passed")


@torch.no_grad()
def test_batch_invariance() -> None:
    torch.manual_seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B = 10
    T = 16000

    model = LogMelSpectrogram().eval().to(device)

    input = torch.randn(B, T, device=device)

    batch_output = model(input)

    single_output = torch.cat(
        [model(input[i : i + 1]) for i in range(B)],
        dim=0,
    )

    assert torch.allclose(
        batch_output,
        single_output,
        atol=1e-5,
    )

    print("Batch invariance test passed")


@torch.no_grad()
def test_streaming() -> "None":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B = 3
    chunk_size = 160
    num_chunks = 100
    T = num_chunks * chunk_size

    model = LogMelSpectrogram().eval().to(device)

    input = torch.randn(B, T, device=device)

    output_train = model(input)

    buffer = model.init_state(B, input.device, input.dtype)

    outputs = []
    for i in range(0, T, chunk_size):
        output_i, buffer = model.step(
            input[:, i : i + chunk_size],
            buffer,
        )
        outputs.append(output_i)

    output_stream = torch.cat(outputs, dim=1)

    assert torch.allclose(output_train, output_stream, atol=1e-5), (
        ((output_train - output_stream) ** 2).mean().sqrt(),
    )

    print("Streaming test passed")


@torch.no_grad()
def test_desync() -> "None":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B = 3
    chunk_size = 160
    offsets = [16, 19, 23]
    num_chunks = max(offsets) + 2
    T = num_chunks * chunk_size

    model = LogMelSpectrogram().eval().to(device)
    input = torch.randn(B, T, device=device)

    pre_buffers = []
    single_outputs = []
    single_buffers = []

    for b, offset_b in enumerate(offsets):
        buffer_b = model.init_state(1, input.device, input.dtype)

        for i in range(offset_b):
            x_i = input[b : b + 1, i * chunk_size : (i + 1) * chunk_size]
            _, buffer_b = model.step(x_i, buffer_b)

        pre_buffers.append(buffer_b.clone())

        x_i = input[
            b : b + 1,
            offset_b * chunk_size : (offset_b + 1) * chunk_size,
        ]
        output_b, buffer_b = model.step(x_i, buffer_b)

        single_outputs.append(output_b)
        single_buffers.append(buffer_b)

    pre_buffer = torch.cat(pre_buffers, dim=0)
    single_output = torch.cat(single_outputs, dim=0)
    single_buffer = torch.cat(single_buffers, dim=0)

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

    output, buffer = model.step_desync(
        x,
        pre_buffer.clone(),
        torch.ones(B, device=device, dtype=torch.bool),
    )

    assert torch.allclose(output, single_output, atol=1e-5), (
        ((output - single_output) ** 2).mean().sqrt(),
    )
    assert torch.allclose(buffer, single_buffer, atol=1e-5), (
        ((buffer - single_buffer) ** 2).mean().sqrt(),
    )

    exec_mask = torch.tensor([True, False, True], device=device)

    output_masked, buffer_masked = model.step_desync(
        x,
        pre_buffer.clone(),
        exec_mask,
    )

    assert torch.allclose(buffer_masked[~exec_mask], pre_buffer[~exec_mask], atol=1e-5)
    assert torch.allclose(buffer_masked[exec_mask], single_buffer[exec_mask], atol=1e-5)
    assert output_masked.shape == output.shape

    print("Desync test passed")


def test_jit() -> "None":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B = 3
    T = 16000
    chunk_size = 160
    num_chunks = T // chunk_size

    model = LogMelSpectrogram().eval().to(device)
    model_jit = torch.jit.script(model)

    input = torch.randn(B, T, device=device)

    # Normal forward
    output = model(input)
    output_jit = model_jit(input)

    assert torch.allclose(output, output_jit, atol=1e-5), (
        ((output - output_jit) ** 2).mean().sqrt(),
    )

    # Streaming
    buffer = model.init_state(B, input.device, input.dtype)
    buffer_jit = model_jit.init_state(B, input.device, input.dtype)

    outputs = []
    outputs_jit = []

    for i in range(num_chunks):
        x_i = input[:, i * chunk_size : (i + 1) * chunk_size]

        output_i, buffer = model.step(x_i, buffer)
        output_i_jit, buffer_jit = model_jit.step(x_i, buffer_jit)

        outputs.append(output_i)
        outputs_jit.append(output_i_jit)

    output = torch.cat(outputs, dim=1)
    output_jit = torch.cat(outputs_jit, dim=1)

    assert torch.allclose(output, output_jit, atol=1e-5), (
        ((output - output_jit) ** 2).mean().sqrt(),
    )
    assert torch.allclose(buffer, buffer_jit, atol=1e-5), (
        ((buffer - buffer_jit) ** 2).mean().sqrt(),
    )

    # Desync
    buffer = model.init_state(B, input.device, input.dtype)
    buffer_jit = model_jit.init_state(B, input.device, input.dtype)

    x = input[:, :chunk_size]
    exec_mask = torch.tensor([True, False, True], device=device)

    output, next_buffer = model.step_desync(x, buffer, exec_mask)
    output_jit, next_buffer_jit = model_jit.step_desync(
        x,
        buffer_jit,
        exec_mask,
    )

    assert torch.allclose(output, output_jit, atol=1e-5), (
        ((output - output_jit) ** 2).mean().sqrt(),
    )
    assert torch.allclose(next_buffer, next_buffer_jit, atol=1e-5), (
        ((next_buffer - next_buffer_jit) ** 2).mean().sqrt(),
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
    chunk_size = 160

    model = LogMelSpectrogram().eval().to(device)

    buffer = model.init_state(B, device, torch.float32)
    static_input = torch.randn(B, chunk_size, device=device)

    for _ in range(3):
        model.step(static_input, buffer)

    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()

    with torch.cuda.graph(graph):
        static_output, static_buffer = model.step(
            static_input,
            buffer,
        )

    new_input = torch.randn(B, chunk_size, device=device)
    static_input.copy_(new_input)

    graph.replay()
    graph_output = static_output.clone()
    graph_buffer = static_buffer.clone()

    eager_output, eager_buffer = model.step(
        new_input,
        buffer,
    )

    assert torch.allclose(graph_output, eager_output, atol=1e-5), (
        ((graph_output - eager_output) ** 2).mean().sqrt(),
    )
    assert torch.allclose(graph_buffer, eager_buffer, atol=1e-5), (
        ((graph_buffer - eager_buffer) ** 2).mean().sqrt(),
    )

    # Desync CUDA graph
    buffer = model.init_state(B, device, torch.float32)
    static_input = torch.randn(B, chunk_size, device=device)
    static_exec_mask = torch.tensor([True, False, True], device=device)

    for _ in range(3):
        model.step_desync(
            static_input,
            buffer,
            static_exec_mask,
        )

    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()

    with torch.cuda.graph(graph):
        static_output, static_buffer = model.step_desync(
            static_input,
            buffer,
            static_exec_mask,
        )

    new_input = torch.randn(B, chunk_size, device=device)
    static_input.copy_(new_input)

    graph.replay()
    graph_output = static_output.clone()
    graph_buffer = static_buffer.clone()

    eager_output, eager_buffer = model.step_desync(
        new_input,
        buffer,
        static_exec_mask,
    )

    assert torch.allclose(graph_output, eager_output, atol=1e-5), (
        ((graph_output - eager_output) ** 2).mean().sqrt(),
    )
    assert torch.allclose(graph_buffer, eager_buffer, atol=1e-5), (
        ((graph_buffer - eager_buffer) ** 2).mean().sqrt(),
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

    class LogMelStreamWrapper(nn.Module):
        def __init__(self, model: "LogMelSpectrogram") -> "None":
            super().__init__()
            self.model = model

        def forward(
            self,
            input: "Tensor",
            buffer: "Tensor",
        ) -> "Tuple[Tensor, Tensor]":
            return self.model.step(input, buffer)

    class LogMelStreamDesyncWrapper(nn.Module):
        def __init__(self, model: "LogMelSpectrogram") -> "None":
            super().__init__()
            self.model = model

        def forward(
            self,
            input: "Tensor",
            buffer: "Tensor",
            exec_mask: "Tensor",
        ) -> "Tuple[Tensor, Tensor]":
            return self.model.step_desync(
                input,
                buffer,
                exec_mask,
            )

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B = 3
    T = 16000
    chunk_size = 160

    model = (
        LogMelSpectrogram(
            stft_backend="conv",
        )
        .eval()
        .to(device)
    )

    input = torch.randn(B, T, device=device)

    # Normal forward ONNX
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
            "output": {0: "batch", 1: "feature_time"},
        },
    )

    session = ort.InferenceSession(f.getvalue())
    output_ort = session.run(
        None,
        {"input": input.cpu().numpy()},
    )[0]
    output_ort = torch.tensor(output_ort, device=device, dtype=output.dtype)

    assert torch.allclose(output, output_ort, atol=1e-5), (
        ((output - output_ort) ** 2).mean().sqrt(),
    )

    # Stream ONNX
    x_i = input[:, :chunk_size]
    buffer = model.init_state(B, input.device, input.dtype)

    output, next_buffer = model.step(x_i, buffer)

    wrapper = LogMelStreamWrapper(model).eval().to(device)

    f = io.BytesIO()
    torch.onnx.export(
        wrapper,
        (x_i, buffer),
        f,
        input_names=["input", "buffer"],
        output_names=["output", "next_buffer"],
        dynamic_axes={
            "input": {0: "batch", 1: "time"},
            "output": {0: "batch", 1: "feature_time"},
            "buffer": {0: "batch"},
            "next_buffer": {0: "batch"},
        },
    )

    session = ort.InferenceSession(f.getvalue())
    outputs_ort = session.run(
        None,
        {
            "input": x_i.cpu().numpy(),
            "buffer": buffer.cpu().numpy(),
        },
    )

    output_ort = torch.tensor(outputs_ort[0], device=device, dtype=output.dtype)
    next_buffer_ort = torch.tensor(
        outputs_ort[1],
        device=device,
        dtype=next_buffer.dtype,
    )

    assert torch.allclose(output, output_ort, atol=1e-5), (
        ((output - output_ort) ** 2).mean().sqrt(),
    )
    assert torch.allclose(next_buffer, next_buffer_ort, atol=1e-5), (
        ((next_buffer - next_buffer_ort) ** 2).mean().sqrt(),
    )

    # Desync stream ONNX
    exec_mask = torch.tensor([True, False, True], device=device)

    output, next_buffer = model.step_desync(
        x_i,
        buffer,
        exec_mask,
    )

    wrapper = LogMelStreamDesyncWrapper(model).eval().to(device)

    f = io.BytesIO()
    torch.onnx.export(
        wrapper,
        (x_i, buffer, exec_mask),
        f,
        input_names=["input", "buffer", "exec_mask"],
        output_names=["output", "next_buffer"],
        dynamic_axes={
            "input": {0: "batch", 1: "time"},
            "output": {0: "batch", 1: "feature_time"},
            "buffer": {0: "batch"},
            "next_buffer": {0: "batch"},
            "exec_mask": {0: "batch"},
        },
    )

    session = ort.InferenceSession(f.getvalue())
    outputs_ort = session.run(
        None,
        {
            "input": x_i.cpu().numpy(),
            "buffer": buffer.cpu().numpy(),
            "exec_mask": exec_mask.cpu().numpy(),
        },
    )

    output_ort = torch.tensor(outputs_ort[0], device=device, dtype=output.dtype)
    next_buffer_ort = torch.tensor(
        outputs_ort[1],
        device=device,
        dtype=next_buffer.dtype,
    )

    assert torch.allclose(output, output_ort, atol=1e-5), (
        ((output - output_ort) ** 2).mean().sqrt(),
    )
    assert torch.allclose(next_buffer, next_buffer_ort, atol=1e-5), (
        ((next_buffer - next_buffer_ort) ** 2).mean().sqrt(),
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
