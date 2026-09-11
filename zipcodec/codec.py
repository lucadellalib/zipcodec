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

"""ZipCodec model."""

import inspect
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn


try:
    from . import deployment as _deployment
    from .endpoints import (
        DictEndpoint,
        EndpointModule,
        EndpointSpec,
        endpoint_spec,
        make_zipcodec_endpoints,
    )
    from .knn import knn as knn_qfeats
    from .utils import load_audio, resample_audio, save_audio
except ImportError:
    import deployment as _deployment
    from endpoints import (
        DictEndpoint,
        EndpointModule,
        EndpointSpec,
        endpoint_spec,
        make_zipcodec_endpoints,
    )
    from knn import knn as knn_qfeats
    from utils import load_audio, resample_audio, save_audio


State = Tuple[Tensor, ...]
COMPONENT_NAMES = (
    "encoder",
    "frontend",
    "compressor",
    "quantizer",
    "decompressor",
    "backend",
    "decoder",
)


try:
    __version__ = version("zipcodec")
except PackageNotFoundError:
    __version__ = "0+unknown"


def _as_state(value: "Any") -> "State":
    if isinstance(value, Tensor):
        return (value,)
    if isinstance(value, tuple) and all(isinstance(x, Tensor) for x in value):
        return value
    raise TypeError("State must be a Tensor or a flat tuple of Tensors")


class ZipCodec(nn.Module):
    """Streaming neural audio codec."""

    def __init__(
        self,
        encoder: "Optional[nn.Module]" = None,
        frontend: "Optional[nn.Module]" = None,
        compressor: "Optional[nn.Module]" = None,
        quantizer: "Optional[nn.Module]" = None,
        decompressor: "Optional[nn.Module]" = None,
        backend: "Optional[nn.Module]" = None,
        decoder: "Optional[nn.Module]" = None,
    ) -> "None":
        super().__init__()
        self.encoder = encoder
        self.frontend = frontend
        self.compressor = compressor
        self.quantizer = quantizer
        self.decompressor = decompressor
        self.backend = backend
        self.decoder = decoder
        self.endpoints = dict(make_zipcodec_endpoints())
        self.model_id = None

    @property
    def sample_rate_input(self) -> "int":
        """Return the input sample rate."""
        return 16000

    @property
    def sample_rate_output(self) -> "int":
        """Return the output sample rate."""
        return 16000

    @property
    def sample_rate(self) -> "int":
        """Return the sample rate."""
        if self.sample_rate_input != self.sample_rate_output:
            raise RuntimeError(
                "`sample_rate` is undefined because input and output sample rates "
                f"differ (input={self.sample_rate_input}, output={self.sample_rate_output}). "
                "Please use `sample_rate_input` or `sample_rate_output` explicitly"
            )
        return self.sample_rate_input

    @property
    def chunk_size(self) -> "int":
        """Return the fixed streaming input chunk size in samples."""
        return 2560

    @property
    def latency(self) -> "float":
        """Return the theoretical latency in milliseconds."""
        return 1000.0 * self.chunk_size / self.sample_rate_input

    def info(self) -> "Dict[str, Any]":
        """Return the model information."""
        return {
            "model_id": self.model_id,
            "version": __version__,
            "sample_rate_input": self.sample_rate_input,
            "sample_rate_output": self.sample_rate_output,
            "chunk_size": self.chunk_size,
            "latency_ms": self.latency,
            "num_total_params": sum(x.numel() for x in self.state_dict().values()),
        }

    @torch.jit.ignore
    def load_audio(self, path: "Union[str, Path]") -> "Tuple[Tensor, int]":
        """Load a PCM WAV file as a float tensor shaped (channels, time)."""
        return load_audio(path)

    @torch.jit.ignore
    def save_audio(
        self,
        path: "Union[str, Path]",
        waveform: "Tensor",
        sample_rate: "int",
        encoding: "str" = "float32",
    ) -> "None":
        """Save a waveform shaped (channels, time) as float32 or 16-bit PCM WAV."""
        save_audio(path, waveform, sample_rate, encoding)

    @torch.jit.ignore
    def resample_audio(
        self,
        waveform: "Tensor",
        orig_freq: "int",
        new_freq: "int",
    ) -> "Tensor":
        """Resample the last axis using Hann-windowed sinc interpolation."""
        return resample_audio(waveform, orig_freq, new_freq)

    def _component(self, name: "str") -> "nn.Module":
        module = getattr(self, name)
        if module is None:
            raise RuntimeError(
                f"Component {name!r} is required for this operation but is not set"
            )
        return module

    def _validate_components(self, names: "Tuple[str, ...]") -> "None":
        for name in names:
            self._component(name)

    @staticmethod
    def _component_method(
        module: "nn.Module",
        method_name: "str",
    ) -> "Optional[Any]":
        method = getattr(module, method_name, None)
        if method is not None:
            return method
        if hasattr(module, "init_state"):
            raise RuntimeError(
                f"{type(module).__name__} defines init_state() but not {method_name}()"
            )
        return None

    @staticmethod
    def _method_state_names(method: "Any") -> "Tuple[str, ...]":
        parameters = tuple(inspect.signature(method).parameters.values())
        positional = tuple(
            parameter
            for parameter in parameters
            if parameter.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        )
        if positional and positional[-1].name == "exec_mask":
            positional = positional[:-1]
        if not positional:
            raise ValueError(f"Invalid cached-inference signature for {method}")
        return tuple(parameter.name for parameter in positional[1:])

    def _component_state_layout(
        self,
        component_names: "Tuple[str, ...]",
        method_name: "Optional[str]",
    ) -> "Tuple[Tuple[str, Tuple[str, ...]], ...]":
        if method_name is None:
            return ()

        layout = []
        for name in component_names:
            module = self._component(name)
            method = self._component_method(module, method_name)
            state_names = () if method is None else self._method_state_names(method)
            layout.append((name, state_names))
        return tuple(layout)

    def _init_component_state(
        self,
        component_names: "Tuple[str, ...]",
        method_name: "Optional[str]",
        batch_size: "int",
        device: "torch.device",
        dtype: "torch.dtype",
    ) -> "State":
        state = []
        init_method_name = (
            "init_state_desync"
            if method_name is not None and method_name.endswith("_desync")
            else "init_state"
        )

        for name, state_names in self._component_state_layout(
            component_names,
            method_name,
        ):
            state_size = len(state_names)
            if state_size == 0:
                continue

            module = self._component(name)
            init_method = getattr(module, init_method_name, None)
            if init_method is None:
                raise RuntimeError(
                    f"{type(module).__name__} requires {state_size} state tensors "
                    f"for {method_name}() but does not define {init_method_name}()"
                )

            component_state = _as_state(init_method(batch_size, device, dtype))
            if len(component_state) != state_size:
                raise RuntimeError(
                    f"{type(module).__name__}.{init_method_name}() returned "
                    f"{len(component_state)} tensors; expected {state_size}"
                )
            state.extend(component_state)

        return tuple(state)

    def init_endpoint_state(
        self,
        endpoint_name: "str",
        batch_size: "int",
        device: "torch.device",
        dtype: "torch.dtype",
    ) -> "State":
        """Initialize the flat state tuple for one endpoint."""
        return self.endpoint(endpoint_name).init_state(
            self,
            batch_size,
            device,
            dtype,
        )

    def init_state(
        self,
        batch_size: "int",
        device: "torch.device",
        dtype: "torch.dtype",
    ) -> "State":
        """Initialize full-codec synchronized state."""
        return self.init_endpoint_state("step", batch_size, device, dtype)

    def init_state_desync(
        self,
        batch_size: "int",
        device: "torch.device",
        dtype: "torch.dtype",
    ) -> "State":
        """Initialize full-codec desynchronized state."""
        return self.init_endpoint_state("step_desync", batch_size, device, dtype)

    def _run_components(
        self,
        input: "Tensor",
        component_names: "Tuple[str, ...]",
        method_name: "str",
        state: "State",
        exec_mask: "Optional[Tensor]" = None,
    ) -> "Tuple[Tensor, State]":
        if not isinstance(state, tuple) or not all(
            isinstance(value, Tensor) for value in state
        ):
            raise TypeError("State must be a flat tuple of Tensors")

        output = input
        next_state = []
        offset = 0

        for name, state_names in self._component_state_layout(
            component_names,
            method_name,
        ):
            state_size = len(state_names)
            module = self._component(name)
            method = self._component_method(module, method_name)
            if method is None:
                output = module(output)
                continue

            component_state = state[offset : offset + state_size]
            if len(component_state) != state_size:
                raise ValueError(
                    f"State for component {name!r} has {len(component_state)} "
                    f"tensors; expected {state_size}"
                )

            if method_name.endswith("_desync"):
                if exec_mask is None:
                    raise TypeError("exec_mask is required for desynchronized steps")
                result = method(output, *component_state, exec_mask)
            else:
                result = method(output, *component_state)

            if not isinstance(result, tuple) or len(result) != state_size + 1:
                raise RuntimeError(
                    f"{type(module).__name__}.{method_name}() must return output "
                    f"followed by {state_size} state tensors"
                )

            output = result[0]
            next_state.extend(result[1:])
            offset += state_size

        if offset != len(state):
            raise ValueError(
                f"Received {len(state)} state tensors but consumed {offset}"
            )

        return output, tuple(next_state)

    def forward(self, wav: "Tensor") -> "Tensor":
        self._validate_components(COMPONENT_NAMES)
        lats = self.wav_to_lats(wav)
        toks = self.lats_to_toks(lats)
        codes = self.toks_to_codes(toks)
        qfeats = self.codes_to_qfeats(codes)
        return self.qfeats_to_wav(qfeats)

    def _validate_endpoint_step_input(
        self,
        endpoint_name: "str",
        input: "Tensor",
    ) -> "None":
        # Compiled adapters validate before entering the traced/exported graph
        if torch.jit.is_tracing() or torch.onnx.is_in_onnx_export():
            return

        endpoint_name = endpoint_name.removesuffix("_desync")
        expected = None
        unit = "frames"
        if endpoint_name in (
            "step",
            "wav_to_frame_feats_step",
            "wav_to_feats_step",
            "wav_to_lats_step",
            "wav_to_toks_step",
            "wav_to_qfeats_step",
            "wav_to_frame_qfeats_step",
            "wav_to_wav_vc_step",
        ):
            expected = self.chunk_size
            unit = "samples"
        elif endpoint_name in (
            "feats_to_lats_step",
            "feats_to_toks_step",
            "feats_to_qfeats_step",
            "lats_to_qfeats_step",
            "lats_to_wav_step",
            "toks_to_qfeats_step",
            "toks_to_frame_qfeats_step",
            "toks_to_wav_step",
            "toks_to_wav_vc_step",
            "codes_to_qfeats_step",
            "codes_to_frame_qfeats_step",
            "qfeats_to_frame_qfeats_step",
            "qfeats_to_wav_step",
        ):
            expected = 1
        elif endpoint_name == "frame_qfeats_to_wav_step":
            expected = self._component("decoder").patch_size

        if expected is not None and (input.ndim < 2 or input.shape[1] != expected):
            actual = input.shape[1] if input.ndim >= 2 else None
            raise ValueError(
                f"{endpoint_name} expects exactly {expected} {unit}, got {actual}"
            )

    def step(
        self,
        wav: "Tensor",
        state: "State",
    ) -> "Tuple[Tensor, State]":
        self._validate_endpoint_step_input("step", wav)
        return self._codec_step(wav, state, "step")

    def step_desync(
        self,
        wav: "Tensor",
        state: "State",
        exec_mask: "Tensor",
    ) -> "Tuple[Tensor, State]":
        self._validate_endpoint_step_input("step_desync", wav)
        return self._codec_step(wav, state, "step_desync", exec_mask)

    def _codec_step(
        self,
        wav: "Tensor",
        state: "State",
        method_name: "str",
        exec_mask: "Optional[Tensor]" = None,
    ) -> "Tuple[Tensor, State]":
        self._validate_components(COMPONENT_NAMES)
        encoder_components = ("encoder", "frontend", "compressor")
        decoder_components = ("decompressor", "backend", "decoder")
        encoder_size = sum(
            len(state_names)
            for _, state_names in self._component_state_layout(
                encoder_components,
                method_name,
            )
        )

        lats, encoder_state = self._run_components(
            wav,
            encoder_components,
            method_name,
            state[:encoder_size],
            exec_mask,
        )
        toks = self.lats_to_toks(lats)
        codes = self.toks_to_codes(toks)
        wav, decoder_state = self._run_components(
            codes,
            decoder_components,
            method_name,
            state[encoder_size:],
            exec_mask,
        )
        return wav, encoder_state + decoder_state

    def wav_to_frame_feats(self, wav: "Tensor") -> "Tensor":
        return self._component("encoder")(wav)

    def wav_to_frame_feats_step(
        self,
        wav: "Tensor",
        state: "State",
    ) -> "Tuple[Tensor, State]":
        self._validate_endpoint_step_input("wav_to_frame_feats_step", wav)
        return self._run_components(
            wav,
            ("encoder",),
            "step",
            state,
        )

    def wav_to_frame_feats_step_desync(
        self, wav: "Tensor", state: "State", exec_mask: "Tensor"
    ) -> "Tuple[Tensor, State]":
        self._validate_endpoint_step_input("wav_to_frame_feats_step_desync", wav)
        return self._run_components(
            wav,
            ("encoder",),
            "step_desync",
            state,
            exec_mask,
        )

    def wav_to_feats(self, wav: "Tensor") -> "Tensor":
        components = ("encoder", "frontend")
        self._validate_components(components)
        output = wav
        for name in components:
            output = self._component(name)(output)
        return output

    def wav_to_feats_step(
        self,
        wav: "Tensor",
        state: "State",
    ) -> "Tuple[Tensor, State]":
        self._validate_endpoint_step_input("wav_to_feats_step", wav)
        return self._run_components(
            wav,
            ("encoder", "frontend"),
            "step",
            state,
        )

    def wav_to_feats_step_desync(
        self, wav: "Tensor", state: "State", exec_mask: "Tensor"
    ) -> "Tuple[Tensor, State]":
        self._validate_endpoint_step_input("wav_to_feats_step_desync", wav)
        return self._run_components(
            wav,
            ("encoder", "frontend"),
            "step_desync",
            state,
            exec_mask,
        )

    def wav_to_lats(self, wav: "Tensor") -> "Tensor":
        components = ("encoder", "frontend", "compressor")
        self._validate_components(components)
        output = wav
        for name in components:
            output = self._component(name)(output)
        return output

    def wav_to_lats_step(
        self,
        wav: "Tensor",
        state: "State",
    ) -> "Tuple[Tensor, State]":
        self._validate_endpoint_step_input("wav_to_lats_step", wav)
        return self._run_components(
            wav,
            ("encoder", "frontend", "compressor"),
            "step",
            state,
        )

    def wav_to_lats_step_desync(
        self, wav: "Tensor", state: "State", exec_mask: "Tensor"
    ) -> "Tuple[Tensor, State]":
        self._validate_endpoint_step_input("wav_to_lats_step_desync", wav)
        return self._run_components(
            wav,
            ("encoder", "frontend", "compressor"),
            "step_desync",
            state,
            exec_mask,
        )

    def wav_to_toks(self, wav: "Tensor") -> "Tensor":
        return self.lats_to_toks(self.wav_to_lats(wav))

    def wav_to_toks_step(
        self,
        wav: "Tensor",
        state: "State",
    ) -> "Tuple[Tensor, State]":
        self._validate_endpoint_step_input("wav_to_toks_step", wav)
        lats, state = self._run_components(
            wav,
            ("encoder", "frontend", "compressor"),
            "step",
            state,
        )
        return self.lats_to_toks(lats), state

    def wav_to_toks_step_desync(
        self, wav: "Tensor", state: "State", exec_mask: "Tensor"
    ) -> "Tuple[Tensor, State]":
        lats, state = self.wav_to_lats_step_desync(wav, state, exec_mask)
        return self.lats_to_toks(lats), state

    def wav_to_qfeats(self, wav: "Tensor") -> "Tensor":
        toks = self.wav_to_toks(wav)
        codes = self.toks_to_codes(toks)
        return self.codes_to_qfeats(codes)

    def wav_to_frame_qfeats(self, wav: "Tensor") -> "Tensor":
        toks = self.wav_to_toks(wav)
        codes = self.toks_to_codes(toks)
        return self.codes_to_frame_qfeats(codes)

    def wav_to_wav_vc(
        self,
        wav: "Tensor",
        matching_set: "Tensor",
        topk: "int" = 4,
        num_splits: "int" = 1,
    ) -> "Tensor":
        """Reconstruct audio after frame-level kNN voice conversion."""
        qfeats = self.wav_to_frame_qfeats(wav)
        matched_qfeats = self.knn(
            qfeats,
            matching_set,
            topk=topk,
            num_splits=num_splits,
        )
        return self.frame_qfeats_to_wav(matched_qfeats)

    def wav_to_wav_vc_step(
        self,
        wav: "Tensor",
        matching_set: "Tensor",
        state: "State",
    ) -> "Tuple[Tensor, State]":
        """Stream audio through frame-level kNN voice conversion."""
        self._validate_endpoint_step_input("wav_to_wav_vc_step", wav)
        encoder_components = ("encoder", "frontend", "compressor")
        qfeat_components = ("decompressor", "backend")
        decoder_components = ("decoder",)
        encoder_size = sum(
            len(state_names)
            for _, state_names in self._component_state_layout(
                encoder_components,
                "step",
            )
        )
        qfeat_size = sum(
            len(state_names)
            for _, state_names in self._component_state_layout(
                qfeat_components,
                "step",
            )
        )
        lats, encoder_state = self._run_components(
            wav,
            encoder_components,
            "step",
            state[:encoder_size],
        )
        toks = self.lats_to_toks(lats)
        codes = self.toks_to_codes(toks)
        qfeats, qfeat_state = self._run_components(
            codes,
            qfeat_components,
            "step",
            state[encoder_size : encoder_size + qfeat_size],
        )
        matched_qfeats = self.knn(qfeats, matching_set)
        wav_vc, decoder_state = self._run_components(
            matched_qfeats,
            decoder_components,
            "step",
            state[encoder_size + qfeat_size :],
        )
        return wav_vc, encoder_state + qfeat_state + decoder_state

    def wav_to_wav_vc_step_desync(
        self,
        wav: "Tensor",
        matching_set: "Tensor",
        state: "State",
        exec_mask: "Tensor",
    ) -> "Tuple[Tensor, State]":
        components = ("encoder", "frontend", "compressor")
        qfeat_components = ("decompressor", "backend")
        encoder_size = sum(
            len(names)
            for _, names in self._component_state_layout(components, "step_desync")
        )
        qfeat_size = sum(
            len(names)
            for _, names in self._component_state_layout(
                qfeat_components, "step_desync"
            )
        )
        lats, encoder_state = self._run_components(
            wav, components, "step_desync", state[:encoder_size], exec_mask
        )
        codes = self.toks_to_codes(self.lats_to_toks(lats))
        qfeats, qfeat_state = self._run_components(
            codes,
            qfeat_components,
            "step_desync",
            state[encoder_size : encoder_size + qfeat_size],
            exec_mask,
        )
        wav_vc, decoder_state = self._run_components(
            self.knn(qfeats, matching_set),
            ("decoder",),
            "step_desync",
            state[encoder_size + qfeat_size :],
            exec_mask,
        )
        return wav_vc, encoder_state + qfeat_state + decoder_state

    def wav_to_frame_qfeats_step(
        self,
        wav: "Tensor",
        state: "State",
    ) -> "Tuple[Tensor, State]":
        self._validate_endpoint_step_input("wav_to_frame_qfeats_step", wav)
        encoder_components = ("encoder", "frontend", "compressor")
        qfeat_components = ("decompressor", "backend")
        encoder_size = sum(
            len(state_names)
            for _, state_names in self._component_state_layout(
                encoder_components,
                "step",
            )
        )
        lats, encoder_state = self._run_components(
            wav,
            encoder_components,
            "step",
            state[:encoder_size],
        )
        toks = self.lats_to_toks(lats)
        codes = self.toks_to_codes(toks)
        qfeats, qfeat_state = self._run_components(
            codes,
            qfeat_components,
            "step",
            state[encoder_size:],
        )
        return qfeats, encoder_state + qfeat_state

    def wav_to_frame_qfeats_step_desync(
        self, wav: "Tensor", state: "State", exec_mask: "Tensor"
    ) -> "Tuple[Tensor, State]":
        self._validate_endpoint_step_input("wav_to_frame_qfeats_step_desync", wav)
        encoder_components = ("encoder", "frontend", "compressor")
        qfeat_components = ("decompressor", "backend")
        encoder_size = sum(
            len(names)
            for _, names in self._component_state_layout(
                encoder_components, "step_desync"
            )
        )
        lats, encoder_state = self._run_components(
            wav, encoder_components, "step_desync", state[:encoder_size], exec_mask
        )
        codes = self.toks_to_codes(self.lats_to_toks(lats))
        qfeats, qfeat_state = self._run_components(
            codes, qfeat_components, "step_desync", state[encoder_size:], exec_mask
        )
        return qfeats, encoder_state + qfeat_state

    def wav_to_qfeats_step(
        self,
        wav: "Tensor",
        state: "State",
    ) -> "Tuple[Tensor, State]":
        self._validate_endpoint_step_input("wav_to_qfeats_step", wav)
        encoder_components = ("encoder", "frontend", "compressor")
        encoder_size = sum(
            len(state_names)
            for _, state_names in self._component_state_layout(
                encoder_components,
                "step",
            )
        )
        lats, encoder_state = self._run_components(
            wav,
            encoder_components,
            "step",
            state[:encoder_size],
        )
        codes = self.toks_to_codes(self.lats_to_toks(lats))
        qfeats, qfeat_state = self._run_components(
            codes,
            ("decompressor",),
            "step",
            state[encoder_size:],
        )
        return qfeats, encoder_state + qfeat_state

    def wav_to_qfeats_step_desync(
        self, wav: "Tensor", state: "State", exec_mask: "Tensor"
    ) -> "Tuple[Tensor, State]":
        self._validate_endpoint_step_input("wav_to_qfeats_step_desync", wav)
        encoder_components = ("encoder", "frontend", "compressor")
        encoder_size = sum(
            len(names)
            for _, names in self._component_state_layout(
                encoder_components, "step_desync"
            )
        )
        lats, encoder_state = self._run_components(
            wav,
            encoder_components,
            "step_desync",
            state[:encoder_size],
            exec_mask,
        )
        codes = self.toks_to_codes(self.lats_to_toks(lats))
        qfeats, qfeat_state = self._run_components(
            codes,
            ("decompressor",),
            "step_desync",
            state[encoder_size:],
            exec_mask,
        )
        return qfeats, encoder_state + qfeat_state

    def feats_to_lats(self, feats: "Tensor") -> "Tensor":
        return self._component("compressor")(feats)

    def feats_to_lats_step(
        self,
        feats: "Tensor",
        state: "State",
    ) -> "Tuple[Tensor, State]":
        self._validate_endpoint_step_input("feats_to_lats_step", feats)
        return self._run_components(feats, ("compressor",), "step", state)

    def feats_to_lats_step_desync(
        self, feats: "Tensor", state: "State", exec_mask: "Tensor"
    ) -> "Tuple[Tensor, State]":
        self._validate_endpoint_step_input("feats_to_lats_step_desync", feats)
        return self._run_components(
            feats, ("compressor",), "step_desync", state, exec_mask
        )

    def feats_to_toks(self, feats: "Tensor") -> "Tensor":
        return self.lats_to_toks(self.feats_to_lats(feats))

    def feats_to_toks_step(
        self,
        feats: "Tensor",
        state: "State",
    ) -> "Tuple[Tensor, State]":
        self._validate_endpoint_step_input("feats_to_toks_step", feats)
        lats, state = self.feats_to_lats_step(feats, state)
        return self.lats_to_toks(lats), state

    def feats_to_toks_step_desync(
        self, feats: "Tensor", state: "State", exec_mask: "Tensor"
    ) -> "Tuple[Tensor, State]":
        lats, state = self.feats_to_lats_step_desync(feats, state, exec_mask)
        return self.lats_to_toks(lats), state

    def feats_to_qfeats(self, feats: "Tensor") -> "Tensor":
        return self.lats_to_qfeats(self.feats_to_lats(feats))

    def feats_to_qfeats_step(
        self,
        feats: "Tensor",
        state: "State",
    ) -> "Tuple[Tensor, State]":
        self._validate_endpoint_step_input("feats_to_qfeats_step", feats)
        compressor_state_size = sum(
            len(names)
            for _, names in self._component_state_layout(("compressor",), "step")
        )
        lats, compressor_state = self._run_components(
            feats, ("compressor",), "step", state[:compressor_state_size]
        )
        qfeats, decompressor_state = self._run_components(
            self.lats_to_codes(lats),
            ("decompressor",),
            "step",
            state[compressor_state_size:],
        )
        return qfeats, compressor_state + decompressor_state

    def feats_to_qfeats_step_desync(
        self, feats: "Tensor", state: "State", exec_mask: "Tensor"
    ) -> "Tuple[Tensor, State]":
        self._validate_endpoint_step_input("feats_to_qfeats_step_desync", feats)
        compressor_state_size = sum(
            len(names)
            for _, names in self._component_state_layout(("compressor",), "step_desync")
        )
        lats, compressor_state = self._run_components(
            feats,
            ("compressor",),
            "step_desync",
            state[:compressor_state_size],
            exec_mask,
        )
        qfeats, decompressor_state = self._run_components(
            self.lats_to_codes(lats),
            ("decompressor",),
            "step_desync",
            state[compressor_state_size:],
            exec_mask,
        )
        return qfeats, compressor_state + decompressor_state

    def lats_to_toks(self, lats: "Tensor") -> "Tensor":
        quantizer = self._component("quantizer")
        encode = getattr(quantizer, "encode", None)
        if encode is not None:
            return encode(lats)
        encoder = getattr(quantizer, "encoder", None)
        if encoder is None:
            raise RuntimeError("Quantizer must define encode() or encoder()")
        return encoder(lats)

    def lats_to_codes(self, lats: "Tensor") -> "Tensor":
        return self.toks_to_codes(self.lats_to_toks(lats))

    def lats_to_qfeats(self, lats: "Tensor") -> "Tensor":
        return self.codes_to_qfeats(self.lats_to_codes(lats))

    def lats_to_qfeats_step(
        self,
        lats: "Tensor",
        state: "State",
    ) -> "Tuple[Tensor, State]":
        self._validate_endpoint_step_input("lats_to_qfeats_step", lats)
        return self._run_components(
            self.lats_to_codes(lats), ("decompressor",), "step", state
        )

    def lats_to_qfeats_step_desync(
        self, lats: "Tensor", state: "State", exec_mask: "Tensor"
    ) -> "Tuple[Tensor, State]":
        self._validate_endpoint_step_input("lats_to_qfeats_step_desync", lats)
        return self._run_components(
            self.lats_to_codes(lats),
            ("decompressor",),
            "step_desync",
            state,
            exec_mask,
        )

    def lats_to_wav(self, lats: "Tensor") -> "Tensor":
        return self.qfeats_to_wav(self.lats_to_qfeats(lats))

    def lats_to_wav_step(
        self,
        lats: "Tensor",
        state: "State",
    ) -> "Tuple[Tensor, State]":
        self._validate_endpoint_step_input("lats_to_wav_step", lats)
        return self._run_components(
            self.lats_to_codes(lats),
            ("decompressor", "backend", "decoder"),
            "step",
            state,
        )

    def lats_to_wav_step_desync(
        self, lats: "Tensor", state: "State", exec_mask: "Tensor"
    ) -> "Tuple[Tensor, State]":
        self._validate_endpoint_step_input("lats_to_wav_step_desync", lats)
        return self._run_components(
            self.lats_to_codes(lats),
            ("decompressor", "backend", "decoder"),
            "step_desync",
            state,
            exec_mask,
        )

    def toks_to_codes(self, toks: "Tensor") -> "Tensor":
        quantizer = self._component("quantizer")
        decode = getattr(quantizer, "decode", None)
        if decode is None:
            raise RuntimeError("Quantizer must define decode()")
        return decode(toks)

    def codes_to_toks(self, codes: "Tensor") -> "Tensor":
        return self.lats_to_toks(codes)

    def toks_to_qfeats(self, toks: "Tensor") -> "Tensor":
        return self.codes_to_qfeats(self.toks_to_codes(toks))

    def toks_to_qfeats_step(
        self,
        toks: "Tensor",
        state: "State",
    ) -> "Tuple[Tensor, State]":
        self._validate_endpoint_step_input("toks_to_qfeats_step", toks)
        return self._run_components(
            self.toks_to_codes(toks), ("decompressor",), "step", state
        )

    def toks_to_qfeats_step_desync(
        self, toks: "Tensor", state: "State", exec_mask: "Tensor"
    ) -> "Tuple[Tensor, State]":
        self._validate_endpoint_step_input("toks_to_qfeats_step_desync", toks)
        return self._run_components(
            self.toks_to_codes(toks),
            ("decompressor",),
            "step_desync",
            state,
            exec_mask,
        )

    def toks_to_frame_qfeats(self, toks: "Tensor") -> "Tensor":
        return self.codes_to_frame_qfeats(self.toks_to_codes(toks))

    def toks_to_frame_qfeats_step(
        self,
        toks: "Tensor",
        state: "State",
    ) -> "Tuple[Tensor, State]":
        self._validate_endpoint_step_input("toks_to_frame_qfeats_step", toks)
        return self._run_components(
            self.toks_to_codes(toks),
            ("decompressor", "backend"),
            "step",
            state,
        )

    def toks_to_frame_qfeats_step_desync(
        self, toks: "Tensor", state: "State", exec_mask: "Tensor"
    ) -> "Tuple[Tensor, State]":
        self._validate_endpoint_step_input("toks_to_frame_qfeats_step_desync", toks)
        return self._run_components(
            self.toks_to_codes(toks),
            ("decompressor", "backend"),
            "step_desync",
            state,
            exec_mask,
        )

    def toks_to_wav(self, toks: "Tensor") -> "Tensor":
        codes = self.toks_to_codes(toks)
        qfeats = self.codes_to_qfeats(codes)
        return self.qfeats_to_wav(qfeats)

    def toks_to_wav_vc(
        self,
        toks: "Tensor",
        matching_set: "Tensor",
        topk: "int" = 4,
        num_splits: "int" = 1,
    ) -> "Tensor":
        """Decode tokens after frame-level kNN voice conversion."""
        codes = self.toks_to_codes(toks)
        qfeats = self.codes_to_frame_qfeats(codes)
        matched_qfeats = self.knn(
            qfeats,
            matching_set,
            topk=topk,
            num_splits=num_splits,
        )
        return self.frame_qfeats_to_wav(matched_qfeats)

    def toks_to_wav_vc_step(
        self,
        toks: "Tensor",
        matching_set: "Tensor",
        state: "State",
    ) -> "Tuple[Tensor, State]":
        """Stream tokens through frame-level kNN voice conversion."""
        self._validate_endpoint_step_input("toks_to_wav_vc_step", toks)
        qfeat_components = ("decompressor", "backend")
        decoder_components = ("decoder",)
        qfeat_size = sum(
            len(state_names)
            for _, state_names in self._component_state_layout(
                qfeat_components,
                "step",
            )
        )
        codes = self.toks_to_codes(toks)
        qfeats, qfeat_state = self._run_components(
            codes,
            qfeat_components,
            "step",
            state[:qfeat_size],
        )
        matched_qfeats = self.knn(qfeats, matching_set)
        wav_vc, decoder_state = self._run_components(
            matched_qfeats,
            decoder_components,
            "step",
            state[qfeat_size:],
        )
        return wav_vc, qfeat_state + decoder_state

    def toks_to_wav_vc_step_desync(
        self,
        toks: "Tensor",
        matching_set: "Tensor",
        state: "State",
        exec_mask: "Tensor",
    ) -> "Tuple[Tensor, State]":
        qfeat_components = ("decompressor", "backend")
        qfeat_size = sum(
            len(names)
            for _, names in self._component_state_layout(
                qfeat_components, "step_desync"
            )
        )
        qfeats, qfeat_state = self._run_components(
            self.toks_to_codes(toks),
            qfeat_components,
            "step_desync",
            state[:qfeat_size],
            exec_mask,
        )
        wav_vc, decoder_state = self._run_components(
            self.knn(qfeats, matching_set),
            ("decoder",),
            "step_desync",
            state[qfeat_size:],
            exec_mask,
        )
        return wav_vc, qfeat_state + decoder_state

    def toks_to_wav_step(
        self,
        toks: "Tensor",
        state: "State",
    ) -> "Tuple[Tensor, State]":
        self._validate_endpoint_step_input("toks_to_wav_step", toks)
        codes = self.toks_to_codes(toks)
        wav, decoder_state = self._run_components(
            codes,
            ("decompressor", "backend", "decoder"),
            "step",
            state,
        )
        return wav, decoder_state

    def toks_to_wav_step_desync(
        self, toks: "Tensor", state: "State", exec_mask: "Tensor"
    ) -> "Tuple[Tensor, State]":
        self._validate_endpoint_step_input("toks_to_wav_step_desync", toks)
        return self._run_components(
            self.toks_to_codes(toks),
            ("decompressor", "backend", "decoder"),
            "step_desync",
            state,
            exec_mask,
        )

    def codes_to_qfeats(self, codes: "Tensor") -> "Tensor":
        return self._component("decompressor")(codes)

    def codes_to_frame_qfeats(self, codes: "Tensor") -> "Tensor":
        components = ("decompressor", "backend")
        self._validate_components(components)
        output = codes
        for name in components:
            output = self._component(name)(output)
        return output

    def codes_to_qfeats_step(
        self,
        codes: "Tensor",
        state: "State",
    ) -> "Tuple[Tensor, State]":
        self._validate_endpoint_step_input("codes_to_qfeats_step", codes)
        return self._run_components(
            codes,
            ("decompressor",),
            "step",
            state,
        )

    def codes_to_qfeats_step_desync(
        self, codes: "Tensor", state: "State", exec_mask: "Tensor"
    ) -> "Tuple[Tensor, State]":
        self._validate_endpoint_step_input("codes_to_qfeats_step_desync", codes)
        return self._run_components(
            codes, ("decompressor",), "step_desync", state, exec_mask
        )

    def codes_to_frame_qfeats_step(
        self,
        codes: "Tensor",
        state: "State",
    ) -> "Tuple[Tensor, State]":
        self._validate_endpoint_step_input("codes_to_frame_qfeats_step", codes)
        return self._run_components(
            codes,
            ("decompressor", "backend"),
            "step",
            state,
        )

    def codes_to_frame_qfeats_step_desync(
        self, codes: "Tensor", state: "State", exec_mask: "Tensor"
    ) -> "Tuple[Tensor, State]":
        self._validate_endpoint_step_input("codes_to_frame_qfeats_step_desync", codes)
        return self._run_components(
            codes,
            ("decompressor", "backend"),
            "step_desync",
            state,
            exec_mask,
        )

    def qfeats_to_wav(self, qfeats: "Tensor") -> "Tensor":
        components = ("backend", "decoder")
        self._validate_components(components)
        output = qfeats
        for name in components:
            output = self._component(name)(output)
        return output

    def qfeats_to_frame_qfeats(self, qfeats: "Tensor") -> "Tensor":
        return self._component("backend")(qfeats)

    def qfeats_to_frame_qfeats_step(
        self,
        qfeats: "Tensor",
        state: "State",
    ) -> "Tuple[Tensor, State]":
        self._validate_endpoint_step_input("qfeats_to_frame_qfeats_step", qfeats)
        return self._run_components(qfeats, ("backend",), "step", state)

    def qfeats_to_frame_qfeats_step_desync(
        self, qfeats: "Tensor", state: "State", exec_mask: "Tensor"
    ) -> "Tuple[Tensor, State]":
        self._validate_endpoint_step_input("qfeats_to_frame_qfeats_step_desync", qfeats)
        return self._run_components(
            qfeats, ("backend",), "step_desync", state, exec_mask
        )

    def qfeats_to_wav_step(
        self,
        qfeats: "Tensor",
        state: "State",
    ) -> "Tuple[Tensor, State]":
        self._validate_endpoint_step_input("qfeats_to_wav_step", qfeats)
        return self._run_components(
            qfeats,
            ("backend", "decoder"),
            "step",
            state,
        )

    def qfeats_to_wav_step_desync(
        self, qfeats: "Tensor", state: "State", exec_mask: "Tensor"
    ) -> "Tuple[Tensor, State]":
        self._validate_endpoint_step_input("qfeats_to_wav_step_desync", qfeats)
        return self._run_components(
            qfeats, ("backend", "decoder"), "step_desync", state, exec_mask
        )

    def frame_qfeats_to_wav(self, frame_qfeats: "Tensor") -> "Tensor":
        return self._component("decoder")(frame_qfeats)

    def frame_qfeats_to_wav_step(
        self,
        frame_qfeats: "Tensor",
        state: "State",
    ) -> "Tuple[Tensor, State]":
        self._validate_endpoint_step_input("frame_qfeats_to_wav_step", frame_qfeats)
        return self._run_components(
            frame_qfeats,
            ("decoder",),
            "step",
            state,
        )

    def frame_qfeats_to_wav_step_desync(
        self, frame_qfeats: "Tensor", state: "State", exec_mask: "Tensor"
    ) -> "Tuple[Tensor, State]":
        self._validate_endpoint_step_input(
            "frame_qfeats_to_wav_step_desync", frame_qfeats
        )
        return self._run_components(
            frame_qfeats,
            ("decoder",),
            "step_desync",
            state,
            exec_mask,
        )

    def knn(
        self,
        frame_qfeats: "Tensor",
        matching_set: "Tensor",
        topk: "int" = 4,
        num_splits: "int" = 1,
    ) -> "Tensor":
        """Match each frame qfeat to the mean of its nearest target frames."""
        return knn_qfeats(
            frame_qfeats,
            matching_set,
            topk=topk,
            num_splits=num_splits,
        )

    def endpoint(self, name: "str") -> "EndpointSpec":
        try:
            endpoint = self.endpoints[name]
        except KeyError as error:
            available = ", ".join(sorted(self.endpoints))
            raise KeyError(
                f"Unknown endpoint {name!r}; available: {available}"
            ) from error
        self._validate_components(endpoint.required_components)
        return endpoint

    def register_endpoint(
        self,
        name: "str",
        input_names: "Union[str, Tuple[str, ...]]",
        output_names: "Union[str, Tuple[str, ...]]",
        required_components: "Tuple[str, ...]",
        state_components: "Tuple[str, ...]" = (),
        state_method: "Optional[str]" = None,
        replace: "bool" = False,
    ) -> "ZipCodec":
        """Register a model method as a compilable dictionary endpoint."""
        if not hasattr(self, name) or not callable(getattr(self, name)):
            raise AttributeError(f"ZipCodec has no callable method {name!r}")
        unknown_components = (set(required_components) | set(state_components)) - set(
            COMPONENT_NAMES
        )
        if unknown_components:
            raise ValueError(f"Unknown components: {sorted(unknown_components)}")
        if name in self.endpoints and not replace:
            raise ValueError(
                f"Endpoint {name!r} is already registered; pass replace=True to replace it"
            )
        self.endpoints[name] = endpoint_spec(
            name,
            input_names,
            output_names,
            required_components,
            state_components,
            state_method,
        )
        return self

    def endpoint_spec(self, name: "str") -> "EndpointSpec":
        """Return the specification registered for an endpoint."""
        try:
            return self.endpoints[name]
        except KeyError as error:
            available = ", ".join(sorted(self.endpoints))
            raise KeyError(
                f"Unknown endpoint {name!r}; available: {available}"
            ) from error

    def components_for_endpoint(self, name: "str") -> "Tuple[str, ...]":
        """Return the model components required by an endpoint."""
        return self.endpoint_spec(name).required_components

    def prune_for_endpoint(self, name: "str") -> "ZipCodec":
        """Release unneeded components by replacing their attributes with ``None``."""
        required = set(self.components_for_endpoint(name))
        for component_name in COMPONENT_NAMES:
            if component_name not in required:
                setattr(self, component_name, None)
        return self

    def endpoint_module(
        self,
        name: "str",
    ) -> "EndpointModule":
        return EndpointModule(self, self.endpoint(name))

    def compile(
        self,
        endpoint_name: "str",
        **kwargs: "Any",
    ) -> "_deployment.TorchCompileEndpoint":
        """Compile an endpoint with :func:`torch.compile`."""
        return _deployment.compile_torch_endpoint(self, endpoint_name, **kwargs)

    def dict_endpoint(self, name: "str") -> "DictEndpoint":
        endpoint = self.endpoint(name)
        return DictEndpoint(EndpointModule(self, endpoint), endpoint)

    def eager(self, endpoint_name: "str") -> "DictEndpoint":
        """Return an eager endpoint with dictionary inputs and outputs."""
        return self.dict_endpoint(endpoint_name)

    def jit(
        self,
        endpoint_name: "str",
        example_inputs: "Dict[str, Any]",
        check_trace: "bool" = True,
        strict: "bool" = True,
        fusion: "bool" = False,
    ) -> "_deployment.JITEndpoint":
        """Trace an endpoint and expose it through the dictionary interface."""
        return _deployment.compile_jit_endpoint(
            self,
            endpoint_name,
            example_inputs,
            check_trace=check_trace,
            strict=strict,
            fusion=fusion,
        )

    def cudagraph(
        self,
        endpoint_name: "str",
        example_inputs: "Dict[str, Any]",
        warmup_steps: "int" = 3,
    ) -> "_deployment.CUDAGraphEndpoint":
        """Capture an endpoint in a CUDA graph."""
        return _deployment.compile_cuda_graph_endpoint(
            self,
            endpoint_name,
            example_inputs,
            warmup_steps=warmup_steps,
        )

    def onnx(
        self,
        endpoint_name: "str",
        example_inputs: "Dict[str, Any]",
        dynamic_axes: "Optional[Dict[str, Dict[int, str]]]" = None,
        opset_version: "int" = 17,
        output_device: "Optional[torch.device]" = None,
        providers: "Optional[Tuple[str, ...]]" = None,
        temporary_directory: "Optional[Any]" = None,
        io_binding: "bool" = False,
    ) -> "_deployment.ONNXEndpoint":
        """Export an endpoint and create an ONNX Runtime callable."""
        return _deployment.compile_onnx_endpoint(
            self,
            endpoint_name,
            example_inputs,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            output_device=output_device,
            providers=providers,
            temporary_directory=temporary_directory,
            io_binding=io_binding,
        )

    def export_onnx(
        self,
        endpoint_name: "str",
        example_inputs: "Dict[str, Any]",
        output_path: "Union[str, Path]",
        dynamic_axes: "Optional[Dict[str, Dict[int, str]]]" = None,
        opset_version: "int" = 17,
        metadata_path: "Optional[Union[str, Path]]" = None,
        example_inputs_path: "Optional[Union[str, Path]]" = None,
    ) -> "Tuple[Path, Path, Path]":
        """Persist an ONNX model, I/O metadata, and example NumPy inputs."""
        return _deployment.export_onnx_endpoint(
            self,
            endpoint_name,
            example_inputs,
            output_path,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            metadata_path=metadata_path,
            example_inputs_path=example_inputs_path,
        )

    def to_config(
        self,
        config: "str",
        pretrained: "bool" = False,
        registry: "Optional[Any]" = None,
    ) -> "None":
        """Save the model configuration and optionally its checkpoint."""
        try:
            from .pretrained import to_config
        except ImportError:
            from pretrained import to_config

        to_config(
            self,
            config,
            pretrained=pretrained,
            registry=registry,
        )

    def to_pretrained(
        self,
        config: "str",
        registry: "Optional[Any]" = None,
    ) -> "None":
        """Save the model configuration and safetensors checkpoint."""
        self.to_config(config, pretrained=True, registry=registry)

    @classmethod
    def from_config(
        cls,
        config: "str",
        pretrained: "bool" = False,
        components: "Optional[Sequence[str]]" = None,
        endpoint: "Optional[str]" = None,
        overrides: "Optional[Dict[str, Any]]" = None,
        registry: "Optional[Any]" = None,
        **kwargs: "Any",
    ) -> "ZipCodec":
        """Build a model from local or Hugging Face configuration."""
        try:
            from .pretrained import from_config
        except ImportError:
            from pretrained import from_config

        return from_config(
            config,
            pretrained=pretrained,
            components=components,
            endpoint=endpoint,
            overrides=overrides,
            registry=registry,
            **kwargs,
        )

    @classmethod
    def from_pretrained(
        cls,
        config: "str",
        components: "Optional[Sequence[str]]" = None,
        endpoint: "Optional[str]" = None,
        overrides: "Optional[Dict[str, Any]]" = None,
        registry: "Optional[Any]" = None,
        **kwargs: "Any",
    ) -> "ZipCodec":
        """Build a model and load its checkpoint."""
        return cls.from_config(
            config,
            pretrained=True,
            components=components,
            endpoint=endpoint,
            overrides=overrides,
            registry=registry,
            **kwargs,
        )
