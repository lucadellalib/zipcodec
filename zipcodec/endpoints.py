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

"""Declarative ZipCodec endpoints and dictionary adapters."""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import torch
from torch import Tensor, nn


__all__ = [
    "DictEndpoint",
    "EndpointModule",
    "EndpointSpec",
    "endpoint_spec",
    "make_zipcodec_endpoints",
]


State = Tuple[Tensor, ...]
Names = Union[str, Tuple[str, ...]]


def _as_state(value: "Any") -> "Tuple[Tensor, ...]":
    if isinstance(value, Tensor):
        return (value,)
    if isinstance(value, tuple) and all(isinstance(x, Tensor) for x in value):
        return value
    raise TypeError("State must be a Tensor or a flat tuple of Tensors")


@dataclass(frozen=True)
class EndpointSpec:
    """Declarative description of a public ZipCodec endpoint."""

    name: str
    method: str
    input_names: Tuple[str, ...]
    output_names: Tuple[str, ...]
    required_components: Tuple[str, ...]
    state_components: Tuple[str, ...] = ()
    state_method: Optional[str] = None

    def __post_init__(self) -> "None":
        if ("state" in self.input_names) != ("next_state" in self.output_names):
            raise ValueError(
                f"Endpoint {self.name!r} must declare both state and next_state"
            )
        if self.state_components and self.state_method is None:
            raise ValueError(
                f"Endpoint {self.name!r} has state components but no state method"
            )

    @property
    def has_state(self) -> "bool":
        return bool(self.state_components)

    def init_state(
        self,
        model: "nn.Module",
        batch_size: "int",
        device: "torch.device",
        dtype: "torch.dtype",
    ) -> "State":
        """Initialize exactly the state consumed by this endpoint."""
        return model._init_component_state(
            self.state_components,
            self.state_method,
            batch_size,
            device,
            dtype,
        )


class EndpointModule(nn.Module):
    """Expose an endpoint using only flat Tensor inputs and outputs."""

    def __init__(
        self,
        model: "nn.Module",
        endpoint: "EndpointSpec",
    ) -> "None":
        super().__init__()
        self.model = model
        self.endpoint = endpoint
        self.state_layout = model._component_state_layout(
            endpoint.state_components,
            endpoint.state_method,
        )
        self.state_size = sum(len(state_names) for _, state_names in self.state_layout)
        self.flat_input_names = self._expand_names(
            endpoint.input_names,
            state_prefix="",
        )
        self.flat_output_names = self._expand_names(
            endpoint.output_names,
            state_prefix="next_",
        )

    def _state_names(self, prefix: "str") -> "Tuple[str, ...]":
        names = []
        for component_name, state_names in self.state_layout:
            names.extend(
                f"{prefix}{component_name}_{state_name}" for state_name in state_names
            )
        return tuple(names)

    def _expand_names(
        self,
        names: "Tuple[str, ...]",
        state_prefix: "str",
    ) -> "Tuple[str, ...]":
        expanded = []
        for name in names:
            if name in ("state", "next_state"):
                expanded.extend(self._state_names(state_prefix))
            else:
                expanded.append(name)
        return tuple(expanded)

    @staticmethod
    def _validate_keys(
        inputs: "Dict[str, Any]",
        expected_names: "Tuple[str, ...]",
    ) -> "None":
        expected = set(expected_names)
        actual = set(inputs)
        missing = expected - actual
        extra = actual - expected
        if missing or extra:
            details = []
            if missing:
                details.append(f"missing={sorted(missing)}")
            if extra:
                details.append(f"extra={sorted(extra)}")
            raise KeyError("Invalid endpoint inputs: " + ", ".join(details))

    def flatten_inputs(self, inputs: "Dict[str, Any]") -> "Tuple[Tensor, ...]":
        self._validate_keys(inputs, self.endpoint.input_names)
        self.model._validate_endpoint_step_input(
            self.endpoint.name,
            inputs[self.endpoint.input_names[0]],
        )
        flat = []
        for name in self.endpoint.input_names:
            if name == "state":
                state = inputs[name]
                if not isinstance(state, tuple) or not all(
                    isinstance(value, Tensor) for value in state
                ):
                    raise TypeError("Endpoint state must be a flat tuple of Tensors")
                if len(state) != self.state_size:
                    raise ValueError(
                        f"Endpoint {self.endpoint.name!r} expects {self.state_size} "
                        f"state tensors, got {len(state)}"
                    )
                flat.extend(state)
            else:
                value = inputs[name]
                if not isinstance(value, Tensor):
                    raise TypeError(f"Input {name!r} must be a Tensor")
                flat.append(value)
        return tuple(flat)

    def unflatten_outputs(
        self,
        outputs: "Tuple[Tensor, ...]",
    ) -> "Dict[str, Any]":
        if len(outputs) != len(self.flat_output_names):
            raise RuntimeError(
                f"Endpoint {self.endpoint.name!r} returned {len(outputs)} tensors; "
                f"expected {len(self.flat_output_names)}"
            )

        result = {}
        offset = 0
        for name in self.endpoint.output_names:
            if name == "next_state":
                result[name] = tuple(outputs[offset : offset + self.state_size])
                offset += self.state_size
            else:
                result[name] = outputs[offset]
                offset += 1
        return result

    def forward(self, *args: "Tensor") -> "Tuple[Tensor, ...]":
        if len(args) != len(self.flat_input_names):
            raise ValueError(
                f"Endpoint {self.endpoint.name!r} expects "
                f"{len(self.flat_input_names)} tensors, got {len(args)}"
            )

        method_args = []
        offset = 0
        for name in self.endpoint.input_names:
            if name == "state":
                method_args.append(tuple(args[offset : offset + self.state_size]))
                offset += self.state_size
            else:
                method_args.append(args[offset])
                offset += 1

        method = getattr(self.model, self.endpoint.method)
        output = method(*method_args)
        output = output if isinstance(output, tuple) else (output,)

        flat_output = []
        for name, value in zip(self.endpoint.output_names, output, strict=True):
            if name == "next_state":
                state = _as_state(value)
                if len(state) != self.state_size:
                    raise RuntimeError(
                        f"Endpoint {self.endpoint.name!r} returned {len(state)} "
                        f"state tensors; expected {self.state_size}"
                    )
                flat_output.extend(state)
            else:
                if not isinstance(value, Tensor):
                    raise TypeError(f"Output {name!r} must be a Tensor")
                flat_output.append(value)

        return tuple(flat_output)


class DictEndpoint:
    """Eager dictionary-style endpoint."""

    def __init__(
        self,
        module: "EndpointModule",
        endpoint: "EndpointSpec",
    ) -> "None":
        self.module = module
        self.endpoint = endpoint

    def __call__(self, inputs: "Dict[str, Any]") -> "Dict[str, Any]":
        outputs = self.module(*self.module.flatten_inputs(inputs))
        return self.module.unflatten_outputs(outputs)


def endpoint_spec(
    name: "str",
    input_names: "Names",
    output_names: "Names",
    required_components: "Tuple[str, ...]",
    state_components: "Tuple[str, ...]" = (),
    state_method: "Optional[str]" = None,
) -> "EndpointSpec":
    """Create an endpoint specification with optional streaming state."""
    inputs = (input_names,) if isinstance(input_names, str) else input_names
    outputs = (output_names,) if isinstance(output_names, str) else output_names
    if not inputs:
        raise ValueError(f"Endpoint {name!r} must declare at least one input")
    if not outputs:
        raise ValueError(f"Endpoint {name!r} must declare at least one output")
    if "state" in inputs or "next_state" in outputs:
        raise ValueError("State names are added automatically from state_components")
    if state_components:
        if inputs[-1] == "exec_mask":
            inputs = inputs[:-1] + ("state", "exec_mask")
        else:
            inputs += ("state",)
        outputs += ("next_state",)
    return EndpointSpec(
        name=name,
        method=name,
        input_names=inputs,
        output_names=outputs,
        required_components=required_components,
        state_components=state_components,
        state_method=state_method,
    )


def _make_zipcodec_endpoints() -> "Dict[str, EndpointSpec]":
    encoder = ("encoder", "frontend", "compressor")
    decoder = ("decompressor", "backend", "decoder")
    full = encoder + ("quantizer",) + decoder

    specs = (
        endpoint_spec("forward", "wav", "wav_rec", full),
        endpoint_spec("step", "wav", "wav_rec", full, encoder + decoder, "step"),
        endpoint_spec(
            "wav_to_frame_feats",
            "wav",
            "frame_feats",
            ("encoder",),
        ),
        endpoint_spec(
            "wav_to_frame_feats_step",
            "wav",
            "frame_feats",
            ("encoder",),
            ("encoder",),
            "step",
        ),
        endpoint_spec(
            "wav_to_feats",
            "wav",
            "feats",
            ("encoder", "frontend"),
        ),
        endpoint_spec(
            "wav_to_feats_step",
            "wav",
            "feats",
            ("encoder", "frontend"),
            ("encoder", "frontend"),
            "step",
        ),
        endpoint_spec("wav_to_lats", "wav", "lats", encoder),
        endpoint_spec(
            "wav_to_lats_step",
            "wav",
            "lats",
            encoder,
            encoder,
            "step",
        ),
        endpoint_spec("wav_to_toks", "wav", "toks", encoder + ("quantizer",)),
        endpoint_spec(
            "wav_to_toks_step",
            "wav",
            "toks",
            encoder + ("quantizer",),
            encoder,
            "step",
        ),
        endpoint_spec(
            "wav_to_qfeats",
            "wav",
            "qfeats",
            encoder + ("quantizer", "decompressor"),
        ),
        endpoint_spec(
            "wav_to_qfeats_step",
            "wav",
            "qfeats",
            encoder + ("quantizer", "decompressor"),
            encoder + ("decompressor",),
            "step",
        ),
        endpoint_spec(
            "wav_to_frame_qfeats",
            "wav",
            "frame_qfeats",
            encoder + ("quantizer", "decompressor", "backend"),
        ),
        endpoint_spec(
            "wav_to_frame_qfeats_step",
            "wav",
            "frame_qfeats",
            encoder + ("quantizer", "decompressor", "backend"),
            encoder + ("decompressor", "backend"),
            "step",
        ),
        endpoint_spec(
            "wav_to_wav_vc",
            ("wav", "matching_set"),
            "wav_vc",
            full,
        ),
        endpoint_spec(
            "wav_to_wav_vc_step",
            ("wav", "matching_set"),
            "wav_vc",
            full,
            encoder + decoder,
            "step",
        ),
        endpoint_spec("feats_to_lats", "feats", "lats", ("compressor",)),
        endpoint_spec(
            "feats_to_lats_step",
            "feats",
            "lats",
            ("compressor",),
            ("compressor",),
            "step",
        ),
        endpoint_spec("feats_to_toks", "feats", "toks", ("compressor", "quantizer")),
        endpoint_spec(
            "feats_to_toks_step",
            "feats",
            "toks",
            ("compressor", "quantizer"),
            ("compressor",),
            "step",
        ),
        endpoint_spec(
            "feats_to_qfeats",
            "feats",
            "qfeats",
            ("compressor", "quantizer", "decompressor"),
        ),
        endpoint_spec(
            "feats_to_qfeats_step",
            "feats",
            "qfeats",
            ("compressor", "quantizer", "decompressor"),
            ("compressor", "decompressor"),
            "step",
        ),
        endpoint_spec("lats_to_toks", "lats", "toks", ("quantizer",)),
        endpoint_spec("lats_to_codes", "lats", "codes", ("quantizer",)),
        endpoint_spec(
            "lats_to_qfeats",
            "lats",
            "qfeats",
            ("quantizer", "decompressor"),
        ),
        endpoint_spec(
            "lats_to_qfeats_step",
            "lats",
            "qfeats",
            ("quantizer", "decompressor"),
            ("decompressor",),
            "step",
        ),
        endpoint_spec("lats_to_wav", "lats", "wav", ("quantizer",) + decoder),
        endpoint_spec(
            "lats_to_wav_step",
            "lats",
            "wav",
            ("quantizer",) + decoder,
            decoder,
            "step",
        ),
        endpoint_spec("toks_to_codes", "toks", "codes", ("quantizer",)),
        endpoint_spec("codes_to_toks", "codes", "toks", ("quantizer",)),
        endpoint_spec(
            "toks_to_qfeats",
            "toks",
            "qfeats",
            ("quantizer", "decompressor"),
        ),
        endpoint_spec(
            "toks_to_qfeats_step",
            "toks",
            "qfeats",
            ("quantizer", "decompressor"),
            ("decompressor",),
            "step",
        ),
        endpoint_spec(
            "toks_to_frame_qfeats",
            "toks",
            "frame_qfeats",
            ("quantizer", "decompressor", "backend"),
        ),
        endpoint_spec(
            "toks_to_frame_qfeats_step",
            "toks",
            "frame_qfeats",
            ("quantizer", "decompressor", "backend"),
            ("decompressor", "backend"),
            "step",
        ),
        endpoint_spec("toks_to_wav", "toks", "wav", ("quantizer",) + decoder),
        endpoint_spec(
            "toks_to_wav_vc",
            ("toks", "matching_set"),
            "wav_vc",
            ("quantizer",) + decoder,
        ),
        endpoint_spec(
            "toks_to_wav_vc_step",
            ("toks", "matching_set"),
            "wav_vc",
            ("quantizer",) + decoder,
            decoder,
            "step",
        ),
        endpoint_spec(
            "toks_to_wav_step",
            "toks",
            "wav",
            ("quantizer",) + decoder,
            decoder,
            "step",
        ),
        endpoint_spec(
            "codes_to_qfeats",
            "codes",
            "qfeats",
            ("decompressor",),
        ),
        endpoint_spec(
            "codes_to_qfeats_step",
            "codes",
            "qfeats",
            ("decompressor",),
            ("decompressor",),
            "step",
        ),
        endpoint_spec(
            "codes_to_frame_qfeats",
            "codes",
            "frame_qfeats",
            ("decompressor", "backend"),
        ),
        endpoint_spec(
            "codes_to_frame_qfeats_step",
            "codes",
            "frame_qfeats",
            ("decompressor", "backend"),
            ("decompressor", "backend"),
            "step",
        ),
        endpoint_spec(
            "qfeats_to_wav",
            "qfeats",
            "wav",
            ("backend", "decoder"),
        ),
        endpoint_spec(
            "qfeats_to_frame_qfeats",
            "qfeats",
            "frame_qfeats",
            ("backend",),
        ),
        endpoint_spec(
            "qfeats_to_frame_qfeats_step",
            "qfeats",
            "frame_qfeats",
            ("backend",),
            ("backend",),
            "step",
        ),
        endpoint_spec(
            "qfeats_to_wav_step",
            "qfeats",
            "wav",
            ("backend", "decoder"),
            ("backend", "decoder"),
            "step",
        ),
        endpoint_spec(
            "frame_qfeats_to_wav",
            "frame_qfeats",
            "wav",
            ("decoder",),
        ),
        endpoint_spec(
            "frame_qfeats_to_wav_step",
            "frame_qfeats",
            "wav",
            ("decoder",),
            ("decoder",),
            "step",
        ),
        endpoint_spec(
            "knn",
            ("frame_qfeats", "matching_set"),
            "matched_frame_qfeats",
            (),
        ),
    )
    desync_specs = tuple(
        endpoint_spec(
            f"{spec.name}_desync",
            spec.input_names[:-1] + ("exec_mask",),
            spec.output_names[:-1],
            spec.required_components,
            spec.state_components,
            "step_desync",
        )
        for spec in specs
        if spec.state_method == "step"
    )
    specs += desync_specs
    return {spec.name: spec for spec in specs}


ZIPCODEC_ENDPOINTS: Mapping[str, EndpointSpec] = MappingProxyType(
    _make_zipcodec_endpoints()
)


def make_zipcodec_endpoints() -> "Mapping[str, EndpointSpec]":
    """Return the cached immutable ZipCodec endpoint registry."""
    return ZIPCODEC_ENDPOINTS
