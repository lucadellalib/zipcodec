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

"""ZipCodec endpoint compilation backends."""

import contextlib
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor, nn


try:
    from .endpoints import EndpointModule, EndpointSpec
except ImportError:
    from endpoints import EndpointModule, EndpointSpec


__all__ = [
    "CUDAGraphEndpoint",
    "JITEndpoint",
    "ONNXEndpoint",
    "TorchCompileEndpoint",
    "compile_cuda_graph_endpoint",
    "compile_jit_endpoint",
    "compile_onnx_endpoint",
    "compile_torch_endpoint",
    "export_onnx_endpoint",
]


_NUMPY_DTYPES = {
    torch.bool: np.bool_,
    torch.uint8: np.uint8,
    torch.int8: np.int8,
    torch.int16: np.int16,
    torch.int32: np.int32,
    torch.int64: np.int64,
    torch.float16: np.float16,
    torch.float32: np.float32,
    torch.float64: np.float64,
}


def _numpy_dtype(dtype: "torch.dtype") -> "Any":
    try:
        return _NUMPY_DTYPES[dtype]
    except KeyError as error:
        raise TypeError(f"ONNX Runtime I/O binding does not support {dtype}") from error


def _ort_device(device: "torch.device") -> "Tuple[str, int]":
    if device.type == "cpu":
        return "cpu", 0
    if device.type == "cuda":
        index = device.index
        if index is None:
            index = torch.cuda.current_device()
        return "cuda", index
    raise ValueError(
        f"ONNX Runtime I/O binding supports only CPU and CUDA tensors, got {device}"
    )


def _require_streaming_endpoint(endpoint: "EndpointSpec", backend: "str") -> "None":
    if not endpoint.has_state:
        raise ValueError(
            f"{backend} supports only streaming endpoints with fixed-size inputs; "
            f"got offline endpoint {endpoint.name!r}"
        )


def _tensor_spec(name: "str", value: "Tensor") -> "Dict[str, Any]":
    return {
        "name": name,
        "shape": list(value.shape),
        "dtype": str(value.dtype).removeprefix("torch."),
    }


def _sha256(path: "Path") -> "str":
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(16 * 1024**2):
            digest.update(chunk)
    return digest.hexdigest()


def _export_onnx(
    module: "EndpointModule",
    args: "Tuple[Tensor, ...]",
    model_path: "Path",
    dynamic_axes: "Optional[Dict[str, Dict[int, str]]]",
    opset_version: "int",
) -> "Tuple[Tensor, ...]":
    was_training = module.training
    module.eval()
    try:
        with torch.no_grad():
            outputs = module(*args)
        if isinstance(outputs, Tensor):
            outputs = (outputs,)
        torch.onnx.export(
            module,
            args,
            str(model_path),
            input_names=list(module.flat_input_names),
            output_names=list(module.flat_output_names),
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            external_data=True,
        )
    finally:
        module.train(was_training)
    return tuple(outputs)


def _convert_onnx_to_fp16(source_path: "Path") -> "Path":
    import onnx

    try:
        from onnxruntime.transformers.float16 import convert_float_to_float16
    except ImportError as error:
        raise ImportError(
            "FP16 ONNX export requires ONNX Runtime; install the development "
            "dependencies with `uv sync --group dev`"
        ) from error

    model = convert_float_to_float16(
        str(source_path),
        disable_shape_infer=True,
        force_fp16_initializers=True,
    )
    converted_path = source_path.with_name(f"{source_path.stem}_fp16.onnx")
    onnx.save_model(
        model,
        str(converted_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=f"{converted_path.name}.data",
        size_threshold=0,
    )
    return converted_path


def _persist_onnx_bundle(source_path: "Path", output_path: "Path") -> "Tuple[str, ...]":
    import onnx

    model = onnx.load(str(source_path), load_external_data=False)
    tensors = []
    graphs = [model.graph]
    while graphs:
        graph = graphs.pop()
        tensors.extend(graph.initializer)
        for sparse_tensor in graph.sparse_initializer:
            tensors.extend((sparse_tensor.values, sparse_tensor.indices))
        for node in graph.node:
            for attribute in node.attribute:
                if attribute.type == onnx.AttributeProto.TENSOR:
                    tensors.append(attribute.t)
                elif attribute.type == onnx.AttributeProto.TENSORS:
                    tensors.extend(attribute.tensors)
                elif attribute.type == onnx.AttributeProto.SPARSE_TENSOR:
                    tensors.extend(
                        (
                            attribute.sparse_tensor.values,
                            attribute.sparse_tensor.indices,
                        )
                    )
                elif attribute.type == onnx.AttributeProto.SPARSE_TENSORS:
                    for sparse_tensor in attribute.sparse_tensors:
                        tensors.extend((sparse_tensor.values, sparse_tensor.indices))
                elif attribute.type == onnx.AttributeProto.GRAPH:
                    graphs.append(attribute.g)
                elif attribute.type == onnx.AttributeProto.GRAPHS:
                    graphs.extend(attribute.graphs)

    external_tensors = [
        tensor
        for tensor in tensors
        if tensor.data_location == onnx.TensorProto.EXTERNAL
    ]
    data_path = output_path.with_suffix(f"{output_path.suffix}.data")
    if not external_tensors:
        shutil.copy2(source_path, output_path)
        data_path.unlink(missing_ok=True)
        return ()

    data_file = tempfile.NamedTemporaryFile(
        prefix=f".{data_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
        delete=False,
    )
    temporary_data_path = Path(data_file.name)
    data_file.close()
    graph_file = tempfile.NamedTemporaryFile(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
        delete=False,
    )
    temporary_graph_path = Path(graph_file.name)
    graph_file.close()

    try:
        copied_ranges = {}
        with temporary_data_path.open("wb") as destination:
            for tensor in external_tensors:
                fields = {entry.key: entry.value for entry in tensor.external_data}
                try:
                    source_location = fields["location"]
                except KeyError as error:
                    raise ValueError(
                        f"External tensor {tensor.name!r} has no location"
                    ) from error
                source = source_path.parent / source_location
                source_offset = int(fields.get("offset", 0))
                source_length = int(
                    fields.get("length", source.stat().st_size - source_offset)
                )
                source_range = (source.resolve(), source_offset, source_length)

                if source_range in copied_ranges:
                    destination_offset = copied_ranges[source_range]
                else:
                    padding = (-destination.tell()) % 64
                    if padding:
                        destination.write(b"\0" * padding)
                    destination_offset = destination.tell()
                    remaining = source_length
                    with source.open("rb") as source_file:
                        source_file.seek(source_offset)
                        while remaining:
                            chunk = source_file.read(min(16 * 1024**2, remaining))
                            if not chunk:
                                raise EOFError(
                                    f"External tensor {tensor.name!r} expected "
                                    f"{source_length} bytes from {source}"
                                )
                            destination.write(chunk)
                            remaining -= len(chunk)
                    copied_ranges[source_range] = destination_offset

                del tensor.external_data[:]
                for key, value in (
                    ("location", data_path.name),
                    ("offset", str(destination_offset)),
                    ("length", str(source_length)),
                ):
                    entry = tensor.external_data.add()
                    entry.key = key
                    entry.value = value

        onnx.save_model(model, str(temporary_graph_path))
        temporary_data_path.replace(data_path)
        temporary_graph_path.replace(output_path)
    finally:
        temporary_data_path.unlink(missing_ok=True)
        temporary_graph_path.unlink(missing_ok=True)

    return (data_path.name,)


def export_onnx_endpoint(
    model: "nn.Module",
    endpoint_name: "str",
    example_inputs: "Dict[str, Any]",
    output_path: "Union[str, Path]",
    dynamic_axes: "Optional[Dict[str, Dict[int, str]]]" = None,
    opset_version: "int" = 17,
    metadata_path: "Optional[Union[str, Path]]" = None,
    example_inputs_path: "Optional[Union[str, Path]]" = None,
) -> "Tuple[Path, Path, Path]":
    """Persist a standalone ONNX endpoint and its deployment metadata."""
    endpoint = model.endpoint(endpoint_name)
    _require_streaming_endpoint(endpoint, "ONNX")
    module = model.endpoint_module(endpoint_name)
    args = module.flatten_inputs(example_inputs)

    model_path = Path(output_path)
    if model_path.suffix.lower() != ".onnx":
        raise ValueError("ONNX output_path must end with '.onnx'")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_metadata_path = (
        model_path.with_suffix(".json")
        if metadata_path is None
        else Path(metadata_path)
    )
    resolved_inputs_path = (
        model_path.with_name(f"{model_path.stem}_inputs.npz")
        if example_inputs_path is None
        else Path(example_inputs_path)
    )
    resolved_metadata_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_inputs_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix="zipcodec-onnx-export-",
        dir=model_path.parent,
    ) as artifact_directory:
        temporary_model_path = Path(artifact_directory) / model_path.name
        outputs = _export_onnx(
            module,
            args,
            temporary_model_path,
            dynamic_axes,
            opset_version,
        )
        if any(value.is_floating_point() for value in args) and all(
            not value.is_floating_point() or value.dtype == torch.float16
            for value in args
        ):
            temporary_model_path = _convert_onnx_to_fp16(temporary_model_path)
        external_data = _persist_onnx_bundle(temporary_model_path, model_path)
    np.savez(
        resolved_inputs_path,
        **{
            name: value.detach().cpu().numpy()
            for name, value in zip(module.flat_input_names, args, strict=True)
        },
    )
    metadata = {
        "format_version": 1,
        "endpoint": endpoint.name,
        "opset_version": opset_version,
        "inference_dtype": next(
            (
                str(value.dtype).removeprefix("torch.")
                for value in args
                if value.is_floating_point()
            ),
            None,
        ),
        "semantic_inputs": list(endpoint.input_names),
        "semantic_outputs": list(endpoint.output_names),
        "inputs": [
            _tensor_spec(name, value)
            for name, value in zip(module.flat_input_names, args, strict=True)
        ],
        "outputs": [
            _tensor_spec(name, value)
            for name, value in zip(module.flat_output_names, outputs, strict=True)
        ],
        "state": {
            "inputs": [
                name
                for name in module.flat_input_names
                if name not in endpoint.input_names
            ],
            "outputs": [
                name
                for name in module.flat_output_names
                if name not in endpoint.output_names
            ],
        },
        "artifacts": {
            "model": model_path.name,
            "external_data": list(external_data),
            "example_inputs": resolved_inputs_path.name,
            "sha256": {
                "model": _sha256(model_path),
                "external_data": {
                    path: _sha256(model_path.parent / path) for path in external_data
                },
                "example_inputs": _sha256(resolved_inputs_path),
            },
        },
    }
    resolved_metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    return model_path, resolved_metadata_path, resolved_inputs_path


@contextlib.contextmanager
def _jit_fusion_disabled() -> "Iterator[None]":
    can_fuse_on_gpu = torch._C._jit_can_fuse_on_gpu()
    texpr_fuser_enabled = torch._C._jit_texpr_fuser_enabled()
    torch._C._jit_override_can_fuse_on_gpu(False)
    torch._C._jit_set_texpr_fuser_enabled(False)
    try:
        yield
    finally:
        torch._C._jit_override_can_fuse_on_gpu(can_fuse_on_gpu)
        torch._C._jit_set_texpr_fuser_enabled(texpr_fuser_enabled)


class JITEndpoint:
    """Traced TorchScript endpoint with semantic dictionary I/O."""

    def __init__(
        self,
        compiled_module: "torch.jit.ScriptModule",
        module: "EndpointModule",
        endpoint: "EndpointSpec",
        fusion: "bool" = True,
    ) -> "None":
        self.compiled_module = compiled_module
        self.module = module
        self.endpoint = endpoint
        self.fusion = fusion

    def __call__(self, inputs: "Dict[str, Any]") -> "Dict[str, Any]":
        flat_inputs = self.module.flatten_inputs(inputs)
        if self.fusion:
            outputs = self.compiled_module(*flat_inputs)
        else:
            with _jit_fusion_disabled():
                outputs = self.compiled_module(*flat_inputs)
        if isinstance(outputs, Tensor):
            outputs = (outputs,)
        return self.module.unflatten_outputs(tuple(outputs))


def compile_jit_endpoint(
    model: "nn.Module",
    endpoint_name: "str",
    example_inputs: "Dict[str, Any]",
    check_trace: "bool" = True,
    strict: "bool" = True,
    fusion: "bool" = True,
) -> "JITEndpoint":
    """Trace an endpoint, optionally disabling JIT fusion when it executes."""
    endpoint = model.endpoint(endpoint_name)
    _require_streaming_endpoint(endpoint, "TorchScript")
    module = model.endpoint_module(endpoint_name)
    args = module.flatten_inputs(example_inputs)
    was_training = module.training
    module.eval()
    try:
        with torch.no_grad():
            compiled_module = torch.jit.trace(
                module,
                args,
                check_trace=check_trace,
                strict=strict,
            )
    finally:
        module.train(was_training)

    return JITEndpoint(
        compiled_module=compiled_module,
        module=module,
        endpoint=endpoint,
        fusion=fusion,
    )


class TorchCompileEndpoint:
    """``torch.compile`` endpoint with semantic dictionary I/O."""

    def __init__(
        self,
        compiled_module: "nn.Module",
        module: "EndpointModule",
        endpoint: "EndpointSpec",
    ) -> "None":
        self.compiled_module = compiled_module
        self.module = module
        self.endpoint = endpoint

    def __call__(self, inputs: "Dict[str, Any]") -> "Dict[str, Any]":
        outputs = self.compiled_module(*self.module.flatten_inputs(inputs))
        if isinstance(outputs, Tensor):
            outputs = (outputs,)
        return self.module.unflatten_outputs(tuple(outputs))


def compile_torch_endpoint(
    model: "nn.Module",
    endpoint_name: "str",
    **kwargs: "Any",
) -> "TorchCompileEndpoint":
    """Compile one endpoint with :func:`torch.compile`."""
    endpoint = model.endpoint(endpoint_name)
    module = model.endpoint_module(endpoint_name)
    return TorchCompileEndpoint(torch.compile(module, **kwargs), module, endpoint)


class CUDAGraphEndpoint:
    """CUDA graph endpoint with semantic dictionary inputs and outputs."""

    def __init__(
        self,
        module: "EndpointModule",
        endpoint: "EndpointSpec",
        example_inputs: "Dict[str, Any]",
        warmup_steps: "int" = 3,
    ) -> "None":
        self.module = module
        self.endpoint = endpoint

        flat_example_inputs = module.flatten_inputs(example_inputs)
        if not flat_example_inputs or not all(
            value.device.type == "cuda" for value in flat_example_inputs
        ):
            raise ValueError("All CUDA graph example inputs must be CUDA tensors")

        self.static_inputs = tuple(
            value.detach().clone() for value in flat_example_inputs
        )
        was_training = module.training
        module.eval()

        try:
            for _ in range(warmup_steps):
                module(*self.static_inputs)
            torch.cuda.synchronize()

            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.static_outputs = module(*self.static_inputs)
        finally:
            module.train(was_training)

    def __call__(self, inputs: "Dict[str, Any]") -> "Dict[str, Any]":
        return self._call_leaf(inputs)

    def _call_leaf(self, inputs: "Dict[str, Any]") -> "Dict[str, Any]":
        flat_inputs = self.module.flatten_inputs(inputs)
        for static_input, value in zip(
            self.static_inputs,
            flat_inputs,
            strict=True,
        ):
            static_input.copy_(value)

        self.graph.replay()
        outputs = tuple(value.clone() for value in self.static_outputs)
        return self.module.unflatten_outputs(outputs)


def compile_cuda_graph_endpoint(
    model: "nn.Module",
    endpoint_name: "str",
    example_inputs: "Dict[str, Any]",
    warmup_steps: "int" = 3,
) -> "CUDAGraphEndpoint":
    """Capture one endpoint in a CUDA graph."""
    endpoint = model.endpoint(endpoint_name)
    _require_streaming_endpoint(endpoint, "CUDA Graph")
    return CUDAGraphEndpoint(
        module=model.endpoint_module(endpoint_name),
        endpoint=endpoint,
        example_inputs=example_inputs,
        warmup_steps=warmup_steps,
    )


class ONNXEndpoint:
    """ONNX Runtime endpoint with semantic dictionary inputs and outputs."""

    def __init__(
        self,
        session: "Any",
        module: "EndpointModule",
        endpoint: "EndpointSpec",
        output_device: "Optional[torch.device]" = None,
        io_binding: "bool" = False,
        input_specs: "Tuple[Tuple[Tuple[int, ...], torch.dtype], ...]" = (),
        output_specs: "Tuple[Tuple[Tuple[int, ...], torch.dtype], ...]" = (),
    ) -> "None":
        self.session = session
        self.module = module
        self.endpoint = endpoint
        self.output_device = output_device
        self.io_binding = io_binding
        self.input_specs = input_specs
        self.output_specs = output_specs

    def __call__(self, inputs: "Dict[str, Any]") -> "Dict[str, Any]":
        return self._call_leaf(inputs)

    def _call_leaf(self, inputs: "Dict[str, Any]") -> "Dict[str, Any]":
        flat_inputs = self.module.flatten_inputs(inputs)
        if self.io_binding:
            return self._call_with_io_binding(flat_inputs)

        ort_inputs = {
            name: value.detach().cpu().numpy()
            for name, value in zip(
                self.module.flat_input_names,
                flat_inputs,
                strict=True,
            )
        }
        ort_outputs = self.session.run(
            list(self.module.flat_output_names),
            ort_inputs,
        )

        outputs = []
        for value in ort_outputs:
            tensor = torch.from_numpy(value)
            if self.output_device is not None:
                tensor = tensor.to(self.output_device)
            outputs.append(tensor)
        return self.module.unflatten_outputs(tuple(outputs))

    def _call_with_io_binding(
        self,
        flat_inputs: "Tuple[Tensor, ...]",
    ) -> "Dict[str, Any]":
        binding = self.session.io_binding()
        bound_inputs = tuple(value.detach().contiguous() for value in flat_inputs)
        actual_specs = tuple(
            (tuple(value.shape), value.dtype) for value in bound_inputs
        )
        if actual_specs != self.input_specs:
            raise ValueError(
                "I/O-bound ONNX endpoints require the same input shapes and "
                f"dtypes used during export; expected {self.input_specs}, "
                f"got {actual_specs}"
            )

        for name, value in zip(
            self.module.flat_input_names,
            bound_inputs,
            strict=True,
        ):
            device_type, device_id = _ort_device(value.device)
            binding.bind_input(
                name,
                device_type,
                device_id,
                _numpy_dtype(value.dtype),
                tuple(value.shape),
                value.data_ptr(),
            )

        if self.output_device is None:
            raise RuntimeError("I/O binding requires an output device")
        device_type, device_id = _ort_device(self.output_device)
        outputs = tuple(
            torch.empty(shape, dtype=dtype, device=self.output_device)
            for shape, dtype in self.output_specs
        )
        for name, value in zip(
            self.module.flat_output_names,
            outputs,
            strict=True,
        ):
            binding.bind_output(
                name,
                device_type,
                device_id,
                _numpy_dtype(value.dtype),
                tuple(value.shape),
                value.data_ptr(),
            )

        binding.synchronize_inputs()
        self.session.run_with_iobinding(binding)
        binding.synchronize_outputs()
        return self.module.unflatten_outputs(outputs)


def compile_onnx_endpoint(
    model: "nn.Module",
    endpoint_name: "str",
    example_inputs: "Dict[str, Any]",
    dynamic_axes: "Optional[Dict[str, Dict[int, str]]]" = None,
    opset_version: "int" = 17,
    output_device: "Optional[torch.device]" = None,
    providers: "Optional[Tuple[str, ...]]" = None,
    temporary_directory: "Optional[Path]" = None,
    io_binding: "bool" = False,
) -> "ONNXEndpoint":
    """Export one endpoint and create an ONNX Runtime callable."""
    endpoint = model.endpoint(endpoint_name)
    _require_streaming_endpoint(endpoint, "ONNX")

    import onnxruntime as ort

    artifact_root = None if temporary_directory is None else Path(temporary_directory)
    if artifact_root is not None:
        artifact_root.mkdir(parents=True, exist_ok=True)

    def compile_endpoint() -> "ONNXEndpoint":
        module = model.endpoint_module(endpoint_name)
        args = module.flatten_inputs(example_inputs)
        was_training = module.training
        module.eval()

        try:
            if io_binding:
                if not args:
                    raise ValueError("I/O binding requires at least one input tensor")
                binding_output_device = (
                    args[0].device
                    if output_device is None
                    else torch.device(output_device)
                )
                with torch.no_grad():
                    example_outputs = module(*args)
                if isinstance(example_outputs, Tensor):
                    example_outputs = (example_outputs,)
                input_specs = tuple((tuple(value.shape), value.dtype) for value in args)
                output_specs = tuple(
                    (tuple(value.shape), value.dtype) for value in example_outputs
                )
            else:
                binding_output_device = output_device
                input_specs = ()
                output_specs = ()

            with tempfile.TemporaryDirectory(
                prefix="zipcodec-onnx-",
                dir=artifact_root,
            ) as artifact_directory:
                model_path = f"{artifact_directory}/model.onnx"
                torch.onnx.export(
                    module,
                    args,
                    model_path,
                    input_names=list(module.flat_input_names),
                    output_names=list(module.flat_output_names),
                    dynamic_axes=dynamic_axes,
                    opset_version=opset_version,
                    external_data=True,
                )
                session = ort.InferenceSession(
                    model_path,
                    providers=None if providers is None else list(providers),
                )
                if io_binding:
                    active_providers = tuple(session.get_providers())
                    bound_devices = {value.device.type for value in args} | {
                        binding_output_device.type
                    }
                    if (
                        "cuda" in bound_devices
                        and "CUDAExecutionProvider" not in active_providers
                    ):
                        raise RuntimeError(
                            "CUDA I/O binding was requested, but ONNX Runtime "
                            "did not activate CUDAExecutionProvider. "
                            f"Active providers: {active_providers}. "
                            "Check that onnxruntime-gpu, CUDA, and cuDNN are "
                            "installed with compatible versions"
                        )
        finally:
            module.train(was_training)

        return ONNXEndpoint(
            session=session,
            module=module,
            endpoint=endpoint,
            output_device=binding_output_device,
            io_binding=io_binding,
            input_specs=input_specs,
            output_specs=output_specs,
        )

    return compile_endpoint()
