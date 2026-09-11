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

"""Configuration and selective pretrained loading for ZipCodec."""

import inspect
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Type

import torch
from torch import Tensor, nn


try:
    from .codec import COMPONENT_NAMES, ZipCodec
    from .endpoints import make_zipcodec_endpoints
    from .modules.erfformer import ErfFormer
    from .modules.mel import LogMelSpectrogram
    from .modules.patch import Patch1d, Unpatch1d
    from .modules.ssq import ScalarSphericalQuantizer
    from .modules.vocos import Vocos
except ImportError:
    from codec import COMPONENT_NAMES, ZipCodec
    from endpoints import make_zipcodec_endpoints

    from modules.erfformer import ErfFormer
    from modules.mel import LogMelSpectrogram
    from modules.patch import Patch1d, Unpatch1d
    from modules.ssq import ScalarSphericalQuantizer
    from modules.vocos import Vocos


__all__ = ["from_config", "to_config"]


Registry = Mapping[str, Type[nn.Module]]
DEFAULT_REGISTRY: Dict[str, Type[nn.Module]] = {
    cls.__name__: cls
    for cls in (
        nn.Identity,
        LogMelSpectrogram,
        Patch1d,
        Unpatch1d,
        ErfFormer,
        ScalarSphericalQuantizer,
        Vocos,
    )
}


def _config_path(path: "str") -> "str":
    return path if path.endswith(".json") else f"{path}.json"


def _checkpoint_stem(config_filename: "str", is_repo: "bool") -> "str":
    if is_repo:
        return "model"
    return os.path.splitext(config_filename)[0]


def _override_config(config: "Dict[str, Any]", path: "str", value: "Any") -> "None":
    keys = path.split(".")
    target = config
    for key in keys[:-1]:
        target = target.setdefault(key, {})
    target[keys[-1]] = value


def _selected_components(
    components: "Optional[Sequence[str]]",
) -> "Tuple[str, ...]":
    if components is None:
        return COMPONENT_NAMES

    selected = tuple(dict.fromkeys(components))
    unknown = set(selected) - set(COMPONENT_NAMES)
    if unknown:
        raise ValueError(
            f"Unknown components: {sorted(unknown)}; available: {list(COMPONENT_NAMES)}"
        )
    return selected


def _module_config(module: "nn.Module") -> "Dict[str, Any]":
    signature = inspect.signature(module.__init__)
    config = {}
    for name, parameter in signature.parameters.items():
        if name == "self" or parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        if not hasattr(module, name):
            raise ValueError(
                f"Cannot serialize {type(module).__name__}: missing attribute {name!r}"
            )
        value = getattr(module, name)
        try:
            json.dumps(value)
        except TypeError as error:
            raise TypeError(
                f"Configuration value {type(module).__name__}.{name} is not JSON serializable"
            ) from error
        config[name] = value
    return config


def _build_modules(
    config: "Dict[str, Any]",
    components: "Tuple[str, ...]",
    registry: "Registry",
) -> "Dict[str, Optional[nn.Module]]":
    modules: Dict[str, Optional[nn.Module]] = {}
    selected = set(components)

    for component_name in COMPONENT_NAMES:
        if component_name not in selected:
            modules[component_name] = None
            continue

        name_key = f"{component_name}_name"
        config_key = f"{component_name}_config"
        class_name = config.get(name_key)
        if class_name is None:
            modules[component_name] = None
            continue
        if class_name not in registry:
            raise ValueError(
                f"Unregistered module {class_name!r}; available: {sorted(registry)}"
            )
        modules[component_name] = registry[class_name](**(config.get(config_key) or {}))

    return modules


def _filter_keys(
    keys: "Sequence[str]",
    components: "Tuple[str, ...]",
) -> "Tuple[str, ...]":
    prefixes = tuple(f"{component}." for component in components)
    return tuple(key for key in keys if key.startswith(prefixes))


def _load_state_dict_checked(
    model: "ZipCodec",
    state_dict: "Dict[str, Tensor]",
) -> "None":
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        raise RuntimeError(f"State dict is missing keys: {list(missing)}")
    if unexpected:
        raise RuntimeError(f"State dict has unexpected keys: {list(unexpected)}")


def _load_local_safetensors(
    checkpoint: "str",
    components: "Tuple[str, ...]",
) -> "Dict[str, Tensor]":
    try:
        from safetensors import safe_open
    except ImportError as error:
        raise ImportError(
            "Install safetensors to load .safetensors checkpoints"
        ) from error

    state_dict = {}
    with safe_open(checkpoint, framework="pt", device="cpu") as file:
        selected_keys = _filter_keys(tuple(file.keys()), components)
        for key in selected_keys:
            state_dict[key] = file.get_tensor(key)
    return state_dict


def _load_remote_safetensors(
    repo_id: "str",
    filename: "str",
    components: "Tuple[str, ...]",
    revision: "Optional[str]" = None,
    token: "Optional[str]" = None,
    **download_kwargs: "Any",
) -> "Dict[str, Tensor]":
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise ImportError(
            "Install huggingface-hub to load remote checkpoints"
        ) from error

    checkpoint = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
        token=token,
        **download_kwargs,
    )
    return _load_local_safetensors(checkpoint, components)


def to_config(
    model: "ZipCodec",
    config: "str",
    pretrained: "bool" = False,
    registry: "Optional[Registry]" = None,
) -> "None":
    """Save a ZipCodec configuration and optionally its safetensors checkpoint."""
    registry = DEFAULT_REGISTRY if registry is None else registry
    config_json = _config_path(config)
    directory = os.path.dirname(config_json)
    if directory:
        os.makedirs(directory, exist_ok=True)

    serialized = {}
    for component_name in COMPONENT_NAMES:
        module = getattr(model, component_name)
        if module is None:
            continue
        class_name = type(module).__name__
        if class_name not in registry or registry[class_name] is not type(module):
            raise ValueError(
                f"Unregistered module {class_name!r}; available: {sorted(registry)}"
            )
        serialized[f"{component_name}_name"] = class_name
        serialized[f"{component_name}_config"] = _module_config(module)

    with open(config_json, "w", encoding="utf-8") as file:
        json.dump(serialized, file, indent=2)

    if pretrained:
        try:
            from safetensors.torch import save_file
        except ImportError as error:
            raise ImportError(
                "Install safetensors to save pretrained ZipCodec models"
            ) from error
        checkpoint = f"{os.path.splitext(config_json)[0]}.safetensors"
        state_dict = {
            key: value.detach().cpu().contiguous()
            for key, value in model.state_dict().items()
        }
        save_file(state_dict, checkpoint)


def from_config(
    config: "str",
    pretrained: "bool" = False,
    components: "Optional[Sequence[str]]" = None,
    endpoint: "Optional[str]" = None,
    overrides: "Optional[Dict[str, Any]]" = None,
    registry: "Optional[Registry]" = None,
    revision: "Optional[str]" = None,
    token: "Optional[str]" = None,
    **download_kwargs: "Any",
) -> "ZipCodec":
    """Build a ZipCodec from local JSON or a Hugging Face repository.

    When ``components`` or ``endpoint`` is provided, only the selected modules
    are instantiated and their tensors are read into memory. Remote safetensors
    checkpoints are downloaded in full and reused from the Hugging Face cache.

    """
    if components is not None and endpoint is not None:
        raise ValueError("Specify either components or endpoint, not both")
    if endpoint is not None:
        endpoints = make_zipcodec_endpoints()
        try:
            components = endpoints[endpoint].required_components
        except KeyError as error:
            available = ", ".join(sorted(endpoints))
            raise KeyError(
                f"Unknown endpoint {endpoint!r}; available: {available}"
            ) from error

    registry = DEFAULT_REGISTRY if registry is None else registry
    selected = _selected_components(components)
    model_id = config
    config_json = _config_path(config)
    is_local = os.path.exists(config_json)

    if is_local:
        with open(config_json, encoding="utf-8") as file:
            model_config = json.load(file)
        repo_id = None
        config_filename = os.path.basename(config_json)
        is_repo = False
    else:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as error:
            raise ImportError(
                "Install huggingface-hub to load remote ZipCodec models"
            ) from error

        is_repo = bool(re.fullmatch(r"[\w.-]+/[\w.-]+", config))
        repo_id = config if is_repo else os.path.dirname(config_json)
        config_filename = "config.json" if is_repo else os.path.basename(config_json)
        local_config = hf_hub_download(
            repo_id=repo_id,
            filename=config_filename,
            revision=revision,
            token=token,
            **download_kwargs,
        )
        with open(local_config, encoding="utf-8") as file:
            model_config = json.load(file)

    if overrides is not None:
        for path, value in overrides.items():
            _override_config(model_config, path, value)

    model = ZipCodec(**_build_modules(model_config, selected, registry))
    model.model_id = model_id
    if not pretrained:
        return model

    checkpoint_stem = _checkpoint_stem(config_filename, is_repo)
    safetensors_filename = f"{checkpoint_stem}.safetensors"

    if is_local:
        checkpoint = str(Path(config_json).with_name(safetensors_filename))
        if os.path.exists(checkpoint):
            state_dict = _load_local_safetensors(checkpoint, selected)
        else:
            pt_checkpoint = str(Path(config_json).with_name(f"{checkpoint_stem}.pt"))
            if components is not None:
                raise FileNotFoundError(
                    "Selective loading requires a .safetensors checkpoint; "
                    f"not found: {checkpoint}"
                )
            state_dict = torch.load(pt_checkpoint, map_location="cpu")
            state_dict = state_dict.get("model", state_dict)
    else:
        assert repo_id is not None
        try:
            state_dict = _load_remote_safetensors(
                repo_id,
                safetensors_filename,
                selected,
                revision=revision,
                token=token,
                **download_kwargs,
            )
        except Exception:
            if components is not None:
                raise
            from huggingface_hub import hf_hub_download

            pt_checkpoint = hf_hub_download(
                repo_id=repo_id,
                filename=f"{checkpoint_stem}.pt",
                revision=revision,
                token=token,
                **download_kwargs,
            )
            state_dict = torch.load(pt_checkpoint, map_location="cpu")
            state_dict = state_dict.get("model", state_dict)

    _load_state_dict_checked(model, state_dict)
    return model
