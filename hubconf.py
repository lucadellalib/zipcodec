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

"""PyTorch Hub entry point."""

from typing import Any, Dict, Optional, Sequence

from zipcodec import ZipCodec


# Keep this list consistent with pyproject.toml
dependencies = [
    "huggingface_hub",
    "numpy",
    "safetensors",
    "torch",
]


def zipcodec(
    config: "str" = "lucadellalib/zipcodec",
    pretrained: "bool" = True,
    components: "Optional[Sequence[str]]" = None,
    endpoint: "Optional[str]" = None,
    overrides: "Optional[Dict[str, Any]]" = None,
    **kwargs: "Any",
) -> "ZipCodec":
    """Load ZipCodec from a local config or a Hugging Face repository.

    Parameters
    ----------
    config : str
        Local JSON configuration path or Hugging Face repository ID.
    pretrained : bool
        Load the checkpoint associated with ``config``.
    components : sequence of str, optional
        Load only these codec components.
    endpoint : str, optional
        Load only the components needed by this endpoint.
    overrides : dict, optional
        Dot-separated configuration values to override.
    **kwargs : Any
        Additional arguments forwarded to Hugging Face Hub downloads.

    """
    return ZipCodec.from_config(
        config,
        pretrained=pretrained,
        components=components,
        endpoint=endpoint,
        overrides=overrides,
        **kwargs,
    )


if __name__ == "__main__":
    model = zipcodec()
    print(
        "Total number of parameters/buffers: "
        f"{sum(x.numel() for x in model.state_dict().values()) / 1e6:.2f}M"
    )
