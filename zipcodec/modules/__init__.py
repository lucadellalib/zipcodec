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

"""ZipCodec building blocks."""

try:
    from .erfformer import ErfFormer
    from .mel import LogMelSpectrogram
    from .patch import Patch1d, Unpatch1d
    from .ssq import ScalarSphericalQuantizer
    from .vocos import Vocos
except ImportError:
    from modules.erfformer import ErfFormer
    from modules.mel import LogMelSpectrogram
    from modules.patch import Patch1d, Unpatch1d
    from modules.ssq import ScalarSphericalQuantizer
    from modules.vocos import Vocos


__all__ = [
    "ErfFormer",
    "LogMelSpectrogram",
    "Patch1d",
    "ScalarSphericalQuantizer",
    "Unpatch1d",
    "Vocos",
]
