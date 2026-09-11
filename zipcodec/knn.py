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

"""Frame-level nearest-neighbor matching utilities."""

from typing import List

import torch
from torch import Tensor, nn


__all__ = ["knn"]


def knn(
    input: "Tensor",
    matching_set: "Tensor",
    topk: "int" = 4,
    num_splits: "int" = 1,
) -> "Tensor":
    """Replace each input frame with the mean of its cosine-nearest frames.

    Parameters
    ----------
    input:
        Query features shaped ``(..., hidden_dim)``.
    matching_set:
        Candidate target-speaker frames shaped ``(num_frames, hidden_dim)``.
    topk:
        Number of nearest target frames to average.
    num_splits:
        Number of matching-set chunks used to reduce peak similarity memory.

    Returns
    -------
    Tensor with the same shape as ``input``.

    """
    # Ranking by descending cosine similarity is equivalent to ranking by
    # ascending cosine distance (1 - similarity), without materializing the
    # distance expression or using cdist
    queries = nn.functional.normalize(input.flatten(end_dim=-2), dim=-1)
    targets = nn.functional.normalize(matching_set, dim=-1)
    split_size = max(
        1,
        (matching_set.shape[0] + num_splits - 1) // num_splits,
    )

    top_similarities: List[Tensor] = []
    top_indices: List[Tensor] = []
    offset = 0
    for target_subset in targets.split(split_size):
        similarities = queries @ target_subset.transpose(0, 1)
        subset_topk = min(topk, target_subset.shape[0])
        values, indices = similarities.topk(
            subset_topk,
            dim=-1,
            largest=True,
        )
        top_similarities.append(values)
        top_indices.append(indices + offset)
        offset += target_subset.shape[0]

    similarities = torch.cat(top_similarities, dim=-1)
    indices = torch.cat(top_indices, dim=-1)
    final_topk = min(topk, similarities.shape[-1])
    _, positions = similarities.topk(final_topk, dim=-1, largest=True)
    nearest_indices = indices.gather(1, positions)
    matched = matching_set[nearest_indices].mean(dim=1)
    return matched.reshape(input.shape)
