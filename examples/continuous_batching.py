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

"""Simulate continuous batching with the desynchronized streaming endpoint."""

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import torch
from torch import Tensor


torch.backends.cudnn.benchmark = True

State = Tuple[Tensor, ...]
DEFAULT_CHECKPOINT = "lucadellalib/zipcodec"
REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_AUDIO_PATHS = (
    REPOSITORY_ROOT / "audios/english/p226_006.wav",
    REPOSITORY_ROOT / "audios/english/p257_003.wav",
    REPOSITORY_ROOT / "audios/english/p287_007.wav",
)


@dataclass
class Stream:
    """One audio request assigned to a batch slot."""

    request_id: int
    path: Path
    wav: Tensor
    offset: int = 0
    outputs: List[Tensor] = field(default_factory=list)


def _should_pause(step: "int", slot: "int", pause_every: "int") -> "bool":
    return pause_every > 0 and (step + slot + 1) % pause_every == 0


def _state_batch_axes(state: "State", single_state: "State") -> "Tuple[int, ...]":
    axes = []
    for index, (value, single_value) in enumerate(
        zip(state, single_state, strict=True)
    ):
        candidates = tuple(
            axis
            for axis, (size, single_size) in enumerate(
                zip(value.shape, single_value.shape, strict=True)
            )
            if size != single_size and single_size == 1
        )
        if len(candidates) != 1:
            raise RuntimeError(
                f"Could not identify the batch axis of state tensor {index}: "
                f"batch shape={tuple(value.shape)}, "
                f"single shape={tuple(single_value.shape)}"
            )
        axes.append(candidates[0])
    return tuple(axes)


def _reset_state_slot(
    state: "State",
    single_state: "State",
    batch_axes: "Tuple[int, ...]",
    slot: "int",
) -> "State":
    reset_state = []
    for value, single_value, batch_axis in zip(
        state,
        single_state,
        batch_axes,
        strict=True,
    ):
        value = value.clone()
        target_index = [slice(None)] * value.ndim
        target_index[batch_axis] = slice(slot, slot + 1)
        value[tuple(target_index)] = single_value
        reset_state.append(value)
    return tuple(reset_state)


def _assign_streams(
    codec: "Any",
    slots: "List[Optional[Stream]]",
    paths: "Sequence[Path]",
    next_request: "int",
    state: "State",
    single_state: "State",
    batch_axes: "Tuple[int, ...]",
) -> "Tuple[int, State]":
    for slot, stream in enumerate(slots):
        if stream is not None or next_request >= len(paths):
            continue
        path = paths[next_request]
        state = _reset_state_slot(state, single_state, batch_axes, slot)
        wav, sample_rate = codec.load_audio(path)
        wav = codec.resample_audio(wav, sample_rate, codec.sample_rate_input)
        slots[slot] = Stream(next_request, path, wav)
        print(f"Assigned request {next_request} ({path}) to slot {slot}")
        next_request += 1
    return next_request, state


@torch.no_grad()
def main(
    audio_paths: "Sequence[Path]" = DEFAULT_AUDIO_PATHS,
    checkpoint: "str" = DEFAULT_CHECKPOINT,
    batch_size: "int" = 2,
    pause_every: "int" = 3,
    output_directory: "Path" = REPOSITORY_ROOT / "outputs/continuous_batching",
) -> "None":
    if batch_size < 2:
        raise ValueError("batch_size must be at least 2")
    if not audio_paths:
        raise ValueError("At least one audio path is required")
    if pause_every < 0 or pause_every == 1:
        raise ValueError("pause_every must be 0 or at least 2")

    device = torch.device("cpu")
    dtype = torch.float32
    codec = (
        torch.hub.load(
            "lucadellalib/zipcodec",
            "zipcodec",
            config=checkpoint,
            trust_repo=True,
        )
        .eval()
        .to(device=device, dtype=dtype)
    )
    endpoint = codec.eager("step_desync")
    chunk_size = codec.chunk_size
    state = codec.init_endpoint_state(
        "step_desync",
        batch_size=batch_size,
        device=device,
        dtype=dtype,
    )
    single_state = codec.init_endpoint_state(
        "step_desync",
        batch_size=1,
        device=device,
        dtype=dtype,
    )
    batch_axes = _state_batch_axes(state, single_state)

    slots: List[Optional[Stream]] = [None] * batch_size
    next_request = 0
    completed = 0
    steps = 0

    while completed < len(audio_paths):
        next_request, state = _assign_streams(
            codec,
            slots,
            audio_paths,
            next_request,
            state,
            single_state,
            batch_axes,
        )

        wav = torch.zeros(batch_size, chunk_size, device=device, dtype=dtype)
        exec_mask = torch.zeros(batch_size, device=device, dtype=torch.bool)
        valid_lengths = [0] * batch_size
        for slot, stream in enumerate(slots):
            if stream is None:
                continue
            if _should_pause(steps, slot, pause_every):
                print(
                    f"Paused request {stream.request_id} in slot {slot}; "
                    "exec_mask=False"
                )
                continue
            chunk = stream.wav[:, stream.offset : stream.offset + chunk_size]
            valid_length = chunk.shape[1]
            wav[slot, :valid_length] = chunk[0]
            exec_mask[slot] = True
            valid_lengths[slot] = valid_length

        result = endpoint(
            {
                "wav": wav,
                "state": state,
                "exec_mask": exec_mask,
            }
        )
        state = result["next_state"]
        wav_rec = result["wav_rec"]
        steps += 1

        for slot, stream in enumerate(slots):
            if stream is None:
                continue
            valid_length = valid_lengths[slot]
            stream.outputs.append(wav_rec[slot : slot + 1, :valid_length].cpu())
            stream.offset += valid_length
            if stream.offset < stream.wav.shape[1]:
                continue

            output_path = output_directory / (
                f"{stream.request_id:03d}_{stream.path.stem}.wav"
            )
            codec.save_audio(
                output_path,
                torch.cat(stream.outputs, dim=1),
                codec.sample_rate_output,
            )
            print(f"Completed request {stream.request_id} in slot {slot}")
            slots[slot] = None
            completed += 1

    print(
        f"Completed {completed} requests in {steps} batched steps "
        f"with {batch_size} slots"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simulate continuous batching with ZipCodec step_desync."
    )
    parser.add_argument(
        "--audio",
        type=Path,
        nargs="+",
        default=None,
        help="Input mono WAV files (default: three files under audios/english).",
    )
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help=(
            "Local config path or Hugging Face repository "
            f"(default: {DEFAULT_CHECKPOINT})."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Number of persistent batch slots (default: 2).",
    )
    parser.add_argument(
        "--pause-every",
        type=int,
        default=3,
        help=(
            "Pause each occupied slot periodically with exec_mask=False; "
            "0 disables simulated pauses; values must be at least 2 "
            "(default: 3)."
        ),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=REPOSITORY_ROOT / "outputs/continuous_batching",
    )
    args = parser.parse_args()
    main(
        audio_paths=DEFAULT_AUDIO_PATHS if args.audio is None else args.audio,
        checkpoint=args.checkpoint,
        batch_size=args.batch_size,
        pause_every=args.pause_every,
        output_directory=args.output_directory,
    )
