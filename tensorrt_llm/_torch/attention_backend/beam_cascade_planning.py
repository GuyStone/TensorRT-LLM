# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Pure page-table construction for the beam-search cascade attention backend.

This module has NO engine dependencies (only ``torch``) so the riskiest logic — splitting
each beam's KV blocks into a shared prefix and a per-beam suffix — can be unit-tested in
isolation (``tests/.../test_beam_cascade_split.py``). The engine-coupled backend that
calls these functions lives in ``beam_cascade.py``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch


@dataclass
class CascadePlan:
    """CSR page tables for the two cascade levels, as host int32 tensors.

    Level 0 (prefix): one query group of ``beam_width`` rows per generation request, all
    attending the shared prefix pages. Level 1 (suffix): one group of 1 row per
    (request, beam), attending that beam's own suffix pages. Query rows are ordered
    (request, beam) to match the engine's generation-token ordering.
    """
    num_gen_queries: int                # == sum(beam_width) over generation requests
    prefix_qo_indptr: torch.Tensor      # [num_gen_reqs + 1]
    prefix_kv_indptr: torch.Tensor      # [num_gen_reqs + 1]
    prefix_kv_indices: torch.Tensor     # [sum prefix pages]
    prefix_last_page_len: torch.Tensor  # [num_gen_reqs]
    suffix_qo_indptr: torch.Tensor      # [num_gen_queries + 1]
    suffix_kv_indptr: torch.Tensor      # [num_gen_queries + 1]
    suffix_kv_indices: torch.Tensor     # [sum suffix pages]
    suffix_last_page_len: torch.Tensor  # [num_gen_queries]


@dataclass
class AppendLayout:
    """Per-beam-as-sequence paged layout for the single new decode token of each beam."""
    kv_indptr: torch.Tensor       # [num_gen_queries + 1]
    kv_indices: torch.Tensor      # [sum full per-beam pages]
    last_page_len: torch.Tensor   # [num_gen_queries]
    batch_indices: torch.Tensor   # [num_gen_queries]  (each new token -> its own seq)
    positions: torch.Tensor       # [num_gen_queries]  (absolute pos = kv_len - 1)

    def to(self, device) -> "AppendLayout":
        """Move all index tensors to ``device`` (append_paged_kv_cache is CUDA-only)."""
        return AppendLayout(self.kv_indptr.to(device), self.kv_indices.to(device),
                            self.last_page_len.to(device),
                            self.batch_indices.to(device), self.positions.to(device))


def per_request_first(flat: List[int],
                      block_ids_per_beam: List[List[List[int]]]) -> List[int]:
    """Gather one representative (first-beam) value per request from a beam-flattened,
    request-major list. Beams of a request share prompt/generated length, so the first
    beam's value is representative. Handles non-uniform beam widths via a running offset.

    The engine builds ``num_cached_tokens_per_seq`` / ``prompt_lens`` with one entry per
    (request, beam) (model_engine.py:3342-3345), so they cannot be indexed by request
    directly.
    """
    assert len(flat) == sum(len(b) for b in block_ids_per_beam), (
        "expected a beam-flattened (request-major) list with one entry per (request, beam)")
    out, off = [], 0
    for beams in block_ids_per_beam:
        out.append(int(flat[off]))
        off += len(beams)
    return out


def longest_common_prefix_len(beams: List[List[int]]) -> int:
    """Length of the longest common *block-id* prefix across all beams.

    Compares ids, not positions: equal-length chains do not imply shared ids
    (kvCacheManager.cpp:2311).
    """
    if not beams:
        return 0
    n = min(len(b) for b in beams)
    first = beams[0]
    for i in range(n):
        if any(b[i] != first[i] for b in beams):
            return i
    return n


def _i32(xs: List[int]) -> torch.Tensor:
    return torch.tensor(xs, dtype=torch.int32)


def build_cascade_page_tables(
    block_ids_per_beam: List[List[List[int]]],
    prompt_lens: List[int],
    kv_lens_per_beam: List[List[int]],
    page_size: int,
) -> CascadePlan:
    """Build the prefix/suffix CSR page tables for a batch of generation requests.

    Args:
        block_ids_per_beam: ``[req][beam] -> [block_id]`` raw per-beam block chains
            (from ``impl.get_batch_cache_block_ids``; lossless, beams equal length).
        prompt_lens: ``[req] -> prompt token length`` (shared-prefix upper bound).
        kv_lens_per_beam: ``[req][beam] -> total kv tokens this beam attends`` including
            the just-appended decode token.
        page_size: tokens per KV block.

    The shared prefix is the longest common block-id prefix, clamped to the floor prompt
    block boundary (``prompt_len // page_size``) so it ends on a page boundary — matching
    the allocator's shared/per-beam split (kvCacheManager.cpp:1597, 2217).
    """
    assert len(prompt_lens) == len(block_ids_per_beam), (
        "prompt_lens must have exactly one entry per request")
    assert len(kv_lens_per_beam) == len(block_ids_per_beam), (
        "kv_lens_per_beam must have exactly one entry per request")

    prefix_qo, prefix_kv_indptr, prefix_kv_indices, prefix_last = [0], [0], [], []
    suffix_qo, suffix_kv_indptr, suffix_kv_indices, suffix_last = [0], [0], [], []

    for req, beams in enumerate(block_ids_per_beam):
        beam_width = len(beams)
        assert beam_width >= 1, "a generation request must have >= 1 beam"

        num_full_prompt_blocks = prompt_lens[req] // page_size
        prefix_n = min(longest_common_prefix_len(beams), num_full_prompt_blocks)
        prefix_len_tokens = prefix_n * page_size

        prefix_qo.append(prefix_qo[-1] + beam_width)
        prefix_kv_indices.extend(beams[0][:prefix_n])
        prefix_kv_indptr.append(len(prefix_kv_indices))
        prefix_last.append(page_size if prefix_n > 0 else 0)

        for beam_idx in range(beam_width):
            kv_len = kv_lens_per_beam[req][beam_idx]
            suffix_len = kv_len - prefix_len_tokens
            assert suffix_len >= 1, (
                f"suffix must contain the current token (req={req}, beam={beam_idx}, "
                f"kv_len={kv_len}, prefix_len={prefix_len_tokens})")
            n_pages = (suffix_len + page_size - 1) // page_size
            suffix_blocks = beams[beam_idx][prefix_n:prefix_n + n_pages]
            assert len(suffix_blocks) == n_pages, (
                "suffix length implies more pages than the beam owns")
            suffix_qo.append(suffix_qo[-1] + 1)
            suffix_kv_indices.extend(suffix_blocks)
            suffix_kv_indptr.append(len(suffix_kv_indices))
            suffix_last.append(suffix_len - (n_pages - 1) * page_size)

    return CascadePlan(
        num_gen_queries=suffix_qo[-1],
        prefix_qo_indptr=_i32(prefix_qo),
        prefix_kv_indptr=_i32(prefix_kv_indptr),
        prefix_kv_indices=_i32(prefix_kv_indices),
        prefix_last_page_len=_i32(prefix_last),
        suffix_qo_indptr=_i32(suffix_qo),
        suffix_kv_indptr=_i32(suffix_kv_indptr),
        suffix_kv_indices=_i32(suffix_kv_indices),
        suffix_last_page_len=_i32(suffix_last),
    )


def build_append_layout(
    block_ids_per_beam: List[List[List[int]]],
    kv_lens_per_beam: List[List[int]],
    page_size: int,
) -> AppendLayout:
    """Layout for ``append_paged_kv_cache``: each (request, beam) is one sequence and the
    new decode token lands in that beam's current last page."""
    indptr, indices, last, batch_idx, positions = [0], [], [], [], []
    row = 0
    for req, beams in enumerate(block_ids_per_beam):
        for beam_idx, blocks in enumerate(beams):
            kv_len = kv_lens_per_beam[req][beam_idx]
            n_pages = (kv_len + page_size - 1) // page_size
            indices.extend(blocks[:n_pages])
            indptr.append(len(indices))
            last.append(kv_len - (n_pages - 1) * page_size)
            batch_idx.append(row)
            positions.append(kv_len - 1)
            row += 1

    return AppendLayout(_i32(indptr), _i32(indices), _i32(last), _i32(batch_idx),
                        _i32(positions))
