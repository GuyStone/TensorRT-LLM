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
"""Beam-search cascade attention backend (``attn_backend="BEAM_CASCADE"``).

During *decode* with beam search, every beam of a request shares the prompt prefix
and diverges only on its short generated suffix. This backend computes attention over
the shared prefix **once** (all beams of a request as one query group) and over each
beam's suffix separately, then merges the two partials with FlashInfer's log-sum-exp
``merge_state``. For long prompts and wide beams (the scoring regime) this cuts
decode-attention HBM traffic on the dominant prefix term by ~beam_width.

Design / verification doc: ``docs/source/blogs/beam-cascade-attention.md``.

Validation status (see that doc, §7):
  * VALIDATED here (FlashInfer only, no engine): the paged two-pass + merge kernel path
    (``tests/.../test_beam_cascade_paged.py``), the dense math
    (``tests/.../test_beam_cascade_parity.py``), and the pure page-table split
    (``tests/.../test_beam_cascade_split.py``).
  * PENDING CI (needs a built engine + model, §7 items 5-8): the engine-coupled wiring —
    KV append layout for beams, ``cache_indirection`` remap, and CUDA-graph capture.
    v1 runs the cascade path EAGER (graph capture is future work).

Key constraints discovered from the real code (cited inline):
  * The inherited ``FlashInferAttentionMetadata.prepare()`` calls
    ``get_batch_cache_indices(request_ids)`` with the default ``beam_width=1`` and asserts
    one beam per request (flashinfer.py:766 -> resource_manager.py:1319), so it is
    beam-incompatible. The beam path builds its layout directly from the lossless raw
    ``impl.get_batch_cache_block_ids`` (resource_manager.py:1316), NOT the lossy
    ``_pack_beam_cache_indices`` packing (resource_manager.py:1326).
  * RoPE is applied to q/k *before* ``forward`` for FlashInfer-family backends
    (attention.py:1017; support_fused_rope()==False), so cached K is already rotated at
    absolute positions and the prefix/suffix split is position-correct with no RoPE logic
    here. This backend MUST keep ``support_fused_rope``/``support_fused_qkv`` == False.
  * FlashInfer wrapper LSE is base-2 and feeds ``merge_state`` directly — NO conversion
    (verified empirically; do not replicate star_flashinfer's ``/np.log2(np.e)``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import flashinfer
import torch

from .beam_cascade_planning import (AppendLayout, CascadePlan,
                                     build_append_layout,
                                     build_cascade_page_tables)
from .flashinfer import FlashInferAttention, FlashInferAttentionMetadata
from .interface import AttentionForwardArgs, merge_attention_forward_args

__all__ = ["BeamCascadeAttention", "BeamCascadeAttentionMetadata"]


# --------------------------------------------------------------------------- #
# Metadata                                                                    #
# --------------------------------------------------------------------------- #
@dataclass(kw_only=True)
class BeamCascadeAttentionMetadata(FlashInferAttentionMetadata):
    # Set unconditionally by model_engine.py:3738/3740 (lives on TrtllmAttentionMetadata,
    # not on the base) — declared here so beam search wires through.
    beam_width: int = 1

    # Built each decode step in prepare(); consumed by forward(). Not ctor args.
    _cascade_active: bool = field(init=False, default=False)
    _cascade_planned: bool = field(init=False, default=False)
    _cascade_plan: Optional[CascadePlan] = field(init=False, default=None)
    _append: Optional[AppendLayout] = field(init=False, default=None)
    _prefix_wrapper: Optional[object] = field(init=False, default=None)
    _suffix_wrapper: Optional[object] = field(init=False, default=None)

    def _cascade_applicable(self) -> bool:
        """Cascade only helps pure-decode beam-search steps."""
        return (self.beam_width > 1 and self.num_generations > 0
                and self.num_contexts == 0 and self.kv_cache_manager is not None)

    def prepare(self) -> None:
        if not self._cascade_applicable():
            # Prefill / mixed / beam_width==1: standard FlashInfer single-stream path
            # (also satisfies the inherited beam_width==1 assumption).
            self._cascade_active = False
            super().prepare()
            return
        # Beam decode: the inherited prepare() asserts beam_width==1, so build the
        # beam-aware layout directly. Wrappers are planned lazily in forward() (eager v1,
        # where head/dtype are known and stream is not capturing).
        self._cascade_active = True
        self._cascade_planned = False
        self._build_beam_decode_tables()

    # ---- engine-coupled (validated in CI; see module docstring) ---- #
    def _build_beam_decode_tables(self) -> None:
        assert self.request_ids is not None
        gen_request_ids = self.request_ids[self.num_contexts:]
        window_size = self._resolve_window_size()

        raw = self.kv_cache_manager.impl.get_batch_cache_block_ids(
            gen_request_ids, window_size)              # [req][beam] -> [block_id]
        block_ids_per_beam = [[list(beam) for beam in req] for req in raw]

        cached = self.kv_cache_params.num_cached_tokens_per_seq[self.num_contexts:]
        prompt_lens = (list(self.prompt_lens[self.num_contexts:])
                       if self.prompt_lens is not None else [int(c) for c in cached])
        # Beams of a request share prompt + equal generated length; +1 for this step.
        kv_lens_per_beam = [[int(cached[i]) + 1] * len(beams)
                            for i, beams in enumerate(block_ids_per_beam)]

        self._cascade_plan = build_cascade_page_tables(block_ids_per_beam,
                                                       prompt_lens,
                                                       kv_lens_per_beam,
                                                       self.page_size)
        self._append = build_append_layout(block_ids_per_beam, kv_lens_per_beam,
                                            self.page_size)
        # TODO(CI): remap suffix block selection through cache_indirection
        #   (self.cache_indirection[g][b][prefix_len:kv_len]) for beams that read a
        #   predecessor's KV after reordering. Prefix region is always beam-0
        #   (sampler.py:3027) and needs no remap.

    def _resolve_window_size(self) -> int:
        # Mirror resource_manager.py:1303-1314 (single window only; VSWA out of scope).
        vec = self.kv_cache_manager.max_attention_window_vec
        if len(vec) > 1:
            raise NotImplementedError(
                "BEAM_CASCADE does not support variable sliding-window attention")
        return vec[0]

    def plan_cascade(self, num_heads: int, num_kv_heads: int, head_dim: int,
                     q_dtype: torch.dtype, kv_dtype: torch.dtype,
                     sm_scale: float) -> None:
        """Plan the prefix/suffix prefill wrappers (eager; called from forward())."""
        dev = self.seq_lens_cuda.device
        if self._prefix_wrapper is None:
            self._prefix_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                self.workspace_buffer, self.kv_layout)
            self._suffix_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                self.workspace_buffer, self.kv_layout)
        p = self._cascade_plan
        # Sync after any KV append and before plan() (flashinfer.py:1191).
        torch.cuda.current_stream().synchronize()
        self._prefix_wrapper.plan(p.prefix_qo_indptr.to(dev),
                                  p.prefix_kv_indptr.to(dev),
                                  p.prefix_kv_indices.to(dev),
                                  p.prefix_last_page_len.to(dev), num_heads,
                                  num_kv_heads, head_dim, self.page_size,
                                  causal=False, sm_scale=sm_scale,
                                  q_data_type=q_dtype, kv_data_type=kv_dtype)
        self._suffix_wrapper.plan(p.suffix_qo_indptr.to(dev),
                                  p.suffix_kv_indptr.to(dev),
                                  p.suffix_kv_indices.to(dev),
                                  p.suffix_last_page_len.to(dev), num_heads,
                                  num_kv_heads, head_dim, self.page_size,
                                  causal=False, sm_scale=sm_scale,
                                  q_data_type=q_dtype, kv_data_type=kv_dtype)
        self._cascade_planned = True


# --------------------------------------------------------------------------- #
# Backend                                                                     #
# --------------------------------------------------------------------------- #
class BeamCascadeAttention(FlashInferAttention):
    """FlashInfer-based attention with a beam-search cascade decode path.

    Delegates to ``FlashInferAttention`` for prefill / mixed / ``beam_width==1`` and adds
    the prefix-once + per-beam-suffix + ``merge_state`` path for pure beam-search decode.
    """

    Metadata = BeamCascadeAttentionMetadata

    @classmethod
    def support_fused_rope(cls) -> bool:
        return False  # RoPE stays external so cached K is pre-rotated (see module docstring)

    @classmethod
    def support_fused_qkv(cls) -> bool:
        return False  # k must arrive separate and pre-rotated

    @classmethod
    def support_mla(cls) -> bool:
        return False  # MLA + beam cascade out of scope (overrides FlashInfer's True)

    def _plan_scale(self) -> float:
        q_scaling = self.q_scaling if self.q_scaling is not None else 1.0
        return 1.0 / (math.sqrt(self.head_dim) * q_scaling)

    def forward(self,
                q: torch.Tensor,
                k: Optional[torch.Tensor],
                v: Optional[torch.Tensor],
                metadata: BeamCascadeAttentionMetadata,
                forward_args: Optional[AttentionForwardArgs] = None,
                **kwargs) -> torch.Tensor:
        if not getattr(metadata, "_cascade_active", False):
            # Non-beam-decode: standard FlashInfer path (append + plan + run).
            return super().forward(q, k, v, metadata, forward_args, **kwargs)

        forward_args = merge_attention_forward_args(forward_args, kwargs)
        output = forward_args.output
        if output is None:
            output = torch.empty_like(q)

        q = q.view(-1, self.num_heads, self.head_dim)
        kv_cache = metadata.kv_cache_manager.get_buffers(
            self.layer_idx, kv_layout=metadata.kv_layout)

        # ---- KV append: write each beam's new token (mirror flashinfer.py:1729) ---- #
        if k is not None and v is not None:
            k = k.view(-1, self.num_kv_heads, self.head_dim)
            v = v.view(-1, self.num_kv_heads, self.head_dim)
            if self.has_fp8_kv_cache:
                k = k.to(torch.float8_e4m3fn)
                v = v.to(torch.float8_e4m3fn)
            ap = metadata._append
            flashinfer.page.append_paged_kv_cache(
                append_key=k, append_value=v, batch_indices=ap.batch_indices,
                positions=ap.positions, paged_kv_cache=kv_cache,
                kv_indices=ap.kv_indices, kv_indptr=ap.kv_indptr,
                kv_last_page_len=ap.last_page_len, kv_layout=metadata.kv_layout)

        # ---- plan once per step (first layer); reused across layers ---- #
        if not metadata._cascade_planned:
            metadata.plan_cascade(self.num_heads, self.num_kv_heads, self.head_dim,
                                  q.dtype, kv_cache.dtype, self._plan_scale())

        # ---- two cascade passes + LSE merge (base-2 LSE; no conversion) ---- #
        v_pfx, s_pfx = metadata._prefix_wrapper.run(q, kv_cache, return_lse=True)
        v_sfx, s_sfx = metadata._suffix_wrapper.run(q, kv_cache, return_lse=True)
        v_merged, _ = flashinfer.cascade.merge_state(v_pfx, s_pfx, v_sfx, s_sfx)

        output.view(-1, self.num_heads, self.head_dim).copy_(v_merged)
        return output
