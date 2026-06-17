# Beam Search + Cascade Attention in TensorRT-LLM — Design & Implementation Plan

**Status:** Proposal / investigation complete
**Author:** next-token-ops
**Repository:** [`GuyStone/TensorRT-LLM`](https://github.com/GuyStone/TensorRT-LLM) (fork of `NVIDIA/TensorRT-LLM`)
**Branch:** `beam-cascade-attention`
**Target:** TensorRT-LLM PyTorch backend (`tensorrt_llm/_torch/`, the path behind `trtllm-serve --backend pytorch`)
**Date:** 2026-06

---

## 0. Repository & Git workflow

All implementation lands on our fork [`GuyStone/TensorRT-LLM`](https://github.com/GuyStone/TensorRT-LLM)
(forked from `NVIDIA/TensorRT-LLM`, default branch `main`), matching the local-clone
convention used for the other engines (`sglang`, `vllm-project`).

**Local checkout** (under `/home/guys/spotify/`, alongside the other engine repos):

```bash
cd /home/guys/spotify
git clone --filter=blob:none https://github.com/GuyStone/TensorRT-LLM.git
cd TensorRT-LLM
git checkout -b beam-cascade-attention
git push -u origin beam-cascade-attention
```

A blobless partial clone (`--filter=blob:none`) keeps the checkout light; full history
refs are present, blobs fetch on demand. The **Path B** work (§5.1) is pure Python and
needs no submodules. Only if **Path A** (§5.2, C++/CUDA) is pursued do you need:

```bash
git submodule update --init --recursive   # cutlass, etc. — for native builds only
```

**Branch & PR strategy**
- Feature branch: `beam-cascade-attention` on `origin` (the fork).
- Keep current with upstream by adding a remote and rebasing periodically:
  ```bash
  git remote add upstream https://github.com/NVIDIA/TensorRT-LLM.git
  git fetch upstream && git rebase upstream/main
  ```
- Open the PR **fork → fork** (`beam-cascade-attention` → `GuyStone/TensorRT-LLM:main`)
  for internal review first; only open an upstream PR to `NVIDIA/TensorRT-LLM` if/when we
  decide to contribute it back.
- This design doc lives in `lpm-benchmark`; optionally mirror it into the branch (e.g.
  `docs/source/blogs/beam-cascade-attention.md`) so the plan travels with the code.

All file paths in §5 and §11 are **relative to the `TensorRT-LLM/` clone root**.

---

## 1. Summary

Add a **cascade attention** decode path for **beam search** to the TRT-LLM PyTorch
backend. Cascade attention computes attention over a long *shared prefix* once and over
each sequence's short *unique suffix* separately, then merges the two partials with an
online-softmax (log-sum-exp) reduction. Beam search is a natural fit: all `B` beams of a
request share the prompt prefix and diverge only on generated tokens.

For our scoring workload (≈2000-token prompt, beam width 15–30, ~5 output tokens) the
shared prefix is **~99.75% of each beam's KV**. Today the decode kernel re-streams that
prefix once *per beam* every step; cascade streams it **once** and broadcasts it across
beams — a ~15–30× reduction in decode-attention HBM traffic on the dominant term.

**This is feasible but non-trivial.** The structural catch: beam search and cascade live
on different attention backends in TRT-LLM today (see §3). The core of the work is joining
them — which cascade itself makes tractable, because it confines beam divergence to a
≤5-token suffix.

---

## 2. Motivation

### 2.1 Workload

From `lpm-benchmark/profiles/qwen3_06b_beam_search.yaml` and the
1.7B equivalent:

| Param            | Value          |
|------------------|----------------|
| input length     | ~2000 tokens   |
| output length    | ~5 tokens      |
| beam width       | 15, 30         |
| model            | Qwen3-0.6B / 1.7B |
| max batch size   | 1 (per current TRT-LLM bench config) |

This is a *reranking / scoring* shape: a long shared prompt, a wide beam to enumerate
top-k continuations, a short output.

### 2.2 Where the cost is

Per decode step, today's beam kernel re-reads each beam's full KV path (prompt included):

```
naive beam decode (per step):  B × prompt_len            = 30 × 2000 = 60,000 KV reads
cascade decode (per step):     1 × prompt_len + B×suffix = 2000 + 30×5 ≈ 2,150 KV reads
                               └─ prompt read ONCE, broadcast across beams
```

The shared prefix dominates, so cascade ≈ `B×` cheaper on decode-attention bandwidth.
This regime clears the empirical floors other engines use to enable cascade at all
(vLLM: prefix ≥ 256 tokens, ≥ 8 sharing sequences — we have 2000 and 15–30).

### 2.3 The caveat we must measure first

With only ~5 output tokens, the one-time 2000-token **prefill** may dominate end-to-end
latency, and cascade only accelerates **decode**. Decode-attention can get 30× cheaper
while wall-clock QPS moves much less. **Gate the whole effort on Phase 0**: profile the
decode-vs-prefill latency split on `bw30_2k_5tok` (`return_perf_metrics: true` is already
on in `lpm-benchmark/engines/trtllm/run_benchmark.sh`). If decode
attention is a meaningful fraction of latency at beam 30, proceed.

---

## 3. Background — current TRT-LLM state (main branch)

Three facts shape the design. All are in the PyTorch backend used by `--backend pytorch`.

### 3.1 Beam search is bolted to the TRTLLM (custom-CUDA) attention backend only

`cache_indirection` — the `[batch, max_beam_width, max_seq_len]` int32 tensor mapping
`(seq, beam, pos) → source beam` — is forced to `None` for every backend except TRTLLM:

```python
# tensorrt_llm/_torch/pyexecutor/model_engine.py
cache_indirection = (self.cache_indirection_attention
                     if self.attn_backend.Metadata is TrtllmAttentionMetadata else None)
```

`interface.py` comments the field as "currently only used for `TrtllmAttentionMetadata`".
So beam search runs exclusively through `torch.ops.trtllm.attention`, and the backend
that ships a cascade kernel (FlashInfer) **does not support beam search today**.

### 3.2 Prompt KV is stored once across beams — but re-read once per beam

`KVCacheManager._pack_beam_cache_indices` (`resource_manager.py`) shares prompt blocks:
*"the first beam owns the shared prompt blocks; for every other beam, append only the
final block when it differs from beam 0's."* That is a **memory** dedup. At the kernel
level, `cache_indirection` is applied across the *entire* sequence, prompt positions
included, so the prompt is re-streamed `B` times during decode. The bandwidth dedup is the
unexploited opportunity. *(This per-beam-read conclusion is inferred from the indirection
semantics, consistent with the gpt-attention docs; not read off the `.cu` kernel.)*

### 3.3 A log-sum-exp merge op exists — but it's MLA-only

`torch.ops.trtllm.merge_chunked_attention_for_mla` (Python wrapper
`TrtllmAttention.merge_attention_for_mla` in `attention_backend/trtllm.py`) already
performs the online-softmax `(O, LSE)` merge for MLA chunked prefill, with softmax stats
laid out `[num_tokens, num_heads, 2]` and a `merge_op` schedule (`2=copy, 1=merge,
0=skip`). Cascade needs exactly this primitive — but a **generic dense-MHA** version is
not exposed in the Python layer. We either reuse FlashInfer's `merge_state` (Path B) or
generalize this op (Path A).

### 3.4 Fixed `max_beam_width`

`max_beam_width` is consumed at init to size preallocated buffers
(`cache_indirection_attention` at `model_engine.py`), the sampler's `BeamSearchStore`, the
KV manager, and CUDA-graph capture — hence the per-beam-width server restart in
`lpm-benchmark/engines/trtllm/run_benchmark.sh`. Cascade does not
change this, though it makes wide beams far cheaper to serve.

---

## 4. Design

### 4.1 The decomposition

For each request and its `B` beams, split each beam's decode attention into two partials
and merge:

```
Level 0  PREFIX:  q = [B current beam tokens]  →  shared prompt KV   (non-causal, paged, deduped)
                  ⇒ (O_pfx, lse_pfx)            # prompt pages referenced ONCE, B query rows

Level 1  SUFFIX:  q = [B current beam tokens]  →  each beam's own ≤5 generated tokens (causal)
                  ⇒ (O_sfx, lse_sfx)            # tiny; the ONLY beam-divergent part

MERGE:   O_beam = merge_state(O_pfx, lse_pfx, O_sfx, lse_sfx)
```

### 4.2 The merge math (numerically stable, safe-max form)

Given partials over disjoint key sets `I` (prefix) and `J` (suffix):

```
O(I)   = Σ_{i∈I} softmax(s_i)·v_i          LSE(I) = log Σ_{i∈I} exp(s_i)

m   = max(LSE(I), LSE(J))
w_I = exp(LSE(I) − m);  w_J = exp(LSE(J) − m)
O   = (w_I·O(I) + w_J·O(J)) / (w_I + w_J)
LSE = m + log(w_I + w_J)
```

This is associative/commutative, so prefix/suffix order is irrelevant and n partials fold
pairwise. FlashInfer's `merge_state`/`merge_states` implement exactly this (and the
safe-max internally). `s` passed to these ops is the **LSE (log-domain)**, not the raw
softmax denominator.

> **Verified finding (Phase 1):** FlashInfer's `s` is the LSE in **base-2** (`log₂ Σ 2ˣ`),
> matching FlashAttention's internal convention — *not* natural log. When both partials
> come from FlashInfer attention wrappers this is automatic (their LSE is already base-2
> and `merge_state` consumes it directly). Only when mixing in a natural-log LSE do you
> need a `1/ln2` conversion. Confirmed in
> `tests/unittest/_torch/attention_backend/test_beam_cascade_parity.py`
> (`test_merge_state_matches_flashinfer`).

### 4.3 Why this makes beam-on-FlashInfer tractable

The hard part of beam search is KV indirection over the *whole* sequence. After the
decomposition, the prefix level needs **no** indirection (all beams read the same shared
pages), and indirection survives only on the ≤5-token suffix. So we don't need a
beam-aware fused kernel — we need per-beam *suffix* block tables plus a predecessor remap
on a tiny tensor. That is the crux of why this integration is achievable in the Python
layer.

---

## 5. Architecture

### 5.1 Path B — new cascade backend on FlashInfer (recommended)

Add a dedicated backend instead of branching inside the hot TRTLLM CUDA path. It is
testable in isolation, reuses FlashInfer's battle-tested cascade/`merge_state` kernels
(no CUDA to write), and keeps beam logic out of `trtllm.py`.

```
tensorrt_llm/_torch/attention_backend/
  beam_cascade.py     # NEW: BeamCascadeAttention + BeamCascadeAttentionMetadata
  utils.py            # register "BEAM_CASCADE" in get_attention_backend()
```

Register it next to the existing entries in `get_attention_backend()` (currently
`VANILLA` / `TRTLLM` / `FLASHINFER` / `FLASHINFER_STAR_ATTENTION`, with a fallback to
`TRTLLM`). Because Path B uses FlashInfer kernels, gate the entry on
`IS_FLASHINFER_AVAILABLE` exactly as the `FLASHINFER` branches already do.

#### Backend skeleton (illustrative — adapt to the real `AttentionBackend` interface)

```python
# tensorrt_llm/_torch/attention_backend/beam_cascade.py
from dataclasses import dataclass
from typing import Optional
import torch
from flashinfer.cascade import merge_state
from flashinfer import BatchPrefillWithPagedKVCacheWrapper

from .interface import AttentionBackend, AttentionMetadata
from .flashinfer import FlashInferAttentionMetadata  # reuse paged-KV plan machinery


@dataclass(kw_only=True)
class BeamCascadeAttentionMetadata(FlashInferAttentionMetadata):
    # cache_indirection MUST stay populated so model_engine's per-step copy lights up.
    cache_indirection: Optional[torch.Tensor] = None

    # Built in prepare(): the two-level layout for the decode batch.
    prompt_lens: Optional[torch.Tensor] = None          # [num_reqs]  shared-prefix length per request
    prefix_kv_indptr: Optional[torch.Tensor] = None     # paged block table for the shared prompt
    prefix_kv_indices: Optional[torch.Tensor] = None
    suffix_kv_indptr: Optional[torch.Tensor] = None     # per-beam divergent suffix block tables
    suffix_kv_indices: Optional[torch.Tensor] = None
    suffix_last_page_len: Optional[torch.Tensor] = None

    def prepare(self):
        super().prepare()  # builds seq_lens, paged indices, etc.
        if self._is_decode_beam_step():
            self._build_cascade_layout()

    def _build_cascade_layout(self):
        # 1. Prefix level: all B beams of a request point qo rows at the SAME prompt pages.
        # 2. Suffix level: per-beam pages for generated tokens, remapped by the
        #    predecessor selection. Slice indirection to the suffix region only:
        #        suffix_indir = cache_indirection[:, :, prompt_len:cur_len]
        #    Because suffix_len ≤ output_len (~5), this is cheap — optionally physically
        #    gather/reorder the suffix KV instead of carrying indirection.
        ...


class BeamCascadeAttention(AttentionBackend[BeamCascadeAttentionMetadata]):
    Metadata = BeamCascadeAttentionMetadata

    def forward(self, q, k, v, metadata, forward_args=None, **kw):
        # Prefill: beams have not forked; the prompt is processed once. Delegate to the
        # standard paged prefill path (no cascade needed).
        if metadata.num_contexts > 0 and metadata.num_generations == 0:
            return self._prefill(q, k, v, metadata)

        # Decode: two passes + merge.
        o_pfx, lse_pfx = self._attend_prefix(q, metadata)   # non-causal, shared KV, return_lse=True
        o_sfx, lse_sfx = self._attend_suffix(q, metadata)   # causal within suffix, per-beam KV
        o_sfx, lse_sfx = _mask_empty_suffix(o_sfx, lse_sfx, metadata)  # see §6.4
        out, _ = merge_state(o_pfx, lse_pfx, o_sfx, lse_sfx)
        return out
```

For the **`max_batch_size 1`** case (our benchmark: one request, `B` beams sharing one
prompt) the simplest correct implementation is FlashInfer's
`MultiLevelCascadeAttentionWrapper(num_levels=2)` directly — level 0 = prompt, level 1 =
per-beam suffix. For **batch > 1** (different prompts per request → no single global
prefix, a "forest"), prefer the explicit two-pass + `merge_state` above; the multi-level
wrapper assumes one shared prefix across the whole batch (the same single-tree limitation
vLLM has).

#### Relevant FlashInfer signatures

```python
# flashinfer/cascade.py
merge_state(v_a, s_a, v_b, s_b) -> (V, S)
#   v_*: [seq, num_heads, head_dim]   s_*: [seq, num_heads]  (s = LSE)

merge_state_in_place(v, s, v_other, s_other, mask: Optional[Tensor]=None) -> None
#   mask: [seq] gates which rows merge — use for the empty-suffix edge (§6.4)

MultiLevelCascadeAttentionWrapper(num_levels, float_workspace_buffer, kv_layout="NHD", ...)
  .plan(qo_indptr_arr, paged_kv_indptr_arr, paged_kv_indices_arr,
        paged_kv_last_page_len, num_qo_heads, num_kv_heads, head_dim, page_size,
        causal=False, ...)
  .run(q, paged_kv_cache)
```

### 5.2 Path A — cascade branch inside TrtllmAttention (higher ceiling, more effort)

Mirror the existing chunked-prefill structure in `trtllm.py`
(`pre_process_for_chunked_prefill` builds segment indptrs + a `merge_op` schedule). Add a
beam-cascade branch in `forward_impl`, and **generalize
`merge_chunked_attention_for_mla` into a dense-MHA `merge_attention_states` op**.

- **Pros:** stays on the already-supported beam path, no FlashInfer dependency, fusion
  potential, the merge op already exists in spirit.
- **Cons:** C++/CUDA work; you own a new kernel/op and its CUDA-graph behavior.

**Recommendation:** build and validate Path B first. Only move to Path A if FlashInfer
launch overhead at small suffix lengths eats the win, or if FlashInfer can't be made a
hard dependency for the serving image.

---

## 6. Beam-search integration details

### 6.1 Sampler interplay — no changes needed

`TorchSampler` (`pyexecutor/sampler.py`) + `sampling_utils.py` already produce all beam
bookkeeping each step: `beam_search_sampling_batch` does top-k over `beam_width*vocab`,
derives `predecessor_beam = next_tokens // vocab_size`, and updates `cache_indirection`
via gather/scatter. The cascade backend **consumes** `cache_indirection` (suffix slice
only); it does not change how beams are scored or selected.

### 6.2 cache_indirection plumbing

`cache_indirection` must remain populated on the metadata so the existing per-step copy in
`model_engine.py` (from the sampler's `cache_indirection_buffer`) still fires. The gate at
§3.1 keys off `Metadata is TrtllmAttentionMetadata`; the new backend must either be added
to that condition or expose an equivalent flag so its `cache_indirection` is wired. The
backend then uses only `cache_indirection[:, :, prompt_len:cur_len]` for the suffix.

### 6.3 KV cache layout

The shared-prefix dedup at the *block* level (§3.2) already gives us the property cascade
needs: all `B` beams' prefix pages are the same physical pages. The prefix level's
`paged_kv_indices` reference those once. Ensure no copy-on-write splits a prefix page per
beam (loses the bandwidth win and risks divergence).

### 6.4 Correctness pitfalls (must-handle)

| # | Pitfall | Handling |
|---|---------|----------|
| 1 | **RoPE positions** — suffix Q/K must be rotated at *absolute* positions continuing from `prompt_len`, not restarted at 0 | keep `pos_encoding_mode`/`rope_*` consistent across levels; if RoPE is pre-applied to cached KV, ensure suffix pages carry correct absolute positions. Silent corruption, no shape error |
| 2 | **Masking split** — prefix = full attention to prompt; suffix = causal *within suffix only* | `causal=True` on suffix level only; prefix non-causal |
| 3 | **Partition** — each beam's latest token in exactly one level | no overlap (double-count via merge) and no gap (token can't attend to itself); verify `kv_len_prefix + kv_len_suffix == total` |
| 4 | **Empty-suffix first step** — at step 0 suffix KV length is 0 → `lse=−∞`, `O` undefined | treat `exp(−∞)=0` (merge contributes nothing) or skip the suffix level; use `merge_state_in_place`'s `mask` arg |
| 5 | **LSE numerics + scale + base** — pass log-domain LSE in FlashInfer's **base-2** convention (§4.2); `sm_scale`/`logits_soft_cap` identical across levels | else merge is mathematically wrong even though it runs. A base mismatch (natural-log vs base-2) silently shifts the merge weights — caught by the Phase 1 cross-check |
| 6 | **Small-beam gate** — kernel-launch overhead can exceed savings for tiny beams/prefixes | fall back to plain beam decode when `beam_width < 8` or `prompt_len < 256` (cf. vLLM heuristic). Our profiles clear it |

---

## 7. Implementation plan (phased)

### Phase 0 — Validate the opportunity (gate) — ~0.5 day
- Profile decode-vs-prefill latency split on `bw30_2k_5tok` for Qwen3-1.7B with current
  TRTLLM backend (`return_perf_metrics: true`).
- **Go/no-go:** proceed only if decode attention is a meaningful fraction of latency at
  beam 30. Record numbers in this doc.

### Phase 1 — Reference correctness, offline — ✅ landed
- Standalone harness `tests/unittest/_torch/attention_backend/test_beam_cascade_parity.py`:
  pure-PyTorch reference proving `full_attention(beam) == merge_state(prefix, suffix)` for
  every beam, across beam widths {1, 4, 15, 30}.
- Locks down the §6.4 pitfalls: RoPE absolute positions (negative test that catches a
  restart-at-0 bug), empty-suffix first step (no NaN), per-beam suffix remap on beam
  reorder, and a cross-check against the real `flashinfer.cascade.merge_state` (which
  surfaced the base-2 LSE convention, §4.2). All pass on GPU.
- Runs standalone (`python …/test_beam_cascade_parity.py`) or under pytest (with the
  repo's full unit-test env).

### Phase 2 — `BeamCascadeAttention` backend — ✅ implemented (engine validation pending CI)
- `tensorrt_llm/_torch/attention_backend/beam_cascade.py`: `BeamCascadeAttention`
  (subclasses `FlashInferAttention`) + `BeamCascadeAttentionMetadata`. Delegates to the
  FlashInfer path for prefill / mixed / `beam_width==1`; the cascade decode path runs the
  explicit 2× `BatchPrefillWithPagedKVCacheWrapper` + `merge_state` (general / forest, so
  Phase 3 is folded in). v1 cascade is **eager** (CUDA-graph capture is future work).
- `tensorrt_llm/_torch/attention_backend/beam_cascade_planning.py`: the pure prefix/suffix
  page-table split + append layout — **no engine deps**, unit-tested in
  `test_beam_cascade_split.py` (incl. the lossless multi-block suffix that
  `_pack_beam_cache_indices` truncates).
- Registered `"BEAM_CASCADE"` in `utils.py` + `attention_backend/__init__.py`; broadened
  the `model_engine.py` `cache_indirection` gate to pass the per-beam buffer to the new
  metadata type.
- Key constraint handled: the inherited `FlashInferAttentionMetadata.prepare()` assumes
  `beam_width==1`, so the beam path builds its KV layout from the lossless
  `impl.get_batch_cache_block_ids`.
- **Pending CI** (§7 items 5-8, needs a built engine + model): end-to-end beam outputs vs
  the TRTLLM backend, `cache_indirection` suffix remap on beam reorder, the
  (request, beam) query-ordering assumption, and CUDA-graph capture.

### Phase 3 — Batched (forest) support — ✅ folded into Phase 2
- The cascade path uses the explicit two-pass + `merge_state` (not the single-tree
  wrapper), so multiple requests with different prefixes / beam widths / suffix lengths
  work; validated in `test_beam_cascade_paged.py::test_two_pass_forest_heterogeneous`.

### Phase 4 — Benchmark & tune — ~2 days
- Add `BEAM_CASCADE` as a variant in `lpm-benchmark/engines/trtllm/run_benchmark.sh`
  (behind a flag/env), A/B vs the default TRTLLM backend across the beam_search profiles.
- Tune the small-beam gate; confirm CUDA-graph compatibility (fixed shapes per beam width).

### Phase 5 (optional) — Path A — if Phase 4 shows launch overhead dominates
- Generalize `merge_chunked_attention_for_mla` to dense MHA; add a cascade branch in
  `TrtllmAttention.forward_impl`.

---

## 8. Testing & validation

- **Numerical:** per-layer output parity vs full-attention beam decode (Phase 1 harness),
  tolerances per dtype (fp16/bf16/fp8). LSE parity check too, not just `O`.
- **End-to-end:** identical generated beams (token ids + scores) vs TRTLLM backend on a
  fixed seed set, beam widths {1, 4, 15, 30}, prompt lengths {256, 2000, 8000}.
- **Edge cases:** beam_width=1 (must degrade to plain decode), output_len=1 (suffix=0 → §6.4),
  prompt shorter than one page, mid-generation beam reordering (predecessor != identity).
- **CUDA graphs:** capture/replay at each fixed beam width; assert no shape drift.

---

## 9. Success criteria

- **Correctness:** beam outputs bit-comparable (ids + scores) to the TRTLLM backend within
  fp tolerance across the test matrix.
- **Performance:** measurable QPS improvement on `bw15_2k_5tok` / `bw30_2k_5tok` vs the
  default TRTLLM backend in the existing harness; no regression at beam_width=1.
- **Cleanliness:** isolated backend, no changes to sampler scoring logic, < ~400 LOC for
  the Path B backend + metadata.

---

## 10. Risks & open questions

- **Prefill dominates (Phase 0 risk):** with 5 output tokens, decode may be a small slice
  of latency → modest end-to-end win. Mitigated by gating on Phase 0.
- **FlashInfer as a hard dependency** in the serving image; and whether the FlashInfer
  paged layout can be fed from TRT-LLM's KV cache manager block tables without a copy.
- **CUDA-graph capture** of a two-pass + merge decode at fixed beam width — needs the
  workspace/indptr buffers preallocated (FlashInfer `use_cuda_graph=True`, buffer arrays).
- **RoPE-in-cache vs RoPE-on-the-fly** for the suffix boundary (§6.4 #1) — confirm which
  the model path uses.
- **Forest case** generality vs the single-tree wrapper — Phase 3 explicit merge resolves
  it but adds indptr bookkeeping.
- **MLA models** (DeepSeek-style): the TRTLLM MLA path hardcodes `beam_width=1`; cascade
  for MLA + beam is out of scope here.

---

## 11. References

- TensorRT-LLM PyTorch backend: <https://nvidia.github.io/TensorRT-LLM/torch.html>
- FlashInfer cascade API: <https://docs.flashinfer.ai/api/cascade.html>
- Cascade Inference (blog, merge math): <https://flashinfer.ai/2024/02/02/cascade-inference.html>
- FlashInfer paper §2.2 (attention-state merge): <https://arxiv.org/abs/2501.01005>
- DeFT — Flash Tree-Attention (closest precedent; beam search is a tree): <https://arxiv.org/abs/2404.00242>
- vLLM cascade heuristic (`use_cascade_attention`): `vllm/v1/attention/backends/flash_attn.py`
- SGLang cascade-kernel request: <https://github.com/sgl-project/sglang/issues/1715>

### Key TRT-LLM files to touch / reference

- `tensorrt_llm/_torch/attention_backend/beam_cascade.py` — **new**
- `tensorrt_llm/_torch/attention_backend/utils.py` — register backend
- `tensorrt_llm/_torch/attention_backend/interface.py` — `AttentionBackend` / `AttentionMetadata`
- `tensorrt_llm/_torch/attention_backend/flashinfer.py` — paged-KV plan machinery to reuse
- `tensorrt_llm/_torch/attention_backend/trtllm.py` — `merge_attention_for_mla` (Path A reference)
- `tensorrt_llm/_torch/pyexecutor/model_engine.py` — `cache_indirection` wiring/gate
- `tensorrt_llm/_torch/pyexecutor/sampler.py`, `sampling_utils.py` — beam bookkeeping (read-only)
- `tensorrt_llm/_torch/pyexecutor/resource_manager.py` — `_pack_beam_cache_indices` (shared prefix blocks)
