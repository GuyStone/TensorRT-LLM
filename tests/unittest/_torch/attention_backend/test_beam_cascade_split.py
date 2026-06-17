# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the pure beam-cascade page-table split logic.

Loads ``beam_cascade_planning.py`` directly (it only imports torch), so it runs without a
built engine. Covers the design's §7 items 2-3: the prefix/suffix split, the lossless
multi-block per-beam suffix (the case the lossy ``_pack_beam_cache_indices`` would have
truncated), prompt-boundary clamping, and the append layout.

Run standalone:
    python tests/unittest/_torch/attention_backend/test_beam_cascade_split.py
"""
from __future__ import annotations

import importlib.util
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_MOD_PATH = os.path.normpath(
    os.path.join(_HERE, "..", "..", "..", "..", "tensorrt_llm", "_torch",
                 "attention_backend", "beam_cascade_planning.py"))
_spec = importlib.util.spec_from_file_location("beam_cascade_planning", _MOD_PATH)
bcp = importlib.util.module_from_spec(_spec)
# Register before exec so @dataclass can resolve cls.__module__ (it looks the module up
# in sys.modules during processing).
sys.modules[_spec.name] = bcp
_spec.loader.exec_module(bcp)


def _lst(t: torch.Tensor):
    return t.tolist()


def test_longest_common_prefix_len():
    assert bcp.longest_common_prefix_len([[10, 11, 12], [10, 11, 13]]) == 2
    assert bcp.longest_common_prefix_len([[10, 11], [10, 11]]) == 2
    assert bcp.longest_common_prefix_len([[10, 12], [11, 13]]) == 0
    # equal length but divergent ids from the start -> 0 (ids, not positions)
    assert bcp.longest_common_prefix_len([[5, 6, 7], [5, 9, 7]]) == 1
    assert bcp.longest_common_prefix_len([]) == 0


def test_single_request_single_block_suffix():
    # page_size 4, prompt 8 tokens (2 blocks), beam_width 2, each beam generated 3 tokens.
    page = 4
    beams = [[[10, 11, 12], [10, 11, 13]]]   # [req][beam] -> blocks
    prompt_lens = [8]
    kv_lens = [[11, 11]]                      # 8 prompt + 3 generated
    p = bcp.build_cascade_page_tables(beams, prompt_lens, kv_lens, page)

    assert p.num_gen_queries == 2
    assert _lst(p.prefix_qo_indptr) == [0, 2]          # one group of 2 beams
    assert _lst(p.prefix_kv_indptr) == [0, 2]
    assert _lst(p.prefix_kv_indices) == [10, 11]       # shared prompt blocks, once
    assert _lst(p.prefix_last_page_len) == [4]         # full page
    assert _lst(p.suffix_qo_indptr) == [0, 1, 2]       # one query per beam
    assert _lst(p.suffix_kv_indptr) == [0, 1, 2]
    assert _lst(p.suffix_kv_indices) == [12, 13]       # each beam's own suffix block
    assert _lst(p.suffix_last_page_len) == [3, 3]      # 3 generated tokens in the page


def test_multi_block_suffix_is_lossless():
    """The case _pack_beam_cache_indices would truncate: each beam's suffix spans >1 block.
    All per-beam suffix blocks must be preserved."""
    page = 4
    # prompt 4 tokens (1 block, id 10 shared); each beam then owns 2 divergent blocks.
    beams = [[[10, 12, 14], [10, 13, 15]]]
    prompt_lens = [4]
    kv_lens = [[11, 11]]                       # 4 prompt + 7 generated -> 2 suffix pages
    p = bcp.build_cascade_page_tables(beams, prompt_lens, kv_lens, page)

    assert _lst(p.prefix_kv_indices) == [10]                 # only the shared prompt block
    # both beams keep BOTH suffix blocks (4 entries total) — not truncated to 1/beam
    assert _lst(p.suffix_kv_indices) == [12, 14, 13, 15]
    assert _lst(p.suffix_kv_indptr) == [0, 2, 4]
    assert _lst(p.suffix_last_page_len) == [3, 3]            # 7 - 4 = 3 in last page


def test_prefix_clamped_to_prompt_boundary():
    """Even if beams share generated blocks, the shared prefix never exceeds full prompt
    blocks (floor prompt_len / page_size)."""
    page = 4
    # beams share [10, 11, 12] by id, but prompt is only 8 tokens (2 full blocks).
    beams = [[[10, 11, 12, 20], [10, 11, 12, 21]]]
    prompt_lens = [8]
    kv_lens = [[16, 16]]                       # 8 prompt + 8 generated
    p = bcp.build_cascade_page_tables(beams, prompt_lens, kv_lens, page)

    assert _lst(p.prefix_kv_indices) == [10, 11]            # clamped to 2 prompt blocks
    # block 12 (shared but generated) belongs to the suffix, not the prefix
    assert _lst(p.suffix_kv_indices) == [12, 20, 12, 21]
    assert _lst(p.suffix_last_page_len) == [4, 4]           # 16 - 8 = 8 tokens -> 2 pages


def test_forest_multiple_requests():
    page = 4
    beams = [
        [[1, 2], [1, 3]],                      # req0: 2 beams
        [[7, 8, 9], [7, 8, 10], [7, 8, 11]],   # req1: 3 beams
    ]
    prompt_lens = [4, 8]
    kv_lens = [[7, 7], [11, 11, 11]]
    p = bcp.build_cascade_page_tables(beams, prompt_lens, kv_lens, page)

    assert p.num_gen_queries == 5                          # 2 + 3 beams
    assert _lst(p.prefix_qo_indptr) == [0, 2, 5]           # groups of 2 and 3
    assert _lst(p.suffix_qo_indptr) == [0, 1, 2, 3, 4, 5]  # one query per beam
    # suffix_kv_indptr length == num_gen_queries + 1
    assert len(p.suffix_kv_indptr) == p.num_gen_queries + 1


def test_append_layout():
    page = 4
    beams = [[[10, 11, 12], [10, 11, 13]]]
    kv_lens = [[11, 11]]
    a = bcp.build_append_layout(beams, kv_lens, page)
    assert _lst(a.kv_indptr) == [0, 3, 6]                  # 3 blocks per beam
    assert _lst(a.kv_indices) == [10, 11, 12, 10, 11, 13]
    assert _lst(a.last_page_len) == [3, 3]                 # 11 - 8 = 3 in last page
    assert _lst(a.batch_indices) == [0, 1]                 # one seq per beam
    assert _lst(a.positions) == [10, 10]                   # new token at pos kv_len-1


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    results = []
    for fn in tests:
        try:
            fn()
            results.append((fn.__name__, "PASS"))
        except Exception as e:  # noqa: BLE001
            results.append((fn.__name__, f"FAIL: {type(e).__name__}: {e}"))
    w = max(len(n) for n, _ in results)
    for n, s in results:
        print(f"  {n.ljust(w)}  {s}")
    failed = [n for n, s in results if s.startswith("FAIL")]
    print(f"\n{'FAILED: ' + ', '.join(failed) if failed else 'ALL PASS'}")
    raise SystemExit(1 if failed else 0)
