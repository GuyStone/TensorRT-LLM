# SPDX-License-Identifier: Apache-2.0
"""Phase 1/2 kernel-level validation for beam-search + cascade attention.

Where `test_beam_cascade_parity.py` proves the *math* of the decomposition in dense
PyTorch, this module proves the **actual paged-KV kernel path** the BEAM_CASCADE backend
will use: a shared-prefix page set read once across all beams, per-beam suffix pages, and
the FlashInfer cascade merge — against a dense full-attention reference.

Two strategies are validated (design §5.1):
  - `MultiLevelCascadeAttentionWrapper` (num_levels=2)  — single shared prefix (batch == 1)
  - explicit 2x `BatchPrefillWithPagedKVCacheWrapper` + `cascade.merge_state` — the GENERAL
    "forest" path (multiple requests, each with its own prefix / beam width / suffix length)

It also confirms (no assertion needed beyond parity) that FlashInfer's wrapper LSE is
base-2 and feeds `merge_state` directly — i.e. when both partials come from FlashInfer,
no LSE base conversion is required.

Requires FlashInfer + CUDA. Run standalone:
    python tests/unittest/_torch/attention_backend/test_beam_cascade_paged.py
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

try:
    import pytest
except ImportError:
    class _Mark:
        def __getattr__(self, _):
            return lambda *a, **k: (lambda fn: fn)

    class _Shim:
        mark = _Mark()

        @staticmethod
        def skip(reason=""):
            raise RuntimeError(reason)

    pytest = _Shim()  # type: ignore

try:
    import flashinfer
    from flashinfer.cascade import merge_state
    _OK = torch.cuda.is_available()
except Exception:
    _OK = False

requires_fi = pytest.mark.skipif(not _OK, reason="requires FlashInfer + CUDA")

DEV = "cuda"
PAGE = 16
H, HKV, HD = 16, 8, 128  # query heads, kv heads (GQA), head_dim
DTYPE = torch.float16
SCALE = 1.0 / math.sqrt(HD)


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


@dataclass
class Req:
    prefix_len: int
    beam_width: int
    suffix_len: int


@dataclass
class Scenario:
    kv_pool: torch.Tensor                       # [pages, 2, PAGE, HKV, HD]  (NHD, 0=K 1=V)
    q: torch.Tensor                             # [N, H, HD]   N = sum(beam_width)
    pfx: dict = field(default_factory=dict)     # qo/indptr/indices/last for the prefix level
    sfx: dict = field(default_factory=dict)     # ... for the suffix level
    dense: list = field(default_factory=list)   # per-query (q, Kfull, Vfull) for the reference


def _i32(values) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.int32, device=DEV)


def build_scenario(reqs: list[Req], seed: int = 0) -> Scenario:
    g = torch.Generator(device=DEV).manual_seed(seed)

    def rnd(*shape):
        return torch.randn(*shape, generator=g, device=DEV, dtype=DTYPE)

    pool_k: list[torch.Tensor] = []
    pool_v: list[torch.Tensor] = []

    def add_seq(k: torch.Tensor, v: torch.Tensor, n: int):
        """Append n tokens of (k, v) as pages; return (page_ids, last_page_len)."""
        npg = _ceil_div(n, PAGE)
        start = len(pool_k)
        for pg in range(npg):
            kp = torch.zeros(PAGE, HKV, HD, device=DEV, dtype=DTYPE)
            vp = torch.zeros(PAGE, HKV, HD, device=DEV, dtype=DTYPE)
            for off in range(PAGE):
                t = pg * PAGE + off
                if t < n:
                    kp[off], vp[off] = k[t], v[t]
            pool_k.append(kp)
            pool_v.append(vp)
        return list(range(start, start + npg)), n - (npg - 1) * PAGE

    pfx = dict(qo=[0], indptr=[0], idx=[], last=[])
    sfx = dict(qo=[0], indptr=[0], idx=[], last=[])
    qs: list[torch.Tensor] = []
    dense: list = []

    for r in reqs:
        kp_, vp_ = rnd(r.prefix_len, HKV, HD), rnd(r.prefix_len, HKV, HD)
        ids, last = add_seq(kp_, vp_, r.prefix_len)
        pfx["idx"] += ids
        pfx["indptr"].append(len(pfx["idx"]))
        pfx["last"].append(last)
        pfx["qo"].append(pfx["qo"][-1] + r.beam_width)  # one prefix group, beam_width queries
        for _ in range(r.beam_width):
            ks_, vs_ = rnd(r.suffix_len, HKV, HD), rnd(r.suffix_len, HKV, HD)
            ids, last = add_seq(ks_, vs_, r.suffix_len)
            sfx["idx"] += ids
            sfx["indptr"].append(len(sfx["idx"]))
            sfx["last"].append(last)
            sfx["qo"].append(sfx["qo"][-1] + 1)         # one suffix group per beam
            qb = rnd(H, HD)
            qs.append(qb)
            dense.append((qb, torch.cat([kp_, ks_]), torch.cat([vp_, vs_])))

    kv_pool = torch.stack([torch.stack(pool_k), torch.stack(pool_v)], dim=1)
    return Scenario(kv_pool=kv_pool, q=torch.stack(qs), pfx=pfx, sfx=sfx, dense=dense)


# --------------------------------------------------------------------------- #
# References / cascade strategies                                             #
# --------------------------------------------------------------------------- #
def _expand(x: torch.Tensor) -> torch.Tensor:  # [..,HKV,HD] -> [..,H,HD]
    return x.repeat_interleave(H // HKV, dim=-2)


def dense_reference(sc: Scenario) -> torch.Tensor:
    outs = []
    for qb, kfull, vfull in sc.dense:
        s = torch.einsum("hd,nhd->hn", qb.float(), _expand(kfull).float()) * SCALE
        outs.append(torch.einsum("hn,nhd->hd", torch.softmax(s, -1), _expand(vfull).float()))
    return torch.stack(outs)


def _prefill(sc: Scenario, lvl: dict):
    ws = torch.empty(128 << 20, dtype=torch.uint8, device=DEV)
    w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, kv_layout="NHD")
    w.plan(_i32(lvl["qo"]), _i32(lvl["indptr"]), _i32(lvl["idx"]), _i32(lvl["last"]),
           H, HKV, HD, PAGE, causal=False, sm_scale=SCALE)
    return w.run(sc.q, sc.kv_pool, return_lse=True)  # (o:[N,H,HD], lse:[N,H] base-2)


def cascade_two_pass(sc: Scenario) -> torch.Tensor:
    """General path: prefix pass + suffix pass + LSE merge (handles the forest case)."""
    o_p, lse_p = _prefill(sc, sc.pfx)
    o_s, lse_s = _prefill(sc, sc.sfx)
    o, _ = merge_state(o_p, lse_p, o_s, lse_s)  # both base-2 -> direct, no conversion
    return o.float()


def cascade_multilevel(sc: Scenario) -> torch.Tensor:
    """Single-shared-prefix path via MultiLevelCascadeAttentionWrapper (batch == 1)."""
    ws = torch.empty(128 << 20, dtype=torch.uint8, device=DEV)
    w = flashinfer.MultiLevelCascadeAttentionWrapper(2, ws, kv_layout="NHD")
    w.plan([_i32(sc.pfx["qo"]), _i32(sc.sfx["qo"])],
           [_i32(sc.pfx["indptr"]), _i32(sc.sfx["indptr"])],
           [_i32(sc.pfx["idx"]), _i32(sc.sfx["idx"])],
           [_i32(sc.pfx["last"]), _i32(sc.sfx["last"])],
           H, HKV, HD, PAGE, causal=False, sm_scale=SCALE)
    return w.run(sc.q, sc.kv_pool).float()


_TOL = dict(atol=2e-3, rtol=2e-3)  # fp16 kernel path


# --------------------------------------------------------------------------- #
# Tests                                                                       #
# --------------------------------------------------------------------------- #
@requires_fi
@pytest.mark.parametrize("beam_width", [1, 4, 15, 30])
@pytest.mark.parametrize("suffix_len", [1, 5])
def test_multilevel_single_request(beam_width, suffix_len):
    sc = build_scenario([Req(prefix_len=200, beam_width=beam_width, suffix_len=suffix_len)])
    torch.testing.assert_close(cascade_multilevel(sc), dense_reference(sc), **_TOL)


@requires_fi
@pytest.mark.parametrize("beam_width", [1, 4, 15, 30])
@pytest.mark.parametrize("suffix_len", [1, 5])
def test_two_pass_single_request(beam_width, suffix_len):
    sc = build_scenario([Req(prefix_len=200, beam_width=beam_width, suffix_len=suffix_len)])
    torch.testing.assert_close(cascade_two_pass(sc), dense_reference(sc), **_TOL)


@requires_fi
def test_two_pass_forest_heterogeneous():
    """Multiple requests, different prefix / beam width / suffix len in one batch."""
    sc = build_scenario([
        Req(prefix_len=300, beam_width=6, suffix_len=3),
        Req(prefix_len=120, beam_width=4, suffix_len=5),
        Req(prefix_len=512, beam_width=8, suffix_len=1),
    ], seed=3)
    torch.testing.assert_close(cascade_two_pass(sc), dense_reference(sc), **_TOL)


@requires_fi
def test_workload_scale():
    """The benchmarked regime: 2000-token shared prefix, beam 30, short suffix."""
    sc = build_scenario([Req(prefix_len=2000, beam_width=30, suffix_len=5)], seed=7)
    ref = dense_reference(sc)
    torch.testing.assert_close(cascade_two_pass(sc), ref, **_TOL)
    torch.testing.assert_close(cascade_multilevel(sc), ref, **_TOL)


if __name__ == "__main__":
    results = []

    def run(name, fn):
        try:
            fn()
            results.append((name, "PASS"))
        except Exception as e:  # noqa: BLE001
            results.append((name, f"FAIL: {type(e).__name__}: {str(e).splitlines()[0]}"))

    if not _OK:
        print("SKIP: requires FlashInfer + CUDA")
        raise SystemExit(0)

    for bw in (1, 4, 15, 30):
        for sl in (1, 5):
            run(f"multilevel[bw={bw},sfx={sl}]", lambda bw=bw, sl=sl:
                test_multilevel_single_request(bw, sl))
            run(f"two_pass[bw={bw},sfx={sl}]", lambda bw=bw, sl=sl:
                test_two_pass_single_request(bw, sl))
    run("two_pass_forest_heterogeneous", test_two_pass_forest_heterogeneous)
    run("workload_scale[P=2000,bw=30]", test_workload_scale)

    width = max(len(n) for n, _ in results)
    for n, s in results:
        print(f"  {n.ljust(width)}  {s}")
    failed = [n for n, s in results if s.startswith("FAIL")]
    print(f"\n{'FAILED: ' + ', '.join(failed) if failed else 'ALL PASS'}")
    raise SystemExit(1 if failed else 0)
