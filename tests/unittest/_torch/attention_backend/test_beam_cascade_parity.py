# SPDX-License-Identifier: Apache-2.0
"""Phase 1 correctness harness for beam-search + cascade attention.

Design doc: docs/source/blogs/beam-cascade-attention.md (§4 decomposition, §6.4 pitfalls).

This is a *standalone* numerical harness — it deliberately does NOT touch the engine.
It proves the cascade decomposition is mathematically exact before any backend work:

    full_attention(beam)  ==  merge_state( prefix_attention, suffix_attention )

for every beam, and it locks down the pitfalls called out in the design (§6.4):
    - RoPE must use ABSOLUTE positions across the prefix/suffix boundary,
    - the empty-suffix first decode step must not produce NaNs,
    - per-beam suffix remap (beam reorder / predecessor selection) stays correct,
    - our reference merge matches FlashInfer's `cascade.merge_state` op (the one the
      engine backend will call).

Run standalone:   python tests/unittest/_torch/attention_backend/test_beam_cascade_parity.py
Run with pytest:  pytest tests/unittest/_torch/attention_backend/test_beam_cascade_parity.py
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

try:
    import pytest
except ImportError:  # allow `python test_...py` without pytest installed
    class _Mark:
        def __getattr__(self, _):
            def deco(*a, **k):
                return (lambda fn: fn)
            return deco

    class _PytestShim:
        mark = _Mark()

        @staticmethod
        def skip(reason=""):
            raise RuntimeError(f"skipped: {reason}")

    pytest = _PytestShim()  # type: ignore

try:
    import flashinfer
    from flashinfer.cascade import merge_state as fi_merge_state
    _HAS_FLASHINFER = True
except Exception:  # pragma: no cover - environment dependent
    _HAS_FLASHINFER = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NEG_INF = float("-inf")
LN2 = math.log(2.0)  # FlashInfer's LSE is base-2 (log2-sum-exp2); see cross-check test


@dataclass
class Cfg:
    beam_width: int = 8
    prefix_len: int = 2000   # the shared prompt — the whole point of cascade
    suffix_len: int = 4      # per-beam generated tokens (≤ output_len)
    num_heads: int = 16
    num_kv_heads: int = 8    # GQA (num_heads % num_kv_heads == 0)
    head_dim: int = 128
    rope_theta: float = 1.0e6
    dtype: torch.dtype = torch.float32
    seed: int = 0

    @property
    def sm_scale(self) -> float:
        return 1.0 / math.sqrt(self.head_dim)


# --------------------------------------------------------------------------- #
# RoPE (NeoX / rotate-half style)                                             #
# --------------------------------------------------------------------------- #
def rope_tables(positions: torch.Tensor, head_dim: int, theta: float,
                dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin of shape [len(positions), head_dim] for the given absolute positions."""
    inv_freq = 1.0 / (theta**(torch.arange(0, head_dim, 2, device=positions.device,
                                            dtype=torch.float32) / head_dim))
    freqs = positions.to(torch.float32)[:, None] * inv_freq[None, :]   # [N, D/2]
    emb = torch.cat((freqs, freqs), dim=-1)                            # [N, D]
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, positions: torch.Tensor, theta: float) -> torch.Tensor:
    """Apply RoPE to x[..., N, H, D] given absolute `positions` [N]."""
    cos, sin = rope_tables(positions, x.shape[-1], theta, x.dtype)     # [N, D]
    cos = cos[:, None, :]                                              # [N, 1, D]
    sin = sin[:, None, :]
    return x * cos + _rotate_half(x) * sin


# --------------------------------------------------------------------------- #
# Attention primitives                                                        #
# --------------------------------------------------------------------------- #
def _expand_kv(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    """[..., Hkv, D] -> [..., H, D] by GQA head repetition."""
    hkv = x.shape[-2]
    if hkv == num_heads:
        return x
    return x.repeat_interleave(num_heads // hkv, dim=-2)


def attn_with_lse(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                  sm_scale: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-query-per-beam attention returning (output, logsumexp).

    q: [B, H, D]   k,v: [B, N, H, D]  (already GQA-expanded, RoPE-applied)
    returns o: [B, H, D]   lse: [B, H]
    """
    if k.shape[1] == 0:  # empty key set (e.g. suffix at the first decode step)
        b, h, d = q.shape
        return (torch.zeros(b, h, d, dtype=q.dtype, device=q.device),
                torch.full((b, h), NEG_INF, dtype=torch.float32, device=q.device))
    scores = torch.einsum("bhd,bnhd->bhn", q.float(), k.float()) * sm_scale  # [B,H,N]
    lse = torch.logsumexp(scores, dim=-1)                                    # [B,H]
    weights = torch.softmax(scores, dim=-1)                                  # [B,H,N]
    o = torch.einsum("bhn,bnhd->bhd", weights, v.float()).to(q.dtype)        # [B,H,D]
    return o, lse


def merge_state_ref(o_a: torch.Tensor, lse_a: torch.Tensor,
                    o_b: torch.Tensor, lse_b: torch.Tensor
                    ) -> tuple[torch.Tensor, torch.Tensor]:
    """Numerically-stable online-softmax merge of two attention partials (§4.2).

    o_*: [B, H, D]   lse_*: [B, H]   (lse is log-sum-exp, log-domain).
    Safe against lse == -inf (empty partial contributes nothing, no NaN).
    """
    m = torch.maximum(lse_a, lse_b)
    m = torch.where(torch.isinf(m), torch.zeros_like(m), m)  # both -inf -> 0, avoid nan
    wa = torch.exp(lse_a - m)
    wb = torch.exp(lse_b - m)
    denom = wa + wb
    o = (wa[..., None] * o_a.float() + wb[..., None] * o_b.float()) / denom[..., None]
    lse = m + torch.log(denom)
    return o.to(o_a.dtype), lse


# --------------------------------------------------------------------------- #
# Scenario construction                                                       #
# --------------------------------------------------------------------------- #
def make_scenario(cfg: Cfg):
    g = torch.Generator(device=DEVICE).manual_seed(cfg.seed)
    B, P, S = cfg.beam_width, cfg.prefix_len, cfg.suffix_len
    Hkv, D = cfg.num_kv_heads, cfg.head_dim

    def rnd(*shape):
        return torch.randn(*shape, generator=g, device=DEVICE, dtype=cfg.dtype)

    return {
        "q_raw": rnd(B, cfg.num_heads, D),       # one decode query per beam
        "kpfx_raw": rnd(P, Hkv, D),              # shared prompt K (same for all beams)
        "vpfx": rnd(P, Hkv, D),                  # shared prompt V
        "ksfx_raw": rnd(B, S, Hkv, D),           # per-beam suffix K
        "vsfx": rnd(B, S, Hkv, D),               # per-beam suffix V
    }


def full_attention(cfg: Cfg, sc: dict, q_pos: int) -> torch.Tensor:
    """Reference: each beam attends to its full key set [prefix ++ own suffix]."""
    B, P, S, H = cfg.beam_width, cfg.prefix_len, cfg.suffix_len, cfg.num_heads
    prefix_pos = torch.arange(P, device=DEVICE)
    suffix_pos = torch.arange(P, P + S, device=DEVICE)
    qpos = torch.tensor([q_pos], device=DEVICE)

    q = apply_rope(sc["q_raw"].unsqueeze(0), qpos, cfg.rope_theta).squeeze(0)  # [B,H,D]
    kp = _expand_kv(apply_rope(sc["kpfx_raw"], prefix_pos, cfg.rope_theta), H)  # [P,H,D]
    vp = _expand_kv(sc["vpfx"], H)                                              # [P,H,D]
    kp = kp.unsqueeze(0).expand(B, P, H, cfg.head_dim).contiguous()            # [B,P,H,D]
    vp = vp.unsqueeze(0).expand(B, P, H, cfg.head_dim).contiguous()

    if S > 0:
        ks = _expand_kv(apply_rope(sc["ksfx_raw"], suffix_pos, cfg.rope_theta), H)  # [B,S,H,D]
        vs = _expand_kv(sc["vsfx"], H)
        k_all = torch.cat([kp, ks], dim=1)
        v_all = torch.cat([vp, vs], dim=1)
    else:
        k_all, v_all = kp, vp
    o, _ = attn_with_lse(q, k_all, v_all, cfg.sm_scale)
    return o


def cascade_attention(cfg: Cfg, sc: dict, q_pos: int, *,
                      suffix_pos_override: torch.Tensor | None = None) -> torch.Tensor:
    """Cascade: shared-prefix partial + per-beam suffix partial, merged via LSE."""
    B, P, S, H = cfg.beam_width, cfg.prefix_len, cfg.suffix_len, cfg.num_heads
    prefix_pos = torch.arange(P, device=DEVICE)
    suffix_pos = (suffix_pos_override if suffix_pos_override is not None
                  else torch.arange(P, P + S, device=DEVICE))
    qpos = torch.tensor([q_pos], device=DEVICE)

    q = apply_rope(sc["q_raw"].unsqueeze(0), qpos, cfg.rope_theta).squeeze(0)   # [B,H,D]

    # ---- Level 0: shared prefix (read ONCE, broadcast across beams) ----
    kp = _expand_kv(apply_rope(sc["kpfx_raw"], prefix_pos, cfg.rope_theta), H)
    vp = _expand_kv(sc["vpfx"], H)
    kp = kp.unsqueeze(0).expand(B, P, H, cfg.head_dim).contiguous()
    vp = vp.unsqueeze(0).expand(B, P, H, cfg.head_dim).contiguous()
    o_pfx, lse_pfx = attn_with_lse(q, kp, vp, cfg.sm_scale)

    # ---- Level 1: per-beam suffix (the only beam-divergent work) ----
    if S > 0:
        ks = _expand_kv(apply_rope(sc["ksfx_raw"], suffix_pos, cfg.rope_theta), H)
        vs = _expand_kv(sc["vsfx"], H)
        o_sfx, lse_sfx = attn_with_lse(q, ks, vs, cfg.sm_scale)
    else:
        o_sfx = torch.zeros_like(o_pfx)
        lse_sfx = torch.full_like(lse_pfx, NEG_INF)

    o, _ = merge_state_ref(o_pfx, lse_pfx, o_sfx, lse_sfx)
    return o


def _tol(dtype: torch.dtype) -> dict:
    return {"atol": 3e-4, "rtol": 3e-4} if dtype == torch.float32 else {"atol": 2e-2, "rtol": 2e-2}


# --------------------------------------------------------------------------- #
# Tests                                                                       #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("beam_width", [1, 4, 15, 30])
@pytest.mark.parametrize("suffix_len", [1, 4])
def test_cascade_matches_full_attention(beam_width, suffix_len):
    """Core parity: cascade == full attention, every beam, across beam widths."""
    cfg = Cfg(beam_width=beam_width, suffix_len=suffix_len)
    sc = make_scenario(cfg)
    q_pos = cfg.prefix_len + cfg.suffix_len
    ref = full_attention(cfg, sc, q_pos)
    got = cascade_attention(cfg, sc, q_pos)
    torch.testing.assert_close(got, ref, **_tol(cfg.dtype))


def test_empty_suffix_first_step():
    """First decode step: suffix length 0 must merge cleanly to the prefix result."""
    cfg = Cfg(beam_width=8, suffix_len=0)
    sc = make_scenario(cfg)
    q_pos = cfg.prefix_len
    ref = full_attention(cfg, sc, q_pos)
    got = cascade_attention(cfg, sc, q_pos)
    assert torch.isfinite(got).all(), "empty-suffix merge produced NaN/Inf"
    torch.testing.assert_close(got, ref, **_tol(cfg.dtype))


def test_rope_absolute_positions_required():
    """Pitfall §6.4#1: restarting suffix RoPE at 0 (instead of absolute) must break parity.

    This guards against the silent-corruption bug — the harness *must* detect it.
    """
    cfg = Cfg(beam_width=8, suffix_len=4)
    sc = make_scenario(cfg)
    q_pos = cfg.prefix_len + cfg.suffix_len
    ref = full_attention(cfg, sc, q_pos)
    wrong = torch.arange(cfg.suffix_len, device=DEVICE)  # WRONG: restarted at 0
    got_wrong = cascade_attention(cfg, sc, q_pos, suffix_pos_override=wrong)
    assert not torch.allclose(got_wrong, ref, atol=1e-2), \
        "wrong suffix RoPE positions should NOT match — harness failed to catch the pitfall"


def test_beam_reorder_remap():
    """Beam reorder: gather each beam's suffix from its predecessor, parity must hold.

    Mirrors the predecessor selection that updates cache_indirection each step (§6.1):
    after top-k, beam b inherits beam `pred[b]`'s suffix KV.
    """
    cfg = Cfg(beam_width=8, suffix_len=4)
    sc = make_scenario(cfg)
    g = torch.Generator(device=DEVICE).manual_seed(123)
    pred = torch.randint(0, cfg.beam_width, (cfg.beam_width,), generator=g, device=DEVICE)

    remapped = dict(sc)
    remapped["ksfx_raw"] = sc["ksfx_raw"][pred].contiguous()  # suffix follows predecessor
    remapped["vsfx"] = sc["vsfx"][pred].contiguous()
    # queries are freshly produced per beam, prefix is shared — neither is remapped.

    q_pos = cfg.prefix_len + cfg.suffix_len
    ref = full_attention(cfg, remapped, q_pos)
    got = cascade_attention(cfg, remapped, q_pos)
    torch.testing.assert_close(got, ref, **_tol(cfg.dtype))


@pytest.mark.skipif(not (_HAS_FLASHINFER and torch.cuda.is_available()),
                    reason="requires FlashInfer + CUDA")
def test_merge_state_matches_flashinfer():
    """Cross-check the reference merge against FlashInfer's `cascade.merge_state` — the
    exact op the engine backend will call.

    IMPORTANT finding (design §6.4#5): FlashInfer's ``s`` is the LSE in **base-2**
    (log2-sum-exp2), matching FlashAttention's internal convention — NOT natural log.
    So the engine must keep FlashInfer attention LSEs in their native base-2 domain when
    merging (automatic when both partials come from FlashInfer wrappers). Mixing a
    natural-log LSE into ``merge_state`` requires the ``1/ln2`` conversion done here.
    """
    torch.manual_seed(7)
    n, h, d = 30, 16, 128  # seq(=beams) x heads x head_dim
    o_a = torch.randn(n, h, d, device="cuda", dtype=torch.float16)
    o_b = torch.randn(n, h, d, device="cuda", dtype=torch.float16)
    lse_a = torch.randn(n, h, device="cuda", dtype=torch.float32)  # natural-log domain
    lse_b = torch.randn(n, h, device="cuda", dtype=torch.float32)

    # natural-log LSE -> base-2 for FlashInfer; convert its base-2 result back to compare.
    fi_o, fi_s = fi_merge_state(o_a, lse_a / LN2, o_b, lse_b / LN2)
    ref_o, ref_lse = merge_state_ref(o_a, lse_a, o_b, lse_b)

    torch.testing.assert_close(fi_o.float(), ref_o.float(), atol=4e-3, rtol=4e-3)
    torch.testing.assert_close(fi_s.float() * LN2, ref_lse.float(), atol=4e-3, rtol=4e-3)


# --------------------------------------------------------------------------- #
# Standalone runner                                                           #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    results: list[tuple[str, str]] = []

    def run(name, fn, *args):
        try:
            fn(*args)
            results.append((name, "PASS"))
        except Exception as e:  # noqa: BLE001
            results.append((name, f"FAIL: {type(e).__name__}: {str(e).splitlines()[0]}"))

    print(f"device={DEVICE}  flashinfer={'yes' if _HAS_FLASHINFER else 'no'}\n")
    for bw in (1, 4, 15, 30):
        for sl in (1, 4):
            run(f"parity[bw={bw},suffix={sl}]", test_cascade_matches_full_attention, bw, sl)
    run("empty_suffix_first_step", test_empty_suffix_first_step)
    run("rope_absolute_positions_required", test_rope_absolute_positions_required)
    run("beam_reorder_remap", test_beam_reorder_remap)
    if _HAS_FLASHINFER and torch.cuda.is_available():
        run("merge_state_matches_flashinfer", test_merge_state_matches_flashinfer)
    else:
        results.append(("merge_state_matches_flashinfer", "SKIP (no flashinfer/cuda)"))

    width = max(len(n) for n, _ in results)
    for name, status in results:
        print(f"  {name.ljust(width)}  {status}")
    failed = [n for n, s in results if s.startswith("FAIL")]
    print(f"\n{'FAILED: ' + ', '.join(failed) if failed else 'ALL PASS'}")
    raise SystemExit(1 if failed else 0)
