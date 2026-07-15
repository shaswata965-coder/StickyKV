"""Tiled GEMV decode-attention tests (design.md §11, Phase 2).

The contract the kernel must uphold:

1. **Equivalence.** ``tiled_gemv_attention`` equals a full fp32 softmax over the
   materialize-then-interleave effective K/V (the Phase-1 path) — for the pure-fp
   case, the two-tier case, and under GQA. This is the correctness anchor.
2. **Never materialize the full fp Q tier.** The kernel consumes the Q store one
   window at a time: it must call ``iter_active_windows`` (not ``gather_active``),
   and the largest fp key tile derived from the Q tier is exactly one window.
3. **Order invariance & robustness.** Empty Q tier, dtype (fp16/fp32), and tile
   ordering all behave.

CPU only, B = 1.
"""

from __future__ import annotations

import pytest
import torch

from modules.quant.gemv import tiled_gemv_attention
from modules.quant.store import QuantizedStore
from modules.quant.effective import materialize_effective_kv, rotate_key_window


# ---------------------------------------------------------------------------
# Fixtures: a real RoPE and a two-tier scenario builder
# ---------------------------------------------------------------------------


class _RealRoPE(torch.nn.Module):
    """Minimal RoPE matching the model's ``apply_rotary_pos_emb`` contract."""

    def __init__(self, head_dim: int, base: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x, position_ids):
        B = position_ids.shape[0]
        inv = self.inv_freq[None, :, None].float().expand(B, -1, 1)
        pos = position_ids[:, None, :].float()
        freqs = (inv @ pos).transpose(1, 2)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)


def _reference_attention(query, eff_k, eff_v, scaling):
    """Full fp32 softmax attention (the anchor the tiled kernel must match).

    query : [H_q, 1, D]; eff_k/eff_v : [H_kv, S, D]. Handles GQA by expanding
    kv heads to query heads (the plain, un-tiled way).
    """
    h_q = query.shape[0]
    h_kv, S, D = eff_k.shape
    n_rep = h_q // h_kv
    q = query.reshape(h_kv, n_rep, D).to(torch.float32)          # [H_kv, rep, D]
    k = eff_k.to(torch.float32)                                  # [H_kv, S, D]
    v = eff_v.to(torch.float32)                                  # [H_kv, S, D]
    logits = torch.einsum("hrd,hsd->hrs", q, k) * scaling        # [H_kv, rep, S]
    w = torch.softmax(logits, dim=-1)
    out = torch.einsum("hrs,hsd->hrd", w, v)                     # [H_kv, rep, D]
    return out.reshape(h_q, 1, D)


def _build_two_tier(
    n_fp_windows=3, n_q_windows=4, ws=2, num_sink=2, H_kv=2, n_rep=1, D=4,
    dtype=torch.float32, seed=0,
):
    """Construct an fp store + a populated QuantizedStore with disjoint window ids.

    Layout: sink (num_sink fp tokens) ‖ fp body windows ‖ (Q windows live only in
    the store). Every window carries its own original absolute positions; window
    ids are contiguous and the fp/Q split is arbitrary but disjoint.
    Returns everything the kernel and the reference both need.
    """
    torch.manual_seed(seed)
    rope = _RealRoPE(D)
    H_q = H_kv * n_rep

    # --- fp store: sink + fp-body windows, post-RoPE, at contiguous positions ---
    fp_ids = list(range(n_fp_windows))                    # fp windows 0..n_fp-1
    T_fp = num_sink + n_fp_windows * ws
    fp_pos = torch.arange(T_fp, dtype=torch.long)
    fp_k_pre = torch.randn(H_kv, T_fp, D, dtype=dtype)
    fp_v = torch.randn(H_kv, T_fp, D, dtype=dtype)
    fp_k = rotate_key_window(fp_k_pre, fp_pos, rope).to(dtype)

    # --- Q store: n_q windows with ids AFTER the fp windows -----------------
    store = QuantizedStore(window_size=ws, head_dim=D, num_kv_heads=H_kv)
    base = num_sink + T_fp  # start Q positions well past the fp store
    for j in range(n_q_windows):
        wid = n_fp_windows + j
        pos = torch.arange(base + j * ws, base + (j + 1) * ws, dtype=torch.long)
        k_pre = torch.randn(H_kv, ws, D, dtype=dtype)
        v = torch.randn(H_kv, ws, D, dtype=dtype)
        store.demote(wid, k_pre, v, pos)

    query = torch.randn(H_q, 1, D, dtype=dtype)
    scaling = D ** -0.5
    return dict(
        rope=rope, query=query, scaling=scaling, store=store,
        fp_k=fp_k, fp_v=fp_v, fp_pos=fp_pos, num_sink=num_sink, ws=ws, dtype=dtype,
    )


def _materialized_reference(s):
    """Reference output via the Phase-1 materialize path + full fp32 softmax."""
    eff_k, eff_v = materialize_effective_kv(
        s["fp_k"], s["fp_v"], s["fp_pos"], s["store"],
        num_sink=s["num_sink"], window_size=s["ws"],
        rope_module=s["rope"], out_dtype=s["dtype"],
    )
    return _reference_attention(s["query"], eff_k, eff_v, s["scaling"])


# ---------------------------------------------------------------------------
# 1. Equivalence to the materialize path
# ---------------------------------------------------------------------------


def test_matches_materialize_two_tier_fp32():
    s = _build_two_tier(dtype=torch.float32)
    ref = _materialized_reference(s)
    out = tiled_gemv_attention(
        s["query"], s["fp_k"], s["fp_v"], s["store"], s["rope"], s["scaling"]
    )
    assert out.shape == (s["query"].shape[0], 1, s["query"].shape[-1])
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5), (out - ref).abs().max()


def test_matches_materialize_pure_fp_empty_q_tier():
    """Empty Q tier ⇒ plain fp attention over the fp store, exactly the anchor."""
    s = _build_two_tier(n_q_windows=0, dtype=torch.float32)
    assert s["store"].num_active_windows == 0
    ref = _reference_attention(s["query"], s["fp_k"], s["fp_v"], s["scaling"])
    out = tiled_gemv_attention(
        s["query"], s["fp_k"], s["fp_v"], s["store"], s["rope"], s["scaling"]
    )
    assert torch.allclose(out, ref, atol=1e-6, rtol=1e-6)
    # store=None must behave identically to an empty store.
    out_none = tiled_gemv_attention(
        s["query"], s["fp_k"], s["fp_v"], None, s["rope"], s["scaling"]
    )
    assert torch.equal(out, out_none)


@pytest.mark.parametrize("n_rep", [1, 2, 4])
def test_matches_materialize_gqa(n_rep):
    s = _build_two_tier(H_kv=2, n_rep=n_rep, dtype=torch.float32)
    ref = _materialized_reference(s)
    out = tiled_gemv_attention(
        s["query"], s["fp_k"], s["fp_v"], s["store"], s["rope"], s["scaling"]
    )
    assert out.shape[0] == 2 * n_rep
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5), (out - ref).abs().max()


def test_matches_materialize_fp16():
    """fp16 KV: looser tol (fp16 dequant/RoPE rounding), still tight vs anchor."""
    s = _build_two_tier(dtype=torch.float16)
    ref = _materialized_reference(s)  # anchor also built from the fp16 eff K/V
    out = tiled_gemv_attention(
        s["query"], s["fp_k"], s["fp_v"], s["store"], s["rope"], s["scaling"]
    )
    assert out.dtype == torch.float16
    assert torch.allclose(out.float(), ref.float(), atol=2e-3, rtol=2e-3), \
        (out.float() - ref.float()).abs().max()


@pytest.mark.parametrize("seed", range(8))
def test_fuzz_matches_materialize(seed):
    s = _build_two_tier(
        n_fp_windows=1 + seed % 4, n_q_windows=1 + (seed * 3) % 5,
        ws=2 + 2 * (seed % 3), num_sink=seed % 3, H_kv=1 + seed % 2,
        n_rep=1 + seed % 3, D=4, dtype=torch.float32, seed=seed,
    )
    ref = _materialized_reference(s)
    out = tiled_gemv_attention(
        s["query"], s["fp_k"], s["fp_v"], s["store"], s["rope"], s["scaling"]
    )
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5), (out - ref).abs().max()


# ---------------------------------------------------------------------------
# 2. Never materialize the full fp Q tier — the core design invariant
# ---------------------------------------------------------------------------


def test_consumes_q_tier_one_window_at_a_time():
    """The kernel must stream the Q tier via iter_active_windows (one window each),
    never gather_active (which stacks the whole tier)."""
    s = _build_two_tier(n_q_windows=5, ws=4, dtype=torch.float32)
    store = s["store"]

    gather_calls = {"n": 0}
    orig_gather = store.gather_active
    def spy_gather(*a, **k):
        gather_calls["n"] += 1
        return orig_gather(*a, **k)
    store.gather_active = spy_gather  # type: ignore[assignment]

    max_tile_tokens = {"n": 0}
    orig_iter = store.iter_active_windows
    def spy_iter(*a, **k):
        for wid, kp, vw, pos in orig_iter(*a, **k):
            # key tile is exactly one window — never the whole tier.
            max_tile_tokens["n"] = max(max_tile_tokens["n"], kp.shape[1])
            yield wid, kp, vw, pos
    store.iter_active_windows = spy_iter  # type: ignore[assignment]

    tiled_gemv_attention(
        s["query"], s["fp_k"], s["fp_v"], store, s["rope"], s["scaling"]
    )

    assert gather_calls["n"] == 0, "kernel must not call gather_active (whole-tier blow-up)"
    assert max_tile_tokens["n"] == s["ws"], (
        f"largest Q key tile must be one window ({s['ws']} tokens), "
        f"got {max_tile_tokens['n']}"
    )


def test_iter_active_windows_yields_one_window_each():
    """Store-level: the lazy iterator yields ws-token tensors, chronological."""
    s = _build_two_tier(n_fp_windows=2, n_q_windows=3, ws=6, dtype=torch.float32)
    store = s["store"]
    seen = list(store.iter_active_windows(out_dtype=torch.float32))
    assert [wid for wid, *_ in seen] == store.active_ids()  # chronological
    for _wid, k, v, pos in seen:
        assert k.shape == (2, s["ws"], 4)   # [H_kv, ws, D] — one window
        assert v.shape == (2, s["ws"], 4)
        assert pos.numel() == s["ws"]


# ---------------------------------------------------------------------------
# 3. Order invariance & finalization robustness
# ---------------------------------------------------------------------------


def test_online_softmax_order_invariance():
    """Combining tiles in any order gives the same output (permutation-invariant
    over keys). Feed the fp store as one tile vs many chunks — identical result."""
    s = _build_two_tier(n_fp_windows=5, n_q_windows=3, ws=2, dtype=torch.float32)
    whole = tiled_gemv_attention(
        s["query"], s["fp_k"], s["fp_v"], s["store"], s["rope"], s["scaling"],
        fp_tile_size=None,
    )
    chunked = tiled_gemv_attention(
        s["query"], s["fp_k"], s["fp_v"], s["store"], s["rope"], s["scaling"],
        fp_tile_size=1,  # one fp token per tile — maximal fragmentation
    )
    assert torch.allclose(whole, chunked, atol=1e-6, rtol=1e-6)


def test_batch_axis_shape_roundtrip():
    """A [1,H_q,1,D] query returns [1,H_q,1,D]; [H_q,1,D] returns [H_q,1,D]."""
    s = _build_two_tier(dtype=torch.float32)
    q = s["query"]
    out3 = tiled_gemv_attention(q, s["fp_k"], s["fp_v"], s["store"], s["rope"], s["scaling"])
    out4 = tiled_gemv_attention(
        q.unsqueeze(0), s["fp_k"], s["fp_v"], s["store"], s["rope"], s["scaling"]
    )
    assert out3.shape == (q.shape[0], 1, q.shape[-1])
    assert out4.shape == (1, q.shape[0], 1, q.shape[-1])
    assert torch.equal(out4[0], out3)


def test_rejects_multi_query_and_multi_batch():
    s = _build_two_tier(dtype=torch.float32)
    H_q, D = s["query"].shape[0], s["query"].shape[-1]
    with pytest.raises(NotImplementedError):  # prefill (T_q > 1) not supported
        tiled_gemv_attention(
            torch.randn(H_q, 3, D), s["fp_k"], s["fp_v"], s["store"], s["rope"], s["scaling"]
        )
    with pytest.raises(NotImplementedError):  # B > 1 not supported
        tiled_gemv_attention(
            torch.randn(2, H_q, 1, D), s["fp_k"], s["fp_v"], s["store"], s["rope"], s["scaling"]
        )


# ---------------------------------------------------------------------------
# 4. Cache integration: WindowedCache.decode_attention (both backends)
# ---------------------------------------------------------------------------


class _FakeModelConfig:
    num_attention_heads = 2
    num_key_value_heads = 2
    hidden_size = 8
    head_dim = 4
    num_hidden_layers = 1


def _cache_with_populated_q_tier(backend):
    """Build a WindowedCache, seed 4 windows, evict once so w1 lands in the Q tier.

    Mirrors test_quant_cache.test_first_eviction_demotes_and_materializes: after
    the eviction the fp store holds w0 + w3 and the Q store holds w1.
    """
    if backend == "flash":
        from modules.windowed_cache.cache import WindowedCache
        from modules.windowed_cache.config import WindowedCacheConfig
    else:
        from modules.windowed_eager_cache.cache import WindowedCache
        from modules.windowed_eager_cache.config import WindowedCacheConfig

    ws, num_sink, H, D = 2, 0, 2, 4
    cfg = WindowedCacheConfig(
        window_size=ws, num_sink_tokens=num_sink, local_window_size=ws,
        cache_budget=0.5, quant_ratio=0.5,
    )
    cache = WindowedCache(
        config=cfg, prefill_len=8, model_config=_FakeModelConfig(),
        kv_dtype=torch.float32, rope_module=_RealRoPE(4),
        num_layers=1, max_tokens=8,
    )
    torch.manual_seed(0)
    rope = cache.rope_module
    T = 4 * ws
    pos = torch.arange(T, dtype=torch.long)
    k_pre = torch.randn(H, T, D)
    v = torch.randn(H, T, D)
    st = cache._states[0]
    st.key_states = rotate_key_window(k_pre, pos, rope).unsqueeze(0).clone()
    st.value_states = v.unsqueeze(0).clone()
    st.position_ids = pos.unsqueeze(0).clone()
    st.window_scores = torch.zeros(1, H, 4)
    st.window_scores[0, :, 0] = 100.0
    st.window_scores[0, :, 1] = 50.0
    st.window_scores[0, :, 2] = 10.0
    st.original_window_ids = torch.tensor([[0, 1, 2, 3]])
    pol = cache._policies[0]
    pol.top_k_fp, pol.N_q, pol.local_windows = 1, 1, 1
    cache._evict_two_tier(0, step=2)
    assert cache._stores[0].active_ids() == [1]  # Q tier populated
    return cache


@pytest.mark.parametrize("backend", ["flash", "eager"])
def test_cache_decode_attention_matches_materialize(backend):
    cache = _cache_with_populated_q_tier(backend)
    H_q, D = 2, 4
    scaling = D ** -0.5
    query = torch.randn(1, H_q, 1, D)

    # Reference: the Phase-1 materialize path + full fp32 softmax.
    eff_k, eff_v = cache._materialize(0)
    ref = _reference_attention(query[0], eff_k[0], eff_v[0], scaling).unsqueeze(0)

    out = cache.decode_attention(0, query, scaling=scaling)
    assert out.shape == (1, H_q, 1, D)
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5), (out - ref).abs().max()


def test_cache_decode_attention_default_scaling():
    """No scaling arg ⇒ D**-0.5, same as the reference."""
    cache = _cache_with_populated_q_tier("flash")
    H_q, D = 2, 4
    query = torch.randn(1, H_q, 1, D)
    eff_k, eff_v = cache._materialize(0)
    ref = _reference_attention(query[0], eff_k[0], eff_v[0], D ** -0.5).unsqueeze(0)
    out = cache.decode_attention(0, query)  # default scaling
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)
