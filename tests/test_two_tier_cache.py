"""Two-tier (fp16 + int4) windowed-cache tests — WS-2 through WS-5.

All CPU, mocked rope modules and hand-built tensors, mirroring the style of
``tests/test_windowed_cache.py``. The q=0 regression parity lives here too:
with ``quant_ratio = 0`` the resolver and the full update()/eviction cycle
must be byte-identical to the legacy single-tier path.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from modules.quant import build_interleaved_position_map
from modules.windowed_cache import cache as cache_mod
from modules.windowed_cache.cache import WindowedCache
from modules.windowed_cache.config import ResolvedConfig, WindowedCacheConfig
from modules.windowed_cache.policy import EvictionPolicy
from modules.windowed_cache.state import CacheState


@dataclass
class _FakeModelConfig:
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    hidden_size: int = 4096
    head_dim: int = 128
    num_hidden_layers: int = 32


# ---------------------------------------------------------------------------
# WS-2 — tier-aware budget resolver (design §7)
# ---------------------------------------------------------------------------


class TestTierResolver:

    def test_q0_matches_legacy_resolver_field_for_field(self):
        """quant_ratio=0 reproduces the single-tier resolve() exactly."""
        model_cfg = _FakeModelConfig()
        base = dict(
            window_size=8, num_sink_tokens=4, local_window_size=0.25,
            cache_budget=0.40,
        )
        legacy = WindowedCacheConfig(**base).resolve(
            100, model_cfg, torch.float16, max_tokens=100
        )
        tiered = WindowedCacheConfig(**base, quant_ratio=0.0).resolve(
            100, model_cfg, torch.float16, max_tokens=100
        )
        assert legacy == tiered  # frozen dataclass — field-for-field equality
        assert tiered.quant_ratio == 0.0
        assert tiered.top_q_windows == 0
        assert tiered.quant_window_bytes == 0

    def test_worked_example_beta_025_q_05(self):
        """design §7: β=0.25, q=0.5 → N_q ≈ 4·N_fp (b_q ≈ b_fp/4 + overhead)."""
        model_cfg = _FakeModelConfig()  # H_kv=8, D=128
        S = 32
        cfg = WindowedCacheConfig(
            window_size=S, num_sink_tokens=4, local_window_size=S,
            cache_budget=0.25, quant_ratio=0.5,
        )
        r = cfg.resolve(16000, model_cfg, torch.float16, max_tokens=0)

        b_fp = r.bytes_per_token * S                       # 4·H_kv·D·S (fp16 K+V)
        b_q = 8 * 128 * S + 4 * 8 * 128 + 4 * 8 * S
        assert r.quant_window_bytes == b_q

        fp_budget_bytes = int(0.5 * r.total_budget_bytes)
        n_fp = (fp_budget_bytes // r.bytes_per_token) // S  # total fp-window capacity
        assert r.top_q_windows == (r.total_budget_bytes - fp_budget_bytes) // b_q
        # ~4× the fp window count for equal memory, minus the scale/zero
        # overhead (4·H_kv·(D+S) bytes/window ≈ 13% of the code bytes here,
        # so ≈ 3.5× at S=32, D=128 — approaching 4× as S grows, §7/§10).
        ratio = r.top_q_windows / n_fp
        assert 3.3 < ratio <= 4.0
        # And the memory identity: Q tier bytes fit inside M_q.
        assert r.top_q_windows * b_q <= r.total_budget_bytes - fp_budget_bytes
        assert b_q < b_fp / 3.3

    def test_sink_local_stay_inside_fp_share(self):
        """top_k (fp evictable) is carved out of the FP share only."""
        model_cfg = _FakeModelConfig()
        cfg = WindowedCacheConfig(
            window_size=8, num_sink_tokens=4, local_window_size=16,
            cache_budget=0.40, quant_ratio=0.5,
        )
        r = cfg.resolve(100, model_cfg, torch.float16, max_tokens=100)
        fp_budget_tokens = int(0.5 * r.total_budget_bytes) // r.bytes_per_token
        assert r.top_k_windows == (fp_budget_tokens - 4 - 16) // 8
        retained_fp = 4 + 16 + r.top_k_windows * 8
        assert retained_fp * r.bytes_per_token <= int(0.5 * r.total_budget_bytes) \
            + r.bytes_per_token * 8  # floor slack < one window

    def test_quant_ratio_validation(self):
        base = dict(
            window_size=8, num_sink_tokens=4, local_window_size=16,
            cache_budget=0.40,
        )
        with pytest.raises(ValueError, match="bool"):
            WindowedCacheConfig(**base, quant_ratio=True)
        with pytest.raises(ValueError, match=r"\[0, 1\)"):
            WindowedCacheConfig(**base, quant_ratio=1.0)
        with pytest.raises(ValueError, match=r"\[0, 1\)"):
            WindowedCacheConfig(**base, quant_ratio=-0.1)
        # int 0 is accepted and normalised to float (score_p pattern).
        cfg = WindowedCacheConfig(**base, quant_ratio=0)
        assert cfg.quant_ratio == 0.0 and isinstance(cfg.quant_ratio, float)

    def test_odd_window_size_rejected_when_quantizing(self):
        with pytest.raises(ValueError, match="even"):
            WindowedCacheConfig(
                window_size=7, num_sink_tokens=4, local_window_size=14,
                cache_budget=0.40, quant_ratio=0.5,
            )
        # ...but odd window_size stays legal with the tier off.
        WindowedCacheConfig(
            window_size=7, num_sink_tokens=4, local_window_size=14,
            cache_budget=0.40, quant_ratio=0.0,
        )

    def test_fp_share_too_small_raises(self):
        model_cfg = _FakeModelConfig()
        cfg = WindowedCacheConfig(
            window_size=8, num_sink_tokens=10, local_window_size=40,
            cache_budget=0.10, quant_ratio=0.9,
        )
        with pytest.raises(ValueError, match="fp share"):
            cfg.resolve(100, model_cfg, torch.float16, max_tokens=50)


# ---------------------------------------------------------------------------
# WS-3 — interleaved position map (design §5 step 4)
# ---------------------------------------------------------------------------


class TestInterleavedPositionMap:

    def test_fp_gaps_land_exactly_over_q_slots(self):
        """fp W1, W5 survive with Q W2 between them: fp positions skip the
        slots W2 occupies, and W2's ledger range fills the gap exactly."""
        S, sink = 4, 2
        fp_ids = torch.tensor([[1, 5]])
        q_ids = torch.tensor([[2]])
        fp_pos, q_starts = build_interleaved_position_map(fp_ids, q_ids, S, sink)
        # Chronological layout: sink [0,1] | W1 [2..5] | W2 [6..9] | W5 [10..13]
        assert fp_pos.tolist() == [[0, 1, 2, 3, 4, 5, 10, 11, 12, 13]]
        assert q_starts.tolist() == [[6]]
        # Union of fp positions and Q ranges is one contiguous arange(T_total).
        q_range = torch.arange(6, 10)
        union = torch.cat([fp_pos[0], q_range]).sort().values
        assert torch.equal(union, torch.arange(sink + 3 * S))

    def test_partial_trailing_fp_window(self):
        """The newest (local) fp window may be partial; Q windows before it
        still get full-size slots."""
        S, sink = 4, 1
        fp_ids = torch.tensor([[0, 3]])
        q_ids = torch.tensor([[1, 2]])
        fp_pos, q_starts = build_interleaved_position_map(
            fp_ids, q_ids, S, sink, fp_tail_len=2
        )
        # sink [0] | W0 [1..4] | W1 [5..8] | W2 [9..12] | W3 [13..14] (partial)
        assert fp_pos.tolist() == [[0, 1, 2, 3, 4, 13, 14]]
        assert q_starts.tolist() == [[5, 9]]

    def test_per_row_divergent_eviction(self):
        """Rows retaining different windows get independent maps."""
        S, sink = 2, 0
        fp_ids = torch.tensor([[0, 4], [2, 4]])
        q_ids = torch.tensor([[1], [0]])
        fp_pos, q_starts = build_interleaved_position_map(fp_ids, q_ids, S, sink)
        # Row 0: W0 [0..1] | W1(Q) [2..3] | W4 [4..5]
        assert fp_pos[0].tolist() == [0, 1, 4, 5]
        assert q_starts[0].tolist() == [2]
        # Row 1: W0(Q) [0..1] | W2 [2..3] | W4 [4..5]
        assert fp_pos[1].tolist() == [2, 3, 4, 5]
        assert q_starts[1].tolist() == [0]

    def test_no_q_windows_reduces_to_contiguous(self):
        S, sink = 4, 2
        fp_ids = torch.tensor([[0, 1, 2]])
        q_ids = torch.zeros(1, 0, dtype=torch.long)
        fp_pos, q_starts = build_interleaved_position_map(fp_ids, q_ids, S, sink)
        assert torch.equal(fp_pos, torch.arange(sink + 3 * S).unsqueeze(0))
        assert q_starts.shape == (1, 0)

    def test_partial_window_must_be_fp_and_newest(self):
        S, sink = 4, 0
        fp_ids = torch.tensor([[0]])
        q_ids = torch.tensor([[1]])  # Q window is newest → partial fp illegal
        with pytest.raises(AssertionError, match="newest"):
            build_interleaved_position_map(fp_ids, q_ids, S, sink, fp_tail_len=2)


# ---------------------------------------------------------------------------
# WS-3 — rerotate_keys(new_position_ids) (design §5 step 5)
# ---------------------------------------------------------------------------


class _NoOpRoPE(torch.nn.Module):
    """cos=1 / sin=0 — identity rotation that still exercises the machinery."""

    def forward(self, x, position_ids):
        seq_len = position_ids.shape[-1]
        D = x.shape[-1]
        cos = torch.ones(1, seq_len, D, dtype=x.dtype, device=x.device)
        sin = torch.zeros(1, seq_len, D, dtype=x.dtype, device=x.device)
        return cos, sin


class TestRerotateKeysNewPositions:

    def _state(self, B=1, H=2, T=6, D=8):
        state = CacheState()
        state.key_states = torch.randn(B, H, T, D)
        state.value_states = torch.randn(B, H, T, D)
        state.position_ids = torch.arange(T).unsqueeze(0).expand(B, -1)
        return state

    def test_position_ids_set_to_gappy_targets(self):
        """Bookkeeping must hold the interleaved (gappy) fp positions, not
        arange(T_retained) — or the next eviction snapshots wrong angles."""
        state = self._state()
        old = state.position_ids.clone()
        gappy = torch.tensor([[0, 1, 2, 3, 8, 9]])  # gap where a Q window sits
        state.rerotate_keys(_NoOpRoPE(), old, new_position_ids=gappy)
        assert torch.equal(state.position_ids, gappy)

    def test_default_none_is_legacy_contiguous(self):
        state = self._state()
        old = torch.tensor([[0, 2, 4, 6, 8, 10]])
        state.rerotate_keys(_NoOpRoPE(), old)
        assert torch.equal(state.position_ids, torch.arange(6).unsqueeze(0))

    def test_explicit_positions_change_rotation(self):
        """A real rotation lands keys at the requested angles: rotating to the
        gappy targets equals rotating a reference to those same positions."""
        try:
            from transformers.models.llama.modeling_llama import (  # noqa: F401
                apply_rotary_pos_emb,
            )
        except ImportError:
            pytest.skip("transformers unavailable")

        class _RealRoPE(torch.nn.Module):
            def forward(self, x, position_ids):
                D = x.shape[-1]
                inv = 1.0 / (100.0 ** (torch.arange(0, D, 2).float() / D))
                ang = position_ids[..., :, None].float() * inv[None, None, :]
                emb = torch.cat([ang, ang], dim=-1)
                return emb.cos(), emb.sin()

        torch.manual_seed(0)
        B, H, T, D = 1, 2, 4, 8
        k0 = torch.randn(B, H, T, D)
        old = torch.tensor([[3, 7, 11, 15]])
        gappy = torch.tensor([[0, 1, 6, 7]])

        state = CacheState()
        state.key_states = k0.clone()
        state.value_states = torch.randn(B, H, T, D)
        state.position_ids = old.clone()
        state.rerotate_keys(_RealRoPE(), old, new_position_ids=gappy)

        # Reference: strip at old, apply at gappy, via the same HF helper.
        rope = _RealRoPE()
        cos_o, sin_o = rope(k0, old)
        _, k_un = apply_rotary_pos_emb(k0, k0, cos_o, -sin_o)
        cos_n, sin_n = rope(k0, gappy)
        _, k_ref = apply_rotary_pos_emb(k_un, k_un, cos_n, sin_n)
        assert torch.allclose(state.key_states, k_ref, atol=1e-6)


# ---------------------------------------------------------------------------
# WS-3 — tier-aware policy (assignments + fp-partition expansion)
# ---------------------------------------------------------------------------


def _resolved(**overrides):
    defaults = dict(
        window_size=4, num_sink_tokens=2, local_tokens=4, top_k_windows=2,
        bytes_per_token=4096, total_budget_bytes=163840,
        total_budget_tokens=40, quant_ratio=0.5, top_q_windows=2,
        quant_window_bytes=1,
    )
    defaults.update(overrides)
    return ResolvedConfig(**defaults)


class TestTierAssignments:

    def test_split_top_band_then_q_band(self):
        policy = EvictionPolicy(_resolved())
        policy.initialize_after_prefill(2 + 6 * 4)  # 6 windows, 1 local
        scores = torch.tensor([[[10.0, 40.0, 30.0, 20.0, 5.0, 0.0]]])
        fp_idx, q_idx = policy.compute_tier_assignments(scores)
        # evictable = [0..4]; ranking: 1(40) 2(30) 3(20) 0(10) 4(5)
        # fp band = top-2 {1,2} + local {5}; Q band = next-2 {3,0}.
        assert fp_idx.tolist() == [[1, 2, 5]]
        assert q_idx.tolist() == [[0, 3]]  # chronologically sorted

    def test_q_band_capped_by_evictable_remainder(self):
        policy = EvictionPolicy(_resolved(top_k_windows=3, top_q_windows=5))
        scores = torch.rand(1, 2, 5)  # 4 evictable + 1 local
        fp_idx, q_idx = policy.compute_tier_assignments(scores)
        assert fp_idx.shape[1] == 4  # 3 fp + local
        assert q_idx.shape[1] == 1   # only 1 evictable window left over

    def test_zero_q_budget_gives_empty_q(self):
        policy = EvictionPolicy(_resolved(top_q_windows=0))
        scores = torch.rand(1, 2, 6)
        fp_idx, q_idx = policy.compute_tier_assignments(scores)
        assert q_idx.shape == (1, 0)
        # And the fp band matches the single-tier method exactly.
        legacy = policy.compute_retain_window_indices(scores)
        assert torch.equal(fp_idx, legacy)

    def test_expand_with_explicit_fp_total(self):
        """Merged indices are translated to fp-store ranks by the caller; the
        expansion then caps by the EXPLICIT fp-store length."""
        policy = EvictionPolicy(_resolved())
        policy.initialize_after_prefill(999)  # deliberately wrong global total
        ranks = torch.tensor([[0, 2]])
        tok = policy.expand_to_token_indices(ranks, total_tokens=2 + 3 * 4 - 1)
        # sink [0,1] + rank0 [2..5] + rank2 [10..13] — 13 is OOB (total 13).
        assert tok.tolist() == [[0, 1, 2, 3, 4, 5, 10, 11, 12]]


# ---------------------------------------------------------------------------
# WS-4 / WS-5 — full two-tier update()/eviction integration
# ---------------------------------------------------------------------------
#
# Geometry: window_size=2, sink=0, local=2 (1 window), prefill=12, fp32 KV
# (bytes_per_token=8192), cache_budget=0.53125, quant_ratio=0.25 → resolves
# to top_k_windows=2 (fp) and top_q_windows=2 (int4).
# Keys/values encode their ORIGINAL WINDOW ID in
# every element, so (a) surviving windows can be read back by value, and
# (b) pre-RoPE quant groups are degenerate (constant) → int4 reconstruction
# is EXACT, making end-to-end equality assertions possible.

_H_KV, _D, _H_Q = 2, 8, 2
_S = 2


class _RealRoPE(torch.nn.Module):
    """Genuine sin/cos rotation so position stamping is actually exercised."""

    def forward(self, x, position_ids):
        D = x.shape[-1]
        inv = 1.0 / (100.0 ** (torch.arange(0, D, 2).float() / D))
        ang = position_ids[..., :, None].float() * inv[None, None, :]
        emb = torch.cat([ang, ang], dim=-1)
        return emb.cos(), emb.sin()


def _wk(vals):
    """Keys [1, H_kv, len(vals), D] whose every element is the token's
    original window id."""
    t = torch.tensor(vals, dtype=torch.float32).view(1, 1, -1, 1)
    return t.expand(1, _H_KV, len(vals), _D).contiguous()


def _sc(vals):
    """Window scores [1, H_q, W] (same across heads)."""
    t = torch.tensor(vals, dtype=torch.float32).view(1, 1, -1)
    return t.expand(1, _H_Q, len(vals)).contiguous()


def _mk_cache(rope=None, quant_ratio=0.25, cache_budget=0.53125):
    cfg = WindowedCacheConfig(
        window_size=_S, num_sink_tokens=0, local_window_size=2,
        cache_budget=cache_budget, quant_ratio=quant_ratio,
    )
    cache = WindowedCache(
        config=cfg, prefill_len=12, model_config=_FakeModelConfig(),
        kv_dtype=torch.float32, rope_module=rope or _NoOpRoPE(),
        num_layers=1, max_tokens=4,
    )
    return cache


def _prefill(cache):
    vals = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    k = _wk(vals)
    return cache.update(k, k.clone(), 0, cache_kwargs={
        "cache_position": torch.arange(12),
        "window_scores": _sc([100, 50, 3, 90, 2, 40]),
    })


def _step(cache, val, scores):
    """One decode step: query position follows get_seq_length (mirroring the
    position-override hook), one new token whose value is its window id."""
    pos = cache.get_seq_length(0)
    k = _wk([val])
    return cache.update(k, k.clone(), 0, cache_kwargs={
        "cache_position": torch.tensor([pos]),
        "window_scores": _sc(scores),
    })


def _drive_to_evict1(cache):
    """Prefill + 3 decode steps; the eviction fires on the 3rd (step 2).
    Outcome: fp = {w0, w3, w7(local)}, Q = {w1, w5}, dropped = {w2, w4, w6}."""
    _prefill(cache)
    _step(cache, 6, [0] * 7)
    _step(cache, 6, [0] * 7)
    return _step(cache, 7, [0] * 8)  # eviction inside this call


def _continue_to_evict2(cache):
    """Two more steps after eviction 1; eviction 2 on step 4.
    Outcome: fp = {w0, w1(PROMOTED), w8(local)}, Q = {w3(fresh demote), w5},
    dropped = {w7}."""
    _step(cache, 7, [0] * 5)
    return _step(cache, 8, [500, 1000, 100, 90, 0, 0])


def _continue_to_evict3(cache):
    """Two more steps after eviction 2; eviction 3 on step 6.
    Outcome: fp = {w0, w8, w9(local)}, Q = {w1(REACTIVATED), w3},
    dropped = {w5}. No fresh quantization happens here."""
    _step(cache, 8, [0] * 5)
    return _step(cache, 9, [1400, 150, 910, 0, 1500, 0])


def _drive_to_evict2(cache):
    _drive_to_evict1(cache)
    return _continue_to_evict2(cache)


class TestTwoTierEviction:

    def test_resolved_geometry(self):
        cache = _mk_cache()
        assert cache.resolved.top_k_windows == 2
        assert cache.resolved.top_q_windows == 2
        assert cache._two_tier

    def test_mixed_tier_eviction_layout(self):
        """Evict 1: merged axis, tier mask, gappy fp positions, ledger ranges,
        effective T_total, and the interleaved effective K/V content."""
        cache = _mk_cache()
        k_eff, v_eff = _drive_to_evict1(cache)
        state = cache._states[0]
        ledger = cache._q_ledgers[0]

        # Merged window axis: both tiers, chronological.
        assert state.original_window_ids.tolist() == [[0, 1, 3, 5, 7]]
        assert state.fp_tier_mask.tolist() == [[True, False, True, False, True]]

        # fp store: w0 (2 tok) + w3 (2 tok) + w7 (1 tok, partial local) at the
        # INTERLEAVED (gappy) positions — gaps exactly where w1/w5 sit.
        assert state.position_ids.tolist() == [[0, 1, 4, 5, 8]]
        assert state.key_states[0, 0, :, 0].tolist() == [0.0, 0.0, 3.0, 3.0, 7.0]

        # Ledger: active w1 at positions [2,3], w5 at [6,7].
        assert ledger.active_count == 2
        assert ledger.entries[1].position_start == 2
        assert ledger.entries[5].position_start == 6

        # Effective length spans both tiers.
        assert cache.get_seq_length(0) == 9

        # update() returned the interleaved effective K/V in chronological
        # window order — int4 reconstruction is exact here (degenerate groups).
        expect = [0.0, 0.0, 1.0, 1.0, 3.0, 3.0, 5.0, 5.0, 7.0]
        assert k_eff.shape == (1, _H_KV, 9, _D)
        assert k_eff[0, 0, :, 0].tolist() == expect
        assert v_eff[0, 0, :, 0].tolist() == expect
        # ...and does NOT alias the fp store.
        assert k_eff.shape[2] != state.key_states.shape[2]

    def test_merged_axis_score_alignment(self):
        """window_scores index i ↔ chronological window id i ↔ i-th physical
        window chunk of the effective K/V, across a mixed-tier eviction."""
        cache = _mk_cache()
        k_eff, _ = _drive_to_evict1(cache)
        state = cache._states[0]
        # Scores gathered on the merged axis (accumulated: prefill values).
        assert state.window_scores[0, 0].tolist() == [100.0, 50.0, 90.0, 40.0, 0.0]
        # Physical chunking of the effective K matches original_window_ids.
        chunk_ids = k_eff[0, 0, ::_S, 0].long().tolist()
        assert chunk_ids == state.original_window_ids[0].tolist()

    def test_promotion_keeps_ledger_entry_dormant(self):
        cache = _mk_cache()
        k_eff, _ = _drive_to_evict2(cache)
        state = cache._states[0]
        ledger = cache._q_ledgers[0]

        assert state.original_window_ids.tolist() == [[0, 1, 3, 5, 8]]
        assert state.fp_tier_mask.tolist() == [[True, True, False, False, True]]
        # w1 was promoted: fp copy live, ledger entry retained DORMANT.
        assert 1 in ledger.entries and not ledger.entries[1].active
        assert ledger.dormant_count == 1
        # w3 freshly demoted + w5 stays → 2 active.
        assert ledger.active_count == 2
        assert ledger.entries[3].position_start == 4
        assert ledger.entries[5].position_start == 6
        # fp store spliced chronologically: w0 | w1(dequantized) | w8.
        assert state.position_ids.tolist() == [[0, 1, 2, 3, 8]]
        assert state.key_states[0, 0, :, 0].tolist() == [0.0, 0.0, 1.0, 1.0, 8.0]
        expect = [0.0, 0.0, 1.0, 1.0, 3.0, 3.0, 5.0, 5.0, 8.0]
        assert k_eff[0, 0, :, 0].tolist() == expect

    def test_redemotion_reactivates_codes_bit_identical(self, monkeypatch):
        """Promote→demote reactivation: the stored codes are reused untouched
        and the quantizer is NOT re-invoked (design §10)."""
        cache = _mk_cache()
        _drive_to_evict1(cache)
        ledger = cache._q_ledgers[0]
        store = cache._q_stores[0]

        # Snapshot w1's codes right after its first (only) quantization.
        slot1 = ledger.entries[1].slot
        k_codes_orig = store.k_codes[slot1].clone()
        v_codes_orig = store.v_codes[slot1].clone()
        k_scale_orig = store.k_scale[slot1].clone()

        _continue_to_evict2(cache)       # w1 promoted (dormant)

        # Spy on the quantizer for the re-demotion eviction.
        calls = []
        orig_qk = cache_mod.quantize_key_windows
        monkeypatch.setattr(
            cache_mod, "quantize_key_windows",
            lambda k_win: (calls.append(1), orig_qk(k_win))[1],
        )
        k_eff, _ = _continue_to_evict3(cache)

        # Re-demotion of w1 (and the w5 drop) triggered ZERO quantizer calls.
        assert calls == []
        assert ledger.entries[1].active
        # Codes + pinned grid are bit-identical (slot may have shifted due to
        # the w5 drop + store compaction).
        slot1_new = ledger.entries[1].slot
        assert torch.equal(store.k_codes[slot1_new], k_codes_orig)
        assert torch.equal(store.v_codes[slot1_new], v_codes_orig)
        assert torch.equal(store.k_scale[slot1_new], k_scale_orig)

        # Final layout sanity.
        state = cache._states[0]
        assert state.original_window_ids.tolist() == [[0, 1, 3, 8, 9]]
        assert state.fp_tier_mask.tolist() == [[True, False, False, True, True]]
        expect = [0.0, 0.0, 1.0, 1.0, 3.0, 3.0, 8.0, 8.0, 9.0]
        assert k_eff[0, 0, :, 0].tolist() == expect
        # Dropped w5's ledger entry is freed and the store compacted gap-free.
        assert 5 not in ledger.entries
        assert store.num_slots == len(ledger.entries)

    def test_two_tier_equals_fat_fp_cache_noop_rope(self):
        """A two-tier cache (top_k=2 fp + top_q=2 int4) retaining windows
        {0,1,3,5,7} produces the SAME effective K/V as a single-tier fp cache
        with top_k=4 retaining the same set — exact, since the per-window
        constant keys quantize losslessly."""
        two = _mk_cache()
        k2, v2 = _drive_to_evict1(two)
        # β=0.63, q=0 → 10 budget tokens → top_k=(10-2)//2 = 4.
        fat = _mk_cache(quant_ratio=0.0, cache_budget=0.63)
        assert fat.resolved.top_k_windows == 4
        kf, vf = _drive_to_evict1(fat)
        assert torch.equal(k2, kf)
        assert torch.equal(v2, vf)
        # Single-tier keeps contiguous positions; two-tier tiles the same
        # arange across fp position_ids ∪ ledger ranges.
        assert fat._states[0].position_ids.tolist() == [list(range(9))]
        ledger = two._q_ledgers[0]
        q_pos = []
        for e in ledger.entries.values():
            q_pos.extend(range(e.position_start, e.position_start + _S))
        union = sorted(two._states[0].position_ids[0].tolist() + q_pos)
        assert union == list(range(9))

    def test_two_tier_equals_fat_fp_cache_real_rope(self):
        """Same equivalence under a REAL rotation: values (never rotated,
        losslessly quantized) match exactly; keys match to within int4 error
        of the un-rotate→quantize→re-rotate path. A mis-stamped position
        would produce O(window-id) errors, far above the tolerance."""
        two = _mk_cache(rope=_RealRoPE())
        k2, v2 = _drive_to_evict1(two)
        fat = _mk_cache(rope=_RealRoPE(), quant_ratio=0.0, cache_budget=0.63)
        kf, vf = _drive_to_evict1(fat)
        assert torch.equal(v2, vf)
        assert torch.allclose(k2, kf, atol=0.25)
        assert (k2 - kf).abs().max() > 0  # quant error is real, not a no-op

    def test_attention_position_invariance_of_interleave(self):
        """softmax(q·Kᵀ)·V is invariant to physical key order — RoPE bakes the
        logical position into each key, so the chronological interleave is a
        scorer requirement, not an attention one (design §5)."""
        torch.manual_seed(0)
        cache = _mk_cache(rope=_RealRoPE())
        k_eff, v_eff = _drive_to_evict1(cache)
        q = torch.randn(1, _H_KV, 1, _D)
        ref = torch.softmax(q @ k_eff.transpose(-2, -1) * _D ** -0.5, -1) @ v_eff
        perm = torch.randperm(k_eff.shape[2])
        got = torch.softmax(
            q @ k_eff[:, :, perm].transpose(-2, -1) * _D ** -0.5, -1
        ) @ v_eff[:, :, perm]
        assert torch.allclose(ref, got, atol=1e-6)

    def test_seq_length_and_query_position_follow_t_total(self):
        """get_seq_length reports T_total (fp+Q) so the position override
        appends new tokens at the effective end (design §5 step 7)."""
        cache = _mk_cache()
        _drive_to_evict1(cache)
        assert cache.get_seq_length(0) == 9        # 5 fp + 4 Q
        _step(cache, 7, [0] * 5)
        # Token appended AT position 9; effective length now 10.
        assert cache.get_seq_length(0) == 10
        assert cache._states[0].position_ids[0, -1].item() == 9

    def test_batch_size_gt1_rejected_with_quant(self):
        cache = _mk_cache()
        vals = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
        k = _wk(vals).expand(2, _H_KV, 12, _D).contiguous()
        cache.update(k, k.clone(), 0, cache_kwargs={
            "cache_position": torch.arange(12),
            "window_scores": _sc([100, 50, 3, 90, 2, 40]).expand(2, _H_Q, 6).contiguous(),
        })
        for pos in (12, 13):
            k1 = _wk([6]).expand(2, _H_KV, 1, _D).contiguous()
            cache.update(k1, k1.clone(), 0, cache_kwargs={
                "cache_position": torch.tensor([pos]),
                "window_scores": _sc([0] * 7).expand(2, _H_Q, 7).contiguous(),
            })
        k1 = _wk([7]).expand(2, _H_KV, 1, _D).contiguous()
        with pytest.raises(NotImplementedError, match="batch-size-1"):
            cache.update(k1, k1.clone(), 0, cache_kwargs={
                "cache_position": torch.tensor([14]),
                "window_scores": _sc([0] * 8).expand(2, _H_Q, 8).contiguous(),
            })

    def test_q0_cache_unchanged_by_two_tier_wiring(self):
        """quant_ratio=0 drives the legacy path: no tier mask, no ledger
        activity, live fp tensors returned (aliasing preserved)."""
        cache = _mk_cache(quant_ratio=0.0)
        assert not cache._two_tier
        k_ret, v_ret = _drive_to_evict1(cache)
        state = cache._states[0]
        assert state.fp_tier_mask is None
        assert cache._q_ledgers[0].active_count == 0
        assert cache._q_stores[0].num_slots == 0
        assert k_ret is state.key_states and v_ret is state.value_states
        # Legacy contiguous rebase.
        assert state.position_ids.tolist() == [list(range(state.seq_length))]


# ---------------------------------------------------------------------------
# WS-5 — flash hook sources effective K; flash/eager score parity
# ---------------------------------------------------------------------------


class TestFlashEagerParity:

    def test_get_effective_keys_matches_update_return(self):
        """The flash hook's aux-SDPA key source (get_effective_keys) is the
        SAME merged-axis tensor eager attention consumes (update()'s return)."""
        cache = _mk_cache()
        k_eff, _ = _drive_to_evict1(cache)
        k_hook = cache.get_effective_keys(0)
        assert torch.equal(k_hook, k_eff)
        # And with the tier off it is the raw fp store (legacy behaviour).
        legacy = _mk_cache(quant_ratio=0.0)
        _drive_to_evict1(legacy)
        assert legacy.get_effective_keys(0) is legacy._states[0].key_states

    def test_flash_and_eager_window_scores_agree_on_merged_axis(self):
        """Score a random query against (a) the flash path — aux SDPA over
        get_effective_keys — and (b) the eager path — attention weights over
        update()'s returned K.  Identical per-window scores on the merged
        axis across a mixed-tier eviction."""
        import torch.nn.functional as F
        from modules.windowed_cache.scorer import (
            reduce_token_scores_to_windows,
        )
        from modules.windowed_eager_cache.scorer import (
            compute_window_scores as eager_scores,
        )

        torch.manual_seed(1)
        cache = _mk_cache(rope=_RealRoPE())
        k_ret, _ = _drive_to_evict1(cache)
        q = torch.randn(1, _H_KV, 1, _D)

        # Flash path (hooks.py): aux SDPA over the cache's effective keys.
        k_flash = cache.get_effective_keys(0)
        aw = torch.matmul(q, k_flash.transpose(-2, -1)) * _D ** -0.5
        aw = F.softmax(aw, dim=-1, dtype=torch.float32).to(q.dtype)
        flash = reduce_token_scores_to_windows(aw.sum(dim=-2), 0, _S)

        # Eager path (hooks.py): real attention over update()'s return.
        aw2 = torch.matmul(q, k_ret.transpose(-2, -1)) * _D ** -0.5
        aw2 = F.softmax(aw2, dim=-1, dtype=torch.float32).to(q.dtype)
        eager = eager_scores(aw2, 0, _S)

        assert flash.shape == eager.shape == (1, _H_KV, 5)  # merged axis
        assert torch.equal(flash, eager)

    def test_flash_hook_source_is_effective_keys(self):
        """Regression pin: hooks.py must not read raw _states[l].key_states."""
        import inspect
        from modules.windowed_cache import hooks as flash_hooks
        src = inspect.getsource(flash_hooks)
        assert "get_effective_keys" in src
        assert "._states[lidx].key_states" not in src

    def test_eager_twin_runs_the_same_two_tier_cycle(self):
        """The eager backend is a byte-identical twin (modulo banner): the
        same drive produces the same mixed-tier layout."""
        from modules.windowed_eager_cache.cache import (
            WindowedCache as EagerCache,
        )
        from modules.windowed_eager_cache.config import (
            WindowedCacheConfig as EagerConfig,
        )
        cfg = EagerConfig(
            window_size=_S, num_sink_tokens=0, local_window_size=2,
            cache_budget=0.53125, quant_ratio=0.25,
        )
        cache = EagerCache(
            config=cfg, prefill_len=12, model_config=_FakeModelConfig(),
            kv_dtype=torch.float32, rope_module=_NoOpRoPE(),
            num_layers=1, max_tokens=4,
        )
        k_eff, v_eff = _drive_to_evict1(cache)
        state = cache._states[0]
        assert state.original_window_ids.tolist() == [[0, 1, 3, 5, 7]]
        assert state.fp_tier_mask.tolist() == [[True, False, True, False, True]]
        expect = [0.0, 0.0, 1.0, 1.0, 3.0, 3.0, 5.0, 5.0, 7.0]
        assert k_eff[0, 0, :, 0].tolist() == expect
        assert cache.get_seq_length(0) == 9


# ---------------------------------------------------------------------------
# WS-6 — config surface + tier telemetry
# ---------------------------------------------------------------------------


class TestConfigSurfaceAndTelemetry:

    def test_yaml_quant_ratio_threads_into_cache_config(self, tmp_path):
        from utils.config import ConfigValidationError, load_config
        yml = tmp_path / "exp.yaml"
        yml.write_text(
            "cache:\n  window_size: 32\n  quant_ratio: 0.5\n", encoding="utf-8"
        )
        cfg = load_config(yml)
        assert cfg.cache.quant_ratio == 0.5
        # Default is off.
        yml2 = tmp_path / "exp2.yaml"
        yml2.write_text("cache:\n  window_size: 32\n", encoding="utf-8")
        assert load_config(yml2).cache.quant_ratio == 0.0
        # Validation mirrors WindowedCacheConfig.
        bad = tmp_path / "bad.yaml"
        bad.write_text(
            "cache:\n  window_size: 7\n  quant_ratio: 0.5\n", encoding="utf-8"
        )
        with pytest.raises(ConfigValidationError, match="even"):
            load_config(bad)

    def test_base_yaml_defaults_tier_off(self):
        from utils.config import load_config
        cfg = load_config("configs/base.yaml")
        assert cfg.cache.quant_ratio == 0.0

    def test_tier_events_recorded_per_eviction(self):
        from modules.windowed_cache.telemetry import NullTelemetry, Telemetry
        cache = _mk_cache()
        cache.telemetry = Telemetry(num_layers=1)
        _drive_to_evict1(cache)
        _continue_to_evict2(cache)
        events = [
            r for r in cache.telemetry.get_records(0) if "tier_promoted" in r
        ]
        assert len(events) == 2
        # Evict 1: 2 fresh demotions, no promotions, 3 dropped windows.
        assert events[0]["tier_demoted"] == 2
        assert events[0]["tier_promoted"] == 0
        assert events[0]["tier_reactivated"] == 0
        assert events[0]["tier_dropped"] == 3
        assert events[0]["tier_active_q_windows"] == 2
        # Evict 2: w1 promoted (→ 1 dormant entry), w3 freshly demoted.
        assert events[1]["tier_promoted"] == 1
        assert events[1]["tier_demoted"] == 1
        assert events[1]["tier_dormant_entries"] == 1
        assert events[1]["tier_active_q_windows"] == 2
        # NullTelemetry stays a no-op.
        nt = NullTelemetry()
        nt.record_tier_events(0, 0, 1, 1, 1, 1, 1, 1)
        assert nt.get_records(0) == []


# ---------------------------------------------------------------------------
# fp16 dtype stability — production kv dtype through the two-tier cycle
# ---------------------------------------------------------------------------


class TestFp16DtypeStability:
    """The production path runs fp16 KV. The rotated Q-tier keys and the
    promoted-window dequants must come back in the store dtypes even if the
    rope module emits fp32 cos/sin (HF ropes cast to x.dtype; a sloppy module
    must not crash the interleave or silently promote the values)."""

    class _FaithfulRoPE(torch.nn.Module):
        def forward(self, x, position_ids):
            D = x.shape[-1]
            inv = 1.0 / (100.0 ** (torch.arange(0, D, 2).float() / D))
            ang = position_ids[..., :, None].float() * inv[None, None, :]
            emb = torch.cat([ang, ang], dim=-1)
            return emb.cos().to(x.dtype), emb.sin().to(x.dtype)

    class _SloppyRoPE(torch.nn.Module):
        def forward(self, x, position_ids):
            D = x.shape[-1]
            inv = 1.0 / (100.0 ** (torch.arange(0, D, 2).float() / D))
            ang = position_ids[..., :, None].float() * inv[None, None, :]
            emb = torch.cat([ang, ang], dim=-1)
            return emb.cos(), emb.sin()  # always fp32

    def _drive_fp16(self, rope):
        cfg = WindowedCacheConfig(
            window_size=_S, num_sink_tokens=0, local_window_size=2,
            cache_budget=0.53125, quant_ratio=0.25,
        )
        cache = WindowedCache(
            config=cfg, prefill_len=12, model_config=_FakeModelConfig(),
            kv_dtype=torch.float16, rope_module=rope, num_layers=1,
            max_tokens=4,
        )
        torch.manual_seed(3)

        def upd(n_tok, w):
            t = cache.get_seq_length(0)
            k = torch.randn(1, _H_KV, n_tok, _D, dtype=torch.float16)
            v = torch.randn(1, _H_KV, n_tok, _D, dtype=torch.float16)
            return cache.update(k, v, 0, cache_kwargs={
                "cache_position": torch.arange(t, t + n_tok),
                "window_scores": _sc(torch.rand(w).tolist()),
            })

        upd(12, 6)
        out = None
        for i in range(7):  # 3 evictions incl. promote/demote churn
            t = cache.get_seq_length(0)
            out = upd(1, (t + 1 - 0 + _S - 1) // _S)
        return cache, out

    def test_faithful_rope_keeps_everything_fp16(self):
        cache, (k_eff, v_eff) = self._drive_fp16(self._FaithfulRoPE())
        assert cache._q_ledgers[0].active_count > 0  # tier actually live
        assert k_eff.dtype == torch.float16
        assert v_eff.dtype == torch.float16
        assert cache._states[0].key_states.dtype == torch.float16

    def test_sloppy_fp32_rope_does_not_crash_interleave(self):
        cache, (k_eff, v_eff) = self._drive_fp16(self._SloppyRoPE())
        assert cache._q_ledgers[0].active_count > 0
        # Values are never rotated — they must stay fp16 regardless.
        assert v_eff.dtype == torch.float16
        assert cache._states[0].value_states.dtype == torch.float16
        assert k_eff.shape[2] == v_eff.shape[2] == cache.get_seq_length(0)


# ---------------------------------------------------------------------------
# Multi-eviction consistency fuzz — sink prefix, odd prefill, random scores
# ---------------------------------------------------------------------------


class TestLongRunConsistency:

    def test_invariants_hold_across_many_evictions(self):
        """Drive 11 decode steps (5 evictions) with random scores, a sink
        prefix, and an odd prefill. At every step: (a) effective length =
        fp + Q tokens, (b) fp position_ids ∪ ledger ranges tile
        arange(T_total) exactly, (c) each physical window chunk of the
        effective K carries its original_window_id (int4 round-trip exact for
        the constant-per-window keys), (d) the Q tier never exceeds its
        budget."""
        torch.manual_seed(7)
        S, sink = 2, 2
        cfg = WindowedCacheConfig(
            window_size=S, num_sink_tokens=sink, local_window_size=2,
            cache_budget=0.46, quant_ratio=0.18,
        )
        cache = WindowedCache(
            config=cfg, prefill_len=13, model_config=_FakeModelConfig(),
            kv_dtype=torch.float32, rope_module=_NoOpRoPE(),
            num_layers=1, max_tokens=11,
        )
        assert cache._two_tier and cache.resolved.top_q_windows >= 1
        assert cache.resolved.top_k_windows >= 1

        def val(tok_idx):  # window id of a token; sink gets a sentinel
            return -5.0 if tok_idx < sink else float((tok_idx - sink) // S)

        def mk_keys(idxs):
            t = torch.tensor([val(i) for i in idxs]).view(1, 1, -1, 1)
            return t.expand(1, _H_KV, len(idxs), _D).contiguous()

        def check(cache, k_eff):
            state = cache._states[0]
            ledger = cache._q_ledgers[0]
            t_eff = cache.get_seq_length(0)
            assert t_eff == state.seq_length + ledger.active_tokens
            assert k_eff.shape[2] == t_eff
            # Position tiling.
            q_pos = []
            for e in ledger.entries.values():
                if e.active:
                    q_pos.extend(range(e.position_start, e.position_start + S))
            union = sorted(state.position_ids[0].tolist() + q_pos)
            assert union == list(range(t_eff)), union
            # Merged-axis chunk identity (skip trailing partial window).
            if state.original_window_ids is not None:
                ids = state.original_window_ids[0].tolist()
                got = k_eff[0, 0, sink::S, 0].tolist()
                for w_pos, owid in enumerate(ids):
                    if sink + w_pos * S + S <= t_eff:  # full window only
                        assert got[w_pos] == float(owid), (w_pos, ids, got)
            # Budget: Q tier never exceeds its window capacity.
            assert ledger.active_count <= cache.resolved.top_q_windows

        n_tok = 13
        k = mk_keys(range(n_tok))
        w0 = (13 - sink + S - 1) // S
        k_eff, _ = cache.update(k, k.clone(), 0, cache_kwargs={
            "cache_position": torch.arange(n_tok),
            "window_scores": _sc(torch.rand(w0).tolist()),
        })
        check(cache, k_eff)

        # Token identity is tracked by NEXT original window id so appended
        # tokens carry the id of the window they actually join.
        for _step_i in range(11):
            state = cache._states[0]
            t_eff = cache.get_seq_length(0)
            # The appended token joins the newest window: full windows so far
            # (over the effective axis) determine its id.
            newest_id = cache._next_original_window_id[0]
            post = t_eff - sink
            joins_new = post % S == 0
            if joins_new:
                tok_val = float(newest_id)
            else:
                tok_val = float(state.original_window_ids[0, -1].item())
            k1 = torch.full((1, _H_KV, 1, _D), tok_val)
            W = (t_eff + 1 - sink + S - 1) // S
            k_eff, _ = cache.update(k1, k1.clone(), 0, cache_kwargs={
                "cache_position": torch.tensor([t_eff]),
                "window_scores": _sc(torch.rand(W).tolist()),
            })
            check(cache, k_eff)


# ---------------------------------------------------------------------------
# Score-hook lag reconciliation — regression for the merged-axis vs physical
# store desync (fp positions / retained-token gather length mismatch).
# ---------------------------------------------------------------------------


class TestScoreLagReconciliation:
    """The flash/eager score hooks score each token on the NEXT forward pass,
    so the token appended on the very step that triggers an eviction is
    physically in the fp store but not yet on the merged window axis. If that
    unscored token opened a new window, `_evict_two_tier` used to mix the
    physical `T_fp_old`/`tail` with the lagged merged fp-window set and the
    interleaved position map diverged from the retained-token gather —
    crashing `rerotate_keys` on a length mismatch. Feeding scores with the
    realistic one-step lag reproduces it; the reconciliation fixes it.

    The parametrized geometries below were verified to CRASH on the pre-fix
    code (``rerotate_keys`` length mismatch); the accounting model config uses
    ``head_dim == _D`` so the resolved tier sizes match the physical windows.
    """

    @dataclass
    class _SmallModelConfig:
        num_attention_heads: int = _H_Q
        num_key_value_heads: int = _H_KV
        hidden_size: int = _H_Q * _D
        head_dim: int = _D
        num_hidden_layers: int = 1

    class _RealRoPESmall(torch.nn.Module):
        def forward(self, x, position_ids):
            D = x.shape[-1]
            inv = 1.0 / (10000.0 ** (torch.arange(0, D, 2).float() / D))
            ang = position_ids[..., :, None].float() * inv[None, None, :]
            emb = torch.cat([ang, ang], dim=-1)
            return emb.cos().to(x.dtype), emb.sin().to(x.dtype)

    def _score_over_effective(self, cache, q, num_sink, S):
        """Mimic the flash hook: aux SDPA over get_effective_keys → windows."""
        import torch.nn.functional as F
        from modules.windowed_cache.scorer import reduce_token_scores_to_windows
        k = cache.get_effective_keys(0)
        aw = torch.matmul(q, k.transpose(-2, -1)) * (_D ** -0.5)
        aw = F.softmax(aw.float(), dim=-1).to(q.dtype)
        return reduce_token_scores_to_windows(aw.sum(dim=-2), num_sink, S)

    @pytest.mark.parametrize("prefill,num_sink", [(8, 0), (10, 2), (12, 0)])
    def test_lagged_scores_survive_eviction(self, prefill, num_sink):
        S = _S
        cfg = WindowedCacheConfig(
            window_size=S, num_sink_tokens=num_sink, local_window_size=0.5,
            cache_budget=0.4, quant_ratio=0.5,
        )
        cache = WindowedCache(
            config=cfg, prefill_len=prefill, model_config=self._SmallModelConfig(),
            kv_dtype=torch.float32, rope_module=self._RealRoPESmall(),
            num_layers=1, max_tokens=30,
        )
        assert cache.resolved.top_q_windows >= 1  # Q tier actually engages
        assert cache.resolved.top_k_windows >= 1
        torch.manual_seed(0)

        def n_windows(t):
            post = max(t - num_sink, 0)
            return (post + S - 1) // S if post > 0 else 0

        def check():
            state = cache._states[0]
            ledger = cache._q_ledgers[0]
            t_eff = cache.get_seq_length(0)
            assert t_eff == state.seq_length + ledger.active_tokens
            # fp position_ids ∪ active ledger ranges tile arange(t_eff) exactly.
            q_pos = []
            for e in ledger.entries.values():
                if e.active:
                    q_pos.extend(range(e.position_start, e.position_start + S))
            union = sorted(state.position_ids[0].tolist() + q_pos)
            assert union == list(range(t_eff))
            assert ledger.active_count <= cache.resolved.top_q_windows
            # The merged fp-window count is reconciled DOWN to the physical
            # store at eviction (never over-tracks it).
            T_fp = state.seq_length
            phys = (T_fp - num_sink + S - 1) // S if T_fp > num_sink else 0
            assert int(state.fp_tier_mask.sum().item()) <= phys

        # Prefill.
        k = torch.randn(1, _H_KV, prefill, _D)
        cache.update(k, k.clone(), 0, cache_kwargs={
            "cache_position": torch.arange(prefill),
            "window_scores": torch.rand(1, _H_Q, n_windows(prefill)),
        })
        # Hook fires AFTER update — one-step lag is the whole point.
        q = torch.randn(1, _H_KV, 1, _D)
        pending = self._score_over_effective(cache, q, num_sink, S)

        for _ in range(30):
            t = cache.get_seq_length(0)
            k1 = torch.randn(1, _H_KV, 1, _D)
            cache.update(k1, k1.clone(), 0, cache_kwargs={
                "cache_position": torch.tensor([t]),
                "window_scores": pending,
            })
            check()
            q = torch.randn(1, _H_KV, 1, _D)
            pending = self._score_over_effective(cache, q, num_sink, S)
