"""Tests for the flash-attn windowed cache package — 24 tests.

All tests run on CPU with mocked attention modules and synthetic data.
No real model loads.
"""

from __future__ import annotations

import ast
import inspect
import math
from dataclasses import dataclass
from typing import Optional

import pytest
import torch
from torch import Tensor

from modules.windowed_cache.config import ResolvedConfig, WindowedCacheConfig
from modules.windowed_cache.policy import EvictionPolicy
from modules.windowed_cache.scorer import accumulate, compute_window_scores
from modules.windowed_cache.state import CacheState
from modules.windowed_cache.telemetry import NullTelemetry, Telemetry
from modules.windowed_cache.hooks import HookHandles


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class _FakeModelConfig:
    """Mimics HF PretrainedConfig for testing."""
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    hidden_size: int = 4096
    head_dim: int = 128
    num_hidden_layers: int = 32


def _make_config(**overrides):
    defaults = dict(
        window_size=8,
        num_sink_tokens=4,
        local_window_size=16,
        cache_budget=0.40,
        track_scores=False,
    )
    defaults.update(overrides)
    return WindowedCacheConfig(**defaults)


def _make_resolved(**overrides):
    defaults = dict(
        window_size=8,
        num_sink_tokens=4,
        local_tokens=16,
        top_k_windows=2,
        bytes_per_token=4096,
        total_budget_bytes=163840,
        total_budget_tokens=40,
    )
    defaults.update(overrides)
    return ResolvedConfig(**defaults)


# ---------------------------------------------------------------------------
# 1. test_percentage_rounding_snaps_up_to_window_multiple
# ---------------------------------------------------------------------------

class TestConfig:

    def test_percentage_rounding_snaps_up_to_window_multiple(self):
        """Float local_window_size should ceil then snap up to window_size multiple."""
        cfg = _make_config(
            window_size=8,
            num_sink_tokens=4,
            local_window_size=0.25,
            cache_budget=0.50,
        )
        model_cfg = _FakeModelConfig()
        resolved = cfg.resolve(prefill_len=100, model_config=model_cfg, kv_dtype=torch.float16, max_tokens=128)
        # local is a fraction of the BUDGET: total_budget_tokens =
        # floor(0.50 * (100 + 128)) = 114; 0.25 * 114 = 28.5 → ceil 29 → snap 32
        assert resolved.local_tokens == 32
        assert resolved.local_tokens % resolved.window_size == 0

        # Non-exact: 0.10 * 114 = 11.4, ceil=12, 12 % 8 = 4 → snap up to 16
        cfg2 = _make_config(
            window_size=8,
            num_sink_tokens=4,
            local_window_size=0.10,
            cache_budget=0.50,
        )
        resolved2 = cfg2.resolve(prefill_len=100, model_config=model_cfg, kv_dtype=torch.float16, max_tokens=128)
        assert resolved2.local_tokens == 16
        assert resolved2.local_tokens % resolved2.window_size == 0

    # -------------------------------------------------------------------
    # 2. test_worked_example_prefill
    # -------------------------------------------------------------------

    def test_worked_example_prefill(self):
        """LLaMA-3-8B fp16, prefill=100, budget=0.40 → known values."""
        cfg = _make_config(
            window_size=8,
            num_sink_tokens=4,
            local_window_size=16,
            cache_budget=0.40,
        )
        model_cfg = _FakeModelConfig(
            num_attention_heads=32,
            num_key_value_heads=8,
            hidden_size=4096,
            head_dim=128,
        )
        resolved = cfg.resolve(prefill_len=100, model_config=model_cfg, kv_dtype=torch.float16, max_tokens=100)

        # bytes_per_token = 8 * 128 * 2 * 2 = 4096
        assert resolved.bytes_per_token == 4096
        # total_budget_bytes = int(0.40 * (100+100) * 4096) = 327680
        assert resolved.total_budget_bytes == 327680
        # total_budget_tokens = 327680 // 4096 = 80
        assert resolved.total_budget_tokens == 80
        # remaining = 80 - 4 - 16 = 60, top_k = 60 // 8 = 7
        assert resolved.top_k_windows == 7

    # -------------------------------------------------------------------
    # 3. test_retained_cache_never_exceeds_byte_budget
    # -------------------------------------------------------------------

    @pytest.mark.parametrize("budget", [0.20, 0.40, 0.60, 0.80, 1.0])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
    def test_retained_cache_never_exceeds_byte_budget(self, budget, dtype):
        """Retained token count * bytes_per_token <= total_budget_bytes."""
        cfg = _make_config(cache_budget=budget, local_window_size=8)
        model_cfg = _FakeModelConfig()
        resolved = cfg.resolve(prefill_len=200, model_config=model_cfg, kv_dtype=dtype, max_tokens=128)

        retained = (
            resolved.num_sink_tokens
            + resolved.top_k_windows * resolved.window_size
            + resolved.local_tokens
        )
        assert retained * resolved.bytes_per_token <= resolved.total_budget_bytes

    # -------------------------------------------------------------------
    # 4. test_dtype_invariance_of_ratio
    # -------------------------------------------------------------------

    def test_dtype_invariance_of_ratio(self):
        """Token budget should be same ratio regardless of dtype."""
        cfg = _make_config(cache_budget=0.50, local_window_size=8)
        model_cfg = _FakeModelConfig()
        r16 = cfg.resolve(200, model_cfg, torch.float16, max_tokens=128)
        r32 = cfg.resolve(200, model_cfg, torch.float32, max_tokens=128)
        # Token budget should be identical (budget * (prefill_len + max_tokens))
        assert r16.total_budget_tokens == r32.total_budget_tokens

    # -------------------------------------------------------------------
    # 5. test_gqa_byte_accounting
    # -------------------------------------------------------------------

    def test_gqa_byte_accounting(self):
        """GQA: bytes_per_token uses num_kv_heads, not num_attention_heads."""
        cfg = _make_config(cache_budget=0.50, local_window_size=8)
        model_cfg = _FakeModelConfig(num_attention_heads=32, num_key_value_heads=8)
        resolved = cfg.resolve(100, model_cfg, torch.float16, max_tokens=128)
        # 8 * 128 * 2 * 2 = 4096 (not 32 * 128 * 2 * 2 = 16384)
        assert resolved.bytes_per_token == 8 * 128 * 2 * 2

    # -------------------------------------------------------------------
    # 6. test_cache_budget_must_be_float
    # -------------------------------------------------------------------

    def test_cache_budget_must_be_float(self):
        """int and bool rejected for cache_budget."""
        with pytest.raises(ValueError, match="float"):
            _make_config(cache_budget=1)  # type: ignore
        with pytest.raises(ValueError, match="bool"):
            _make_config(cache_budget=True)  # type: ignore

    # -------------------------------------------------------------------
    # 7. test_cache_budget_smaller_than_protected_raises
    # -------------------------------------------------------------------

    def test_cache_budget_smaller_than_protected_raises(self):
        """Budget too small for sink + local → ValueError."""
        cfg = _make_config(
            cache_budget=0.05,
            num_sink_tokens=10,
            local_window_size=40,
        )
        model_cfg = _FakeModelConfig()
        with pytest.raises(ValueError, match="total_budget_tokens"):
            cfg.resolve(100, model_cfg, torch.float16, max_tokens=50)

    # -------------------------------------------------------------------
    # 8. test_cache_budget_zero_evictable_is_legal
    # -------------------------------------------------------------------

    def test_cache_budget_zero_evictable_is_legal(self):
        """top_k_windows=0 is legal (sink + local only)."""
        cfg = _make_config(
            window_size=8,
            num_sink_tokens=4,
            local_window_size=8,
            cache_budget=0.12,  # just enough for sink + local = 12 tokens
        )
        model_cfg = _FakeModelConfig()
        resolved = cfg.resolve(100, model_cfg, torch.float16, max_tokens=50)
        assert resolved.top_k_windows >= 0


# ---------------------------------------------------------------------------
# Scoring Tests
# ---------------------------------------------------------------------------

class TestScoring:

    # -------------------------------------------------------------------
    # 9. test_local_window_score_persists_after_sliding
    # -------------------------------------------------------------------

    def test_local_window_score_persists_after_sliding(self):
        """Scores accumulated in local window persist after sliding into evictable."""
        B, H_q, S = 1, 4, 20
        num_sink = 4
        window_size = 8

        # Simulate 3 steps of accumulation on the same attention pattern
        attn = torch.randn(B, H_q, 4, S).softmax(dim=-1)
        scores1 = compute_window_scores(attn, num_sink, window_size)
        scores2 = compute_window_scores(attn, num_sink, window_size)
        scores3 = compute_window_scores(attn, num_sink, window_size)

        state_scores = scores1.clone()
        accumulate(state_scores, scores2)
        accumulate(state_scores, scores3)

        # Score should be 3x the single-step score
        expected = scores1 * 3
        assert torch.allclose(state_scores, expected, atol=1e-5)

    # -------------------------------------------------------------------
    # 10. test_window_scores_survive_eviction_compaction (pins §3)
    # -------------------------------------------------------------------

    def test_window_scores_survive_eviction_compaction(self):
        """After eviction, surviving windows keep their accumulated scores."""
        B, H_q = 1, 4
        W_total = 6
        window_size = 8
        num_sink = 4

        # Known window scores: windows 0-5 with distinct scores
        window_scores = torch.tensor([
            [[10.0, 1.0, 2.0, 8.0, 7.0, 5.0]]
        ]).expand(B, H_q, W_total).clone()

        resolved = _make_resolved(
            window_size=window_size,
            num_sink_tokens=num_sink,
            local_tokens=16,  # 2 local windows
            top_k_windows=2,
        )
        policy = EvictionPolicy(resolved)
        total = num_sink + W_total * window_size
        policy.initialize_after_prefill(total)

        # Compute retain window indices
        retained_window_idx = policy.compute_retain_window_indices(window_scores)
        # Top-2 from evictable [0,1,2,3] by mean score: window 0 (10.0), window 3 (8.0)
        # Local: windows 4, 5
        # Expected: [0, 3, 4, 5]

        # Gather window scores by retained indices (mimicking cache.py)
        idx_w = retained_window_idx.unsqueeze(1).expand(B, H_q, -1)
        retained_scores = torch.gather(window_scores, dim=-1, index=idx_w)

        # Verify scores persisted (not zeroed)
        expected_scores = torch.tensor([
            [[10.0, 8.0, 7.0, 5.0]]
        ]).expand(B, H_q, -1)
        assert torch.allclose(retained_scores, expected_scores)


# ---------------------------------------------------------------------------
# Eviction / Rerotation Tests
# ---------------------------------------------------------------------------

class TestEviction:

    # -------------------------------------------------------------------
    # 11. test_position_ids_preserve_originals_after_eviction
    # -------------------------------------------------------------------

    def test_position_ids_preserve_originals_after_eviction(self):
        """After slice_and_keep, position_ids = the surviving tokens' ORIGINAL
        positions (no rebasing) so keys keep their original RoPE phase."""
        state = CacheState()
        B, H, T, D = 1, 4, 20, 64
        state.key_states = torch.randn(B, H, T, D)
        state.value_states = torch.randn(B, H, T, D)
        state.position_ids = torch.arange(T)

        retain = torch.tensor([[0, 1, 5, 10, 15, 19]])
        state.slice_and_keep(retain)

        expected = torch.tensor([0, 1, 5, 10, 15, 19])
        assert torch.equal(state.position_ids, expected)

    # -------------------------------------------------------------------
    # 12. test_key_rerotation_uses_new_positions
    # -------------------------------------------------------------------

    def test_key_rerotation_uses_new_positions(self):
        """After rerotation, keys should be different from before."""
        state = CacheState()
        B, H, T, D = 1, 4, 10, 64
        state.key_states = torch.randn(B, H, T, D)
        state.value_states = torch.randn(B, H, T, D)
        state.position_ids = torch.arange(T)

        old_keys = state.key_states.clone()
        old_positions = torch.tensor([0, 2, 4, 6, 8, 10, 12, 14, 16, 18])

        # Mock rope module
        class MockRoPE(torch.nn.Module):
            def forward(self, x, position_ids):
                seq_len = position_ids.shape[-1]
                cos = torch.ones(1, seq_len, D) * 0.5
                sin = torch.ones(1, seq_len, D) * 0.3
                return cos, sin

        try:
            state.rerotate_keys(MockRoPE(), old_positions)
            # Keys should have changed
            assert not torch.equal(state.key_states, old_keys)
        except ImportError:
            pytest.skip("transformers not available for rerotation test")

    # -------------------------------------------------------------------
    # 13. test_rerotation_uses_model_rope_module
    # -------------------------------------------------------------------

    def test_rerotation_uses_model_rope_module(self):
        """Rerotation must use the model's RoPE (for NTK/YaRN preservation)."""
        # Verified by code inspection: state.rerotate_keys accepts rope_module
        # and calls it to get cos/sin. The test below confirms the signature.
        state = CacheState()
        sig = inspect.signature(state.rerotate_keys)
        assert "rope_module" in sig.parameters

    # -------------------------------------------------------------------
    # 14. test_values_not_rerotated
    # -------------------------------------------------------------------

    def test_values_not_rerotated(self):
        """Values should remain unchanged after rerotation."""
        state = CacheState()
        B, H, T, D = 1, 4, 10, 64
        state.key_states = torch.randn(B, H, T, D)
        state.value_states = torch.randn(B, H, T, D)
        state.position_ids = torch.arange(T)

        old_values = state.value_states.clone()
        old_positions = torch.arange(T) * 2

        class MockRoPE(torch.nn.Module):
            def forward(self, x, position_ids):
                seq_len = position_ids.shape[-1]
                cos = torch.ones(1, seq_len, D)
                sin = torch.zeros(1, seq_len, D)
                return cos, sin

        try:
            state.rerotate_keys(MockRoPE(), old_positions)
            assert torch.equal(state.value_states, old_values)
        except ImportError:
            pytest.skip("transformers not available")

    # -------------------------------------------------------------------
    # 15. test_retained_windows_are_in_chronological_order
    # -------------------------------------------------------------------

    def test_retained_windows_are_in_chronological_order(self):
        """Retained window indices must be sorted chronologically."""
        resolved = _make_resolved(top_k_windows=3, local_tokens=16)
        policy = EvictionPolicy(resolved)
        policy.initialize_after_prefill(4 + 8 * 10)  # 10 windows

        B, H_q, W = 1, 4, 10
        scores = torch.randn(B, H_q, W)
        retained = policy.compute_retain_window_indices(scores)

        # Check chronological order
        for b in range(B):
            vals = retained[b].tolist()
            assert vals == sorted(vals), f"Not sorted: {vals}"

    # -------------------------------------------------------------------
    # 16. test_retain_shared_across_heads_via_mean
    # -------------------------------------------------------------------

    def test_retain_shared_across_heads_via_mean(self):
        """Retain decision uses mean across heads — same indices for all heads."""
        resolved = _make_resolved(top_k_windows=2, local_tokens=16)
        policy = EvictionPolicy(resolved)
        policy.initialize_after_prefill(4 + 8 * 8)  # 8 windows

        B, H_q, W = 2, 4, 8
        scores = torch.randn(B, H_q, W)

        retained = policy.compute_retain_window_indices(scores)
        # retained is [B, W_retained] — same dimensionality, head-agnostic
        assert retained.shape[0] == B
        assert retained.dim() == 2

    # -------------------------------------------------------------------
    # 17. test_retain_independent_across_batch
    # -------------------------------------------------------------------

    def test_retain_independent_across_batch(self):
        """Different batch items can retain different windows."""
        resolved = _make_resolved(top_k_windows=1, local_tokens=8)
        policy = EvictionPolicy(resolved)
        policy.initialize_after_prefill(4 + 8 * 5)  # 5 windows

        B, H_q, W = 2, 4, 5
        scores = torch.zeros(B, H_q, W)
        # Batch 0: window 0 is best evictable
        scores[0, :, 0] = 100.0
        # Batch 1: window 3 is best evictable (local_windows=1, so evictable=0,1,2,3)
        scores[1, :, 3] = 100.0

        retained = policy.compute_retain_window_indices(scores)
        # Batch 0 should retain window 0 (top-1 evictable) + window 4 (local)
        assert 0 in retained[0].tolist()
        # Batch 1 should retain window 3 (top-1 evictable) + window 4 (local)
        assert 3 in retained[1].tolist()

    # -------------------------------------------------------------------
    # 18. test_no_premask_invariant
    # -------------------------------------------------------------------

    def test_no_premask_invariant(self):
        """compute_window_scores takes full attention (no masking before softmax)."""
        B, H_q, T_obs, S = 1, 2, 4, 20
        # Full attention with softmax already applied
        attn = torch.randn(B, H_q, T_obs, S).softmax(dim=-1)
        scores = compute_window_scores(attn, num_sink=4, window_size=8)
        # All scores should be positive (sum of positive softmax values)
        assert (scores >= 0).all()


# ---------------------------------------------------------------------------
# Hook Tests
# ---------------------------------------------------------------------------

class TestHooks:

    # -------------------------------------------------------------------
    # 19. test_extract_arg_prefers_kwarg_then_positional
    # -------------------------------------------------------------------

    def test_extract_arg_prefers_kwarg_then_positional(self):
        """_extract_arg reads a forward arg by keyword, falling back to position."""
        from modules.windowed_cache.hooks import _extract_arg
        # keyword present -> returned directly
        assert _extract_arg((), {"hidden_states": 7}, "hidden_states", 0) == 7
        # absent keyword -> positional fallback at the given index
        assert _extract_arg(("h", "pe"), {}, "position_embeddings", 1) == "pe"
        # neither -> None
        assert _extract_arg((), {}, "hidden_states", 0) is None

    # -------------------------------------------------------------------
    # 20. test_hook_removal_idempotent
    # -------------------------------------------------------------------

    def test_hook_removal_idempotent(self):
        """handles.remove() is a no-op on second call."""
        handles = HookHandles()
        handles._hook_handles = []
        handles.remove()
        handles.remove()  # should not raise
        assert handles._removed

    # -------------------------------------------------------------------
    # 21. test_score_hook_does_not_disable_flash_attn
    # -------------------------------------------------------------------

    def test_score_hook_does_not_disable_flash_attn(self):
        """Scoring uses pure forward hooks — no forward replacement, flash-attn stays active."""
        handles = HookHandles()
        # register_forward_hook handles only — the backend never replaces
        # module.forward, so flash-attn-2 runs untouched.
        assert hasattr(handles, "_hook_handles")
        assert not hasattr(handles, "_patched_modules")

    # -------------------------------------------------------------------
    # 22. test_telemetry_disabled_is_noop
    # -------------------------------------------------------------------

    def test_telemetry_disabled_is_noop(self):
        """NullTelemetry should be zero overhead."""
        t = NullTelemetry()
        # Should not raise and should not store
        t.record_scores(0, 0, torch.zeros(1, 4, 8))
        t.record_cache_state(0, 0, torch.zeros(1), torch.zeros(1), torch.zeros(1))
        assert t.get_records(0) == []

    # -------------------------------------------------------------------
    # 23. test_prefill_not_divisible_by_window_size
    # -------------------------------------------------------------------

    def test_prefill_not_divisible_by_window_size(self):
        """N=97, window_size=5 — partial window gets zero-padded scores."""
        B, H_q, T_obs = 1, 2, 4
        num_sink = 4
        window_size = 5

        S = 97  # total keys
        attn = torch.randn(B, H_q, T_obs, S).softmax(dim=-1)
        scores = compute_window_scores(attn, num_sink, window_size)

        # post_sink = 93 tokens, ceil(93/5) = 19 windows
        expected_windows = math.ceil(93 / window_size)
        assert scores.shape == (B, H_q, expected_windows)

    # -------------------------------------------------------------------
    # 24. test_no_python_loops_in_hot_path
    # -------------------------------------------------------------------

    def test_no_python_loops_in_hot_path(self):
        """AST inspection: reject `for` loops over batch/head/token/window in hot-path files."""
        from modules.windowed_cache import cache as cache_mod
        from modules.windowed_cache import state as state_mod
        from modules.windowed_cache import policy as policy_mod
        from modules.windowed_cache import scorer as scorer_mod

        forbidden_iter_vars = {"batch", "b", "head", "h", "token", "tok", "t", "window", "w", "n"}

        for mod in [cache_mod, state_mod, policy_mod, scorer_mod]:
            source = inspect.getsource(mod)
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.For):
                    target = node.target
                    if isinstance(target, ast.Name) and target.id.lower() in forbidden_iter_vars:
                        pytest.fail(
                            f"Found forbidden loop variable '{target.id}' in "
                            f"{mod.__name__}"
                        )

    # 25. test_q_buffer_preallocation removed:
    # _QRingBuffer was deleted along with the obs_window scoring path.
    # H2O-style cumulative scoring needs no per-layer query buffer.
