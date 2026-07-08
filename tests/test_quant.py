"""WS-1 tests — quantizer / store / ledger (modules/quant).

All CPU, no transformers dependency. Pins the design.md §2 numerics:
fp16-pinned grid, round-half-even, degenerate-group exactness, nibble
packing along the scale-group axis, and ledger dormancy/reactivation.
"""

from __future__ import annotations

import pytest
import torch

from modules.quant import (
    QuantizedStore,
    QuantLedger,
    dequantize,
    dequantize_key_windows,
    dequantize_value_windows,
    pack_nibbles,
    quantize,
    quantize_key_windows,
    quantize_value_windows,
    quantize_with_grid,
    unpack_nibbles,
)


# ---------------------------------------------------------------------------
# Quantizer numerics
# ---------------------------------------------------------------------------


class TestQuantizerNumerics:

    def test_round_trip_error_within_bound(self):
        """|x̂ − x| ≤ scale/2 (+ fp16-grid slop at the clamp boundary)."""
        torch.manual_seed(0)
        x = (torch.randn(8, 4, 64) * 3).to(torch.float16)
        codes, scale, zero = quantize(x, dim=-1)
        xhat = dequantize(codes, scale, zero, out_dtype=torch.float32)
        err = (xhat - x.float()).abs()
        # The fp16-rounded grid can shift each code centre by up to
        # 15·ulp(scale)/2; 0.51·scale absorbs it comfortably.
        bound = 0.51 * scale.float() + 1e-6
        assert (err <= bound).all(), f"max err {err.max()} vs bound {bound.min()}"

    def test_codes_are_int4_range(self):
        x = torch.randn(4, 32)
        codes, _, _ = quantize(x, dim=-1)
        assert codes.dtype == torch.uint8
        assert int(codes.min()) >= 0 and int(codes.max()) <= 15

    def test_degenerate_group_exactness(self):
        """mx == mn → scale = 1, all codes 0, x̂ = mn exactly."""
        x = torch.full((2, 3, 16), 2.75, dtype=torch.float16)
        codes, scale, zero = quantize(x, dim=-1)
        assert (codes == 0).all()
        assert (scale == 1.0).all()
        xhat = dequantize(codes, scale, zero, out_dtype=torch.float16)
        assert torch.equal(xhat, x)

    def test_degenerate_group_non_fp16_value(self):
        """Constant fp32 group not representable in fp16 still yields all-zero
        codes and x̂ == the fp16-stored zero exactly."""
        x = torch.full((1, 8), 3.14159, dtype=torch.float32)
        codes, scale, zero = quantize(x, dim=-1)
        assert (codes == 0).all()
        xhat = dequantize(codes, scale, zero, out_dtype=torch.float16)
        assert torch.equal(xhat, zero.expand_as(xhat))

    def test_fp16_grid_idempotence(self):
        """quant → dequant → re-quant against the SAME fp16 grid gives
        bit-identical codes (zero structural drift, design §2/§10)."""
        torch.manual_seed(1)
        x = (torch.randn(6, 8, 32) * 2).to(torch.float16)
        codes, scale, zero = quantize(x, dim=-1)
        # Strict form: dequant kept in fp32 (code centres are exact).
        xhat32 = dequantize(codes, scale, zero, out_dtype=torch.float32)
        codes2 = quantize_with_grid(xhat32, scale, zero)
        assert torch.equal(codes, codes2)
        # fp16 write-back form on well-conditioned data (the v1 read path).
        xhat16 = dequantize(codes, scale, zero, out_dtype=torch.float16)
        codes3 = quantize_with_grid(xhat16, scale, zero)
        assert torch.equal(codes, codes3)

    def test_quantizes_against_fp16_grid_not_fp32(self):
        """Codes must be fit to the fp16-ROUNDED scale/zero (design §2)."""
        # A range whose fp32 scale differs from its fp16 rounding.
        x = torch.tensor([[0.0, 0.1, 0.2, 1.0000123]], dtype=torch.float32)
        codes, scale, zero = quantize(x, dim=-1)
        expected = quantize_with_grid(x, scale, zero)  # fp16 grid
        assert torch.equal(codes, expected)

    def test_round_half_even(self):
        """torch.round (banker's rounding) is the pinned rounding mode."""
        # grid: zero=0, scale=1 → values at exact halves.
        scale = torch.ones(1, dtype=torch.float16)
        zero = torch.zeros(1, dtype=torch.float16)
        x = torch.tensor([0.5, 1.5, 2.5, 3.5], dtype=torch.float32)
        codes = quantize_with_grid(x, scale, zero)
        assert codes.tolist() == [0, 2, 2, 4]

    def test_clamp_before_cast(self):
        """Out-of-grid values clamp to [0, 15] — never wrap through the cast."""
        scale = torch.ones(1, dtype=torch.float16)
        zero = torch.zeros(1, dtype=torch.float16)
        x = torch.tensor([-7.0, 300.0], dtype=torch.float32)
        codes = quantize_with_grid(x, scale, zero)
        assert codes.tolist() == [0, 15]


# ---------------------------------------------------------------------------
# Nibble packing
# ---------------------------------------------------------------------------


class TestPacking:

    def test_pack_unpack_round_trip(self):
        torch.manual_seed(2)
        codes = torch.randint(0, 16, (3, 5, 8, 32), dtype=torch.uint8)
        assert torch.equal(unpack_nibbles(pack_nibbles(codes)), codes)

    def test_even_index_in_low_nibble(self):
        codes = torch.tensor([1, 2, 3, 4], dtype=torch.uint8)
        packed = pack_nibbles(codes)
        assert packed.tolist() == [1 | (2 << 4), 3 | (4 << 4)]

    def test_odd_group_axis_asserted(self):
        with pytest.raises(AssertionError, match="even"):
            pack_nibbles(torch.zeros(3, dtype=torch.uint8))

    def test_packed_dequant_equals_unpacked(self):
        """Packed layout behind the same API == the unpacked bring-up variant."""
        torch.manual_seed(3)
        k = (torch.randn(4, 2, 16, 8)).to(torch.float16)  # [N, H, S, D]
        # Unpacked reference: quantize channel-major directly.
        x = k.permute(0, 1, 3, 2)
        codes_u, scale_u, zero_u = quantize(x, dim=-1)
        ref = dequantize(codes_u, scale_u, zero_u).permute(0, 1, 3, 2)
        # Packed path through the window API.
        codes_p, scale_p, zero_p = quantize_key_windows(k)
        got = dequantize_key_windows(codes_p, scale_p, zero_p)
        assert torch.equal(ref.contiguous(), got)
        assert torch.equal(unpack_nibbles(codes_p), codes_u)


# ---------------------------------------------------------------------------
# Window-level K/V granularity (design §2)
# ---------------------------------------------------------------------------


class TestWindowGranularity:

    def test_key_shapes_channel_major(self):
        N, H, S, D = 3, 2, 16, 8
        codes, scale, zero = quantize_key_windows(torch.randn(N, H, S, D))
        assert codes.shape == (N, H, D, S // 2)      # 2 tokens/byte
        assert scale.shape == (N, H, D)              # per (head, channel, window)
        assert scale.dtype == torch.float16 and zero.dtype == torch.float16

    def test_value_shapes_token_major(self):
        N, H, S, D = 3, 2, 16, 8
        codes, scale, zero = quantize_value_windows(torch.randn(N, H, S, D))
        assert codes.shape == (N, H, S, D // 2)      # 2 channels/byte
        assert scale.shape == (N, H, S)              # per (head, token)

    def test_key_group_is_token_axis(self):
        """A per-channel constant key window must reconstruct exactly (each
        (head, channel) group over the window is degenerate)."""
        N, H, S, D = 1, 2, 8, 4
        base = torch.randn(N, H, 1, D).to(torch.float16)
        k = base.expand(N, H, S, D).contiguous()
        codes, scale, zero = quantize_key_windows(k)
        got = dequantize_key_windows(codes, scale, zero)
        assert torch.equal(got, k)

    def test_value_group_is_channel_axis(self):
        """A per-token constant value row reconstructs exactly."""
        N, H, S, D = 1, 2, 8, 4
        base = torch.randn(N, H, S, 1).to(torch.float16)
        v = base.expand(N, H, S, D).contiguous()
        codes, scale, zero = quantize_value_windows(v)
        got = dequantize_value_windows(codes, scale, zero)
        assert torch.equal(got, v)

    def test_value_round_trip_bound(self):
        torch.manual_seed(4)
        v = (torch.randn(4, 2, 16, 8) * 2).to(torch.float16)
        codes, scale, zero = quantize_value_windows(v)
        got = dequantize_value_windows(codes, scale, zero, out_dtype=torch.float32)
        err = (got - v.float()).abs()
        bound = 0.51 * scale.float().unsqueeze(-1) + 1e-6
        assert (err <= bound).all()


# ---------------------------------------------------------------------------
# Store + ledger
# ---------------------------------------------------------------------------


def _window_payload(n, H=2, S=8, D=4, seed=0):
    torch.manual_seed(seed)
    k = torch.randn(n, H, S, D).to(torch.float16)
    v = torch.randn(n, H, S, D).to(torch.float16)
    return quantize_key_windows(k) + quantize_value_windows(v)


class TestStoreLedger:

    def test_append_and_gather(self):
        store = QuantizedStore()
        payload = _window_payload(3)
        first = store.append(*payload)
        assert first == 0 and store.num_slots == 3
        first2 = store.append(*_window_payload(2, seed=1))
        assert first2 == 3 and store.num_slots == 5
        kc, ks, kz = store.gather_keys(torch.tensor([1, 4]))
        assert kc.shape[0] == 2
        assert torch.equal(kc[0], store.k_codes[1])

    def test_compact_preserves_content(self):
        store = QuantizedStore()
        store.append(*_window_payload(4))
        keep = torch.tensor([0, 2, 3])
        want = store.k_codes[keep].clone()
        store.compact(keep)
        assert store.num_slots == 3
        assert torch.equal(store.k_codes, want)

    def test_ledger_lifecycle_and_slot_remap(self):
        ledger = QuantLedger(window_size=8)
        store = QuantizedStore()
        store.append(*_window_payload(3))
        for owid, slot in [(5, 0), (9, 1), (2, 2)]:
            ledger.add(owid, slot)
        assert ledger.active_count == 3 and ledger.active_tokens == 24

        # Promote window 9 → dormant (codes retained).
        ledger.deactivate(9)
        assert ledger.active_count == 2 and ledger.dormant_count == 1

        # Drop window 5 outright; compact the store.
        ledger.drop(5)
        keep = ledger.compact_store_slots()
        assert keep.tolist() == [1, 2]
        store.compact(keep)
        assert store.num_slots == 2
        # Slots remapped: owid 9 (was slot 1) → 0, owid 2 (was slot 2) → 1.
        assert ledger.entries[9].slot == 0
        assert ledger.entries[2].slot == 1

        # Re-demotion = reactivation, not re-quant.
        ledger.reactivate(9)
        assert ledger.active_count == 2 and ledger.dormant_count == 0

    def test_active_view_sorted_by_original_window_id(self):
        ledger = QuantLedger(window_size=4)
        for owid, slot in [(7, 0), (1, 1), (4, 2)]:
            ledger.add(owid, slot)
        ledger.deactivate(4)  # dormant → excluded from the view
        ledger.set_position_starts([7, 1], [20, 4])
        owids, slots, starts = ledger.active_view()
        assert owids.tolist() == [1, 7]
        assert slots.tolist() == [1, 0]
        assert starts.tolist() == [4, 20]

    def test_position_start_updates_are_int_writes(self):
        """set_position_starts touches only position_start — codes and slots
        (the immutable/store fields) are untouched."""
        ledger = QuantLedger(window_size=4)
        ledger.add(3, 0)
        before_slot = ledger.entries[3].slot
        ledger.set_position_starts([3], [12])
        assert ledger.entries[3].position_start == 12
        assert ledger.entries[3].slot == before_slot
