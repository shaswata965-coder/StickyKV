"""QuantizedStore — dense, gap-free int4 code store for one layer's Q tier.

Holds the packed codes and pinned fp16 scale/zero for every ledger-tracked
window (active **and** dormant) along a window-slot axis. Which slot belongs
to which window — and whether it is active — lives in the
:class:`~modules.quant.ledger.QuantLedger`; the store is pure tensor storage.

Keys are stored **pre-RoPE** (design §2, §5): RoPE is applied fresh at read
using each window's current ``position_range``. Slot shapes are fixed by
``(H_kv, D, window_size)``, so the store is a plain dense stack:

- ``k_codes``  ``[N, H_kv, D, S/2]`` uint8 (channel-major, 2 tokens/byte)
- ``k_scale`` / ``k_zero``  ``[N, H_kv, D]`` fp16
- ``v_codes``  ``[N, H_kv, S, D/2]`` uint8 (token-major, 2 channels/byte)
- ``v_scale`` / ``v_zero``  ``[N, H_kv, S]`` fp16

``append`` adds windows at the end; ``compact`` gathers surviving slots
(the ledger shifts its slot offsets to match). Codes/scales are **never**
rewritten in place — a window's slot content is immutable for its lifetime
(no re-quantization, design §3, §10).

v1 scope: batch-size 1 — one store per layer (per-row primitives; B>1 is a
later extension, design §10).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor


class QuantizedStore:
    """Dense int4 code store + pinned fp16 grids for one layer (B=1)."""

    __slots__ = ("k_codes", "k_scale", "k_zero", "v_codes", "v_scale", "v_zero")

    def __init__(self) -> None:
        self.k_codes: Optional[Tensor] = None
        self.k_scale: Optional[Tensor] = None
        self.k_zero: Optional[Tensor] = None
        self.v_codes: Optional[Tensor] = None
        self.v_scale: Optional[Tensor] = None
        self.v_zero: Optional[Tensor] = None

    @property
    def num_slots(self) -> int:
        return 0 if self.k_codes is None else self.k_codes.shape[0]

    def append(
        self,
        k_codes: Tensor, k_scale: Tensor, k_zero: Tensor,
        v_codes: Tensor, v_scale: Tensor, v_zero: Tensor,
    ) -> int:
        """Append *n* windows (leading axis) and return the first new slot index."""
        first = self.num_slots
        if self.k_codes is None:
            self.k_codes = k_codes.contiguous()
            self.k_scale = k_scale.contiguous()
            self.k_zero = k_zero.contiguous()
            self.v_codes = v_codes.contiguous()
            self.v_scale = v_scale.contiguous()
            self.v_zero = v_zero.contiguous()
        else:
            self.k_codes = torch.cat([self.k_codes, k_codes], dim=0)
            self.k_scale = torch.cat([self.k_scale, k_scale], dim=0)
            self.k_zero = torch.cat([self.k_zero, k_zero], dim=0)
            self.v_codes = torch.cat([self.v_codes, v_codes], dim=0)
            self.v_scale = torch.cat([self.v_scale, v_scale], dim=0)
            self.v_zero = torch.cat([self.v_zero, v_zero], dim=0)
        return first

    def gather_keys(self, slots: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Return ``(k_codes, k_scale, k_zero)`` for the given slot indices."""
        return (
            self.k_codes.index_select(0, slots),
            self.k_scale.index_select(0, slots),
            self.k_zero.index_select(0, slots),
        )

    def gather_values(self, slots: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Return ``(v_codes, v_scale, v_zero)`` for the given slot indices."""
        return (
            self.v_codes.index_select(0, slots),
            self.v_scale.index_select(0, slots),
            self.v_zero.index_select(0, slots),
        )

    def compact(self, keep_slots: Tensor) -> None:
        """Drop every slot not in *keep_slots* (ascending order preserved).

        The caller (ledger) is responsible for shifting its per-entry slot
        offsets to the new positions (``old keep_slots[i] → new i``).
        """
        if self.k_codes is None:
            return
        if keep_slots.numel() == 0:
            self.__init__()
            return
        self.k_codes = self.k_codes.index_select(0, keep_slots).contiguous()
        self.k_scale = self.k_scale.index_select(0, keep_slots).contiguous()
        self.k_zero = self.k_zero.index_select(0, keep_slots).contiguous()
        self.v_codes = self.v_codes.index_select(0, keep_slots).contiguous()
        self.v_scale = self.v_scale.index_select(0, keep_slots).contiguous()
        self.v_zero = self.v_zero.index_select(0, keep_slots).contiguous()
