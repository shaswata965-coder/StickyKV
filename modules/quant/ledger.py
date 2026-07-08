"""Per-window ledger for the Q tier (design.md §6).

One :class:`QuantLedger` per layer (B=1 in v1). Each entry is keyed by
``original_window_id`` and tracks a window across evictions:

===================  ========  =====================================================
field                mutable?  purpose
===================  ========  =====================================================
original_window_id   no        chronological identity; drives the interleaved sort
codes / scale / zero no        live in the :class:`QuantizedStore` at ``slot``;
                               written exactly once at first demotion, never again
slot                 yes       offset into the Q store; shifts as the store compacts
position_start       yes       first position of the window's contiguous
                               ``position_range`` in the interleaved map; updated
                               every eviction (O(Q_windows) int writes)
active               yes       dormant entries (promoted windows) keep codes + grid
                               but are excluded from reads and the interleave
===================  ========  =====================================================

Entries **persist through promotion** as dormant; a later re-demotion is a
pure reactivation (no re-quantization — design §10). An entry is freed only
when its window is dropped outright.

The eviction-cadence bookkeeping is per-entry Python ints (the design pins it
at O(Q_windows) integer assignments); the per-step **read path** consumes the
cached :meth:`active_view` tensors, rebuilt only when the ledger mutates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor


@dataclass
class LedgerEntry:
    original_window_id: int   # immutable
    slot: int                 # mutable — shifts on store compaction
    active: bool              # False = dormant (promoted; codes retained)
    position_start: int = -1  # mutable — rewritten every eviction


class QuantLedger:
    """Ledger of Q-tier windows for one layer (B=1)."""

    def __init__(self, window_size: int) -> None:
        self.window_size = window_size
        self.entries: Dict[int, LedgerEntry] = {}
        self._view: Optional[Tuple[Tensor, Tensor, Tensor]] = None

    # -- introspection ---------------------------------------------------

    def __contains__(self, owid: int) -> bool:
        return owid in self.entries

    def is_dormant(self, owid: int) -> bool:
        e = self.entries.get(owid)
        return e is not None and not e.active

    @property
    def active_count(self) -> int:
        return sum(1 for e in self.entries.values() if e.active)

    @property
    def dormant_count(self) -> int:
        return sum(1 for e in self.entries.values() if not e.active)

    @property
    def active_tokens(self) -> int:
        """Effective Q-tier token count (every Q window is a full window)."""
        return self.active_count * self.window_size

    # -- mutation (eviction cadence only) --------------------------------

    def add(self, owid: int, slot: int) -> None:
        """Register a first-time demotion (codes just written to *slot*)."""
        assert owid not in self.entries, f"window {owid} already has a ledger entry"
        self.entries[owid] = LedgerEntry(owid, slot, active=True)
        self._view = None

    def reactivate(self, owid: int) -> None:
        """Re-demotion of a previously-promoted window: flip dormant → active.

        Zero compute and zero added error — the stored codes + pinned grid are
        reused as-is (design §10).
        """
        e = self.entries[owid]
        assert not e.active, f"window {owid} is already active"
        e.active = True
        self._view = None

    def deactivate(self, owid: int) -> None:
        """Promotion Q→K: keep the entry (codes + grid) dormant."""
        e = self.entries[owid]
        assert e.active, f"window {owid} is already dormant"
        e.active = False
        self._view = None

    def drop(self, owid: int) -> None:
        """Window dropped outright — free its entry (store slot freed at compact)."""
        del self.entries[owid]
        self._view = None

    def set_position_starts(self, owids: List[int], starts: List[int]) -> None:
        """Write each surviving Q window's new ``position_range`` start
        (design §5 step 6 — integer assignments only)."""
        for owid, start in zip(owids, starts):
            self.entries[owid].position_start = int(start)
        self._view = None

    def compact_store_slots(self) -> Tensor:
        """Compute the surviving slot list and shift entry offsets to match.

        Returns the ascending ``keep_slots`` tensor to pass to
        ``QuantizedStore.compact``; every entry's ``slot`` is rewritten to its
        new (rank) position.
        """
        keep = sorted(e.slot for e in self.entries.values())
        remap = {old: new for new, old in enumerate(keep)}
        for e in self.entries.values():
            e.slot = remap[e.slot]
        self._view = None
        return torch.tensor(keep, dtype=torch.long)

    # -- read path (per step) --------------------------------------------

    def active_view(self) -> Tuple[Tensor, Tensor, Tensor]:
        """Return ``(owids, slots, position_starts)`` of the **active** entries,
        sorted by ``original_window_id`` (chronological interleave order).

        Cached between mutations so the per-step read path pays no Python
        iteration after the first call following an eviction.
        """
        if self._view is None:
            act = sorted(
                (e for e in self.entries.values() if e.active),
                key=lambda e: e.original_window_id,
            )
            self._view = (
                torch.tensor([e.original_window_id for e in act], dtype=torch.long),
                torch.tensor([e.slot for e in act], dtype=torch.long),
                torch.tensor([e.position_start for e in act], dtype=torch.long),
            )
        return self._view
