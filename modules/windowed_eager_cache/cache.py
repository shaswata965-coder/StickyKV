"""WindowedCache — HuggingFace Cache integration for windowed KV cache.

Orchestration only.  No scoring math, no Top-K math, no attention computation,
no RoPE math — only calls into :mod:`state` and :mod:`policy`.

NOTE: This module is byte-identical to ``modules/windowed_cache/cache.py``
(backends only differ in their ``hooks.py``). Any change here MUST be mirrored
to the flash twin until the duplication is refactored away.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor

try:
    from transformers import Cache as _HFCacheBase
except ImportError:
    _HFCacheBase = object  # type: ignore[assignment,misc]

from modules.quant import (
    QuantLedger,
    QuantizedStore,
    apply_rope_to_keys,
    build_interleaved_position_map,
    dequantize_key_windows,
    dequantize_value_windows,
    materialize_effective_kv,
    quantize_key_windows,
    quantize_value_windows,
)

from .config import ResolvedConfig, WindowedCacheConfig
from .policy import EvictionPolicy
from .scorer import accumulate
from .state import CacheState
from .telemetry import NullTelemetry, Telemetry


class WindowedCache(_HFCacheBase):
    """Windowed KV cache with H2O-style cumulative eviction.

    Parameters
    ----------
    config : WindowedCacheConfig
        User-facing configuration.
    prefill_len : int
        Number of tokens in the prompt (used for budget resolution).
    model_config
        HuggingFace ``PretrainedConfig`` or compatible.
    kv_dtype : torch.dtype
        Data type of the KV cache tensors.
    rope_module : nn.Module
        The model's rotary embedding module, used to re-rotate surviving keys
        to contiguous positions after every eviction (KVPress
        ``KeyRerotationPress`` methodology).
    num_layers : int
        Number of transformer layers.
    telemetry : Telemetry, optional
        Telemetry recorder.  Defaults to :class:`NullTelemetry`.
    """

    def __init__(
        self,
        config: WindowedCacheConfig,
        prefill_len: int,
        model_config: Any,
        kv_dtype: torch.dtype,
        rope_module: torch.nn.Module,
        num_layers: int,
        max_tokens: int,
        telemetry: Optional[Telemetry] = None,
    ) -> None:
        if isinstance(_HFCacheBase, type) and _HFCacheBase is not object:
            try:
                super().__init__()
            except (TypeError, ValueError):
                # transformers >= 4.50 changed Cache.__init__ to require
                # `layers` or `layer_class_to_replicate`. We manage our own
                # per-layer state (self._states) and override the full Cache
                # interface, so skipping the base init is safe.
                pass

        self.config = config
        self.resolved = config.resolve(prefill_len, model_config, kv_dtype, max_tokens)
        self.rope_module = rope_module
        self.num_layers = num_layers
        self.telemetry = telemetry if telemetry is not None else NullTelemetry()

        # Per-layer state and policy
        self._states: List[CacheState] = [CacheState() for _ in range(num_layers)]
        self._policies: List[EvictionPolicy] = [
            EvictionPolicy(self.resolved) for _ in range(num_layers)
        ]
        self._generation_step: List[int] = [0] * num_layers
        self._prefill_done: List[bool] = [False] * num_layers
        # Running counter of the next original-sequence window ID to assign
        # when new windows appear (post-eviction or as generation extends the cache).
        # Without this, the W_new > W_old branch would emit compact-space indices
        # that collide with surviving original IDs.
        self._next_original_window_id: List[int] = [0] * num_layers

        # Shared scratch for cache_kwargs communication with hooks
        self.cache_kwargs: Dict[int, Dict[str, Any]] = {
            i: {} for i in range(num_layers)
        }

        # Two-tier quantization state (design.md §4–§6). One int4 store +
        # per-window ledger per layer, from the SHARED modules.quant package.
        # With quant_ratio = 0 these stay empty and every two-tier branch
        # below is skipped — the single-tier path is untouched.
        self._two_tier: bool = (
            self.resolved.quant_ratio > 0.0 and self.resolved.top_q_windows > 0
        )
        self._q_stores: List[QuantizedStore] = [
            QuantizedStore() for _ in range(num_layers)
        ]
        self._q_ledgers: List[QuantLedger] = [
            QuantLedger(self.resolved.window_size) for _ in range(num_layers)
        ]

    # -----------------------------------------------------------------
    # HF Cache interface
    # -----------------------------------------------------------------

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Return current sequence length for *layer_idx*.

        With a live Q tier this is the **effective** length ``T_total``
        (fp + Q tokens) — HF sizes the causal mask from it and the
        position-override hook uses it as ``past_seen``, so reporting the
        fp-only length would desync the query position from the effective
        key count (design §5 step 7).
        """
        n = self._states[layer_idx].seq_length
        if self._two_tier:
            n += self._q_ledgers[layer_idx].active_tokens
        return n

    def get_effective_keys(self, layer_idx: int) -> Optional[Tensor]:
        """Effective key tensor for *layer_idx* — both tiers, merged
        chronologically (design §8, §9).

        The flash score hook sources its auxiliary SDPA keys here: reading the
        raw fp ``key_states`` would miss the Q windows and the interleaving.
        On the single-tier path this is exactly the raw fp keys. The returned
        tensor is freshly built (per-layer transient) — it does not alias the
        stored fp cache when Q windows are active.
        """
        state = self._states[layer_idx]
        if not self._two_tier or state.key_states is None:
            return state.key_states
        k_eff, _ = materialize_effective_kv(
            state,
            self._q_stores[layer_idx],
            self._q_ledgers[layer_idx],
            self.rope_module,
            self.resolved.window_size,
            keys_only=True,
        )
        return k_eff

    def get_max_length(self) -> Optional[int]:
        """Return ``None`` — windowed cache doesn't have a static max."""
        return None

    def update(
        self,
        key_states: Tensor,
        value_states: Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Append new KV states and optionally evict.

        Steps:
        1. ``state.append(k, v, pos)``
        2. Pull pre-computed window scores from *cache_kwargs*.
        3. Accumulate into ``state.window_scores``.
        4. If ``policy.should_evict(step)``:
           a. Two-step retain: window indices → token indices.
           b. Snapshot survivors' original positions, then ``state.slice_and_keep``.
           c. ``state.rerotate_keys`` — strip + re-apply RoPE at contiguous
              positions ``[0..T_retained-1]`` (always; KVPress methodology).
           d. Gather ``state.window_scores`` by retained window indices.
        5. Return ``(state.key_states, state.value_states)``.
        """
        state = self._states[layer_idx]
        policy = self._policies[layer_idx]

        # Extract position_ids from cache_kwargs if provided
        pos = None
        if cache_kwargs is not None and "cache_position" in cache_kwargs:
            pos = cache_kwargs["cache_position"]

        # 1. Append
        state.append(key_states, value_states, pos)
        n_new = key_states.shape[2]
        policy.extend_total_after_append(n_new)

        # Detect prefill vs generation
        is_prefill = not self._prefill_done[layer_idx]
        if is_prefill:
            policy.initialize_after_prefill(state.seq_length)
            self._prefill_done[layer_idx] = True

        # 2. Pull pre-computed window scores
        merged_kwargs = {}
        if cache_kwargs is not None:
            merged_kwargs.update(cache_kwargs)
        merged_kwargs.update(self.cache_kwargs.get(layer_idx, {}))
        new_window_scores = merged_kwargs.get("window_scores")

        # Clear consumed scores so a hook that silently returns on the next step
        # does not cause stale values to be re-accumulated.
        layer_kwargs = self.cache_kwargs.get(layer_idx)
        if layer_kwargs is not None and "window_scores" in layer_kwargs:
            del layer_kwargs["window_scores"]

        # 3. Initialize or accumulate window_scores
        if new_window_scores is not None:
            if state.window_scores is None:
                state.window_scores = new_window_scores.clone()
                # Initialize identity mapping: compact index i == original window i.
                # Stored per row ([B, W]) so divergent per-row eviction keeps each
                # row's surviving window identities independently.
                W = new_window_scores.shape[-1]
                B_w = new_window_scores.shape[0]
                state.original_window_ids = (
                    torch.arange(W, device=new_window_scores.device, dtype=torch.long)
                    .unsqueeze(0)
                    .expand(B_w, -1)
                    .contiguous()
                )
                self._next_original_window_id[layer_idx] = W
                if self._two_tier:
                    # Merged-axis tier map: every window is born fp; windows
                    # only enter the Q tier through eviction-time demotion.
                    state.fp_tier_mask = torch.ones(
                        B_w, W, dtype=torch.bool,
                        device=new_window_scores.device,
                    )
            else:
                # Handle size mismatch: new scores may cover more windows
                W_old = state.window_scores.shape[-1]
                W_new = new_window_scores.shape[-1]
                if W_new > W_old:
                    pad = torch.zeros(
                        state.window_scores.shape[0],
                        state.window_scores.shape[1],
                        W_new - W_old,
                        device=state.window_scores.device,
                        dtype=state.window_scores.dtype,
                    )
                    state.window_scores = torch.cat(
                        [state.window_scores, pad], dim=-1
                    )
                    # Extend original_window_ids for new windows using the
                    # running original-sequence counter, not compact-space
                    # indices (which would collide with surviving IDs).
                    if state.original_window_ids is not None:
                        n_extra = W_new - W_old
                        start_id = self._next_original_window_id[layer_idx]
                        B_w = state.original_window_ids.shape[0]
                        extra = (
                            torch.arange(
                                start_id, start_id + n_extra,
                                device=state.original_window_ids.device,
                                dtype=torch.long,
                            )
                            .unsqueeze(0)
                            .expand(B_w, -1)
                        )
                        state.original_window_ids = torch.cat(
                            [state.original_window_ids, extra], dim=1
                        )
                        self._next_original_window_id[layer_idx] = start_id + n_extra
                        if state.fp_tier_mask is not None:
                            # New windows are always born in the fp tier.
                            state.fp_tier_mask = torch.cat(
                                [
                                    state.fp_tier_mask,
                                    torch.ones(
                                        B_w, n_extra, dtype=torch.bool,
                                        device=state.fp_tier_mask.device,
                                    ),
                                ],
                                dim=1,
                            )
                elif W_new < W_old:
                    # Symmetric pad on the incoming scores so in-place += works
                    # without changing accumulate's contract. No
                    # original_window_ids change: no new windows appeared.
                    pad = torch.zeros(
                        new_window_scores.shape[0],
                        new_window_scores.shape[1],
                        W_old - W_new,
                        device=new_window_scores.device,
                        dtype=new_window_scores.dtype,
                    )
                    new_window_scores = torch.cat(
                        [new_window_scores, pad], dim=-1
                    )
                accumulate(state.window_scores, new_window_scores)

        # 4. Eviction
        step = self._generation_step[layer_idx]
        should_evict = not is_prefill and policy.should_evict(step)

        if should_evict and state.window_scores is not None and self._two_tier:
            # Two-tier eviction cycle (design §5) — spans both tiers jointly.
            self._evict_two_tier(layer_idx, step)
        elif should_evict and state.window_scores is not None:
            B = state.key_states.shape[0]
            H_q = state.window_scores.shape[1]

            # state.window_scores holds per-window Lp POWER-SUMS (Σ A^p)
            # accumulated across prefill + decode. Take the 1/p root now to get
            # the Lp scores used for ranking. p == 1 is a plain sum (identity
            # root), so ranking_scores is state.window_scores unchanged and this
            # path stays byte-identical to the prior behaviour. The stored
            # window_scores stay in power-space (gathered below) so accumulation
            # continues correctly after compaction.
            p = self.resolved.score_p
            ranking_scores = (
                state.window_scores if p == 1.0
                else state.window_scores.pow(1.0 / p)
            )

            # a. Two-step retain
            retained_window_idx = policy.compute_retain_window_indices(
                ranking_scores
            )
            retain_token_idx = policy.expand_to_token_indices(retained_window_idx)

            # Telemetry
            self.telemetry.record_scores(
                layer_idx, step, ranking_scores, retain_token_idx
            )

            # b. Snapshot survivors' original positions before compaction so
            #    rerotate_keys can strip RoPE with the correct (original) angles.
            #    position_ids is [B, T]; gather per row so each row's snapshot
            #    matches the tokens it actually keeps.
            old_positions = torch.gather(
                state.position_ids, 1,
                retain_token_idx.to(state.position_ids.device),
            ).clone()

            # c. Compact K/V (gather survivors contiguous in memory).
            state.slice_and_keep(retain_token_idx)

            # d. Re-rotate surviving keys to contiguous positions
            #    [0..T_retained-1] (KVPress KeyRerotationPress). This is the only
            #    eviction path: keys are rebased AND the query's RoPE position is
            #    overridden to the compacted cache length every step
            #    (install_position_override_hook), so query<->key relative phase
            #    stays exact. Because the override sets the query position
            #    explicitly, this is correct independent of how HF derives
            #    cache_position across transformers versions.
            state.rerotate_keys(self.rope_module, old_positions)

            # e. Gather window_scores by retained_window_idx
            idx_w = retained_window_idx.unsqueeze(1).expand(B, H_q, -1)
            state.window_scores = torch.gather(
                state.window_scores, dim=-1, index=idx_w
            ).contiguous()

            # f. Keep original_window_ids in sync with the surviving windows.
            #    Gather per row ([B, W]) because rows may retain different windows.
            if state.original_window_ids is not None:
                state.original_window_ids = torch.gather(
                    state.original_window_ids, 1,
                    retained_window_idx.to(state.original_window_ids.device),
                ).contiguous()

            # Update policy
            policy.set_total_after_compaction(state.seq_length)

        # Advance generation step (only after prefill is done)
        if not is_prefill:
            self._generation_step[layer_idx] = step + 1

        # 5. Return. With a live Q tier, the return is a freshly-built
        # interleaved effective tensor (fp ‖ dequantized+RoPE'd Q windows in
        # chronological order — design §5, §8), NOT the live fp cache: callers
        # must not assume it aliases the stored key/value states. On the
        # single-tier path (and before the first demotion) it is the live fp
        # cache, exactly as before.
        if self._two_tier:
            return materialize_effective_kv(
                state,
                self._q_stores[layer_idx],
                self._q_ledgers[layer_idx],
                self.rope_module,
                self.resolved.window_size,
            )
        return state.key_states, state.value_states

    # -----------------------------------------------------------------
    # Two-tier eviction cycle (design.md §5)
    # -----------------------------------------------------------------

    def _evict_two_tier(self, layer_idx: int, step: int) -> None:
        """Run one eviction spanning both tiers jointly (design §5 steps 1–7).

        1. Rank all windows on the merged axis by accumulated scores.
        2. Assign tiers: top ``N_fp`` (incl. sink+local) → fp, next ``N_q`` →
           int4, rest dropped.
        3. Move boundary-crossers — demote (un-rotate once, quantize against a
           freshly-pinned fp16 grid, **or reactivate** the dormant ledger
           entry: no re-quantization, ever), promote (dequantize, entry goes
           dormant), drop (free the ledger entry).
        4. Build the interleaved position map over ALL survivors.
        5. Re-rotate the fp tier to its (gappy) slots in that map.
        6. Update each Q window's ledger ``position_range`` — O(Q) int writes.
        7. The query position follows ``get_seq_length() == T_total``.

        B = 1 only in v1 (the read path is a per-row construct — design §10).
        """
        state = self._states[layer_idx]
        policy = self._policies[layer_idx]
        store = self._q_stores[layer_idx]
        ledger = self._q_ledgers[layer_idx]
        S = self.resolved.window_size
        num_sink = self.resolved.num_sink_tokens

        B = state.key_states.shape[0]
        if B != 1:
            raise NotImplementedError(
                "Two-tier quantization (quant_ratio > 0) is batch-size-1 only "
                "in v1; see the ragged-batching design before enabling B > 1."
            )
        H_q = state.window_scores.shape[1]
        T_fp_old = state.seq_length

        # Reconcile the merged-axis bookkeeping with the physical fp store.
        # The score hook scores each token on the NEXT forward pass, so the
        # token(s) appended on the very step that triggers this eviction are
        # already physically in the fp store (they are the newest local tokens)
        # but are not yet represented in window_scores / original_window_ids /
        # fp_tier_mask — the merged axis lags the store by however many windows
        # those unscored tokens opened. Left unreconciled, the physical
        # T_fp_old / tail (which count the untracked window) and the merged
        # fp-window set disagree: build_interleaved_position_map shrinks the
        # last tracked window to `tail` while expand_to_token_indices keeps it
        # full, so the interleaved position map and the retained-token gather
        # diverge and rerotate_keys crashes on the length mismatch. The missing
        # windows are always the newest (highest owid) and always fp, so append
        # them as zero-score fp windows (retained by locality, not by score).
        phys_fp_windows = (
            (T_fp_old - num_sink + S - 1) // S if T_fp_old > num_sink else 0
        )
        n_fp_tracked = (
            int(state.fp_tier_mask.sum().item())
            if state.fp_tier_mask is not None
            else state.window_scores.shape[-1]
        )
        delta = phys_fp_windows - n_fp_tracked
        if delta > 0:
            B_w = state.window_scores.shape[0]
            state.window_scores = torch.cat(
                [
                    state.window_scores,
                    torch.zeros(
                        B_w, H_q, delta,
                        device=state.window_scores.device,
                        dtype=state.window_scores.dtype,
                    ),
                ],
                dim=-1,
            )
            start_id = self._next_original_window_id[layer_idx]
            extra_ids = (
                torch.arange(
                    start_id, start_id + delta,
                    device=state.original_window_ids.device, dtype=torch.long,
                )
                .unsqueeze(0)
                .expand(state.original_window_ids.shape[0], -1)
            )
            state.original_window_ids = torch.cat(
                [state.original_window_ids, extra_ids], dim=1
            )
            self._next_original_window_id[layer_idx] = start_id + delta
            if state.fp_tier_mask is not None:
                state.fp_tier_mask = torch.cat(
                    [
                        state.fp_tier_mask,
                        torch.ones(
                            state.fp_tier_mask.shape[0], delta,
                            dtype=torch.bool, device=state.fp_tier_mask.device,
                        ),
                    ],
                    dim=1,
                )

        # Root the accumulated Lp power-sums for ranking (identical to the
        # single-tier path; stored scores stay in power-space).
        p = self.resolved.score_p
        ranking_scores = (
            state.window_scores if p == 1.0
            else state.window_scores.pow(1.0 / p)
        )

        # 1–2. Rank on the merged axis, split into tiers.
        fp_idx, q_idx = policy.compute_tier_assignments(ranking_scores)

        owids = state.original_window_ids  # [1, W], ascending (chronological)
        W = owids.shape[1]
        device = owids.device
        cur_fp = state.fp_tier_mask
        if cur_fp is None:
            cur_fp = torch.ones(B, W, dtype=torch.bool, device=device)

        fp_sel = torch.zeros(B, W, dtype=torch.bool, device=device)
        fp_sel[0, fp_idx[0]] = True
        q_sel = torch.zeros(B, W, dtype=torch.bool, device=device)
        q_sel[0, q_idx[0]] = True

        stay_fp = cur_fp & fp_sel        # fp windows keeping their tier
        demote = cur_fp & q_sel          # K→Q boundary-crossers
        promote = (~cur_fp) & fp_sel     # Q→K boundary-crossers
        stay_q = (~cur_fp) & q_sel       # Q windows keeping their tier
        dropped = ~(fp_sel | q_sel)

        # Old fp-store window rank of each merged window (valid where cur_fp):
        # the fp windows' relative merged order equals their physical order,
        # so rank is a cumsum over the fp-tier mask (design §5).
        fp_rank = torch.cumsum(cur_fp.long(), dim=1) - 1  # [1, W]

        # ---- 3a. Demotions (K→Q). First-time: un-rotate once + quantize
        # against a freshly-pinned fp16 grid. Seen-before: reactivate the
        # dormant entry — codes + grid reused, zero recompute (design §10).
        demote_pos = demote[0].nonzero().view(-1)
        demote_owids = owids[0, demote_pos].tolist()
        fresh_sel = [i for i, owid in enumerate(demote_owids) if owid not in ledger]
        n_reactivated = len(demote_owids) - len(fresh_sel)
        for owid in demote_owids:
            if owid in ledger:
                ledger.reactivate(owid)
        if fresh_sel:
            fresh_ranks = fp_rank[0, demote_pos[fresh_sel]]     # [n_f]
            offs = torch.arange(S, device=device, dtype=torch.long)
            dem_tok = (
                num_sink + fresh_ranks.unsqueeze(-1) * S + offs
            ).view(-1)                                          # [n_f * S]
            dem_tok_dev = dem_tok.to(state.key_states.device)
            k_dem = state.key_states.index_select(2, dem_tok_dev)
            v_dem = state.value_states.index_select(2, dem_tok_dev)
            dem_old_pos = state.position_ids.index_select(
                1, dem_tok.to(state.position_ids.device)
            )
            k_dem_pre = apply_rope_to_keys(
                self.rope_module, k_dem, dem_old_pos, inverse=True
            )
            n_f = len(fresh_sel)
            H_kv, D = k_dem.shape[1], k_dem.shape[3]
            k_win = k_dem_pre[0].view(H_kv, n_f, S, D).permute(1, 0, 2, 3)
            v_win = v_dem[0].view(H_kv, n_f, S, D).permute(1, 0, 2, 3)
            first_slot = store.append(
                *quantize_key_windows(k_win), *quantize_value_windows(v_win)
            )
            for i, owid in enumerate(
                owids[0, demote_pos[fresh_sel]].tolist()
            ):
                ledger.add(owid, first_slot + i)

        # ---- 3b. Promotions (Q→K): dequantize, keep the entry DORMANT
        # (codes + pinned grid retained for a free future re-demotion).
        prom_pos = promote[0].nonzero().view(-1)
        prom_ids = owids[0, prom_pos]
        n_prom = prom_ids.numel()
        if n_prom > 0:
            prom_slots = torch.tensor(
                [ledger.entries[owid].slot for owid in prom_ids.tolist()],
                dtype=torch.long,
            )
            # Dequantize into each store's own dtype (they can diverge if a
            # non-dtype-faithful rope module promoted the keys).
            k_prom_win = dequantize_key_windows(
                *store.gather_keys(prom_slots),
                out_dtype=state.key_states.dtype,
            )  # [n_prom, H_kv, S, D] — pre-RoPE
            v_prom_win = dequantize_value_windows(
                *store.gather_values(prom_slots),
                out_dtype=state.value_states.dtype,
            )
            for owid in prom_ids.tolist():
                ledger.deactivate(owid)

        # ---- 3c. Drops: free ledger entries (active Q windows dropped, and
        # dormant entries whose fp window is dropped outright), then compact
        # the Q store so it stays gap-free (ledger slots shift with it).
        n_ledger_dropped = 0
        for owid in owids[0, dropped[0].nonzero().view(-1)].tolist():
            if owid in ledger:
                ledger.drop(owid)
                n_ledger_dropped += 1
        if n_ledger_dropped > 0:
            store.compact(ledger.compact_store_slots())

        # ---- 4. Interleaved position map over ALL survivors, both tiers,
        # sorted by original_window_id (design §5 step 4). The trailing local
        # window may be partial; it is always fp and always the newest.
        fp_ids_new = owids[stay_fp | promote].view(1, -1)
        q_ids_new = owids[stay_q | demote].view(1, -1)
        tail = (T_fp_old - num_sink) % S
        fp_positions, q_starts = build_interleaved_position_map(
            fp_ids_new, q_ids_new, S, num_sink,
            fp_tail_len=None if tail == 0 else tail,
        )

        # ---- 6. Ledger position_range update — O(Q_windows) integer writes,
        # no tensor movement, no re-quant (design §5 step 6).
        ledger.set_position_starts(
            q_ids_new[0].tolist(), q_starts[0].tolist()
        )

        # ---- 5. Rebuild the fp store. Gather the tokens of windows STAYING
        # fp (tier-aware expansion: fp-store ranks, explicit fp token cap;
        # Q windows get no token gather — design §5).
        stay_ranks = fp_rank[stay_fp].view(1, -1)
        retain_token_idx = policy.expand_to_token_indices(
            stay_ranks, total_tokens=T_fp_old
        )

        self.telemetry.record_scores(
            layer_idx, step, ranking_scores, retain_token_idx
        )

        old_positions = torch.gather(
            state.position_ids, 1,
            retain_token_idx.to(state.position_ids.device),
        ).clone()
        state.slice_and_keep(retain_token_idx)

        if n_prom == 0:
            # Re-rotate survivors straight to their interleaved (gappy) slots;
            # position_ids bookkeeping becomes those gappy positions.
            state.rerotate_keys(
                self.rope_module, old_positions, new_position_ids=fp_positions
            )
        else:
            # Splice the promoted windows in chronologically: un-rotate the
            # survivors once, merge with the promoted PRE-RoPE windows by
            # original_window_id, then rotate everything at the interleaved fp
            # positions in a single pass.
            k_surv_pre = apply_rope_to_keys(
                self.rope_module, state.key_states, old_positions, inverse=True
            )
            H_kv, D = k_surv_pre.shape[1], k_surv_pre.shape[3]
            k_prom_flat = (
                k_prom_win.permute(1, 0, 2, 3).reshape(1, H_kv, n_prom * S, D)
            )
            v_prom_flat = (
                v_prom_win.permute(1, 0, 2, 3).reshape(1, H_kv, n_prom * S, D)
            )
            # Per-token sort keys: sink first (-1), then window id; stable so
            # within-window token order is preserved.
            stay_ids = owids[stay_fp].view(-1)
            sizes = torch.full(
                (stay_ids.numel(),), S, dtype=torch.long, device=device
            )
            if tail != 0:
                sizes[-1] = tail  # newest window is always fp and last
            tok_ids = torch.cat([
                torch.full((num_sink,), -1, dtype=torch.long, device=device),
                torch.repeat_interleave(stay_ids, sizes),
                torch.repeat_interleave(prom_ids.view(-1), S),
            ])
            order = torch.argsort(tok_ids, stable=True).to(
                state.key_states.device
            )
            k_pre_all = torch.cat([k_surv_pre, k_prom_flat], dim=2
                                  ).index_select(2, order)
            v_all = torch.cat([state.value_states, v_prom_flat], dim=2
                              ).index_select(2, order)
            state.key_states = apply_rope_to_keys(
                self.rope_module, k_pre_all, fp_positions
            )
            state.value_states = v_all.contiguous()
            state.position_ids = fp_positions.contiguous().clone()

        # ---- Merged-axis bookkeeping: scores / ids / tier mask gathered to
        # the survivors of BOTH tiers (ascending = chronological).
        keep_idx = (fp_sel | q_sel)[0].nonzero().view(1, -1)
        idx_w = keep_idx.unsqueeze(1).expand(B, H_q, -1)
        state.window_scores = torch.gather(
            state.window_scores, dim=-1, index=idx_w
        ).contiguous()
        state.original_window_ids = torch.gather(
            owids, 1, keep_idx
        ).contiguous()
        state.fp_tier_mask = torch.gather(fp_sel, 1, keep_idx).contiguous()

        # ---- 7. Effective totals: policy tracks T_total (fp + Q), matching
        # get_seq_length() and the query-position override.
        policy.set_total_after_compaction(
            state.seq_length + ledger.active_tokens
        )

        self.telemetry.record_tier_events(
            layer_idx, step,
            promoted=n_prom,
            demoted=len(demote_owids),
            reactivated=n_reactivated,
            dropped=int(dropped.sum().item()),
            dormant_entries=ledger.dormant_count,
            active_q_windows=ledger.active_count,
        )

    def reorder_cache(self, beam_idx: Tensor) -> None:
        """Beam search is out of scope (v1)."""
        raise NotImplementedError(
            "WindowedCache does not support beam search (reorder_cache). "
            "Use greedy or sampling decoding."
        )
