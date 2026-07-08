"""WindowedCacheConfig and ResolvedConfig — typed, validated cache configuration.

``WindowedCacheConfig`` is the user-facing configuration dataclass.
``ResolvedConfig`` is the resolved (frozen) form with concrete integer counts
derived from byte-based budget accounting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Union

import torch


# ---------------------------------------------------------------------------
# ResolvedConfig (frozen, output of resolve())
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResolvedConfig:
    """Resolved cache configuration with concrete integer counts.

    Produced by :meth:`WindowedCacheConfig.resolve`.  All fields are ints
    (or the original ``window_size`` / ``num_sink_tokens``).
    """

    window_size: int
    num_sink_tokens: int
    local_tokens: int          # resolved post percentage-rounding
    top_k_windows: int         # may be 0 (legal — sink + local only)
    bytes_per_token: int
    total_budget_bytes: int
    total_budget_tokens: int
    score_p: float = 1.0    # Lp-norm exponent for query pooling (prefill + decode)
    # -- two-tier quantization (design.md §7); all-zero when the feature is off --
    quant_ratio: float = 0.0   # q — fraction of the MEMORY budget given to int4
    top_q_windows: int = 0     # N_q — int4-tier window capacity (from b_q, not b_fp)
    quant_window_bytes: int = 0  # b_q — bytes per int4 window incl. fp16 scale/zero


# ---------------------------------------------------------------------------
# WindowedCacheConfig (user-facing)
# ---------------------------------------------------------------------------


@dataclass
class WindowedCacheConfig:
    """User-facing configuration for the windowed KV cache.

    Parameters
    ----------
    window_size : int
        Size of each scoring window in tokens.  Must be > 0.
    num_sink_tokens : int
        Number of sink tokens always retained at the start.  Must be >= 0.
    local_window_size : int | float
        If int: number of local tokens (must be a multiple of *window_size*).
        If float: ratio in (0, 1] of the **cache budget** (not the full
        context) -- ``local ~= ratio * total_budget_tokens``, ``ceil`` then
        snap up to the nearest *window_size* multiple.  Guarantees the local
        region can never exceed the cache budget.
    cache_budget : float
        Fraction of full-cache memory to retain, in (0, 1].
        Must be ``float`` — ``int`` and ``bool`` are rejected with clear errors.
    track_scores : bool
        Enable telemetry recording.  Default ``False``.
    score_p : float
        Exponent ``p`` for Lp-norm pooling over query rows: the per-key score
        is ``s_j = (Σ_i A_ij^p)^(1/p)``, accumulated continuously across
        **prefill and decode**.  The cache stores per-window power-sums
        ``Σ_i A_ij^p`` (additive, so accumulation stays a plain ``+=``) and
        takes the ``1/p`` root at eviction time.  ``p = 1`` (default) is the
        plain H2O cumulative sum and is byte-identical to the prior behaviour.
        ``p > 1`` rewards keys attended intensely by a few queries over keys
        attended diffusely by many; because the largest term dominates the
        power-sum, a strong early spike is guarded against dilution by later
        diffuse attention.
    quant_ratio : float
        ``q`` — fraction of the **memory** budget (not the window count) given
        to the int4 Q tier (design.md §7): ``M_fp = (1−q)·M_budget``,
        ``M_q = q·M_budget``, ``N_q = M_q / b_q`` with ``b_q`` the int4 window
        byte cost (packed K+V codes **plus** the fp16 key and value scale/zero
        overhead). ``q = 0`` (default) disables the Q tier entirely — the cache
        is byte-identical to the single-tier behaviour. Sink + local windows
        always stay inside the fp share. Bit-width is fixed at 4 and the quant
        group is the eviction window in v1; scale dtype is fixed fp16 (§2) —
        none of these are knobs. Requires an even ``window_size`` (nibble
        packing pairs 2 tokens per byte).

    Notes
    -----
    Eviction always **compacts and re-rotates** (KVPress ``KeyRerotationPress``
    methodology): surviving keys are gathered contiguous in memory, their RoPE
    rotation is stripped and re-applied at contiguous positions
    ``[0..T_retained-1]``, and the query position is overridden to the compacted
    cache length each step (see ``install_position_override_hook`` in
    ``hooks.py``).  There is no keep-original-positions path.

    Scoring is H2O-style cumulative: every query row contributes to the
    per-key score at every step.  There is no observation window.
    """

    window_size: int
    num_sink_tokens: int
    local_window_size: Union[int, float]
    cache_budget: float
    track_scores: bool = False
    score_p: float = 1.0
    quant_ratio: float = 0.0

    def __post_init__(self) -> None:
        # -- window_size --
        if not isinstance(self.window_size, int) or isinstance(self.window_size, bool):
            raise ValueError(
                f"window_size must be a positive int, got {self.window_size!r}"
            )
        if self.window_size <= 0:
            raise ValueError(
                f"window_size must be > 0, got {self.window_size}"
            )

        # -- num_sink_tokens --
        if not isinstance(self.num_sink_tokens, int) or isinstance(self.num_sink_tokens, bool):
            raise ValueError(
                f"num_sink_tokens must be a non-negative int, got {self.num_sink_tokens!r}"
            )
        if self.num_sink_tokens < 0:
            raise ValueError(
                f"num_sink_tokens must be >= 0, got {self.num_sink_tokens}"
            )

        # -- cache_budget (must be float, not int, not bool) --
        if isinstance(self.cache_budget, bool):
            raise ValueError(
                f"cache_budget must be a float in (0, 1], got bool {self.cache_budget!r}. "
                f"bool is rejected because it subclasses int."
            )
        if isinstance(self.cache_budget, int):
            raise ValueError(
                f"cache_budget must be a float ratio in (0, 1], got int {self.cache_budget}. "
                f"Use e.g. 0.40 instead of 40."
            )
        if not isinstance(self.cache_budget, float):
            raise ValueError(
                f"cache_budget must be a float in (0, 1], got {type(self.cache_budget).__name__}"
            )
        if not (0.0 < self.cache_budget <= 1.0):
            raise ValueError(
                f"cache_budget must be in (0, 1], got {self.cache_budget}"
            )

        # -- local_window_size --
        if isinstance(self.local_window_size, bool):
            raise ValueError("local_window_size must be int or float, got bool")
        if isinstance(self.local_window_size, int):
            if self.local_window_size <= 0:
                raise ValueError(
                    f"local_window_size as int must be > 0, got {self.local_window_size}"
                )
            if self.local_window_size % self.window_size != 0:
                raise ValueError(
                    f"local_window_size as int ({self.local_window_size}) must be a "
                    f"multiple of window_size ({self.window_size})"
                )
        elif isinstance(self.local_window_size, float):
            if not (0.0 < self.local_window_size <= 1.0):
                raise ValueError(
                    f"local_window_size as float must be in (0, 1], "
                    f"got {self.local_window_size}"
                )
        else:
            raise ValueError(
                f"local_window_size must be int or float, "
                f"got {type(self.local_window_size).__name__}"
            )

        # -- score_p (Lp-norm exponent; bool rejected, must be >= 1) --
        if isinstance(self.score_p, bool):
            raise ValueError(
                f"score_p must be a number >= 1, got bool {self.score_p!r}"
            )
        if not isinstance(self.score_p, (int, float)):
            raise ValueError(
                f"score_p must be int or float, got "
                f"{type(self.score_p).__name__}"
            )
        if self.score_p < 1.0:
            raise ValueError(
                f"score_p must be >= 1 (p=1 is plain-sum / H2O), got "
                f"{self.score_p}"
            )
        # Normalize to float so the downstream pow() exponent is unambiguous.
        self.score_p = float(self.score_p)

        # -- quant_ratio (Q-tier memory fraction; bool rejected, [0, 1)) --
        if isinstance(self.quant_ratio, bool):
            raise ValueError(
                f"quant_ratio must be a number in [0, 1), got bool "
                f"{self.quant_ratio!r}"
            )
        if not isinstance(self.quant_ratio, (int, float)):
            raise ValueError(
                f"quant_ratio must be int or float, got "
                f"{type(self.quant_ratio).__name__}"
            )
        if not (0.0 <= self.quant_ratio < 1.0):
            raise ValueError(
                f"quant_ratio must be in [0, 1) — q = 1 leaves no fp budget "
                f"for sink + local — got {self.quant_ratio}"
            )
        self.quant_ratio = float(self.quant_ratio)
        if self.quant_ratio > 0.0 and self.window_size % 2 != 0:
            raise ValueError(
                f"quant_ratio > 0 requires an even window_size (int4 nibble "
                f"packing pairs 2 tokens per byte), got {self.window_size}"
            )

    # -----------------------------------------------------------------
    # resolve() — pure function, no mutation
    # -----------------------------------------------------------------

    def resolve(
        self,
        prefill_len: int,
        model_config: Any,
        kv_dtype: torch.dtype,
        max_tokens: int,
    ) -> ResolvedConfig:
        """Return a :class:`ResolvedConfig` with concrete int counts.

        Pure function; doesn't mutate *self*.  Floor-division on byte→token
        conversion guarantees the retained cache never exceeds the byte budget.

        Parameters
        ----------
        prefill_len : int
            Number of tokens in the prefill (prompt).
        model_config
            HuggingFace ``PretrainedConfig`` (or compatible object) with
            ``num_key_value_heads``, ``num_attention_heads``, ``hidden_size``,
            and optionally ``head_dim``.
        kv_dtype : torch.dtype
            Data type of the KV cache tensors (e.g. ``torch.float16``).
        max_tokens : int
            Maximum number of tokens to be generated.  The budget is sized
            against the full expected sequence (prefill + generation) so the
            cache is not undersized when the output is long.
        """
        num_kv_heads = getattr(
            model_config,
            "num_key_value_heads",
            getattr(model_config, "num_attention_heads", None),
        )
        if num_kv_heads is None:
            raise ValueError(
                "model_config must have num_key_value_heads or num_attention_heads"
            )
        head_dim = getattr(model_config, "head_dim", None)
        if head_dim is None:
            num_heads = getattr(model_config, "num_attention_heads", None)
            hidden = getattr(model_config, "hidden_size", None)
            if num_heads is None or hidden is None:
                raise ValueError(
                    "model_config must provide head_dim or (num_attention_heads + hidden_size)"
                )
            head_dim = hidden // num_heads

        element_size = torch.tensor([], dtype=kv_dtype).element_size()
        # K + V, each shaped [num_kv_heads, head_dim] per token
        bytes_per_token = num_kv_heads * head_dim * element_size * 2

        # Total byte budget and token budget
        total_budget_bytes = int(self.cache_budget * (prefill_len + max_tokens) * bytes_per_token)
        total_budget_tokens = total_budget_bytes // bytes_per_token

        # Two-tier split (design.md §7): the MEMORY budget is divided
        # M_fp = (1−q)·M_budget / M_q = q·M_budget. Sink + local live inside
        # M_fp. With q = 0 the fp share IS the whole budget — every value below
        # reduces to the single-tier arithmetic bit-for-bit.
        q = self.quant_ratio
        fp_budget_bytes = (
            total_budget_bytes if q == 0.0 else int((1.0 - q) * total_budget_bytes)
        )
        q_budget_bytes = total_budget_bytes - fp_budget_bytes
        fp_budget_tokens = fp_budget_bytes // bytes_per_token

        # b_q — bytes per int4 window: packed K+V codes at 2 codes/byte
        # (H_kv·D·S/2 each) plus the pinned fp16 scale/zero pairs — per
        # (head, channel) for keys, per (head, token) for values (§2, §7).
        # The resolver MUST use b_q, not b_fp, for the Q tier: the int4 tier
        # holds ~4× the windows of equal fp memory (minus scale overhead).
        quant_window_bytes = (
            num_kv_heads * head_dim * self.window_size      # packed int4 K + V
            + 4 * num_kv_heads * head_dim                   # key scale+zero, fp16
            + 4 * num_kv_heads * self.window_size           # value scale+zero, fp16
        )
        top_q_windows = 0 if q == 0.0 else q_budget_bytes // quant_window_bytes

        # Resolve local_window_size to concrete int.
        # Float local_window_size is a fraction of the CACHE BUDGET (not the
        # full post-sink context): local ~= ratio * total_budget_tokens, then
        # ceil and snap up to a window_size multiple. This guarantees the local
        # region can never exceed the budget. (The previous post-sink-relative
        # formula could make the local region alone larger than the whole
        # budget, which either crashed resolve() or starved top-K retention.)
        # A float ratio resolves against the FP share of the budget (== the
        # whole budget when q = 0), since sink + local must fit inside M_fp.
        if isinstance(self.local_window_size, float):
            raw = self.local_window_size * fp_budget_tokens
            ceiled = math.ceil(raw)
            remainder = ceiled % self.window_size
            if remainder != 0:
                ceiled += self.window_size - remainder
            local_tokens = ceiled
        else:
            local_tokens = self.local_window_size

        # Top-K evictable fp windows (sink + local stay inside the fp share)
        remaining = fp_budget_tokens - self.num_sink_tokens - local_tokens
        if remaining < 0:
            tier = "total_budget_tokens" if q == 0.0 else "fp share of the budget"
            hint = (
                "Increase cache_budget or reduce sink/local sizes."
                if q == 0.0
                else "Increase cache_budget, lower quant_ratio, or reduce sink/local sizes."
            )
            raise ValueError(
                f"{tier} ({fp_budget_tokens}) < "
                f"num_sink_tokens ({self.num_sink_tokens}) + local_tokens ({local_tokens}). "
                f"{hint}"
            )
        top_k_windows = remaining // self.window_size

        return ResolvedConfig(
            window_size=self.window_size,
            num_sink_tokens=self.num_sink_tokens,
            local_tokens=local_tokens,
            top_k_windows=top_k_windows,
            bytes_per_token=bytes_per_token,
            total_budget_bytes=total_budget_bytes,
            total_budget_tokens=total_budget_tokens,
            score_p=self.score_p,
            quant_ratio=q,
            top_q_windows=top_q_windows,
            quant_window_bytes=quant_window_bytes if q > 0.0 else 0,
        )
