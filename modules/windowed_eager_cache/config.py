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
        If float: ratio in (0, 1] of post-sink tokens — ``ceil`` then snap up
        to the nearest *window_size* multiple.
    cache_budget : float
        Fraction of full-cache memory to retain, in (0, 1].
        Must be ``float`` — ``int`` and ``bool`` are rejected with clear errors.
    track_scores : bool
        Enable telemetry recording.  Default ``False``.

    Notes
    -----
    Scoring is H2O-style cumulative: every query row contributes to the
    per-key score at every step.  There is no observation window.
    """

    window_size: int
    num_sink_tokens: int
    local_window_size: Union[int, float]
    cache_budget: float
    track_scores: bool = False

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

    # -----------------------------------------------------------------
    # resolve() — pure function, no mutation
    # -----------------------------------------------------------------

    def resolve(
        self,
        prefill_len: int,
        model_config: Any,
        kv_dtype: torch.dtype,
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
        total_budget_bytes = int(self.cache_budget * prefill_len * bytes_per_token)
        total_budget_tokens = total_budget_bytes // bytes_per_token

        # Resolve local_window_size to concrete int
        post_sink_tokens = prefill_len - self.num_sink_tokens
        if isinstance(self.local_window_size, float):
            raw = self.local_window_size * post_sink_tokens
            ceiled = math.ceil(raw)
            remainder = ceiled % self.window_size
            if remainder != 0:
                ceiled += self.window_size - remainder
            local_tokens = ceiled
        else:
            local_tokens = self.local_window_size

        # Top-K evictable windows
        remaining = total_budget_tokens - self.num_sink_tokens - local_tokens
        if remaining < 0:
            raise ValueError(
                f"total_budget_tokens ({total_budget_tokens}) < "
                f"num_sink_tokens ({self.num_sink_tokens}) + local_tokens ({local_tokens}). "
                f"Increase cache_budget or reduce sink/local sizes."
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
        )
