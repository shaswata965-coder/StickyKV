"""Pure scoring functions for windowed KV cache.

Two functions:
- ``compute_window_scores`` — reduces ``[B, H_q, T_obs, S]`` attention to
  ``[B, H_q, W]`` per-window scores.
- ``accumulate`` — in-place ``+=`` wrapper for unit testability.
"""

from __future__ import annotations

import torch
from torch import Tensor

from einops import reduce


def compute_window_scores(
    attn: Tensor,
    num_sink: int,
    window_size: int,
) -> Tensor:
    """Reduce full attention weights to per-window scores.

    Algorithm:
    1. Sum over T_obs query rows → per-token received attention ``[B, H_q, S]``.
    2. Strip sink prefix (never scored).
    3. Right-pad trailing partial window with zeros.
    4. ``einops.reduce('b h (w s) -> b h w', 'sum')``.

    Parameters
    ----------
    attn : Tensor
        Shape ``[B, H_q, T_obs, S]``, post-softmax attention weights.
    num_sink : int
        Number of sink tokens to strip from the key dimension.
    window_size : int
        Window size for aggregation.

    Returns
    -------
    Tensor
        Shape ``[B, H_q, W]``.  Sink tokens are **not** represented.
    """
    # 1. Sum over T_obs query rows → per-token scores [B, H_q, S]
    token_scores = attn.sum(dim=-2)

    # 2. Strip sink prefix
    post_sink = token_scores[..., num_sink:]  # [B, H_q, S_post]

    # 3. Right-pad to make divisible by window_size
    s_post = post_sink.shape[-1]
    remainder = s_post % window_size
    if remainder != 0:
        pad_size = window_size - remainder
        post_sink = torch.nn.functional.pad(post_sink, (0, pad_size), value=0.0)

    # 4. einops.reduce to window scores
    window_scores = reduce(
        post_sink, "b h (w s) -> b h w", "sum", s=window_size
    )
    return window_scores


def accumulate(state_scores: Tensor, new_scores: Tensor) -> Tensor:
    """Accumulate new window scores into existing state scores (in-place +=).

    Parameters
    ----------
    state_scores : Tensor
        Shape ``[B, H_q, W]``, running cumulative scores.
    new_scores : Tensor
        Shape ``[B, H_q, W]``, scores from the latest step.

    Returns
    -------
    Tensor
        The mutated *state_scores* tensor (same storage).
    """
    state_scores += new_scores
    return state_scores
