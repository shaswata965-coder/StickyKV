"""Pure scoring functions for windowed KV cache.

Two functions:
- ``compute_window_scores`` — reduces ``[B, H_q, T, S]`` attention to
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
    p: float = 1.0,
) -> Tensor:
    """Reduce full attention weights to per-window Lp power-sums.

    Algorithm:
    1. Reduce over T query rows → per-key power-sum ``Σ_i A_ij^p`` giving
       ``[B, H_q, S]`` (``p = 1`` is a plain sum). The ``1/p`` root is NOT
       applied here — the cache accumulates these power-sums across prefill and
       decode (they are additive) and roots once at eviction time.
    2. Strip sink prefix (never scored).
    3. Right-pad trailing partial window with zeros.
    4. ``einops.reduce('b h (w s) -> b h w', 'sum')``.

    Parameters
    ----------
    attn : Tensor
        Shape ``[B, H_q, T, S]``, post-softmax attention weights.
    num_sink : int
        Number of sink tokens to strip from the key dimension.
    window_size : int
        Window size for aggregation.
    p : float
        Lp-norm exponent for pooling over the query axis. ``p = 1`` (default)
        is the plain H2O cumulative sum. ``p > 1`` emphasises concentrated
        (peaky) attention. Applies to every pass (prefill and decode); the
        power-sums are additive so decode folds in one query row at a time, and
        the ``1/p`` root is deferred to the cache at eviction time.

    Returns
    -------
    Tensor
        Shape ``[B, H_q, W]`` — per-window power-sums ``Σ_i A_ij^p`` (pre-root).
        Sink tokens are **not** represented.
    """
    # 1. Reduce over T query rows → per-key power-sum [B, H_q, S].
    #    p == 1 is the plain H2O sum. p > 1 accumulates the p-th powers
    #    Σ_i A_ij^p (Lp-norm pooling over the query axis, PRE-root): keys
    #    attended intensely by a few queries dominate keys attended diffusely by
    #    many. The root is NOT taken here — the cache accumulates these
    #    power-sums continuously across prefill and decode (they are additive)
    #    and applies the 1/p root once at eviction time. The power is taken in
    #    fp32 so small softmax probabilities do not underflow before the sum.
    if p != 1.0:
        token_scores = attn.to(torch.float32).pow(p).sum(dim=-2)
    else:
        token_scores = attn.sum(dim=-2)

    # 2-4. Strip sink, pad, window-reduce (sum over the tokens in each window).
    return reduce_token_scores_to_windows(token_scores, num_sink, window_size)


def reduce_token_scores_to_windows(
    token_scores: Tensor,
    num_sink: int,
    window_size: int,
) -> Tensor:
    """Reduce per-token received-attention to per-window scores.

    This is steps 2-4 of :func:`compute_window_scores`, split out so callers
    that build ``token_scores`` incrementally (e.g. the flash hook accumulating
    over query-row chunks to bound peak memory at full context) can reuse the
    exact same sink-strip + pad + window-sum without materializing the full
    ``[B, H, T, S]`` attention matrix.

    Parameters
    ----------
    token_scores : Tensor
        Shape ``[B, H_q, S]`` — per-key total received attention (already
        summed over the query dimension).
    num_sink, window_size : int

    Returns
    -------
    Tensor
        Shape ``[B, H_q, W]``.  Sink tokens are **not** represented.
    """
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
