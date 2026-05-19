"""Evaluation metrics — pure tensor ops, fully vectorized.

Provides:

**Jaccard (Suite A):**
- ``jaccard_topk`` — per (step, layer, head) Jaccard similarity.
- ``aggregate_per_layer`` — mean over heads.
- ``aggregate_global`` — mean over heads and layers.
- ``final_step_heterogeneity`` — std across heads at last step.

**Faithfulness (Suite B):**
- ``lir`` — Layer Information Retention.
- ``missed_mass`` — 1 - LIR.
- ``kl_inverse`` — inverse KL divergence on retained positions.
- ``global_lir`` — mean LIR across layers and heads.

All functions are fully vectorized. No Python loops over steps/layers/heads.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Suite A — Jaccard
# ---------------------------------------------------------------------------


def jaccard_topk(ours_topk: Tensor, base_topk: Tensor) -> Tensor:
    """Compute per-(step, layer, head) Jaccard similarity of Top-K sets.

    Both inputs: ``[num_steps, num_layers, H_q, top_k]``
    Returns:     ``[num_steps, num_layers, H_q]``

    Fully vectorized — uses set intersection/union via broadcasting.
    """
    assert ours_topk.shape == base_topk.shape, (
        f"Shape mismatch: ours={ours_topk.shape}, base={base_topk.shape}"
    )
    # Expand for pairwise comparison:
    # ours: [S, L, H, K, 1], base: [S, L, H, 1, K]
    ours_exp = ours_topk.unsqueeze(-1)  # [S, L, H, K, 1]
    base_exp = base_topk.unsqueeze(-2)  # [S, L, H, 1, K]

    # matches[..., i, j] = True iff ours[i] == base[j]
    matches = ours_exp == base_exp  # [S, L, H, K, K]

    # Intersection: count of ours elements that appear in base
    # For each ours element (dim -2), check if any base element matches (dim -1)
    intersection = matches.any(dim=-1).sum(dim=-1).float()  # [S, L, H]

    # Union = |A| + |B| - |A ∩ B|
    K = ours_topk.shape[-1]
    union = 2.0 * K - intersection  # [S, L, H]

    # Avoid division by zero (both sets empty — shouldn't happen with K > 0)
    jaccard = torch.where(
        union > 0,
        intersection / union,
        torch.ones_like(intersection),
    )
    return jaccard


def aggregate_per_layer(j: Tensor) -> Tensor:
    """Mean Jaccard over heads → ``[num_steps, num_layers]``."""
    return j.mean(dim=-1)


def aggregate_global(j: Tensor) -> Tensor:
    """Mean Jaccard over heads and layers → ``[num_steps]``."""
    return j.mean(dim=(-2, -1))


def final_step_heterogeneity(j: Tensor) -> Tensor:
    """Std of Jaccard across heads at the last step → ``[num_layers]``."""
    return j[-1].std(dim=-1)


# ---------------------------------------------------------------------------
# Suite B — Faithfulness
# ---------------------------------------------------------------------------


def lir(full_attn: Tensor, retained_positions: Tensor) -> Tensor:
    """Layer Information Retention.

    Parameters
    ----------
    full_attn : Tensor
        Shape ``[num_steps, num_layers, H_q, max_cache_len]``.
        Full-cache attention probabilities from the base run.
    retained_positions : Tensor
        Shape ``[num_steps, num_layers, max_retained]``.
        Indices of retained cache positions in the ours run.
        Padded with -1 for variable-length retain sets.

    Returns
    -------
    Tensor
        Shape ``[num_steps, num_layers, H_q]``.
        LIR ∈ [0, 1]. Higher = better retention.
    """
    S, L, H, C = full_attn.shape
    _, _, R = retained_positions.shape

    # Mask out padding (-1 entries)
    valid_mask = retained_positions >= 0  # [S, L, R]

    # Clamp positions to valid range for gather (padding → 0, masked out later)
    pos_clamped = retained_positions.clamp(min=0)  # [S, L, R]

    # Expand positions for gather across heads:
    # full_attn: [S, L, H, C] — gather on dim=-1
    # pos_clamped: [S, L, R] → [S, L, H, R]
    pos_expanded = pos_clamped.unsqueeze(2).expand(S, L, H, R)  # [S, L, H, R]

    # Gather attention at retained positions
    gathered = torch.gather(full_attn, dim=-1, index=pos_expanded)  # [S, L, H, R]

    # Zero out padded positions
    valid_expanded = valid_mask.unsqueeze(2).expand(S, L, H, R)  # [S, L, H, R]
    gathered = gathered * valid_expanded.float()

    # Sum over retained positions → LIR
    lir_val = gathered.sum(dim=-1)  # [S, L, H]
    return lir_val


def missed_mass(full_attn: Tensor, retained_positions: Tensor) -> Tensor:
    """Missed mass = 1 - LIR.

    Parameters
    ----------
    full_attn, retained_positions : see :func:`lir`.

    Returns
    -------
    Tensor
        Shape ``[num_steps, num_layers, H_q]``.
    """
    return 1.0 - lir(full_attn, retained_positions)


def kl_inverse(
    full_attn: Tensor,
    ours_attn: Tensor,
    retained_positions: Tensor,
    eps: float = 1e-9,
) -> Tensor:
    """Inverse KL divergence: KL(ours || base_restricted).

    Parameters
    ----------
    full_attn : Tensor
        Shape ``[num_steps, num_layers, H_q, max_cache_len]``.
    ours_attn : Tensor
        Shape ``[num_steps, num_layers, H_q, max_retained]``.
        Attention from ours run over retained positions.
    retained_positions : Tensor
        Shape ``[num_steps, num_layers, max_retained]``.
    eps : float
        Epsilon for numerical stability.

    Returns
    -------
    Tensor
        Shape ``[num_steps, num_layers, H_q]``.
    """
    S, L, H, C = full_attn.shape
    _, _, R = retained_positions.shape

    valid_mask = retained_positions >= 0  # [S, L, R]
    pos_clamped = retained_positions.clamp(min=0)

    # Gather base attention at retained positions
    pos_expanded = pos_clamped.unsqueeze(2).expand(S, L, H, R)
    base_at_retained = torch.gather(full_attn, dim=-1, index=pos_expanded)  # [S, L, H, R]

    # Zero out padding
    valid_expanded = valid_mask.unsqueeze(2).expand(S, L, H, R).float()
    base_at_retained = base_at_retained * valid_expanded

    # Renormalize base over retained positions
    base_sum = base_at_retained.sum(dim=-1, keepdim=True).clamp(min=eps)  # [S, L, H, 1]
    base_restricted = base_at_retained / base_sum  # [S, L, H, R]

    # Clamp both distributions for numerical stability
    p = base_restricted.clamp(min=eps)  # target distribution
    q = ours_attn.clamp(min=eps)  # predicted distribution

    # Ensure ours_attn is normalized
    q = q / q.sum(dim=-1, keepdim=True).clamp(min=eps)

    # KL(q || p) = sum q * log(q / p)
    log_q = torch.log(q)
    kl = F.kl_div(log_q, p, reduction="none", log_target=False).sum(dim=-1)  # [S, L, H]

    return kl


def global_lir(per_head_lir: Tensor) -> Tensor:
    """Global LIR = mean over layers and heads.

    Parameters
    ----------
    per_head_lir : Tensor
        Shape ``[num_steps, num_layers, H_q]``.

    Returns
    -------
    Tensor
        Shape ``[num_steps]``.
    """
    return per_head_lir.mean(dim=(-2, -1))
