"""Evaluation metrics — pure tensor ops, fully vectorized.

Provides (Jaccard, Suite A):
- ``jaccard_topk`` — per (step, layer, head) Jaccard similarity.
- ``aggregate_per_layer`` — mean over heads.
- ``aggregate_global`` — mean over heads and layers.
- ``final_step_heterogeneity`` — std across heads at last step.

All functions are fully vectorized. No Python loops over steps/layers/heads.
"""

from __future__ import annotations

import torch
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

    # Mask out -1 sentinel padding so it doesn't pairwise-match itself.
    neg_mask = (ours_exp < 0) | (base_exp < 0)
    # matches[..., i, j] = True iff ours[i] == base[j] (and neither is padding)
    matches = (ours_exp == base_exp) & ~neg_mask  # [S, L, H, K, K]

    # Intersection: count of ours elements that appear in base
    # For each ours element (dim -2), check if any base element matches (dim -1)
    intersection = matches.any(dim=-1).sum(dim=-1).float()  # [S, L, H]

    # Union = |A| + |B| - |A ∩ B|, accounting for sentinel-padded entries
    # which contribute 0 to either set.
    ours_valid = (ours_topk >= 0).sum(dim=-1).float()  # [S, L, H]
    base_valid = (base_topk >= 0).sum(dim=-1).float()  # [S, L, H]
    union = ours_valid + base_valid - intersection  # [S, L, H]

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
    """Std of Jaccard across heads at the last step → ``[num_layers]``.

    Uses population std (``correction=0``) so the result is well-defined even
    when there is only one query head (H=1) — the common case in the
    faithfulness runner where the head axis is a dummy dimension of size 1
    added around the already-head-pooled top-K indices.
    """
    return j[-1].std(dim=-1, correction=0)
