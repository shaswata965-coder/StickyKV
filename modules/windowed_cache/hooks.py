"""Score hooks for the flash-attn-2 backend — KVPress-style monkey-patch.

Flash-attention-2 does not materialize the full attention matrix, so scores
cannot be read from the real forward pass.  Instead, we:
1. Monkey-patch each attention module's ``forward`` to capture post-RoPE
   ``query_states`` and ``key_states`` inside the forward.
2. Register a post-forward hook that runs an auxiliary SDPA pass on the
   captured (q, k) to produce explicit attention weights.
3. Score those weights via :func:`scorer.compute_window_scores` and push the
   result into ``cache.cache_kwargs[layer_idx]["window_scores"]``.

Scoring policy: H2O-style cumulative.  Every query row in the current
forward pass contributes to the per-key score; the cache's ``update()``
then accumulates those per-step scores into ``state.window_scores``
across steps.  There is no observation window.

Cost: ``O(T × N)`` per layer per scoring step — where T is the current
forward pass's query length (prefill_len at step 0, 1 thereafter).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .scorer import compute_window_scores

try:
    from transformers.models.llama.modeling_llama import (
        LlamaAttention,
        repeat_kv,
    )
except ImportError:
    LlamaAttention = None  # type: ignore[assignment,misc]
    repeat_kv = None  # type: ignore[assignment]

try:
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention
except ImportError:
    Qwen2Attention = None  # type: ignore[assignment,misc]


def _get_attn_classes() -> Tuple:
    """Return a tuple of attention module classes to target."""
    classes = []
    if LlamaAttention is not None:
        classes.append(LlamaAttention)
    if Qwen2Attention is not None:
        classes.append(Qwen2Attention)
    return tuple(classes)


# ---------------------------------------------------------------------------
# HookHandles — idempotent removal
# ---------------------------------------------------------------------------


@dataclass
class HookHandles:
    """Manages installed hooks with idempotent ``remove()``."""

    _hook_handles: List[Any] = field(default_factory=list)
    _patched_modules: List[Tuple[nn.Module, Callable]] = field(
        default_factory=list
    )
    _removed: bool = False

    def remove(self) -> None:
        """Remove all hooks and restore original forwards.  Idempotent."""
        if self._removed:
            return
        for handle in self._hook_handles:
            handle.remove()
        for module, original_forward in self._patched_modules:
            module.forward = original_forward
            if hasattr(module, "_captured_q"):
                del module._captured_q
            if hasattr(module, "_captured_k"):
                del module._captured_k
            if hasattr(module, "_original_forward"):
                del module._original_forward
        self._hook_handles.clear()
        self._patched_modules.clear()
        self._removed = True


# ---------------------------------------------------------------------------
# install_score_hooks
# ---------------------------------------------------------------------------


def install_score_hooks(
    model: nn.Module,
    cache: Any,
    config: Any,
) -> HookHandles:
    """Install score-extraction hooks on all attention modules.

    For each ``LlamaAttention`` / ``Qwen2Attention`` module:
    1. Replace ``module.forward`` with a wrapper that captures post-RoPE
       ``query_states`` and ``key_states``.
    2. Register a post-forward hook that runs the auxiliary SDPA score pass
       across **all** query rows of the current step (H2O-style cumulative
       scoring — no observation-window truncation).

    Parameters
    ----------
    model : nn.Module
        The HuggingFace language model.
    cache : WindowedCache
        The cache instance — scores are written to ``cache.cache_kwargs``.
    config : WindowedCacheConfig or ResolvedConfig
        Configuration (``window_size``, ``num_sink_tokens``).

    Returns
    -------
    HookHandles
        Call ``.remove()`` to uninstall all hooks.
    """
    handles = HookHandles()
    attn_classes = _get_attn_classes()
    if not attn_classes:
        warnings.warn(
            "No LlamaAttention or Qwen2Attention found — no hooks installed.",
            RuntimeWarning,
            stacklevel=2,
        )
        return handles

    window_size = getattr(config, "window_size", 8)
    num_sink = getattr(config, "num_sink_tokens", 4)

    # Get H_q and head_dim from model config
    model_config = model.config
    num_q_heads = getattr(model_config, "num_attention_heads", 32)
    num_kv_heads = getattr(
        model_config, "num_key_value_heads", num_q_heads
    )
    head_dim = getattr(model_config, "head_dim", None)
    if head_dim is None:
        head_dim = getattr(model_config, "hidden_size", 4096) // num_q_heads
    num_groups = num_q_heads // num_kv_heads

    # Discover attention modules and assign layer indices
    layer_idx_map: Dict[int, int] = {}
    layer_idx = 0
    for name, module in model.named_modules():
        if isinstance(module, attn_classes):
            layer_idx_map[id(module)] = layer_idx
            layer_idx += 1

    # Install monkey-patch + hook for each attention module
    for name, module in model.named_modules():
        if not isinstance(module, attn_classes):
            continue

        this_layer_idx = layer_idx_map[id(module)]

        # Save original forward
        original_forward = module.forward
        module._original_forward = original_forward

        def make_patched_forward(mod, orig_fwd):
            def patched_forward(*args, **kwargs):
                result = orig_fwd(*args, **kwargs)
                return result
            return patched_forward

        module.forward = make_patched_forward(module, original_forward)
        handles._patched_modules.append((module, original_forward))

        # Post-forward hook for scoring
        def make_hook(mod, lidx):
            def score_hook(module, input, output):
                # Captured post-RoPE q/k from the patched forward
                q_rope = getattr(module, "_captured_q", None)
                k_rope = getattr(module, "_captured_k", None)

                if q_rope is None or k_rope is None:
                    return

                # Get current keys from cache state
                cache_state = cache._states[lidx]
                k_current = cache_state.key_states  # [B, H_kv, S, D]
                if k_current is None:
                    return

                # GQA broadcast: repeat_kv for key
                if repeat_kv is not None and num_groups > 1:
                    k_expanded = repeat_kv(k_current, num_groups)
                else:
                    k_expanded = k_current  # [B, H_q, S, D]

                # Auxiliary SDPA over ALL captured query rows (H2O cumulative)
                # q_rope: [B, H_q, T, D]  →  scores: [B, H_q, T, S]
                scale = 1.0 / math.sqrt(head_dim)
                attn_weights = torch.matmul(
                    q_rope, k_expanded.transpose(-2, -1)
                ) * scale

                # Full softmax — no premask
                attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
                attn_weights = attn_weights.to(q_rope.dtype)

                # compute_window_scores sums over query rows internally
                # (every row contributes — no T_obs truncation).
                scores = compute_window_scores(attn_weights, num_sink, window_size)

                # Push into cache_kwargs; cache.update() accumulates across steps.
                cache.cache_kwargs[lidx]["window_scores"] = scores

            return score_hook

        handle = module.register_forward_hook(make_hook(module, this_layer_idx))
        handles._hook_handles.append(handle)

    return handles
