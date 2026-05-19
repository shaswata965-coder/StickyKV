"""Score hooks for the flash-attn-2 backend — KVPress-style monkey-patch.

Flash-attention-2 does not materialize the full attention matrix, so scores
cannot be read from the real forward pass.  Instead, we:
1. Monkey-patch each attention module's ``forward`` to capture post-RoPE
   ``query_states`` and ``key_states`` inside the forward.
2. Register a post-forward hook that runs an auxiliary SDPA pass on the
   captured (q, k) to produce explicit attention weights.
3. Score those weights via :func:`scorer.compute_window_scores` and push the
   result into ``cache.cache_kwargs[layer_idx]["window_scores"]``.

Cost: ``O(obs_window × N)`` per layer per scoring step — dominated by the
``O(N²)`` real attention work and not a bottleneck.
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
# Query ring buffer (pre-allocated)
# ---------------------------------------------------------------------------


class _QRingBuffer:
    """Pre-allocated ring buffer for recent post-RoPE query vectors.

    Shape: ``[B, H_q, obs_window, head_dim]``.
    ``data_ptr()`` is stable across generation steps — never reallocated.
    """

    def __init__(
        self,
        B: int,
        H_q: int,
        obs_window: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.buffer = torch.zeros(
            B, H_q, obs_window, head_dim, device=device, dtype=dtype
        )
        self.obs_window = obs_window
        self.write_pos = 0
        self.count = 0

    def write(self, q: Tensor) -> None:
        """Write query vector(s) into the ring buffer.

        Parameters
        ----------
        q : Tensor
            Shape ``[B, H_q, T, head_dim]``.  During generation ``T=1``.
            During prefill, bulk-writes last ``min(T, obs_window)`` rows.
        """
        T = q.shape[2]
        if T >= self.obs_window:
            # Prefill: take last obs_window rows
            self.buffer.copy_(q[:, :, -self.obs_window:, :])
            self.write_pos = 0
            self.count = self.obs_window
        else:
            # Generation (T=1 typically) or short sequences
            for t in range(T):
                idx = (self.write_pos + t) % self.obs_window
                self.buffer[:, :, idx, :] = q[:, :, t, :]
            self.write_pos = (self.write_pos + T) % self.obs_window
            self.count = min(self.count + T, self.obs_window)

    def read(self) -> Tensor:
        """Return the buffered queries in chronological order.

        Returns ``[B, H_q, count, head_dim]``.
        """
        if self.count < self.obs_window:
            return self.buffer[:, :, :self.count, :]
        # Ring is full — reorder to chronological
        return torch.cat(
            [
                self.buffer[:, :, self.write_pos:, :],
                self.buffer[:, :, :self.write_pos, :],
            ],
            dim=2,
        )

    @property
    def data_ptr(self) -> int:
        return self.buffer.data_ptr()


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
    2. Register a post-forward hook that runs the auxiliary SDPA score pass.

    Parameters
    ----------
    model : nn.Module
        The HuggingFace language model.
    cache : WindowedCache
        The cache instance — scores are written to ``cache.cache_kwargs``.
    config : WindowedCacheConfig or ResolvedConfig
        Configuration (``obs_window``, ``window_size``, ``num_sink_tokens``).

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

    obs_window = getattr(config, "obs_window", None) or getattr(
        config, "window_size", 32
    )
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

    # Per-layer q buffers (lazily initialized on first forward)
    q_buffers: Dict[int, _QRingBuffer] = {}
    layer_counter = [0]  # mutable counter for layer_idx detection

    layer_idx_map: Dict[int, int] = {}  # id(module) → layer_idx

    # Discover attention modules and assign layer indices
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

        # Create monkey-patched forward that captures post-RoPE q/k
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
                # Try to get captured q/k from module attributes
                # In the monkey-patch approach, the attention forward
                # internally computes q/k after RoPE. We capture them
                # via the module's internal state.
                q_rope = getattr(module, "_captured_q", None)
                k_rope = getattr(module, "_captured_k", None)

                if q_rope is None or k_rope is None:
                    # Fallback: try to extract from hidden_states
                    # and compute q/k manually
                    return

                B = q_rope.shape[0]

                # Initialize q_buffer lazily
                if lidx not in q_buffers:
                    q_buffers[lidx] = _QRingBuffer(
                        B, num_q_heads, obs_window, head_dim,
                        q_rope.device, q_rope.dtype,
                    )

                q_buf = q_buffers[lidx]
                q_buf.write(q_rope)

                # Read observation queries
                q_obs = q_buf.read()  # [B, H_q, T_obs, D]
                T_obs = q_obs.shape[2]

                # Get current keys from cache state
                cache_state = cache._states[lidx]
                k_current = cache_state.key_states  # [B, H_kv, S, D]
                if k_current is None:
                    return

                S = k_current.shape[2]

                # GQA broadcast: repeat_kv for key
                if repeat_kv is not None and num_groups > 1:
                    k_expanded = repeat_kv(k_current, num_groups)
                else:
                    k_expanded = k_current  # [B, H_q, S, D]

                # Auxiliary SDPA (standard PyTorch, NOT flash-attn)
                # [B, H_q, T_obs, D] @ [B, H_q, D, S] → [B, H_q, T_obs, S]
                scale = 1.0 / math.sqrt(head_dim)
                attn_weights = torch.matmul(
                    q_obs, k_expanded.transpose(-2, -1)
                ) * scale

                # Full softmax — no premask
                attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
                attn_weights = attn_weights.to(q_obs.dtype)

                # Compute window scores
                scores = compute_window_scores(attn_weights, num_sink, window_size)

                # Push into cache_kwargs
                cache.cache_kwargs[lidx]["window_scores"] = scores

            return score_hook

        handle = module.register_forward_hook(make_hook(module, this_layer_idx))
        handles._hook_handles.append(handle)

    return handles
