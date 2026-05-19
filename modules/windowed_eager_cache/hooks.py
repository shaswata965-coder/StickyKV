"""Score hooks for the eager-attention backend — plain ``forward_hook``.

When ``attn_implementation="eager"``, HF's eager attention materializes the
full softmax-attention tensor and returns it via the module output tuple
(gated by ``output_attentions=True``).  A plain ``register_forward_hook``
reads it directly — no monkey-patch, no captured q/k, no auxiliary pass.

Runner contract: the runner **must** pass ``output_attentions=True`` to
``model.generate(...)`` / ``model.forward(...)``.  Without it, HF returns
``None`` for attn_weights and the hook warns once and skips.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .scorer import compute_window_scores

try:
    from transformers.models.llama.modeling_llama import LlamaAttention
except ImportError:
    LlamaAttention = None  # type: ignore[assignment,misc]

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
# Attention-row ring buffer (pre-allocated, with post-eviction reallocation)
# ---------------------------------------------------------------------------


class _AttnRingBuffer:
    """Pre-allocated ring buffer for recent attention rows.

    Shape: ``[B, H_q, obs_window, current_cache_len]``.

    Unlike the flash backend's q-buffer, the last dim scales with cache length
    and must be reallocated after eviction.

    Memory at max_cache_len=7500, H_q=32, obs_window=32, fp16:
    ``1 × 32 × 32 × 7500 × 2 ≈ 15 MB`` per layer.
    """

    def __init__(
        self,
        B: int,
        H_q: int,
        obs_window: int,
        cache_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.B = B
        self.H_q = H_q
        self.obs_window = obs_window
        self.device = device
        self.dtype = dtype
        self.buffer = torch.zeros(
            B, H_q, obs_window, cache_len, device=device, dtype=dtype
        )
        self.write_pos = 0
        self.count = 0

    def write(self, attn_row: Tensor) -> None:
        """Write attention row(s) into the ring buffer.

        Parameters
        ----------
        attn_row : Tensor
            Shape ``[B, H_q, T, S]``.
            During generation ``T=1``; during prefill ``T=N``.
        """
        T = attn_row.shape[2]
        S = attn_row.shape[3]
        buf_S = self.buffer.shape[3]

        # Handle cache length growth: pad buffer if needed
        if S > buf_S:
            new_buf = torch.zeros(
                self.B, self.H_q, self.obs_window, S,
                device=self.device, dtype=self.dtype,
            )
            new_buf[:, :, :, :buf_S] = self.buffer
            self.buffer = new_buf

        if T >= self.obs_window:
            # Prefill: take last obs_window rows
            self.buffer[:, :, :, :S] = attn_row[:, :, -self.obs_window:, :]
            self.write_pos = 0
            self.count = self.obs_window
        else:
            # Generation: write T rows at ring positions
            for t in range(T):
                idx = (self.write_pos + t) % self.obs_window
                self.buffer[:, :, idx, :S] = attn_row[:, :, t, :S]
            self.write_pos = (self.write_pos + T) % self.obs_window
            self.count = min(self.count + T, self.obs_window)

    def read(self) -> Tensor:
        """Return buffered attention rows in chronological order.

        Returns ``[B, H_q, count, S]``.
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

    def _reallocate(self, new_cache_len: int) -> None:
        """Reallocate buffer for a new cache length (after eviction).

        Old contents are discarded — column indices no longer reference the
        same physical tokens after compaction.
        """
        self.buffer = torch.zeros(
            self.B, self.H_q, self.obs_window, new_cache_len,
            device=self.device, dtype=self.dtype,
        )
        self.write_pos = 0
        self.count = 0

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
    _removed: bool = False

    def remove(self) -> None:
        """Remove all hooks.  Idempotent."""
        if self._removed:
            return
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
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

    For each ``LlamaAttention`` / ``Qwen2Attention`` module, registers a
    ``forward_hook`` that reads ``attn_weights`` from the module output tuple
    (requires ``output_attentions=True``).

    Parameters
    ----------
    model : nn.Module
        The HuggingFace language model.
    cache : WindowedCache
        The cache instance — scores are written to ``cache.cache_kwargs``.
    config : WindowedCacheConfig or ResolvedConfig
        Configuration.

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

    model_config = model.config
    num_q_heads = getattr(model_config, "num_attention_heads", 32)

    # Per-layer attn buffers and state
    attn_buffers: Dict[int, _AttnRingBuffer] = {}
    warned_once = [False]

    # Discover attention modules
    layer_idx_map: Dict[int, int] = {}
    layer_idx = 0
    for name, module in model.named_modules():
        if isinstance(module, attn_classes):
            layer_idx_map[id(module)] = layer_idx
            layer_idx += 1

    # Track cache lengths for reallocation detection
    prev_cache_lens: Dict[int, int] = {}

    for name, module in model.named_modules():
        if not isinstance(module, attn_classes):
            continue

        this_layer_idx = layer_idx_map[id(module)]

        def make_hook(lidx):
            def score_hook(module, input, output):
                # output = (hidden_states, attn_weights, past_key_value)
                # when output_attentions=True
                if not isinstance(output, tuple) or len(output) < 2:
                    if not warned_once[0]:
                        warnings.warn(
                            "Eager hook: output is not a tuple with attn_weights. "
                            "Ensure output_attentions=True is passed to model.forward().",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                        warned_once[0] = True
                    return

                attn_weights = output[1]

                if attn_weights is None:
                    if not warned_once[0]:
                        warnings.warn(
                            "Eager hook: attn_weights is None. "
                            "Ensure output_attentions=True is passed to model.generate(). "
                            "Without attention weights, scoring is disabled and eviction "
                            "degrades to sink+local only.",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                        warned_once[0] = True
                    return

                # attn_weights: [B, H_q, T, S]
                B = attn_weights.shape[0]
                H_q = attn_weights.shape[1]
                T = attn_weights.shape[2]
                S = attn_weights.shape[3]

                # Initialize or reallocate buffer
                if lidx not in attn_buffers:
                    attn_buffers[lidx] = _AttnRingBuffer(
                        B, H_q, obs_window, S,
                        attn_weights.device, attn_weights.dtype,
                    )
                    prev_cache_lens[lidx] = S
                else:
                    # Check for cache length change (post-eviction reallocation)
                    if S != prev_cache_lens.get(lidx, S):
                        attn_buffers[lidx]._reallocate(S)
                        prev_cache_lens[lidx] = S

                buf = attn_buffers[lidx]

                # Write attention rows
                if T > 1:
                    # Prefill: bulk-write last min(T, obs_window) rows
                    rows_to_write = attn_weights[:, :, -min(T, obs_window):, :]
                    buf.write(rows_to_write)
                else:
                    # Generation: single row
                    buf.write(attn_weights)

                # Update prev cache len
                prev_cache_lens[lidx] = S

                # Read observation window and compute scores
                obs = buf.read()  # [B, H_q, T_obs, S]
                scores = compute_window_scores(obs, num_sink, window_size)

                # Push into cache_kwargs
                cache.cache_kwargs[lidx]["window_scores"] = scores

            return score_hook

        handle = module.register_forward_hook(make_hook(this_layer_idx))
        handles._hook_handles.append(handle)

    return handles
