#!/usr/bin/env python3
"""Standalone demo: greedy generation with StickyKV windowed cache.

Uses the eager-attention backend with:
  - window_size   = 8
  - local_window  = 32  (tokens, must be multiple of window_size)
  - cache_budget  = 0.2 (20% of prefill KV retained)

Requires: transformers, torch, einops, accelerate
Model   : TinyLlama/TinyLlama-1.1B-Chat-v1.0 (fits in ~2 GB)
"""

from __future__ import annotations

import sys, os, time

# ── make project root importable ──────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from modules.windowed_eager_cache import (
    WindowedCache,
    WindowedCacheConfig,
    install_score_hooks,
)

# ── Configuration ─────────────────────────────────────────────────────
MODEL_NAME    = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
PROMPT        = "The future of artificial intelligence is"
MAX_NEW_TOKENS = 128
WINDOW_SIZE   = 8
LOCAL_WINDOW  = 32          # tokens (must be multiple of WINDOW_SIZE)
CACHE_BUDGET  = 0.2         # retain 20% of prefill KV cache
NUM_SINK      = 4           # always-retained leading tokens
DTYPE         = torch.float16
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"


def main() -> None:
    print("=" * 64)
    print("StickyKV  —  Windowed Cache Demo (eager backend)")
    print("=" * 64)
    print(f"  Model         : {MODEL_NAME}")
    print(f"  Device        : {DEVICE}")
    print(f"  Window size   : {WINDOW_SIZE}")
    print(f"  Local window  : {LOCAL_WINDOW} tokens")
    print(f"  Cache budget  : {CACHE_BUDGET:.0%}")
    print(f"  Sink tokens   : {NUM_SINK}")
    print(f"  Max new tokens: {MAX_NEW_TOKENS}")
    print()

    # ── 1. Load model & tokenizer ─────────────────────────────────────
    print("[1/5] Loading tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[2/5] Loading model …")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=DTYPE,
        attn_implementation="eager",
        device_map="auto" if DEVICE == "cuda" else None,
    )
    if DEVICE != "cuda":
        model = model.to(DEVICE)
    model.eval()

    n_layers = model.config.num_hidden_layers
    print(f"       {n_layers} layers, "
          f"{model.config.num_attention_heads} attn heads, "
          f"{model.config.num_key_value_heads} KV heads")

    # ── 2. Tokenize prompt (= prefill) ────────────────────────────────
    input_ids = tokenizer.encode(PROMPT, return_tensors="pt").to(model.device)
    prefill_len = input_ids.shape[1]
    print(f"\n[3/5] Prompt ({prefill_len} tokens): \"{PROMPT}\"")

    # ── 3. Build cache + hooks ────────────────────────────────────────
    print("[4/5] Building windowed cache …")

    # Locate the RoPE module (needed for key rerotation after eviction)
    rope = None
    for name, mod in model.named_modules():
        if "rotary" in name.lower() or "rope" in name.lower():
            rope = mod
            break
    if rope is None:
        for name, mod in model.named_modules():
            if hasattr(mod, "rotary_emb"):
                rope = mod.rotary_emb
                break
    if rope is None:
        rope = torch.nn.Identity()
        print("       ⚠ Could not find RoPE module; using Identity (keys won't be rerotated)")

    cache_config = WindowedCacheConfig(
        window_size=WINDOW_SIZE,
        num_sink_tokens=NUM_SINK,
        local_window_size=LOCAL_WINDOW,
        cache_budget=CACHE_BUDGET,
    )
    cache = WindowedCache(
        config=cache_config,
        prefill_len=prefill_len,
        model_config=model.config,
        kv_dtype=DTYPE,
        rope_module=rope,
        num_layers=n_layers,
    )
    hooks = install_score_hooks(model, cache, cache_config)
    resolved = cache.resolved
    print(f"       Budget    : {resolved.total_budget_tokens} tokens "
          f"({resolved.total_budget_bytes:,} bytes)")
    print(f"       Top-K win : {resolved.top_k_windows}")
    print(f"       Local tok : {resolved.local_tokens}")

    # ── 4. Generate ───────────────────────────────────────────────────
    print(f"\n[5/5] Generating {MAX_NEW_TOKENS} tokens (greedy) …\n")
    generated_ids: list[int] = []
    t0 = time.perf_counter()

    with torch.no_grad():
        for step in range(MAX_NEW_TOKENS):
            if step == 0:
                inp = input_ids
            else:
                inp = torch.tensor([[generated_ids[-1]]], device=model.device)

            outputs = model(
                input_ids=inp,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
                output_attentions=True,   # required for eager score hooks
            )

            # Greedy: pick argmax of last position
            next_token = int(outputs.logits[:, -1, :].argmax(dim=-1).item())
            generated_ids.append(next_token)

            # Stop on EOS
            if next_token == tokenizer.eos_token_id:
                break

    elapsed = time.perf_counter() - t0
    hooks.remove()

    # ── 5. Print results ──────────────────────────────────────────────
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    total_tokens = len(generated_ids)

    print("-" * 64)
    print(f"Prompt : {PROMPT}")
    print(f"Output : {generated_text}")
    print("-" * 64)

    # Cache stats
    cache_seq_len = cache.get_seq_length(0)
    full_seq_len  = prefill_len + total_tokens
    print(f"\nStats:")
    print(f"  Generated tokens  : {total_tokens}")
    print(f"  Time              : {elapsed:.2f}s  ({total_tokens / elapsed:.1f} tok/s)")
    print(f"  Full seq length   : {full_seq_len}")
    print(f"  Cache seq length  : {cache_seq_len}  "
          f"({cache_seq_len / full_seq_len:.1%} of full)")
    print(f"  Eviction steps    : {cache._generation_step[0]}")
    print()


if __name__ == "__main__":
    main()
