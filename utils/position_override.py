"""Query-position override hook — KVPress-style compacted-length positioning.

The windowed cache compacts **and re-rotates** surviving keys to contiguous
positions ``[0..N_survivor-1]`` on every eviction (KVPress ``KeyRerotationPress``
methodology).  For the query<->key relative RoPE phase to stay correct, the
query at each step must sit at the **compacted** cache length, not its original
absolute position.

HuggingFace ``generate`` passes the base model an explicit *monotonic*
``position_ids`` (derived from the growing attention mask), so just re-rotating
keys and reporting a compacted ``get_seq_length()`` is **not** enough — the
query position has to be actively overridden.  This forward pre-hook does that,
mirroring KVPress's pipeline, which sets ``context_length = cache.get_seq_length()``
and positions the query at ``arange(context_length, context_length + q_len)``.

Because the position is set *explicitly* (rather than relying on how HF derives
``cache_position``), this is correct independent of the transformers version.

Scope: this is a **B=1** construct (matches KVPress, LongBench and the parity
runners, which are all batch-size 1).  B>1 left-padded batching cannot be
represented by the single nulled 2D mask used here once per-row eviction
diverges — see the ragged-batching design before enabling the override for B>1.
"""

from __future__ import annotations

from typing import Any

import torch


def install_position_override_hook(model: Any, cache: Any, handles: Any) -> None:
    """Register a forward pre-hook that forces the query position to follow the
    *compacted* cache length, and append its handle to *handles*.

    Parameters
    ----------
    model : nn.Module
        The HF causal-LM.  Its decoder (``model.get_decoder()``, falling back to
        ``model.model`` / ``model``) is the module whose forward consumes
        ``position_ids`` / ``cache_position`` / ``attention_mask`` and builds the
        causal mask.
    cache : WindowedCache
        The windowed cache; ``get_seq_length(0)`` gives the compacted length.
    handles : HookHandles
        The hook registry whose ``_hook_handles`` list the new handle is appended
        to, so it is removed together with the score hooks.
    """
    if hasattr(model, "get_decoder"):
        decoder = model.get_decoder()
    else:
        decoder = getattr(model, "model", model)

    def pre_hook(module, args, kwargs):
        # Query length + device from input_ids (preferred) or inputs_embeds.
        input_ids = kwargs.get("input_ids")
        if input_ids is None and len(args) > 0 and torch.is_tensor(args[0]):
            input_ids = args[0]
        inputs_embeds = kwargs.get("inputs_embeds")
        if input_ids is not None:
            q_len = input_ids.shape[1]
            device = input_ids.device
        elif inputs_embeds is not None:
            q_len = inputs_embeds.shape[1]
            device = inputs_embeds.device
        else:
            return None  # nothing to anchor on; leave the call untouched

        # past_seen is the *compacted* length after the previous step's eviction.
        past_seen = cache.get_seq_length(0)
        new_cp = torch.arange(
            past_seen, past_seen + q_len, device=device, dtype=torch.long
        )

        # Override both: position_ids drives RoPE (query + the score hook's
        # recomputed query), cache_position drives the causal mask and the
        # position the newly-appended key is stored at.
        kwargs["cache_position"] = new_cp
        kwargs["position_ids"] = new_cp.unsqueeze(0)

        # generate grows a 2D attention_mask to the *original* (monotonic) length;
        # after an eviction that disagrees with the compacted cache length. For
        # B=1 drop it so the model builds a pure causal mask from the compacted
        # length. (Prefill and the pre-eviction decode steps still match, so the
        # mask is left intact there.)
        am = kwargs.get("attention_mask")
        if (
            am is not None
            and torch.is_tensor(am)
            and am.dim() == 2
            and am.shape[0] == 1
            and am.shape[-1] != past_seen + q_len
        ):
            kwargs["attention_mask"] = None

        return args, kwargs

    handle = decoder.register_forward_pre_hook(pre_hook, with_kwargs=True)
    handles._hook_handles.append(handle)
