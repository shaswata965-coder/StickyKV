"""OursParityRunner — windowed cache, either backend (Suite A).

Uses the full windowed cache system from modules.windowed_cache (flash) or
modules.windowed_eager_cache (eager), selected via config.cache.backend_package.
Teacher-forces from the base npz's generated_tokens — never samples.
"""
from __future__ import annotations
import json, math, time
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import torch
from torch import Tensor
from data.corpus_loader import CorpusLoader
from utils.cache_factory import get_cache_classes, validate_backend_attn_pairing
from utils.config import ExperimentConfig, ParityValidationError
from utils.env_capture import capture_environment
from utils.hashing import sha256_string, sha256_tokenizer
from utils.logger import get_logger

log = get_logger(__name__)

def _load_base_npz(path: str) -> dict:
    """Load base npz and extract arrays + metadata."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Base run npz not found: {p}")
    data = np.load(str(p), allow_pickle=True)
    meta_str = str(data["metadata_json"][0])
    meta = json.loads(meta_str)
    arrays = {k: data[k] for k in data.files if k != "metadata_json"}
    return {"arrays": arrays, "metadata": meta}

class OursParityRunner:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        # Fail-fast: validate backend-attn pairing before any I/O or model load.
        validate_backend_attn_pairing(
            config.cache.backend_package, config.model.attn_implementation
        )
        # Resolve the cache class trio once; reuse throughout run().
        (
            self._WindowedCache,
            self._WindowedCacheConfig,
            self._install_score_hooks,
        ) = get_cache_classes(config.cache.backend_package)

    def run(self) -> Path:
        cfg = self.config
        p = cfg.parity
        w = cfg.window
        log.info("=== Ours Parity Runner (backend=%s) ===", cfg.cache.backend_package)
        # 2. Load base npz
        base_path = cfg.base_run_npz
        if not base_path:
            raise ValueError("config.base_run_npz is required for parity_ours mode")
        base = _load_base_npz(base_path)
        base_meta = base["metadata"]
        base_gen_tokens = base["arrays"]["generated_tokens"]
        log.info("Base npz loaded: %d generated tokens", len(base_gen_tokens))
        # 3. Validate identicality
        from utils.config import validate_parity_pair
        validate_parity_pair(base_meta, cfg)
        # Check article_sha at runtime
        loader = CorpusLoader(p.dataset)
        articles = loader.load()
        article_text = articles[p.article_index]
        article_sha = sha256_string(article_text)
        if base_meta.get("article_sha") and article_sha != base_meta["article_sha"]:
            raise ParityValidationError(
                f"Article SHA mismatch: base={base_meta['article_sha']}, ours={article_sha}")
        # 4. Cache classes (resolved in __post_init__)
        WC, WCC, install_hooks = (
            self._WindowedCache,
            self._WindowedCacheConfig,
            self._install_score_hooks,
        )
        # 5. Load model
        from transformers import AutoModelForCausalLM, AutoTokenizer
        dtypes = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
        tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, revision=cfg.model.revision)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tok_sha = sha256_tokenizer(tokenizer)
        if base_meta.get("tokenizer_sha") and tok_sha != base_meta["tokenizer_sha"]:
            raise ParityValidationError(
                f"Tokenizer SHA mismatch: base={base_meta['tokenizer_sha']}, ours={tok_sha}")
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model.name, revision=cfg.model.revision,
            torch_dtype=dtypes.get(cfg.model.dtype, torch.float16),
            attn_implementation=cfg.model.attn_implementation, device_map="auto")
        model.eval()
        # 6. Tokenize prefill
        tokens = tokenizer.encode(article_text, return_tensors="pt",
                                  add_special_tokens=True)[:, :p.prefill_len].to(model.device)
        n_layers = model.config.num_hidden_layers
        # 7. Create cache + hooks
        budget = cfg.cache.cache_budget or 0.5
        cache_config = WCC(
            window_size=w.window_size, num_sink_tokens=w.num_sink_tokens,
            local_window_size=w.local_window_size, cache_budget=budget,
            obs_window=w.obs_window)
        # Get rope module
        rope = None
        for name, mod in model.named_modules():
            if "rotary" in name.lower() or "rope" in name.lower():
                rope = mod; break
        if rope is None:
            for name, mod in model.named_modules():
                if hasattr(mod, "rotary_emb"):
                    rope = mod.rotary_emb; break
        cache = WC(config=cache_config, prefill_len=p.prefill_len,
                    model_config=model.config,
                    kv_dtype=dtypes.get(cfg.model.dtype, torch.float16),
                    rope_module=rope or torch.nn.Identity(),
                    num_layers=n_layers)
        hooks = install_hooks(model, cache, cache_config)
        # 8. Teacher-forced generation
        ns, ws_sz, ow, tk = w.num_sink_tokens, w.window_size, w.obs_window, w.top_k_windows
        acc_scores: List[Optional[Tensor]] = [None] * n_layers
        all_topk, all_ws, all_evict = [], [], []
        gen_kwargs: Dict[str, Any] = {}
        if cfg.cache.backend_package == "eager":
            gen_kwargs["output_attentions"] = True
        t0 = time.time()
        try:
            with torch.no_grad():
                for step in range(p.gen_len):
                    if step == 0:
                        inp = tokens.clone()
                    else:
                        # Teacher-force: use base npz's token
                        forced_tok = int(base_gen_tokens[step - 1])
                        inp = torch.tensor([[forced_tok]], device=model.device)
                    out = model(input_ids=inp, past_key_values=cache, use_cache=True,
                                return_dict=True, **gen_kwargs)
                    # Record eviction
                    evicted = any(cache._generation_step[li] > 0 and
                                  cache._generation_step[li] % ws_sz == 0
                                  for li in range(n_layers))
                    all_evict.append(evicted)
                    # Accumulate attention scores when the model returned them.
                    # Gating on data presence (not backend identity) keeps the
                    # backend abstraction clean: flash returns no attentions.
                    if getattr(out, "attentions", None):
                        for li in range(n_layers):
                            a = out.attentions[li]
                            ts = a[..., -ow:, :].sum(dim=-2)
                            if acc_scores[li] is None:
                                acc_scores[li] = ts.clone()
                            else:
                                oS = acc_scores[li].shape[-1]
                                nS = ts.shape[-1]
                                if nS > oS:
                                    acc_scores[li] = torch.cat([acc_scores[li],
                                        torch.zeros(ts.shape[0], ts.shape[1], nS-oS,
                                                    device=ts.device, dtype=ts.dtype)], -1)
                                elif nS < oS:
                                    ts = torch.nn.functional.pad(ts, (0, oS-nS))
                                acc_scores[li][..., :max(nS,oS)] += ts[..., :max(nS,oS)]
                    # Compute window scores from cache state or accumulated scores
                    step_tk, step_ws = [], []
                    for li in range(n_layers):
                        # Try cache state first
                        cs = cache._states[li]
                        if cs.window_scores is not None:
                            ws_v = cs.window_scores  # [B, H_q, W]
                        elif acc_scores[li] is not None:
                            ac = acc_scores[li]
                            ps = ac[..., ns:]
                            Sp = ps.shape[-1]
                            rem = Sp % ws_sz
                            if rem: ps = torch.nn.functional.pad(ps, (0, ws_sz-rem))
                            W = ps.shape[-1] // ws_sz
                            ws_v = ps.reshape(ps.shape[0], ps.shape[1], W, ws_sz).sum(-1)
                        else:
                            step_tk.append(np.zeros(min(tk, 1), dtype=np.int64))
                            step_ws.append(np.zeros((1, 1), dtype=np.float16))
                            continue
                        W = ws_v.shape[-1]
                        lws = w.local_window_size
                        if isinstance(lws, float):
                            Sp = max(W * ws_sz, 1)
                            lt = math.ceil(lws * Sp)
                            r2 = lt % ws_sz
                            if r2: lt += ws_sz - r2
                            lnw = lt // ws_sz
                        else:
                            lnw = lws // ws_sz
                        lnw = min(lnw, W)
                        eW = W - lnw
                        if eW > 0 and tk > 0:
                            ev = ws_v[..., :eW].mean(dim=1)
                            k = min(tk, eW)
                            step_tk.append(ev.topk(k, dim=-1).indices[0].cpu().numpy())
                        else:
                            step_tk.append(np.zeros(min(tk, max(W,1)), dtype=np.int64))
                        step_ws.append(ws_v[0].cpu().to(torch.float16).numpy())
                    all_topk.append(np.stack(step_tk, 0))
                    all_ws.append(np.stack(step_ws, 0))
                    if (step+1) % 100 == 0: log.info("Step %d/%d", step+1, p.gen_len)
        finally:
            # Always remove hooks, even if generation raises mid-loop.
            hooks.remove()
        elapsed = time.time() - t0
        log.info("Done: %.1fs", elapsed)
        # Pad arrays
        mW = max(x.shape[-1] for x in all_ws)
        mK = max(x.shape[-1] for x in all_topk)
        pws = [np.pad(x, [(0,0),(0,mW-x.shape[-1])]) if x.shape[-1]<mW else x for x in all_ws]
        ptk = [np.pad(x, [(0,0),(0,mK-x.shape[-1])], constant_values=-1) if x.shape[-1]<mK else x for x in all_topk]
        # Fix varying H_q dimension in window scores
        mH = max(x.shape[-2] for x in pws)
        pws2 = []
        for x in pws:
            if x.shape[-2] < mH:
                x = np.pad(x, [(0,0),(0,mH-x.shape[-2]),(0,0)])
            pws2.append(x)
        pws = pws2
        St = p.prefill_len + p.gen_len - ns
        if isinstance(w.local_window_size, float):
            lr = math.ceil(w.local_window_size * St)
            r2 = lr % ws_sz
            if r2: lr += ws_sz - r2
        else:
            lr = w.local_window_size
        env = capture_environment()
        meta = {"schema_version": "1.0", "mode": "parity_ours",
                "seed": cfg.run.seed, "dataset": p.dataset,
                "article_id": p.article_index, "article_sha": article_sha,
                "tokenizer_sha": tok_sha, "prefill_len": p.prefill_len,
                "gen_len": p.gen_len, "window_size": w.window_size,
                "num_sink_tokens": ns, "local_window_size_resolved": lr,
                "obs_window": ow, "top_k_windows": tk,
                "model_name": cfg.model.name, "model_revision": cfg.model.revision,
                "dtype": cfg.model.dtype,
                "attn_implementation": cfg.model.attn_implementation,
                "cache_backend": "windowed",
                "cache_backend_package": cfg.cache.backend_package,
                "cache_budget": budget, **env,
                "run_started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)),
                "run_finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        od = Path(cfg.telemetry.output_dir); od.mkdir(parents=True, exist_ok=True)
        be = cfg.cache.backend_package or "unknown"
        npz = Path(cfg.output_path) if cfg.output_path else od / f"parity_ours_{be}_{p.dataset}_{article_sha[:8]}.npz"
        npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(npz),
            top_window_indices=np.stack(ptk, 0),
            window_scores=np.stack(pws, 0),
            eviction_step_mask=np.array(all_evict, dtype=bool),
            generated_tokens=base_gen_tokens,
            metadata_json=np.array([json.dumps(meta)], dtype=object))
        with open(npz.with_suffix(".meta.json"), "w") as f:
            json.dump(meta, f, indent=2, default=str)
        log.info("Saved: %s", npz)
        return npz
