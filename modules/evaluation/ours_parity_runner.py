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

        # Global knobs from data config (must match base run for parity).
        num_samples_cfg = max(1, int(getattr(cfg.data, "num_samples", 1)))
        max_tokens = getattr(cfg.data, "max_tokens", None)
        prefill_len = int(max_tokens) if max_tokens else int(p.prefill_len)

        # 2. Load base npz
        base_path = cfg.base_run_npz
        if not base_path:
            raise ValueError("config.base_run_npz is required for parity_ours mode")
        base = _load_base_npz(base_path)
        base_meta = base["metadata"]
        base_gen_tokens = base["arrays"]["generated_tokens"]

        # Detect base NPZ schema: legacy 1D [num_steps] or new 2D [num_samples, num_steps]
        if base_gen_tokens.ndim == 1:
            base_gen_tokens = base_gen_tokens[np.newaxis, :]   # [1, num_steps]
        num_samples_base = base_gen_tokens.shape[0]

        # Read the per-sample shas list (new schema) or fall back to single sha (legacy)
        base_shas = base_meta.get("article_shas")
        if not base_shas:
            base_shas = [base_meta.get("article_sha")]

        # Determine effective num_samples: must match base; warn if config disagrees
        if num_samples_cfg != num_samples_base:
            log.warning(
                "num_samples=%d in config but base npz has %d samples; "
                "using base value (%d) for parity.",
                num_samples_cfg, num_samples_base, num_samples_base,
            )
        num_samples = num_samples_base
        log.info("Base npz loaded: %d sample(s), %d generated tokens each",
                 num_samples, base_gen_tokens.shape[1])

        # 3. Validate identicality
        from utils.config import validate_parity_pair
        validate_parity_pair(base_meta, cfg)

        # 4. Load corpus + validate per-sample article shas
        loader = CorpusLoader(p.dataset)
        articles = loader.load()
        samples_shas: List[str] = []
        for sample_idx in range(num_samples):
            article_idx = p.article_index + sample_idx
            if article_idx >= len(articles):
                raise ParityValidationError(
                    f"article_index+sample {article_idx} out of range "
                    f"({len(articles)} articles)"
                )
            sha = sha256_string(articles[article_idx])
            samples_shas.append(sha)
            base_sha_i = base_shas[sample_idx] if sample_idx < len(base_shas) else None
            if base_sha_i and sha != base_sha_i:
                raise ParityValidationError(
                    f"Sample {sample_idx} article SHA mismatch: "
                    f"base={base_sha_i}, ours={sha}"
                )

        # 5. Cache classes (resolved in __post_init__)
        WC, WCC, install_hooks = (
            self._WindowedCache,
            self._WindowedCacheConfig,
            self._install_score_hooks,
        )

        # 6. Load model + tokenizer (once across all samples)
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
        n_layers = model.config.num_hidden_layers

        # Get rope module once
        rope = None
        for name, mod in model.named_modules():
            if "rotary" in name.lower() or "rope" in name.lower():
                rope = mod; break
        if rope is None:
            for name, mod in model.named_modules():
                if hasattr(mod, "rotary_emb"):
                    rope = mod.rotary_emb; break

        ns, ws_sz, ow, tk = w.num_sink_tokens, w.window_size, w.obs_window, w.top_k_windows
        budget = cfg.cache.cache_budget or 0.5

        # Per-sample storage
        samples_topk: List[np.ndarray] = []
        samples_ws: List[np.ndarray] = []
        samples_evict: List[np.ndarray] = []

        t0 = time.time()

        for sample_idx in range(num_samples):
            article_idx = p.article_index + sample_idx
            article_text = articles[article_idx]
            log.info(
                "── Sample %d/%d (article %d, sha=%s) ──",
                sample_idx + 1, num_samples, article_idx, samples_shas[sample_idx][:8],
            )

            tokens = tokenizer.encode(article_text, return_tensors="pt",
                                      add_special_tokens=True)[:, :prefill_len].to(model.device)

            # Fresh cache + hooks per sample (cache state must reset).
            cache_config = WCC(
                window_size=w.window_size, num_sink_tokens=w.num_sink_tokens,
                local_window_size=w.local_window_size, cache_budget=budget,
                obs_window=w.obs_window)
            cache = WC(config=cache_config, prefill_len=prefill_len,
                       model_config=model.config,
                       kv_dtype=dtypes.get(cfg.model.dtype, torch.float16),
                       rope_module=rope or torch.nn.Identity(),
                       num_layers=n_layers)
            hooks = install_hooks(model, cache, cache_config)

            sample_gen_tokens = base_gen_tokens[sample_idx]   # [num_steps]
            acc_scores: List[Optional[Tensor]] = [None] * n_layers
            all_topk, all_ws, all_evict = [], [], []
            gen_kwargs: Dict[str, Any] = {}
            if cfg.cache.backend_package == "eager":
                gen_kwargs["output_attentions"] = True

            try:
                with torch.no_grad():
                    for step in range(p.gen_len):
                        if step == 0:
                            inp = tokens.clone()
                        else:
                            forced_tok = int(sample_gen_tokens[step - 1])
                            inp = torch.tensor([[forced_tok]], device=model.device)
                        out = model(input_ids=inp, past_key_values=cache, use_cache=True,
                                    return_dict=True, **gen_kwargs)
                        evicted = any(cache._generation_step[li] > 0 and
                                      cache._generation_step[li] % ws_sz == 0
                                      for li in range(n_layers))
                        all_evict.append(evicted)
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
                        step_tk, step_ws = [], []
                        for li in range(n_layers):
                            cs = cache._states[li]
                            if cs.window_scores is not None:
                                ws_v = cs.window_scores
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
                        if (step+1) % 100 == 0:
                            log.info("  Step %d/%d", step+1, p.gen_len)
            finally:
                hooks.remove()

            # Pad per-step arrays within this sample
            mW = max(x.shape[-1] for x in all_ws)
            mK = max(x.shape[-1] for x in all_topk)
            pws = [np.pad(x, [(0,0),(0,mW-x.shape[-1])]) if x.shape[-1]<mW else x for x in all_ws]
            ptk = [np.pad(x, [(0,0),(0,mK-x.shape[-1])], constant_values=-1) if x.shape[-1]<mK else x for x in all_topk]
            mH = max(x.shape[-2] for x in pws)
            pws = [np.pad(x, [(0,0),(0,mH-x.shape[-2]),(0,0)]) if x.shape[-2]<mH else x for x in pws]

            samples_topk.append(np.stack(ptk, 0))
            samples_ws.append(np.stack(pws, 0))
            samples_evict.append(np.array(all_evict, dtype=bool))

            # Memory hygiene: free per-sample cache, hooks, and tensors before next sample.
            del cache, cache_config, acc_scores, all_topk, all_ws, all_evict, tokens
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc as _gc; _gc.collect()

        # Align K, W, H_q across samples (could differ if eviction trajectories diverged)
        max_K = max(x.shape[-1] for x in samples_topk)
        max_W = max(x.shape[-1] for x in samples_ws)
        max_H = max(x.shape[-2] for x in samples_ws)
        aligned_topk, aligned_ws = [], []
        for tkarr, wsarr in zip(samples_topk, samples_ws):
            if tkarr.shape[-1] < max_K:
                tkarr = np.pad(tkarr, [(0,0),(0,0),(0, max_K - tkarr.shape[-1])],
                               constant_values=-1)
            if wsarr.shape[-2] < max_H:
                wsarr = np.pad(wsarr, [(0,0),(0,0),(0, max_H - wsarr.shape[-2]),(0,0)])
            if wsarr.shape[-1] < max_W:
                wsarr = np.pad(wsarr, [(0,0),(0,0),(0,0),(0, max_W - wsarr.shape[-1])])
            aligned_topk.append(tkarr)
            aligned_ws.append(wsarr)

        top_window_indices = np.stack(aligned_topk, 0)
        window_scores = np.stack(aligned_ws, 0)
        eviction_step_mask = np.stack(samples_evict, 0)

        elapsed = time.time() - t0
        log.info("Done: %d samples, %.1fs", num_samples, elapsed)

        St = prefill_len + p.gen_len - ns
        if isinstance(w.local_window_size, float):
            lr = math.ceil(w.local_window_size * St)
            r2 = lr % ws_sz
            if r2: lr += ws_sz - r2
        else:
            lr = w.local_window_size

        env = capture_environment()
        meta = {
            "schema_version": "1.1",                  # bumped: leading sample axis
            "mode": "parity_ours",
            "seed": cfg.run.seed,
            "dataset": p.dataset,
            "article_id": p.article_index,
            "article_index_start": p.article_index,
            "num_samples": num_samples,
            "article_shas": samples_shas,
            "article_sha": samples_shas[0],
            "tokenizer_sha": tok_sha,
            "prefill_len": prefill_len,
            "max_tokens": max_tokens,
            "gen_len": p.gen_len,
            "window_size": w.window_size,
            "num_sink_tokens": ns,
            "local_window_size_resolved": lr,
            "obs_window": ow,
            "top_k_windows": tk,
            "model_name": cfg.model.name,
            "model_revision": cfg.model.revision,
            "dtype": cfg.model.dtype,
            "attn_implementation": cfg.model.attn_implementation,
            "cache_backend": "windowed",
            "cache_backend_package": cfg.cache.backend_package,
            "cache_budget": budget,
            **env,
            "run_started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)),
            "run_finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        od = Path(cfg.telemetry.output_dir); od.mkdir(parents=True, exist_ok=True)
        be = cfg.cache.backend_package or "unknown"
        npz = Path(cfg.output_path) if cfg.output_path else od / f"parity_ours_{be}_{p.dataset}_{samples_shas[0][:8]}.npz"
        npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(npz),
            top_window_indices=top_window_indices,
            window_scores=window_scores,
            eviction_step_mask=eviction_step_mask,
            generated_tokens=base_gen_tokens,           # carry-through from base [num_samples, num_steps]
            metadata_json=np.array([json.dumps(meta)], dtype=object),
        )
        with open(npz.with_suffix(".meta.json"), "w") as f:
            json.dump(meta, f, indent=2, default=str)
        log.info("Saved: %s", npz)
        return npz
