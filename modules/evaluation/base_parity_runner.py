"""BaseParityRunner — vanilla model, no monkey-patch (Suite A).

Runs the stock HF model with DynamicCache and output_attentions=True.
Computes Top-K window rankings from attention matrices at each step.
No hooks, no eviction. Saves generated tokens for ours runner replay.
"""
from __future__ import annotations
import json, math, time
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import torch
from torch import Tensor
from data.corpus_loader import CorpusLoader
from utils.config import ExperimentConfig
from utils.env_capture import capture_environment
from utils.hashing import sha256_string, sha256_tokenizer
from utils.logger import get_logger

log = get_logger(__name__)

class BaseParityRunner:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config

    def run(self) -> Path:
        cfg = self.config
        p = cfg.parity
        w = cfg.window
        log.info("=== Base Parity Runner ===")
        # 1. Load article
        loader = CorpusLoader(p.dataset)
        articles = loader.load()
        article_text = articles[p.article_index]
        article_sha = sha256_string(article_text)
        log.info("Article SHA: %s", article_sha)
        # 2. Load model + tokenizer
        from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
        dtypes = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
        tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, revision=cfg.model.revision)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tok_sha = sha256_tokenizer(tokenizer)
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model.name, revision=cfg.model.revision,
            torch_dtype=dtypes.get(cfg.model.dtype, torch.float16),
            attn_implementation="eager", device_map="auto")
        model.eval()
        # 3. Tokenize
        tokens = tokenizer.encode(article_text, return_tensors="pt",
                                  add_special_tokens=True)[:, :p.prefill_len].to(model.device)
        n_layers = model.config.num_hidden_layers
        acc_scores: List[Optional[Tensor]] = [None] * n_layers
        all_topk, all_ws, gen_toks = [], [], []
        ns, ws_sz, ow, tk = w.num_sink_tokens, w.window_size, w.obs_window, w.top_k_windows
        t0 = time.time()
        input_ids = tokens.clone()
        pkv = DynamicCache()
        with torch.no_grad():
            for step in range(p.gen_len):
                inp = input_ids if step == 0 else next_tok.unsqueeze(0)
                out = model(input_ids=inp, past_key_values=pkv, use_cache=True,
                            output_attentions=True, return_dict=True)
                pkv = out.past_key_values
                next_tok = out.logits[:, -1, :].argmax(dim=-1)
                gen_toks.append(next_tok.item())
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
                        acc_scores[li][..., :nS] += ts
                step_tk, step_ws = [], []
                for li in range(n_layers):
                    ac = acc_scores[li]
                    ps = ac[..., ns:]
                    Sp = ps.shape[-1]
                    rem = Sp % ws_sz
                    if rem: ps = torch.nn.functional.pad(ps, (0, ws_sz-rem))
                    W = ps.shape[-1] // ws_sz
                    ws_v = ps.reshape(ps.shape[0], ps.shape[1], W, ws_sz).sum(-1)
                    lws = w.local_window_size
                    if isinstance(lws, float):
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
                        step_tk.append(np.zeros(min(tk, W), dtype=np.int64))
                    step_ws.append(ws_v[0].cpu().to(torch.float16).numpy())
                all_topk.append(np.stack(step_tk, 0))
                all_ws.append(np.stack(step_ws, 0))
                if (step+1) % 100 == 0: log.info("Step %d/%d", step+1, p.gen_len)
        elapsed = time.time() - t0
        log.info("Done: %.1fs (%.1f tok/s)", elapsed, p.gen_len / max(elapsed, 1e-6))
        # Pad arrays
        mW = max(x.shape[-1] for x in all_ws)
        mK = max(x.shape[-1] for x in all_topk)
        pws = [np.pad(x, [(0,0),(0,0),(0,mW-x.shape[-1])]) if x.shape[-1]<mW else x for x in all_ws]
        ptk = [np.pad(x, [(0,0),(0,mK-x.shape[-1])], constant_values=-1) if x.shape[-1]<mK else x for x in all_topk]
        # Resolve local_window_size
        St = p.prefill_len + p.gen_len - ns
        if isinstance(w.local_window_size, float):
            lr = math.ceil(w.local_window_size * St)
            r2 = lr % ws_sz
            if r2: lr += ws_sz - r2
        else:
            lr = w.local_window_size
        env = capture_environment()
        meta = {"schema_version": "1.0", "mode": "parity_base",
                "seed": cfg.run.seed, "dataset": p.dataset,
                "article_id": p.article_index, "article_sha": article_sha,
                "tokenizer_sha": tok_sha, "prefill_len": p.prefill_len,
                "gen_len": p.gen_len, "window_size": w.window_size,
                "num_sink_tokens": ns, "local_window_size_resolved": lr,
                "obs_window": ow, "top_k_windows": tk,
                "model_name": cfg.model.name, "model_revision": cfg.model.revision,
                "dtype": cfg.model.dtype, "attn_implementation": "eager",
                "cache_backend": "dynamic", "cache_backend_package": None,
                "cache_budget": None, **env,
                "run_started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)),
                "run_finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        od = Path(cfg.telemetry.output_dir); od.mkdir(parents=True, exist_ok=True)
        npz = Path(cfg.output_path) if cfg.output_path else od / f"parity_base_{p.dataset}_{article_sha[:8]}.npz"
        npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(npz),
            top_window_indices=np.stack(ptk, 0),
            window_scores=np.stack(pws, 0),
            eviction_step_mask=np.zeros(p.gen_len, dtype=bool),
            generated_tokens=np.array(gen_toks, dtype=np.int64),
            metadata_json=np.array([json.dumps(meta)], dtype=object))
        with open(npz.with_suffix(".meta.json"), "w") as f:
            json.dump(meta, f, indent=2, default=str)
        log.info("Saved: %s", npz)
        return npz
