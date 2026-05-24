"""FaithfulnessRunner — pure post-processing over parity npzs (Suite B).

Reads both base and ours npz files.  For every (step, layer) computes five
distribution-comparison metrics between ours' score vector and base's score
vector over the *retained* window set:

    cos_sim      — cosine similarity             ∈ [-1, 1]  (higher = better)
    pearson      — Pearson correlation           ∈ [-1, 1]  (higher = better)
    spearman     — Spearman rank correlation     ∈ [-1, 1]  (higher = better)
    kl_ours_base — KL divergence KL(ours ‖ base) ≥ 0        (lower  = better)
    mass_ratio   — base_mass / ours_mass                    (≈1 = well-matched)

No model loaded — pure tensor ops.
"""
from __future__ import annotations
import json, math, time
from pathlib import Path
from typing import Any, Dict
import numpy as np
import torch
import torch.nn.functional as F
from utils.config import ExperimentConfig, ParityValidationError
from utils.hashing import sha256_file
from utils.logger import get_logger
from utils import metrics as M

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Cosine similarity between two 1-D vectors."""
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=1).clamp(-1, 1).squeeze()


def _pearson(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Pearson correlation between two 1-D vectors."""
    a_c = a - a.mean()
    b_c = b - b.mean()
    return (a_c * b_c).sum() / (a_c.norm() * b_c.norm()).clamp(min=eps)


def _spearman(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Spearman rank correlation (Pearson on ranks)."""
    a_rank = a.argsort().argsort().float()
    b_rank = b.argsort().argsort().float()
    return _pearson(a_rank, b_rank)


def _kl(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """KL(P ‖ Q) from non-negative score vectors P, Q (normalised internally)."""
    p = p.clamp(min=0)
    q = q.clamp(min=0)
    n = p.shape[0]
    p_prob = (p + eps) / (p.sum() + eps * n)
    q_prob = (q + eps) / (q.sum() + eps * n)
    return (p_prob * (p_prob.log() - q_prob.log())).sum().clamp(min=0)

# ---------------------------------------------------------------------------
# NPZ loader
# ---------------------------------------------------------------------------

def _load_npz(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"NPZ not found: {p}")
    data = np.load(str(p), allow_pickle=True)
    meta_str = str(data["metadata_json"][0])
    meta = json.loads(meta_str)
    arrays = {k: data[k] for k in data.files if k != "metadata_json"}
    return {"arrays": arrays, "metadata": meta, "path": str(p)}

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class FaithfulnessRunner:
    """Suite B — faithfulness metrics from paired parity npzs."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config

    def run(self) -> Path:
        cfg = self.config
        fc  = cfg.faithfulness
        log.info("=== Faithfulness Runner ===")
        base = _load_npz(fc.base_npz_path)
        ours = _load_npz(fc.ours_npz_path)
        self._validate_alignment(base["metadata"], ours["metadata"])
        results = self._compute_metrics(base, ours)
        return self._write(results, base, ours, cfg)

    # ------------------------------------------------------------------
    def _validate_alignment(self, bm: dict, om: dict) -> None:
        checks = ["article_sha", "seed", "prefill_len", "gen_len",
                  "window_size", "num_sink_tokens", "model_name"]
        mismatches = []
        for f in checks:
            bv, ov = bm.get(f), om.get(f)
            if bv is not None and ov is not None and bv != ov:
                mismatches.append(f"  {f}: base={bv!r}, ours={ov!r}")
        if mismatches:
            raise ParityValidationError(
                "Faithfulness alignment failed:\n" + "\n".join(mismatches))
        if bm.get("mode") != "parity_base":
            log.warning("Base npz mode is %r, expected 'parity_base'", bm.get("mode"))
        if om.get("mode") != "parity_ours":
            log.warning("Ours npz mode is %r, expected 'parity_ours'", om.get("mode"))

    # ------------------------------------------------------------------
    def _compute_metrics(self, base: dict, ours: dict) -> dict:
        ba, oa = base["arrays"], ours["arrays"]

        # Require new retained-window arrays from ours npz.
        if "retained_window_ids" not in oa or "retained_window_scores" not in oa:
            raise KeyError(
                "Ours npz is missing 'retained_window_ids' / 'retained_window_scores'. "
                "Re-run OursParityRunner to generate an updated npz."
            )

        # ── load tensors ──────────────────────────────────────────────
        # Window scores: legacy [T, L, H, W] → new [S, T, L, H, W]
        base_ws  = torch.from_numpy(ba["window_scores"].astype(np.float32))
        # Top-K indices for Jaccard: legacy [T, L, K] → new [S, T, L, K]
        base_tk  = torch.from_numpy(ba["top_window_indices"].astype(np.int64))
        ours_tk  = torch.from_numpy(oa["top_window_indices"].astype(np.int64))
        # New retained arrays: [S, T, L, M] and [S, T, L, H, M]
        ours_rid = torch.from_numpy(oa["retained_window_ids"].astype(np.int64))
        ours_rsc = torch.from_numpy(oa["retained_window_scores"].astype(np.float32))

        # Normalise to per-sample form
        if base_tk.dim() == 3:
            base_tk  = base_tk.unsqueeze(0)
            ours_tk  = ours_tk.unsqueeze(0)
        if base_ws.dim() == 4:
            base_ws  = base_ws.unsqueeze(0)
        if ours_rid.dim() == 3:
            ours_rid = ours_rid.unsqueeze(0)
            ours_rsc = ours_rsc.unsqueeze(0)

        num_samples = min(base_ws.shape[0], ours_rid.shape[0])

        # Align K for Jaccard (truncate to min side to avoid -1 inflation)
        bK, oK = base_tk.shape[-1], ours_tk.shape[-1]
        if bK != oK:
            minK    = min(bK, oK)
            base_tk = base_tk[..., :minK]
            ours_tk = ours_tk[..., :minK]

        om          = ours["metadata"]
        ws_sz       = int(om.get("window_size", 8))
        ns          = int(om.get("num_sink_tokens", 0))
        prefill_len = int(om.get("prefill_len", 0))

        # ── per-sample accumulation ───────────────────────────────────
        per_sample_jaccard = []
        per_sample_cos, per_sample_prs, per_sample_spm = [], [], []
        per_sample_kl,  per_sample_mr                  = [], []

        for s in range(num_samples):
            b_tk_s  = base_tk[s]    # [T, L, K]
            o_tk_s  = ours_tk[s]    # [T, L, K]
            b_ws_s  = base_ws[s]    # [T, L, H, W_pad]
            o_rid_s = ours_rid[s]   # [T, L, M]
            o_rsc_s = ours_rsc[s]   # [T, L, H, M]

            # Jaccard (unchanged — uses top-K evictable indices)
            j = M.jaccard_topk(o_tk_s.unsqueeze(2), b_tk_s.unsqueeze(2))  # [T, L, 1]
            per_sample_jaccard.append(j)

            T, L, _, W_pad = b_ws_s.shape
            cos_s = torch.zeros(T, L)
            prs_s = torch.zeros(T, L)
            spm_s = torch.zeros(T, L)
            kl_s  = torch.zeros(T, L)
            mr_s  = torch.zeros(T, L)

            for t in range(T):
                bws   = b_ws_s[t]    # [L, H, W_pad]
                # Post-step seq length is prefill_len + (t+1); subtract sinks.
                Sp_t  = max(1, prefill_len + t + 1 - ns)
                W_act = min(math.ceil(Sp_t / ws_sz), W_pad)

                for li in range(L):
                    # Retained window IDs (sorted by original pos, -1 padded)
                    rid_full = o_rid_s[t, li]                      # [M]
                    valid    = (rid_full >= 0) & (rid_full < W_act)
                    idx      = valid.nonzero(as_tuple=True)[0]     # positions in [M]
                    n_ret    = idx.shape[0]
                    if n_ret == 0:
                        continue

                    ret_ids = rid_full[idx]                        # [n_ret] original IDs

                    # Ours' scores for retained windows (mean over heads)
                    o_sc = o_rsc_s[t, li, :, idx].mean(dim=0)     # [n_ret]
                    # Base's scores for same windows (mean over heads)
                    b_sc = bws[li, :, ret_ids].mean(dim=0)         # [n_ret]

                    # 1. Cosine similarity
                    cos_s[t, li] = _cosine(o_sc, b_sc)

                    if n_ret < 2:
                        continue

                    # 2. Pearson correlation
                    prs_s[t, li] = _pearson(o_sc, b_sc)

                    # 3. Spearman rank correlation
                    spm_s[t, li] = _spearman(o_sc, b_sc)

                    # 4. KL(ours ‖ base)
                    kl_s[t, li] = _kl(o_sc, b_sc)

                    # 5. Mass ratio: base_mass / ours_mass over retained windows
                    mr_s[t, li] = b_sc.sum() / o_sc.sum().clamp(min=1e-8)

            per_sample_cos.append(cos_s)
            per_sample_prs.append(prs_s)
            per_sample_spm.append(spm_s)
            per_sample_kl.append(kl_s)
            per_sample_mr.append(mr_s)

        # ── stack & mean across samples ───────────────────────────────
        def _smean(lst: list) -> torch.Tensor:
            return torch.stack(lst, 0).mean(0)

        jaccard_stack = torch.stack(per_sample_jaccard, 0)  # [S, T, L, 1]
        jaccard       = jaccard_stack.mean(0)               # [T, L, 1]

        cos        = _smean(per_sample_cos)   # [T, L]
        pearson    = _smean(per_sample_prs)   # [T, L]
        spearman   = _smean(per_sample_spm)   # [T, L]
        kl         = _smean(per_sample_kl)    # [T, L]
        mass_ratio = _smean(per_sample_mr)    # [T, L]

        jaccard_per_layer = M.aggregate_per_layer(jaccard)   # [T, L]
        jaccard_global    = M.aggregate_global(jaccard)      # [T]
        heterogeneity     = M.final_step_heterogeneity(jaccard)  # [L]

        return {
            "jaccard":           jaccard.numpy(),
            "jaccard_per_layer": jaccard_per_layer.numpy(),
            "jaccard_global":    jaccard_global.numpy(),
            "heterogeneity":     heterogeneity.numpy(),
            "cos_sim":           cos.numpy(),         # [T, L]
            "pearson":           pearson.numpy(),     # [T, L]
            "spearman":          spearman.numpy(),    # [T, L]
            "kl_ours_base":      kl.numpy(),          # [T, L]
            "mass_ratio":        mass_ratio.numpy(),  # [T, L]
            "num_samples":       np.array([num_samples], dtype=np.int64),
            "per_sample_jaccard_global": jaccard_stack.mean(dim=(2, 3)).numpy(),  # [S, T]
        }

    # ------------------------------------------------------------------
    def _write(self, results: dict, base: dict, ours: dict,
               cfg: ExperimentConfig) -> Path:
        od = Path(cfg.telemetry.output_dir)
        od.mkdir(parents=True, exist_ok=True)
        meta = {
            "schema_version": "2.0",
            "base_npz_path":   base["path"],
            "base_npz_sha256": sha256_file(base["path"]),
            "ours_npz_path":   ours["path"],
            "ours_npz_sha256": sha256_file(ours["path"]),
            "run_finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        npz_path = od / "faithfulness_results.npz"
        if cfg.output_path:
            npz_path = Path(cfg.output_path)
        npz_path.parent.mkdir(parents=True, exist_ok=True)

        save_arrays = {
            "jaccard":           results["jaccard"],
            "jaccard_per_layer": results["jaccard_per_layer"],
            "jaccard_global":    results["jaccard_global"],
            "heterogeneity":     results["heterogeneity"],
            "cos_sim":           results["cos_sim"],
            "pearson":           results["pearson"],
            "spearman":          results["spearman"],
            "kl_ours_base":      results["kl_ours_base"],
            "mass_ratio":        results["mass_ratio"],
            "metadata_json":     np.array([json.dumps(meta)], dtype=object),
        }
        for opt in ("num_samples", "per_sample_jaccard_global"):
            if opt in results:
                save_arrays[opt] = results[opt]

        np.savez_compressed(str(npz_path), **save_arrays)
        with open(npz_path.with_suffix(".meta.json"), "w") as f:
            json.dump(meta, f, indent=2, default=str)
        log.info("Saved faithfulness: %s", npz_path)
        return npz_path
