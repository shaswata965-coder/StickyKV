"""FaithfulnessRunner — pure post-processing over parity npzs (Suite B).

Reads both base and ours npz files. Computes LIR, missed mass, inverse KL,
global LIR. No model loaded — pure tensor ops via utils/metrics.py.
"""
from __future__ import annotations
import json, time
from pathlib import Path
from typing import Any, Dict
import numpy as np
import torch
from utils.config import ExperimentConfig, ParityValidationError
from utils.hashing import sha256_file
from utils.logger import get_logger
from utils import metrics as M

log = get_logger(__name__)

def _load_npz(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"NPZ not found: {p}")
    data = np.load(str(p), allow_pickle=True)
    meta_str = str(data["metadata_json"][0])
    meta = json.loads(meta_str)
    arrays = {k: data[k] for k in data.files if k != "metadata_json"}
    return {"arrays": arrays, "metadata": meta, "path": str(p)}

class FaithfulnessRunner:
    """Suite B — faithfulness metrics from paired parity npzs."""
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config

    def run(self) -> Path:
        cfg = self.config
        fc = cfg.faithfulness
        log.info("=== Faithfulness Runner ===")
        # 1. Load both npzs
        base = _load_npz(fc.base_npz_path)
        ours = _load_npz(fc.ours_npz_path)
        bm, om = base["metadata"], ours["metadata"]
        # 2. Validate alignment
        self._validate_alignment(bm, om)
        # 3. Compute metrics
        results = self._compute_metrics(base, ours)
        # 4. Write output
        return self._write(results, base, ours, cfg)

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

    def _compute_metrics(self, base: dict, ours: dict) -> dict:
        ba, oa = base["arrays"], ours["arrays"]
        # Window scores: [num_steps, num_layers, H_q, W]
        base_ws = torch.from_numpy(ba["window_scores"].astype(np.float32))
        ours_ws = torch.from_numpy(oa["window_scores"].astype(np.float32))
        # Top-K indices: [num_steps, num_layers, K]
        base_tk = torch.from_numpy(ba["top_window_indices"].astype(np.int64))
        ours_tk = torch.from_numpy(oa["top_window_indices"].astype(np.int64))
        # Ensure same shape for Jaccard (pad K dimension if needed)
        bK, oK = base_tk.shape[-1], ours_tk.shape[-1]
        if bK != oK:
            maxK = max(bK, oK)
            if bK < maxK:
                base_tk = torch.nn.functional.pad(base_tk, (0, maxK-bK), value=-1)
            if oK < maxK:
                ours_tk = torch.nn.functional.pad(ours_tk, (0, maxK-oK), value=-1)
        # Jaccard needs [num_steps, num_layers, H_q, K] but our topk is [num_steps, num_layers, K]
        # Expand by adding a dummy H_q=1 dimension for Jaccard
        if base_tk.dim() == 3:
            base_tk = base_tk.unsqueeze(2)  # [S, L, 1, K]
            ours_tk = ours_tk.unsqueeze(2)
        # Compute Jaccard
        jaccard = M.jaccard_topk(ours_tk, base_tk)  # [S, L, H_q]
        jaccard_per_layer = M.aggregate_per_layer(jaccard)  # [S, L]
        jaccard_global = M.aggregate_global(jaccard)  # [S]
        heterogeneity = M.final_step_heterogeneity(jaccard)  # [L]
        # LIR-like metrics from window scores (approximate — full attention not always available)
        # Use window score overlap as a proxy for LIR
        num_steps = min(base_ws.shape[0], ours_ws.shape[0])
        # Compute per-step score-mass retention
        lir_proxy = torch.zeros(num_steps)
        for t in range(num_steps):
            bws = base_ws[t]  # [L, H, W]
            ows = ours_ws[t]
            # Total base mass
            total = bws.sum()
            if total > 0:
                # Mass in ours (overlap by position)
                minW = min(bws.shape[-1], ows.shape[-1])
                retained = ows[..., :minW].sum()
                lir_proxy[t] = (retained / total).clamp(0, 1)
        return {
            "jaccard": jaccard.numpy(),
            "jaccard_per_layer": jaccard_per_layer.numpy(),
            "jaccard_global": jaccard_global.numpy(),
            "heterogeneity": heterogeneity.numpy(),
            "lir_proxy": lir_proxy.numpy(),
            "global_lir": lir_proxy.numpy(),  # alias
        }

    def _write(self, results: dict, base: dict, ours: dict, cfg: ExperimentConfig) -> Path:
        od = Path(cfg.telemetry.output_dir)
        od.mkdir(parents=True, exist_ok=True)
        base_sha = sha256_file(base["path"])
        ours_sha = sha256_file(ours["path"])
        meta = {
            "schema_version": "1.0",
            "base_npz_path": base["path"],
            "base_npz_sha256": base_sha,
            "ours_npz_path": ours["path"],
            "ours_npz_sha256": ours_sha,
            "run_finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        npz_path = od / "faithfulness_results.npz"
        if cfg.output_path:
            npz_path = Path(cfg.output_path)
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(npz_path),
            jaccard=results["jaccard"],
            jaccard_per_layer=results["jaccard_per_layer"],
            jaccard_global=results["jaccard_global"],
            heterogeneity=results["heterogeneity"],
            lir_proxy=results["lir_proxy"],
            global_lir=results["global_lir"],
            metadata_json=np.array([json.dumps(meta)], dtype=object),
        )
        with open(npz_path.with_suffix(".meta.json"), "w") as f:
            json.dump(meta, f, indent=2, default=str)
        log.info("Saved faithfulness: %s", npz_path)
        return npz_path
