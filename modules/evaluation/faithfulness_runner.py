"""FaithfulnessRunner — pure post-processing over parity npzs (Suite B).

Reads both base and ours npz files. Computes LIR, missed mass, inverse KL,
global LIR. No model loaded — pure tensor ops via utils/metrics.py.
"""
from __future__ import annotations
import json, math, time
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
        # Window scores: legacy [num_steps, num_layers, H_q, W]
        #                new    [num_samples, num_steps, num_layers, H_q, W]
        base_ws = torch.from_numpy(ba["window_scores"].astype(np.float32))
        ours_ws = torch.from_numpy(oa["window_scores"].astype(np.float32))
        # Top-K indices: legacy [num_steps, num_layers, K]
        #                new    [num_samples, num_steps, num_layers, K]
        base_tk = torch.from_numpy(ba["top_window_indices"].astype(np.int64))
        ours_tk = torch.from_numpy(oa["top_window_indices"].astype(np.int64))

        # Normalise to per-sample form (always add a leading sample axis).
        # Legacy NPZs (rank 3 topk, rank 4 ws) get a sample-axis of size 1.
        if base_tk.dim() == 3:
            base_tk = base_tk.unsqueeze(0)
            ours_tk = ours_tk.unsqueeze(0)
        if base_ws.dim() == 4:
            base_ws = base_ws.unsqueeze(0)
            ours_ws = ours_ws.unsqueeze(0)

        num_samples = min(base_tk.shape[0], ours_tk.shape[0])

        # Align K across base/ours by truncating to the smaller side.
        # Padding the smaller with -1 was wrong: -1 slots never match any
        # valid index, inflating the Jaccard denominator and deflating the
        # score.  Truncating base to min(bK, oK) means we ask "do the
        # windows ours kept appear in base's top-minK?" — the correct
        # question given that ours has a bounded cache budget.
        bK, oK = base_tk.shape[-1], ours_tk.shape[-1]
        if bK != oK:
            minK = min(bK, oK)
            base_tk = base_tk[..., :minK]
            ours_tk = ours_tk[..., :minK]

        # Derive parameters needed for per-step LIR computation.
        # local windows are never evicted, so they must be added to the
        # evictable top-K when measuring how much of base's attention mass
        # ours retains.
        om          = ours["metadata"]
        ws_sz       = int(om.get("window_size", 8))
        lwr         = int(om.get("local_window_size_resolved", 0))
        lnw_max     = max(0, lwr // ws_sz)    # local window count (end-of-run upper bound)
        prefill_len = int(om.get("prefill_len", 0))
        ns          = int(om.get("num_sink_tokens", 0))

        # Compute per-sample metrics, then mean across the sample axis.
        per_sample_jaccard = []
        per_sample_lir = []
        _diag_printed = False
        for s in range(num_samples):
            b_tk_s = base_tk[s]            # [num_steps, num_layers, K]
            o_tk_s = ours_tk[s]
            b_ws_s = base_ws[s]            # [num_steps, num_layers, H_q, W]

            # jaccard_topk wants [num_steps, num_layers, H_q, K]; add dummy H_q=1 dim.
            j = M.jaccard_topk(o_tk_s.unsqueeze(2), b_tk_s.unsqueeze(2))   # [S, L, 1]
            per_sample_jaccard.append(j)

            # LIR: fraction of BASE attention mass that falls on positions
            # ours retained.  Correct formula:
            #   LIR = bws[:, :, ours_retained_orig_wins].sum() / bws.sum()
            # where ours_retained_orig_wins = (top-K evictable, already in
            # original-sequence space after the index-alignment fix) UNION
            # (local windows = last lnw windows of the base sequence at that
            # step, which ours never evicts).
            # This uses BASE scores only — ours' redistributed attention mass
            # is not involved, avoiding the post-eviction inflation artifact.
            num_steps_s = b_ws_s.shape[0]
            n_layers_s  = b_ws_s.shape[1]
            W_padded    = b_ws_s.shape[-1]   # padded max width — do NOT use for indexing
            lir_s = torch.zeros(num_steps_s)
            for t in range(num_steps_s):
                bws   = b_ws_s[t]          # [L, H, W_padded] — trailing cols are zero
                total  = bws.sum()
                if total <= 0:
                    continue
                # Actual window count at step t: base adds one token per step
                # starting from prefill_len tokens at step 0.
                # base_parity_runner uses ceil so we match that here.
                Sp_t     = max(1, prefill_len + t - ns)
                W_actual = min(math.ceil(Sp_t / ws_sz), W_padded)
                # Local windows = last lnw_t of the VALID portion.
                # bws.shape[-1] is W_padded (zero-filled beyond W_actual), so
                # using it for local_ids would point into zero-padding for most
                # steps and contribute nothing — hence the collapsed LIR.
                lnw_t     = min(lnw_max, W_actual)
                local_ids = torch.arange(W_actual - lnw_t, W_actual, dtype=torch.long)
                retained_mass = torch.zeros(1)
                for li in range(n_layers_s):
                    # Evictable top-K for this layer in original-sequence space
                    ev_ids = o_tk_s[t, li]                    # [K]
                    ev_ids = ev_ids[ev_ids >= 0]              # strip -1 padding
                    ev_ids = ev_ids[ev_ids < W_actual]        # clamp to valid range
                    all_ids = torch.unique(torch.cat([local_ids, ev_ids]))
                    retained_mass += bws[li, :, all_ids].sum()
                lir_s[t] = (retained_mass / total).clamp(0, 1)
                # Diagnostic at three representative steps for sample 0
                diag_steps = {0, num_steps_s // 4, num_steps_s // 2,
                               3 * num_steps_s // 4, num_steps_s - 1}
                if s == 0 and t in diag_steps:
                    n_retained    = len(all_ids)
                    all_valid_mass = bws[:, :, :W_actual].sum().item()
                    ret_mass_v    = retained_mass.item()
                    uniform_lir   = n_retained / W_actual if W_actual > 0 else 0.0
                    log.info(
                        "[LIR diag] s=%d t=%d/%d  W_actual=%d lnw_t=%d "
                        "n_local=%d n_ev_ids=%d n_all_ids=%d  "
                        "ret_mass=%.4f total=%.4f all_valid=%.4f  "
                        "lir=%.4f  uniform_lir=%.4f  ev_ids[:5]=%s",
                        s, t, num_steps_s - 1,
                        W_actual, lnw_t,
                        len(local_ids), len(ev_ids), n_retained,
                        ret_mass_v, total.item(), all_valid_mass,
                        lir_s[t].item(), uniform_lir,
                        ev_ids[:5].tolist(),
                    )
            per_sample_lir.append(lir_s)

        # Stack and mean across samples. Each tensor has the same per-sample shape.
        jaccard_stack = torch.stack(per_sample_jaccard, dim=0)   # [num_samples, S, L, 1]
        jaccard = jaccard_stack.mean(dim=0)                       # [S, L, 1]
        lir_stack = torch.stack(per_sample_lir, dim=0)            # [num_samples, S]
        lir_proxy = lir_stack.mean(dim=0)                         # [S]

        jaccard_per_layer = M.aggregate_per_layer(jaccard)        # [S, L]
        jaccard_global = M.aggregate_global(jaccard)              # [S]
        heterogeneity = M.final_step_heterogeneity(jaccard)       # [L]

        return {
            "jaccard": jaccard.numpy(),
            "jaccard_per_layer": jaccard_per_layer.numpy(),
            "jaccard_global": jaccard_global.numpy(),
            "heterogeneity": heterogeneity.numpy(),
            "lir_proxy": lir_proxy.numpy(),
            "global_lir": lir_proxy.numpy(),                      # alias
            "num_samples": np.array([num_samples], dtype=np.int64),
            "per_sample_jaccard_global": jaccard_stack.mean(dim=(2, 3)).numpy(),  # [num_samples, S]
            "per_sample_lir_proxy": lir_stack.numpy(),            # [num_samples, S]
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
        # Persist all results (including optional per-sample breakdowns) plus metadata.
        save_arrays = {
            "jaccard": results["jaccard"],
            "jaccard_per_layer": results["jaccard_per_layer"],
            "jaccard_global": results["jaccard_global"],
            "heterogeneity": results["heterogeneity"],
            "lir_proxy": results["lir_proxy"],
            "global_lir": results["global_lir"],
            "metadata_json": np.array([json.dumps(meta)], dtype=object),
        }
        for opt in ("num_samples", "per_sample_jaccard_global", "per_sample_lir_proxy"):
            if opt in results:
                save_arrays[opt] = results[opt]
        np.savez_compressed(str(npz_path), **save_arrays)
        with open(npz_path.with_suffix(".meta.json"), "w") as f:
            json.dump(meta, f, indent=2, default=str)
        log.info("Saved faithfulness: %s", npz_path)
        return npz_path
