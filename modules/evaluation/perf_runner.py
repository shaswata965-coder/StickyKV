"""PerfRunner — wall-clock benchmarks (Suite C).

TTFT, throughput (tok/s), TPOT, peak memory across the backend x budget
factorial. Supports eager (Kaggle-safe) and flash_attn (Ampere+) backends.
Gracefully skips configs that OOM or require unavailable flash-attn.
"""
from __future__ import annotations
import json, time, gc
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import torch
from utils.config import ExperimentConfig
from utils.env_capture import capture_environment
from utils.logger import get_logger

log = get_logger(__name__)

def _flash_attn_available() -> bool:
    try:
        import flash_attn  # type: ignore
        return True
    except ImportError:
        return False

class PerfRunner:
    """Suite C — TTFT, throughput, TPOT benchmarks."""
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config

    def run(self) -> List[Path]:
        cfg = self.config
        pc = cfg.perf
        log.info("=== Performance Runner ===")
        flash_ok = _flash_attn_available()
        if not flash_ok:
            log.info("flash-attn not available — flash configs will be skipped")
        env = capture_environment()
        # Try to lock GPU clocks
        clocks_locked = False
        if pc.enable_clock_locking:
            clocks_locked = self._try_lock_clocks()
        # Build (prefill_len, gen_len) grid. Explicit `grid:` wins; otherwise
        # fall back to legacy `prefill_lengths` x scalar `gen_len`.
        if pc.grid:
            cells = [(int(g["prefill_len"]), int(g["gen_len"])) for g in pc.grid]
        else:
            cells = [(p, pc.gen_len) for p in pc.prefill_lengths]
        output_paths = []
        for prefill_len, gen_len in cells:
            log.info("--- Cell: prefill=%d gen=%d ---", prefill_len, gen_len)
            result = self._run_prefill(prefill_len, gen_len, pc, cfg, flash_ok, env)
            result["clocks_locked"] = clocks_locked
            result["gen_len"] = gen_len
            path = self._save(result, prefill_len, gen_len, cfg, env, clocks_locked)
            output_paths.append(path)
        return output_paths

    def _run_prefill(self, prefill_len: int, gen_len: int, pc, cfg, flash_ok: bool, env: dict) -> dict:
        configs = pc.configs
        n_configs = len(configs)
        n_runs = pc.num_measurement_runs
        ttft = np.full((n_configs, n_runs), np.nan)
        throughput = np.full((n_configs, n_runs), np.nan)
        tpot = np.full((n_configs, n_runs), np.nan)
        peak_mem = np.full((n_configs, n_runs), np.nan)
        skipped = np.zeros(n_configs, dtype=bool)
        names = []
        attn_impls = []
        for ci, c in enumerate(configs):
            name = c.get("name", f"config_{ci}")
            names.append(name)
            attn_impls.append(c.get("attn_implementation", "eager"))
            # Check flash requirement
            if c.get("requires_flash_attn", False) and not flash_ok:
                if pc.skip_if_flash_attn_unavailable:
                    log.info("Skipping %s (flash-attn unavailable)", name)
                    skipped[ci] = True
                    continue
            log.info("Running config: %s", name)
            try:
                measurements = self._measure_config(c, prefill_len, gen_len, pc, cfg)
                for ri, m in enumerate(measurements):
                    ttft[ci, ri] = m["ttft_ms"]
                    throughput[ci, ri] = m["throughput_tokps"]
                    tpot[ci, ri] = m["tpot_ms"]
                    peak_mem[ci, ri] = m["peak_memory_mb"]
            except torch.cuda.OutOfMemoryError:
                if pc.skip_if_oom:
                    log.warning("OOM on %s at prefill %d — skipping", name, prefill_len)
                    skipped[ci] = True
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                else:
                    raise
            except Exception as e:
                log.warning("Error on %s: %s — skipping", name, e)
                skipped[ci] = True
        return {"names": names, "attn_impls": attn_impls, "ttft": ttft,
                "throughput": throughput, "tpot": tpot, "peak_mem": peak_mem,
                "skipped": skipped}

    def _measure_config(self, c: dict, prefill_len: int, gen_len: int, pc, cfg) -> List[dict]:
        from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
        dtypes = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
        torch_dtype = dtypes.get(cfg.model.dtype, torch.float16)
        attn_impl = c.get("attn_implementation", "eager")
        tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, revision=cfg.model.revision)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model.name, revision=cfg.model.revision,
            torch_dtype=torch_dtype, attn_implementation=attn_impl,
            device_map="auto")
        model.eval()
        # Create input
        input_ids = torch.randint(100, 30000, (1, prefill_len), device=model.device)
        # Setup cache
        cache_backend = c.get("cache_backend", "dynamic")
        cache_pkg = c.get("cache_package")
        budget = c.get("cache_budget")
        # Windowed-cache setup: resolve classes + RoPE once, but DO NOT install
        # hooks here. Each warmup/measurement iteration creates a fresh cache,
        # so hooks must be (re)installed on the cache the model actually
        # receives, then removed before the next iteration.
        WC = WCC = install_hooks = None
        cc = None
        rope = None
        if cache_backend == "windowed" and cache_pkg:
            from utils.cache_factory import get_cache_classes
            WC, WCC, install_hooks = get_cache_classes(cache_pkg)
            w = cfg.window
            cc = WCC(window_size=w.window_size, num_sink_tokens=w.num_sink_tokens,
                      local_window_size=w.local_window_size,
                      cache_budget=budget if budget is not None else 0.5)
            # Two-pass RoPE discovery (mirrors ours_parity_runner.py).
            for nm, mod in model.named_modules():
                if "rotary" in nm.lower() or "rope" in nm.lower():
                    rope = mod; break
            if rope is None:
                for nm, mod in model.named_modules():
                    if hasattr(mod, "rotary_emb"):
                        rope = mod.rotary_emb; break
            if rope is None:
                from utils.config import ConfigValidationError
                raise ConfigValidationError(
                    "Could not locate a RoPE module on the model. WindowedCache "
                    "requires a rotary embedding module for key rerotation."
                )

        def _make_windowed_cache():
            return WC(config=cc, prefill_len=prefill_len,
                      model_config=model.config, kv_dtype=torch_dtype,
                      rope_module=rope,
                      num_layers=model.config.num_hidden_layers)

        gen_kwargs = {}
        if attn_impl == "eager" and c.get("install_hooks_for_measurement"):
            gen_kwargs["output_attentions"] = True
        # Warmup
        for _ in range(pc.num_warmup_runs):
            hooks = None
            try:
                with torch.no_grad():
                    if cache_backend == "windowed" and cache_pkg:
                        pkv = _make_windowed_cache()
                        hooks = install_hooks(model, pkv, cc)
                    else:
                        pkv = DynamicCache() if cache_backend == "dynamic" else None
                    model(input_ids=input_ids, past_key_values=pkv,
                          use_cache=True, return_dict=True, **gen_kwargs)
            finally:
                if hooks is not None:
                    hooks.remove()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        # Measurement runs
        measurements = []
        for ri in range(pc.num_measurement_runs):
            hooks = None
            try:
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                # TTFT — synchronize BEFORE t0 so prior async work doesn't taint
                # the measurement and the sync's own latency isn't timed.
                if torch.cuda.is_available(): torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.no_grad():
                    if cache_backend == "windowed" and cache_pkg:
                        pkv = _make_windowed_cache()
                        hooks = install_hooks(model, pkv, cc)
                    else:
                        pkv = DynamicCache() if cache_backend == "dynamic" else None
                    out = model(input_ids=input_ids, past_key_values=pkv,
                                use_cache=True, return_dict=True, **gen_kwargs)
                if torch.cuda.is_available(): torch.cuda.synchronize()
                t1 = time.perf_counter()
                ttft_ms = (t1 - t0) * 1000
                # Generate tokens for TPOT
                pkv = out.past_key_values
                next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                if torch.cuda.is_available(): torch.cuda.synchronize()
                t2 = time.perf_counter()
                with torch.no_grad():
                    for _ in range(gen_len - 1):
                        out = model(input_ids=next_tok, past_key_values=pkv,
                                    use_cache=True, return_dict=True, **gen_kwargs)
                        pkv = out.past_key_values
                        next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                if torch.cuda.is_available(): torch.cuda.synchronize()
                t3 = time.perf_counter()
                gen_time = t3 - t2
                tpot_ms = (gen_time / max(gen_len - 1, 1)) * 1000
                # End-to-end throughput includes prefill (TTFT) + decode time; this
                # mirrors the legacy field name but is NOT decode-only.
                throughput_tokps = gen_len / max(gen_time + (t1-t0), 1e-9)
                peak_mb = 0.0
                if torch.cuda.is_available():
                    peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
                measurements.append({"ttft_ms": ttft_ms, "throughput_tokps": throughput_tokps,
                                     "tpot_ms": tpot_ms, "peak_memory_mb": peak_mb})
            finally:
                if hooks is not None:
                    hooks.remove()
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        # Cleanup
        del model
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return measurements

    def _save(self, result: dict, prefill_len: int, gen_len: int, cfg, env: dict, clocks_locked: bool) -> Path:
        od = Path(cfg.telemetry.output_dir); od.mkdir(parents=True, exist_ok=True)
        npz_path = od / f"perf_prefill{prefill_len}_gen{gen_len}.npz"
        meta = {
            "prefill_len": prefill_len,
            "gen_len": gen_len,
            **env,
            "clocks_locked": clocks_locked,
        }
        np.savez_compressed(
            str(npz_path),
            config_names=np.array(result["names"], dtype=object),
            attn_implementations=np.array(result["attn_impls"], dtype=object),
            ttft_ms=result["ttft"],
            throughput_tokps=result["throughput"],
            tpot_ms=result["tpot"],
            peak_memory_mb=result["peak_mem"],
            skipped_mask=result["skipped"],
            metadata_json=np.array([json.dumps(meta)], dtype=object),
        )
        log.info("Saved perf: %s", npz_path)
        return npz_path

    def _try_lock_clocks(self) -> bool:
        import subprocess
        try:
            r = subprocess.run(["nvidia-smi", "-lgc", "1400,1400"],
                               capture_output=True, text=True, timeout=10)
            if r.returncode == 0:
                log.info("GPU clocks locked"); return True
            log.warning("nvidia-smi -lgc failed: %s", r.stderr)
        except Exception as e:
            log.warning("Clock locking failed: %s", e)
        return False
