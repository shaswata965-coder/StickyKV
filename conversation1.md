# Conversation 1 — Session Notes

- `main.py` is the single entry point; routes to runners via `run.mode` from config
- `utils/config.py` holds all typed dataclasses; precedence is: dataclass defaults → base.yaml → experiment YAML → CLI overrides
- All YAML configs inherit from `base.yaml` via `_base_` to avoid repeating shared settings
- `utils/cache_factory.py` is called by the runner (not first) to select the correct cache class and hook installer
- Hooks are installed before `model.generate()` is called; `hooks.py` is the first active file when tokens hit the model
- Suite C (perf) does not use real text — it feeds `torch.randint` random tokens to the model
- TTFT = single forward pass time over full prefill; TPOT = decode loop time / (gen_len - 1); Throughput = total tokens / total wall time; Peak memory via `torch.cuda.max_memory_allocated()`
- Random token input is fine for latency/memory benchmarking but does not reflect real eviction behavior — Suites A, B, D cover quality
- Kaggle config bug: `_base_: base.yaml` not `configs/base.yaml` — path resolves relative to the config file's directory
- Eager backend requires `install_hooks_for_measurement: true` in the perf config entry to pass `output_attentions=True`; without it scoring silently degrades to sink+local only
- Suite C (perf) autogenerates random token IDs as input — it does not take any real prompt or load any dataset
