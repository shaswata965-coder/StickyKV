# modules.quant — shared two-tier quantization package (design.md §2–§6)
#
# Single copy imported by BOTH cache backends (windowed_cache and
# windowed_eager_cache). Pure-tensor, CPU-testable; the only transformers
# dependency is the rope module passed in at read time.

from .effective import apply_rope_to_keys, materialize_effective_kv
from .ledger import LedgerEntry, QuantLedger
from .positions import build_interleaved_position_map
from .quantizer import (
    compute_grid,
    dequantize,
    dequantize_key_windows,
    dequantize_value_windows,
    pack_nibbles,
    quantize,
    quantize_key_windows,
    quantize_value_windows,
    quantize_with_grid,
    unpack_nibbles,
)
from .store import QuantizedStore

__all__ = [
    "LedgerEntry",
    "QuantLedger",
    "QuantizedStore",
    "apply_rope_to_keys",
    "build_interleaved_position_map",
    "compute_grid",
    "materialize_effective_kv",
    "dequantize",
    "dequantize_key_windows",
    "dequantize_value_windows",
    "pack_nibbles",
    "quantize",
    "quantize_key_windows",
    "quantize_value_windows",
    "quantize_with_grid",
    "unpack_nibbles",
]
