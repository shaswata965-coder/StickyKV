"""Tests for the faithfulness runner and metrics module."""
from __future__ import annotations
import ast
from pathlib import Path
import pytest


class TestFaithfulness:
    def test_faithfulness_rejects_unaligned_npz(self):
        from utils.config import ParityValidationError
        from modules.evaluation.faithfulness_runner import FaithfulnessRunner
        runner = FaithfulnessRunner.__new__(FaithfulnessRunner)
        bm = {"article_sha": "abc", "seed": 42, "prefill_len": 100,
              "gen_len": 10, "window_size": 8, "num_sink_tokens": 4, "model_name": "t"}
        om = dict(bm, article_sha="xyz")
        with pytest.raises(ParityValidationError):
            runner._validate_alignment(bm, om)

    def test_metrics_vectorized(self):
        src = Path("utils/metrics.py").read_text()
        tree = ast.parse(src)
        # Check function bodies for for-loops (allow module-level)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                for child in ast.walk(node):
                    if isinstance(child, ast.For):
                        pytest.fail(f"for-loop in {node.name}")
