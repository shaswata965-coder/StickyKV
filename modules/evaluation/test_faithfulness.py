"""Tests for faithfulness metrics (Suite B)."""
from __future__ import annotations
import ast
from pathlib import Path
import numpy as np
import pytest
import torch
from utils.metrics import lir, missed_mass, kl_inverse, global_lir


class TestFaithfulness:
    def test_lir_plus_missed_mass_equals_one(self):
        full_attn = torch.softmax(torch.randn(2, 3, 4, 10), dim=-1)
        retained = torch.tensor([[0,1,2,5,7]]).unsqueeze(0).expand(2, 3, -1)
        total = lir(full_attn, retained) + missed_mass(full_attn, retained)
        assert torch.allclose(total, torch.ones_like(total), atol=1e-5)

    def test_lir_is_one_when_no_eviction(self):
        full_attn = torch.softmax(torch.randn(2, 3, 4, 8), dim=-1)
        retained = torch.arange(8).unsqueeze(0).unsqueeze(0).expand(2, 3, -1)
        assert torch.allclose(lir(full_attn, retained), torch.ones(2,3,4), atol=1e-5)

    def test_lir_monotone_in_budget(self):
        torch.manual_seed(42)
        full_attn = torch.softmax(torch.randn(5, 2, 4, 20), dim=-1)
        lirs = []
        for n in [5, 10, 15, 20]:
            r = torch.arange(n).unsqueeze(0).unsqueeze(0).expand(5, 2, -1)
            lirs.append(lir(full_attn, r).mean().item())
        for i in range(len(lirs)-1):
            assert lirs[i] <= lirs[i+1] + 1e-5

    def test_kl_inv_zero_when_distributions_match(self):
        full_attn = torch.softmax(torch.randn(2, 3, 4, 10), dim=-1)
        retained = torch.arange(10).unsqueeze(0).unsqueeze(0).expand(2, 3, -1)
        kl = kl_inverse(full_attn, full_attn, retained)
        assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-4)

    def test_kl_inv_handles_zero_mass(self):
        full_attn = torch.zeros(1, 1, 1, 5); full_attn[0,0,0,0] = 1.0
        retained = torch.tensor([[[0, 1, 2]]])
        ours_attn = torch.tensor([[[[0.8, 0.1, 0.1]]]])
        kl = kl_inverse(full_attn, ours_attn, retained, eps=1e-9)
        assert not torch.isnan(kl).any() and not torch.isinf(kl).any()

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
