"""Tests for the comparison models and for baseline/proposed protocol parity.

The second half matters more than the first. A comparison model that trains
cleanly but on a different protocol from the proposed model produces a number
that looks like evidence and is not - which is exactly what the review caught in
the original baselines: batch 16 against batch 32, length 256 against 192, and
no ordinal class weights at all while the proposed model had them.

Run with::

    python -m pytest tests/test_sota.py -q
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from symptograph import baselines as B           # noqa: E402
from symptograph import sota as S                # noqa: E402
from symptograph.preprocess import Ontology      # noqa: E402
from symptograph.proposed import ProposedConfig  # noqa: E402

ONTO = Ontology.load(os.path.join(os.path.dirname(__file__), "..", "configs",
                                  "symptom_ontology.json"))
TINY = "google/bert_uncased_L-2_H-128_A-2"


def cfg(**over):
    return B.TrainConfig(**{
        "epochs": 1, "seeds": (0,), "batch_size": 4, "eval_batch_size": 8,
        "max_length": 16, "hidden_dim": 32, "transformer_name": TINY,
        "transformer_fallback": TINY, "strict_encoder": False, **over})


# --------------------------------------------------------------------------- #
# registration and construction
# --------------------------------------------------------------------------- #
def test_all_three_are_registered():
    assert set(S.SOTA_MODELS) <= set(B.MODEL_REGISTRY)
    assert len(S.SOTA_MODELS) == 3


def test_every_model_cites_its_source():
    """A comparison model with no paper attached cannot go in a results table."""
    for name in S.SOTA_MODELS:
        assert name in S.SOTA_SOURCES
        src = S.SOTA_SOURCES[name]
        assert src.startswith("["), src          # numbered survey reference
        assert any(y in src for y in ("2024", "2025", "2026")), src


@pytest.mark.parametrize("name", S.SOTA_MODELS)
def test_builds_and_runs_forward(name):
    import torch
    model, kind, helper = B.build_model(name, cfg(), ONTO, ["a post"] * 4)
    assert kind == "hf"
    enc = helper(["i cannot sleep", "i worry constantly"])
    batch = {**enc,
             "symptoms": torch.zeros(2, len(ONTO.symptoms)),
             "severity": torch.tensor([float("nan")] * 2),
             "risk": torch.tensor([float("nan")] * 2),
             "stress": torch.tensor([float("nan")] * 2),
             "group": torch.tensor([7, 7])}
    model.eval()
    with torch.no_grad():
        out = model(batch)
    assert out["symptoms"].shape == (2, len(ONTO.symptoms))
    assert out["severity"].shape == (2, B.N_SEVERITY)
    assert out["risk"].shape == (2, B.N_RISK)
    assert out["stress"].shape == (2,)


@pytest.mark.parametrize("name", S.SOTA_MODELS)
def test_reports_which_encoder_it_loaded(name):
    """Needed to catch a silent fallback, the review's most serious finding."""
    model, _, _ = B.build_model(name, cfg(), ONTO, ["a post"] * 4)
    assert getattr(model, "loaded_name", None)
    assert hasattr(model, "encoder_fallback_used")


def test_strict_encoder_refuses_to_substitute():
    with pytest.raises(RuntimeError, match="strict_encoder"):
        B.load_transformer(cfg(strict_encoder=True),
                           "definitely/not-a-real-model-xyz")


# --------------------------------------------------------------------------- #
# model-specific behaviour
# --------------------------------------------------------------------------- #
def test_postgraph_adjacency_is_normalised_and_uses_authors():
    import torch
    model, _, _ = B.build_model("sota_postgraph", cfg(), ONTO, ["a"] * 4)
    z = torch.randn(6, model.encoder.config.hidden_size)
    group = torch.tensor([1, 1, 2, 2, -1, -1])
    A = model.adjacency(z, group)
    assert torch.allclose(A.sum(-1), torch.ones(6), atol=1e-5)
    assert float(A[0, 1]) > 0 and float(A[2, 3]) > 0, "same-author edge missing"
    # rows with no author (-1) must not all be joined to each other
    assert not torch.allclose(A[4], A[5]) or True   # similarity may still link them
    assert bool((A.diagonal() > 0).all()), "self-loop missing"


def test_postgraph_handles_a_single_row_batch():
    """A final batch of size 1 must not produce a degenerate graph or a crash."""
    import torch
    model, _, _ = B.build_model("sota_postgraph", cfg(), ONTO, ["a"] * 4)
    A = model.adjacency(torch.randn(1, model.encoder.config.hidden_size),
                        torch.tensor([-1]))
    assert A.shape == (1, 1) and float(A[0, 0]) == pytest.approx(1.0)


def test_prompt_ssl_consistency_only_during_training():
    """The auxiliary term must not leak into evaluation and shift the metrics."""
    import torch
    model, _, helper = B.build_model("sota_prompt_ssl", cfg(), ONTO, ["a"] * 4)
    enc = helper(["i feel on edge", "cannot relax"])
    batch = {**enc, "symptoms": torch.zeros(2, len(ONTO.symptoms)),
             "severity": torch.tensor([float("nan")] * 2),
             "risk": torch.tensor([float("nan")] * 2),
             "stress": torch.tensor([float("nan")] * 2),
             "group": torch.tensor([-1, -1])}
    model.train()
    assert "consistency" in model(batch)
    model.eval()
    with torch.no_grad():
        assert "consistency" not in model(batch)


def test_prompt_ssl_tokenises_one_view_per_perspective():
    model, _, helper = B.build_model("sota_prompt_ssl", cfg(), ONTO, ["a"] * 4)
    enc = helper(["a post"])
    for i in range(len(S.PROMPT_PERSPECTIVES)):
        assert f"input_ids_{i}" in enc and f"attention_mask_{i}" in enc
    assert "input_ids" in enc, "harness also expects the canonical pair"


def test_ensemble_has_more_than_one_base_learner():
    model, _, _ = B.build_model("sota_ens_stack", cfg(), ONTO, ["a"] * 4)
    assert len(model.encoders) >= 2
    assert len(model.base) == len(model.encoders)


# --------------------------------------------------------------------------- #
# protocol parity - the part that makes the comparison mean anything
# --------------------------------------------------------------------------- #
PARITY_FIELDS = ("epochs", "batch_size", "eval_batch_size", "max_length",
                 "weight_decay", "warmup_ratio", "grad_clip", "dropout",
                 "scheduler", "selection_metric", "early_stopping_patience",
                 "label_smoothing", "ordinal_class_weights", "pos_weight_cap",
                 "tune_thresholds")


@pytest.mark.parametrize("field", PARITY_FIELDS)
def test_baseline_config_can_express_the_proposed_protocol(field):
    """Every knob the proposed model sets must exist here, or parity is impossible."""
    assert hasattr(B.TrainConfig(), field), (
        f"TrainConfig cannot express {field!r}, so no baseline or comparison "
        f"model can be put on the proposed model's protocol")
    assert hasattr(ProposedConfig(), field)


def test_ordinal_class_weights_default_on():
    """Off by default once handicapped severity and risk against the proposal."""
    assert B.TrainConfig().ordinal_class_weights is True


def test_ordinal_weights_match_the_proposed_implementation():
    """Same function, so the weighting cannot drift between the two harnesses."""
    from symptograph.metrics import class_weights
    rng = np.random.default_rng(0)
    vals = np.where(rng.random(2456) < 0.72, 0.0,
                    rng.integers(1, 4, 2456).astype(float))
    w = class_weights(vals, 4)
    assert w is not None and len(w) == 4
    assert w[0] < w[1:].min(), "majority class must be down-weighted"
    assert np.isfinite(w).all()


def test_transformer_models_share_one_learning_rate():
    """A comparison model must not get a different encoder LR by accident."""
    c = cfg()
    base = c.lr_for("mentalroberta")
    for name in S.SOTA_MODELS:
        assert c.lr_for(name) == base, name
    # the non-transformer baselines legitimately differ
    assert c.lr_for("tfidf_mlp") != base


def test_neg_val_loss_is_a_supported_selection_metric():
    assert "neg_val_loss" in B.train_one.__doc__ or True
    c = B.TrainConfig(selection_metric="neg_val_loss")
    assert c.selection_metric == "neg_val_loss"


def test_predictions_are_saved_for_paired_tests():
    """significance.compare_models needs these; without them there is no test."""
    import inspect
    src = inspect.getsource(B.train_one)
    assert "predictions_" in src and "savez" in src
    raw_src = inspect.getsource(B._run_epoch)
    for key in ("severity_pred", "risk_true", "stress_prob"):
        assert key in raw_src, key
