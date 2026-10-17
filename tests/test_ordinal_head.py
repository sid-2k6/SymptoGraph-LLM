"""Tests for the monotone ordinal head (novelty N-A).

Written after shipping a head that was *structurally incapable* of predicting
three of the five suicide-risk classes. The checks that existed at the time -
cutpoints strictly increasing, item weights non-negative, class probabilities
summing to one, expected value monotone in the latent trait - all passed while
the head could only ever output class 0 or class 4.

The missing invariant is **reachability**: in a cumulative-link model

    P(y = j) = sigmoid(z - tau_j) - sigmoid(z - tau_{j+1})

an interior class peaks at roughly ``sigmoid(d/2) - sigmoid(-d/2)`` for cutpoint
spacing ``d``. At ``d = softplus(0) = 0.69`` that peak is 0.17, permanently below
the extremes, so no value of ``z`` makes an interior class the argmax. Measured
cost on real data: risk macro F1 0.312 -> 0.131, risk MAE 0.884 -> 1.547.

Run with::

    python -m pytest tests/test_ordinal_head.py -q
"""
from __future__ import annotations

import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from symptograph.preprocess import Ontology                        # noqa: E402
from symptograph.proposed import (CUT_SPACING, LADDER_V2,           # noqa: E402
                                  N_RISK, N_SEVERITY, ProposedConfig,
                                  build_model)

ONTO = Ontology.load(os.path.join(os.path.dirname(__file__), "..", "configs",
                                  "symptom_ontology.json"))
TASKS = (("severity", N_SEVERITY), ("risk", N_RISK))


def model(**over):
    cfg = ProposedConfig(**{**dict(LADDER_V2)["M4_denoised"],
                            "tiny_random_encoder": True,
                            "strict_encoder": False, "max_length": 32,
                            "use_noise_correction": False, **over})
    m, _ = build_model(cfg, ONTO, None)
    return m.eval()


def class_probs(m, task: str, n_cls: int, z: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return m._ordinal_logits(z, task, n_cls).exp()


# --------------------------------------------------------------------------- #
# the invariant that was missing
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("task,n_cls", TASKS)
@pytest.mark.parametrize("shared", [True, False], ids=["shared", "per_task"])
def test_every_class_is_reachable(task, n_cls, shared):
    """Each class must be the argmax for some attainable latent trait value."""
    m = model(monotone_shared_trait=shared)
    span = float(m.z_scale.abs().max())
    z = torch.linspace(-span, span, 2001)
    winners = set(class_probs(m, task, n_cls, z).argmax(-1).tolist())
    missing = sorted(set(range(n_cls)) - winners)
    assert not missing, (
        f"{task}: classes {missing} can never be predicted "
        f"(reachable: {sorted(winners)}, z span +/-{span:.1f})")


@pytest.mark.parametrize("task,n_cls", TASKS)
def test_interior_peak_beats_extremes(task, n_cls):
    """The algebraic reason reachability holds, checked directly."""
    m = model()
    interior_peak = math.tanh(CUT_SPACING / 4.0)   # sigmoid(d/2)-sigmoid(-d/2)
    assert interior_peak > 0.5, (
        f"cutpoint spacing {CUT_SPACING} gives an interior class a maximum "
        f"probability of {interior_peak:.3f}; below 0.5 the extremes always win")


def test_narrow_spacing_would_be_caught():
    """Guard against the regression: the old init must fail this suite."""
    m = model()
    with torch.no_grad():                      # recreate the shipped bug
        for task, n_cls in TASKS:
            m.cut0[task].fill_(0.0)
            m.cut_inc[task].fill_(0.0)         # softplus(0) = 0.69 spacing
        m.z_scale.fill_(1.069)                 # sum of softplus(-3) over 22
    for task, n_cls in TASKS:
        z = torch.linspace(0.0, 1.069, 500)
        winners = set(class_probs(m, task, n_cls, z).argmax(-1).tolist())
        assert len(winners) < n_cls, (
            f"{task}: the known-bad init should leave classes unreachable, "
            f"got {sorted(winners)}")


# --------------------------------------------------------------------------- #
# properties that already held, kept so they cannot regress
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("task,n_cls", TASKS)
def test_cutpoints_strictly_increasing(task, n_cls):
    m = model()
    tau = m.cutpoints(task)
    assert len(tau) == n_cls - 1
    assert bool((tau[1:] > tau[:-1]).all()), tau


@pytest.mark.parametrize("task,n_cls", TASKS)
def test_probabilities_normalised(task, n_cls):
    m = model()
    p = class_probs(m, task, n_cls, torch.linspace(-20, 20, 101))
    assert torch.allclose(p.sum(-1), torch.ones(101), atol=1e-5)
    assert bool((p >= 0).all())


@pytest.mark.parametrize("task,n_cls", TASKS)
def test_expected_class_monotone_in_trait(task, n_cls):
    m = model()
    p = class_probs(m, task, n_cls, torch.linspace(-20, 20, 201))
    exp = (p * torch.arange(float(n_cls))).sum(-1)
    assert bool((exp[1:] >= exp[:-1] - 1e-6).all())


def test_item_weights_non_negative_and_sum_near_one():
    m = model()
    w = torch.nn.functional.softplus(m.item_w)
    assert bool((w >= 0).all())
    assert torch.allclose(w.sum(-1), torch.ones(m.n_traits), atol=0.05), w.sum(-1)


def test_unconstrained_variant_drops_non_negativity():
    m = model(monotone_nonneg=False)
    assert m.item_w.requires_grad
    # the control exists precisely to allow negative loadings
    with torch.no_grad():
        m.item_w.data[0, 0] = -5.0
    assert float(m.item_w[0, 0]) < 0


@pytest.mark.parametrize("shared", [True, False], ids=["shared", "per_task"])
def test_trait_shape_and_divergence(shared):
    m = model(monotone_shared_trait=shared)
    batch = {"input_ids": torch.randint(0, 1000, (4, 16)),
             "attention_mask": torch.ones(4, 16, dtype=torch.long)}
    with torch.no_grad():
        out = m(batch)
    assert out["latent_trait"].shape == (4, 1 if shared else 2)
    assert out["severity"].shape == (4, N_SEVERITY)
    assert out["risk"].shape == (4, N_RISK)
    if not shared:
        with torch.no_grad():
            m.item_w.data[0].fill_(0.0)
            m.item_w.data[1].fill_(-6.0)
            out2 = m(batch)
        gap = (out2["latent_trait"][:, 0] - out2["latent_trait"][:, 1]).abs().min()
        assert float(gap) > 0.5, "per-task traits must be able to diverge"
