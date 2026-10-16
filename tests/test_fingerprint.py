"""Resume-guard tests, written because this broke twice in a row.

The guard decides whether a checkpoint may be resumed. It has to satisfy two
requirements that pull against each other:

1. **Never resume into a different network.** Loading a state dict from a
   different architecture either raises a shape error or, worse, silently
   loads and produces numbers attributed to the wrong model.
2. **Never refuse a run a change cannot affect.** ``M0_flat`` has no monotone
   head, so ``monotone_shared_trait`` says nothing about it. Refusing there
   tells the user to ``rm -rf`` finished runs for no reason - which nearly cost
   an hour of L4 time.

Both failures I actually shipped are pinned below as regression cases: a field
added after a checkpoint was written (``monotone_shared_trait``), and a field
that *became* conditional after a checkpoint was written
(``monotone_nonneg``/``monotone_residual``).

Run with::

    python -m pytest tests/test_fingerprint.py -q
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from symptograph.preprocess import Ontology                      # noqa: E402
from symptograph.proposed import (CONDITIONAL_ARCH_FIELDS,        # noqa: E402
                                  LADDER_V2, ProposedConfig,
                                  architecture_fingerprint,
                                  fingerprint_changes)

ONTO = Ontology.load(os.path.join(os.path.dirname(__file__), "..", "configs",
                                  "symptom_ontology.json"))
FLAGS = dict(LADDER_V2)

# Fields the v2 commit introduced. A checkpoint written before it has none of
# them, and genuinely contains no monotone-head parameters.
V2_ADDED = ("use_monotone_ordinal", "use_noise_correction",
            "freeze_encoder_layers", "monotone_nonneg", "monotone_residual",
            "monotone_shared_trait")


def fp(rung: str, **over):
    cfg = ProposedConfig(**{**FLAGS.get(rung, {}), **over})
    return architecture_fingerprint(rung, cfg, ONTO)


def stored_fields(rung: str, version: str, **over) -> dict:
    """The ``fingerprint_fields`` a given version of the code would have written."""
    _, f = fp(rung, **over)
    f = dict(f)
    if version == "current":
        return f
    if version == "v2_first":
        # monotone_shared_trait did not exist yet, and nonneg/residual were
        # written unconditionally - even for rungs with no monotone head.
        f.pop("monotone_shared_trait", None)
        f.setdefault("monotone_nonneg", True)
        f.setdefault("monotone_residual", True)
        return f
    if version == "v1":
        for k in V2_ADDED:
            f.pop(k, None)
        return f
    raise ValueError(version)


def predates_v2_architecture(rung: str) -> bool:
    """Could a pre-v2 checkpoint hold this rung's weights? Only without N-A/N-B."""
    cfg = ProposedConfig(**FLAGS[rung])
    return not cfg.use_monotone_ordinal and not cfg.use_noise_correction


# --------------------------------------------------------------------------- #
# must resume
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("rung", list(FLAGS))
@pytest.mark.parametrize("version", ["v2_first", "current"],
                         ids=["ckpt_predates_shared_trait", "ckpt_current"])
def test_same_config_resumes(rung, version):
    """A checkpoint from any v2 build resumes under the same config."""
    stored = stored_fields(rung, version)
    now_hash, now_fields = fp(rung)
    changed = fingerprint_changes("stale-hash", stored, now_hash, now_fields)
    assert changed == {}, f"{rung} ({version}) wrongly refused: {changed}"


@pytest.mark.parametrize("rung", list(FLAGS))
def test_pre_v2_checkpoint(rung):
    """A pre-v2 checkpoint resumes only where it really holds the same network.

    ``M0_flat`` and ``M1_bottleneck`` are architecturally unchanged since v1, so
    refusing them would discard finished runs. ``M2``-``M4`` add the monotone
    head or the noise channel, which v1 weights cannot represent, so those must
    be refused.
    """
    stored = stored_fields(rung, "v1")
    now_hash, now_fields = fp(rung)
    changed = fingerprint_changes("stale-hash", stored, now_hash, now_fields)
    if predates_v2_architecture(rung):
        assert changed == {}, f"{rung} wrongly refused: {changed}"
    else:
        assert changed, f"{rung} must not load pre-v2 weights"


def test_identical_hash_short_circuits():
    h, f = fp("M2_monotone")
    assert fingerprint_changes(h, f, h, f) == {}


def test_unrelated_rung_unaffected_by_monotone_fields():
    """The exact bug: M0/M1 have no monotone head, so its settings are noise."""
    for rung in ("M0_flat", "M1_bottleneck"):
        _, fields = fp(rung)
        assert not [k for k in fields if k.startswith("monotone")], \
            f"{rung} fingerprint should carry no monotone fields: {fields}"


# --------------------------------------------------------------------------- #
# must refuse
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("rung,over,expect", [
    # the correction that prompted all this: shared trait -> one per task
    ("M2_monotone", {"monotone_shared_trait": False}, "monotone_shared_trait"),
    ("M2_monotone", {"monotone_nonneg": False}, "monotone_nonneg"),
    ("M2_monotone", {"monotone_residual": False}, "monotone_residual"),
    # turning the head off entirely
    ("M2_monotone", {"use_monotone_ordinal": False}, "use_monotone_ordinal"),
    # weak rows concatenated vs streamed
    ("M3_weak", {"weak_as_stream": False}, "weak_as_stream"),
    # plain architecture changes
    ("M1_bottleneck", {"max_length": 128}, "max_length"),
    ("M1_bottleneck", {"encoder_name": "roberta-base"}, "encoder_name"),
    ("M1_bottleneck", {"freeze_encoder_layers": 4}, "freeze_encoder_layers"),
    ("M1_bottleneck", {"dropout": 0.3}, "dropout"),
    ("M1_bottleneck", {"use_graph": True}, "use_graph"),
    ("M1_bottleneck", {"concept_gated_pooling": False}, "concept_gated_pooling"),
    ("M0_flat", {"use_bottleneck": True}, "use_bottleneck"),
    ("M4_denoised", {"use_noise_correction": False}, "use_noise_correction"),
])
def test_real_change_is_refused(rung, over, expect):
    stored = stored_fields(rung, "current")
    now_hash, now_fields = fp(rung, **over)
    changed = fingerprint_changes("stale-hash", stored, now_hash, now_fields)
    assert expect in changed, f"{expect} change not caught: {changed}"


@pytest.mark.parametrize("version", ["v2_first"])
def test_real_change_still_refused_on_legacy_checkpoint(version):
    """Backward compatibility must not become a hole that lets anything pass."""
    stored = stored_fields("M2_monotone", version)
    now_hash, now_fields = fp("M2_monotone", monotone_shared_trait=False)
    changed = fingerprint_changes("stale", stored, now_hash, now_fields)
    assert "monotone_shared_trait" in changed, changed


def test_dropped_field_is_still_reported():
    """A field removed from the architecture outright is a genuine mismatch."""
    stored = stored_fields("M1_bottleneck", "current")
    stored["some_removed_setting"] = 7
    now_hash, now_fields = fp("M1_bottleneck")
    changed = fingerprint_changes("stale", stored, now_hash, now_fields)
    assert "some_removed_setting" in changed


def test_different_rung_refused():
    stored = stored_fields("M1_bottleneck", "current")
    now_hash, now_fields = fp("M2_monotone")
    assert fingerprint_changes("stale", stored, now_hash, now_fields)


# --------------------------------------------------------------------------- #
# the invariant behind the whole design
# --------------------------------------------------------------------------- #
def test_conditional_fields_appear_exactly_when_component_is_on():
    for name, applies in CONDITIONAL_ARCH_FIELDS.items():
        for rung, flags in LADDER_V2:
            cfg = ProposedConfig(**flags)
            _, fields = architecture_fingerprint(rung, cfg, ONTO)
            assert (name in fields) == bool(applies(cfg)), (
                f"{name} presence in {rung} does not match its condition")


# --------------------------------------------------------------------------- #
# the v1 ladder must not change underneath notebooks 06 and 07
# --------------------------------------------------------------------------- #
from symptograph.proposed import LADDER                            # noqa: E402

V1_LADDER_FLAGS = dict(LADDER)


@pytest.mark.parametrize("rung", list(V1_LADDER_FLAGS))
def test_v1_rung_gains_no_v2_component(rung):
    """A v1 rung must not pick up N-A or N-B from a config default.

    ``LADDER`` entries never mention ``use_monotone_ordinal``, so a default of
    True would have given every v1 rung a monotone head - changing results that
    are already written to Drive and cited in the design document.
    """
    cfg = ProposedConfig(**V1_LADDER_FLAGS[rung])
    assert cfg.use_monotone_ordinal is False
    assert cfg.use_noise_correction is False
    assert cfg.freeze_encoder_layers == 0


@pytest.mark.parametrize("rung", list(V1_LADDER_FLAGS))
@pytest.mark.parametrize("stream", [False, True],
                         ids=["concatenated", "weak_stream"])
def test_v1_checkpoint_still_resumes(rung, stream):
    """Existing proposed/ and proposed_stream/ runs stay resumable."""
    flags = V1_LADDER_FLAGS[rung]
    over = {"weak_as_stream": True} if stream and flags.get("use_weak") else {}
    cfg = ProposedConfig(**flags, **over)
    now_hash, now_fields = architecture_fingerprint(rung, cfg, ONTO)
    stored = {k: v for k, v in now_fields.items() if k not in V2_ADDED}
    changed = fingerprint_changes("stale", stored, now_hash, now_fields)
    assert changed == {}, f"{rung} wrongly refused: {changed}"
