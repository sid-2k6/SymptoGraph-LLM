#!/usr/bin/env python3
"""Fit the LLM labeller's per-symptom noise channel (novelty N-B).

A single confidence temperature rescales every label by the same amount, so it
cannot fix *systematic* bias. Measured on this project's own labeller:
``low_self_esteem`` is called present 80% of the time where gold prevalence is
24%, and Cohen's kappa ranges from ~0.76 (suicidal thoughts) down to ~0.4
(hopelessness, depressed mood). Those are different failures per symptom, and
they need a per-symptom correction.

This estimates, for each symptom k::

    tpr[k] = P(LLM says present | gold says present)
    fpr[k] = P(LLM says present | gold says absent)

from ``validation_labels.jsonl`` (the labeller run over the gold symptoms) and
``labelled.parquet``. Training then pushes the model's clean probability through
that channel before comparing it to the weak label.

Identifiability
---------------
Only the gold-labelled symptoms can be measured. The 11 weak-only symptoms have
no gold labels in any split, so their channel is **estimated by pooling within
instrument** (PHQ-9 / GAD-7 / extra) and every such symptom is marked
``"source": "pooled"``. That is an assumption, not a measurement, and the paper
has to say so. It is also why N-B cannot be fully validated until the in-house
anxiety gold set exists.

Usage::

    python scripts/fit_noise_channel.py \
        --validation  DRIVE/weak_labels/validation_labels.jsonl \
        --labelled    DRIVE/processed/labelled.parquet \
        --out         DRIVE/weak_labels/noise_channel.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from symptograph.jsonio import write_json          # noqa: E402
from symptograph.preprocess import Ontology        # noqa: E402

# Laplace smoothing, so a symptom with few gold positives cannot produce
# tpr = 0 or 1 and a correction that collapses the loss.
PRIOR = 2.0


def fit(validation_jsonl: str, labelled_parquet: str, onto: Ontology,
        min_support: int = 20) -> dict:
    rows = []
    with open(validation_jsonl, encoding="utf-8") as fh:
        for line in fh:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not rows:
        raise SystemExit(f"no parsable records in {validation_jsonl}")
    llm = pd.DataFrame([{"post_id": r["post_id"],
                         **{f"llm_{k}": v for k, v in r["labels"].items()}}
                        for r in rows])
    gold = pd.read_parquet(labelled_parquet,
                           columns=["post_id"] + onto.columns)
    df = llm.merge(gold, on="post_id", how="inner")
    print(f"{len(llm):,} labeller records, {len(df):,} joined to gold")

    per, measured = {}, 0
    for s in onto.symptoms:
        lc, gc = f"llm_{s}", onto.col(s)
        rec = {"tpr": None, "fpr": None, "source": "pooled",
               "instrument": onto.instrument(s)}
        if lc in df.columns and gc in df.columns:
            sub = df[(df[lc] >= 0) & (df[gc] >= 0)]
            pos, neg = sub[sub[gc] == 1], sub[sub[gc] == 0]
            rec.update(n_gold_pos=int(len(pos)), n_gold_neg=int(len(neg)))
            if len(pos) >= min_support and len(neg) >= min_support:
                rec["tpr"] = float((pos[lc].sum() + PRIOR)
                                   / (len(pos) + 2 * PRIOR))
                rec["fpr"] = float((neg[lc].sum() + PRIOR)
                                   / (len(neg) + 2 * PRIOR))
                rec["source"] = "measured"
                rec["llm_positive_rate"] = round(float(sub[lc].mean()), 4)
                rec["gold_positive_rate"] = round(float(sub[gc].mean()), 4)
                measured += 1
        per[s] = rec

    # pool the unmeasurable symptoms within instrument, then globally
    done = [v for v in per.values() if v["source"] == "measured"]
    if not done:
        raise SystemExit("no symptom had enough paired gold/LLM labels to fit")
    g_tpr = float(np.mean([v["tpr"] for v in done]))
    g_fpr = float(np.mean([v["fpr"] for v in done]))
    for s, rec in per.items():
        if rec["source"] == "measured":
            continue
        same = [v for v in done if v["instrument"] == rec["instrument"]]
        rec["tpr"] = float(np.mean([v["tpr"] for v in same])) if same else g_tpr
        rec["fpr"] = float(np.mean([v["fpr"] for v in same])) if same else g_fpr
        rec["source"] = "pooled_by_instrument" if same else "pooled_global"

    return {
        "per_symptom": per,
        "n_measured": measured,
        "n_pooled": len(per) - measured,
        "global_tpr": round(g_tpr, 4),
        "global_fpr": round(g_fpr, 4),
        "min_support": min_support,
        "laplace_prior": PRIOR,
        "caveat": ("Symptoms marked pooled_by_instrument or pooled_global have "
                   "NO gold labels, so their channel is assumed rather than "
                   "measured. Report this; it is the same measurement gap that "
                   "prevents N3 from being validated directly."),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--validation", required=True)
    ap.add_argument("--labelled", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ontology", default=None)
    ap.add_argument("--min-support", type=int, default=20)
    a = ap.parse_args()

    onto = Ontology.load(a.ontology) if a.ontology else Ontology.load()
    ch = fit(a.validation, a.labelled, onto, a.min_support)

    tbl = pd.DataFrame([
        {"symptom": s, "instrument": v["instrument"], "source": v["source"],
         "tpr": round(v["tpr"], 3), "fpr": round(v["fpr"], 3),
         "gold_rate": v.get("gold_positive_rate"),
         "llm_rate": v.get("llm_positive_rate"),
         "n_pos": v.get("n_gold_pos"), "n_neg": v.get("n_gold_neg")}
        for s, v in ch["per_symptom"].items()])
    print()
    print(tbl.to_string(index=False))
    print(f"\nmeasured {ch['n_measured']} / {len(tbl)} symptoms; "
          f"{ch['n_pooled']} pooled")
    over = tbl[(tbl.source == "measured") & (tbl.llm_rate.notna())
               & (tbl.llm_rate > 2 * tbl.gold_rate)]
    if len(over):
        print("\nlabeller over-calls these by >2x the gold rate "
              "(exactly what a single temperature cannot fix):")
        print(over[["symptom", "gold_rate", "llm_rate", "fpr"]].to_string(index=False))
    write_json(a.out, ch)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
