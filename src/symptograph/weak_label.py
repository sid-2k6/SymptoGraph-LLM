"""LLM weak supervision for the unannotated symptoms (novelty N3).

Eleven of the 22 unified symptoms have no gold labels in any open corpus, mostly
the GAD-7 anxiety items. This module labels them with an open instruction-tuned
LLM and attaches a calibrated confidence to every label, so the concept loss can
be weighted rather than treating noisy labels as certain.

Scope decision
--------------
Labelling all 1,017,959 weak-pool posts is not feasible: at a realistic 8
posts/second it is roughly 35 GPU-hours, and Colab sessions do not last that
long. The plan is therefore:

* a stratified subsample (default 50,000 posts) receives LLM symptom labels;
* the full pool stays available for self-supervised encoder pretraining, which
  needs no LLM;
* the paper reports a learning curve over subsample size, which is stronger
  evidence than a single large run.

Reliability evidence
--------------------
The weak-only symptoms have no gold labels, so weak-label quality cannot be
measured on them directly. Two things close that gap:

1. Run the same labeller over the *gold* symptoms on the labelled split and
   measure per-symptom agreement and calibration. This establishes whether the
   labeller is trustworthy at all.
2. Hand-annotate a small anxiety set (``build_annotation_sample``) to validate
   the weak-only symptoms directly.

Everything here is resumable. Progress is appended to JSONL after each batch, so
a disconnected session loses at most one batch.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .preprocess import Ontology

DEFAULT_MODELS = {
    "large": "Qwen/Qwen2.5-7B-Instruct",          # needs ~16 GB in fp16 (A100, L4)
    "quantised": "Qwen/Qwen2.5-7B-Instruct-AWQ",  # ~6 GB, fits a T4
    "small": "Qwen/Qwen2.5-3B-Instruct",          # fallback
}


# --------------------------------------------------------------------------- #
# subsampling
# --------------------------------------------------------------------------- #
def build_subsample(weak_pool_path: str, n: int = 50_000, split: str = "train",
                    strata=("disorder_weak", "period"), seed: int = 0,
                    control_share: float = 0.15,
                    columns=("post_id", "text", "subreddit", "period",
                             "disorder_weak", "split", "author_hash")) -> pd.DataFrame:
    """Stratified subsample of the weak pool, balanced across disorder and period.

    Controls are 57% of the pool but contribute little symptom signal, so their
    share is capped at ``control_share``. The remaining quota is spread evenly
    over the clinical disorder groups, which keeps rare conditions such as
    bipolar and schizophrenia represented instead of being swamped by
    r/depression.
    """
    df = pd.read_parquet(weak_pool_path, columns=list(columns))
    df = df[df["split"] == split]
    df = df[df["disorder_weak"].notna()]

    is_control = df["disorder_weak"].isin(["control", "mixed"])
    clinical, control = df[~is_control], df[is_control]

    n_control = int(round(n * control_share))
    n_clinical = n - n_control

    def take(frame: pd.DataFrame, quota: int) -> pd.DataFrame:
        if frame.empty or quota <= 0:
            return frame.head(0)
        cells = list(frame.groupby(list(strata)))
        per = max(quota // max(len(cells), 1), 1)
        picked, leftover = [], 0
        for _, cell in cells:
            k = min(per, len(cell))
            picked.append(cell.sample(n=k, random_state=seed))
            leftover += per - k
        out = pd.concat(picked) if picked else frame.head(0)
        if leftover > 0:                      # redistribute from the larger cells
            remaining = frame.drop(index=out.index)
            extra = min(leftover, len(remaining))
            if extra:
                out = pd.concat([out, remaining.sample(n=extra, random_state=seed + 1)])
        return out.head(quota)

    sample = pd.concat([take(clinical, n_clinical), take(control, n_control)])
    return sample.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def build_annotation_sample(weak_pool_path: str, out_csv: str, n: int = 800,
                            anxiety_subreddits=("anxiety", "healthanxiety", "socialanxiety"),
                            control_n: int = 200, seed: int = 0) -> pd.DataFrame:
    """Posts for the in-house anxiety gold set that validates the weak-only symptoms.

    Writes a CSV with one blank column per weak-only symptom plus an evidence
    span column, ready for two annotators to fill in independently.
    """
    onto = Ontology.load()
    df = pd.read_parquet(weak_pool_path,
                         columns=["post_id", "text", "subreddit", "period", "split"])
    df = df[df["split"] == "train"]
    anx = df[df["subreddit"].isin(anxiety_subreddits)].sample(
        n=min(n - control_n, len(df)), random_state=seed)
    ctl = df[~df["subreddit"].isin(anxiety_subreddits)].sample(n=control_n, random_state=seed)
    out = pd.concat([anx, ctl]).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    weak_only = weak_only_symptoms(onto, gold_symptoms=None)
    for s in weak_only:
        out[s] = ""
    out["evidence_spans"] = ""
    out["annotator"] = ""
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    out.to_csv(out_csv, index=False)
    return out


def weak_only_symptoms(onto: Ontology, gold_symptoms=None) -> list[str]:
    """Symptoms with no gold supervision in the open corpora."""
    if gold_symptoms is None:
        gold = set()
        for src, mapping in onto.label_map.items():
            gold.update(mapping.values())
    else:
        gold = set(gold_symptoms)
    return [s for s in onto.symptoms if s not in gold]


# --------------------------------------------------------------------------- #
# prompting and parsing
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = (
    "You are a careful clinical annotator. You read a social media post and decide, "
    "for each listed symptom, whether the post provides evidence that the author is "
    "experiencing it. Judge only what the text supports. Do not infer a diagnosis. "
    "Reply with JSON only."
)


def build_prompt(text: str, symptoms: list[str], onto: Ontology,
                 max_chars: int = 3000) -> str:
    """One prompt covering every target symptom, with its clinical definition."""
    defs = {m["name"]: m["definition"] for m in onto.meta}
    lines = [f'- "{s}": {defs.get(s, s)}' for s in symptoms]
    schema = ", ".join(f'"{s}": {{"present": 0 or 1, "confidence": 0.0 to 1.0}}'
                       for s in symptoms[:2])
    body = text[:max_chars]
    return (
        f"Post:\n\"\"\"\n{body}\n\"\"\"\n\n"
        f"Symptoms to judge:\n" + "\n".join(lines) + "\n\n"
        f"For each symptom return 'present' (1 if the post gives evidence, else 0) and "
        f"'confidence' (how certain you are, 0.0 to 1.0).\n"
        f"Reply with a single JSON object: {{{schema}, ...}}\n"
        f"Include every symptom exactly once. No commentary."
    )


_JSON_BLOCK = re.compile(r"\{.*\}", re.S)


def parse_response(raw: str, symptoms: list[str]) -> dict | None:
    """Tolerant parse of the model's JSON. Returns None if unusable.

    Handles code fences, leading prose and trailing text. A response missing
    every symptom is rejected so the caller can retry rather than silently
    recording nulls.
    """
    if not raw:
        return None
    m = _JSON_BLOCK.search(raw.replace("```json", " ").replace("```", " "))
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        try:
            obj = json.loads(re.sub(r",\s*([}\]])", r"\1", m.group(0)))
        except json.JSONDecodeError:
            return None
    if not isinstance(obj, dict):
        return None

    out, found = {}, 0
    for s in symptoms:
        v = obj.get(s)
        if isinstance(v, dict):
            present, conf = v.get("present"), v.get("confidence")
        elif isinstance(v, (int, float, bool)):
            present, conf = v, None
        else:
            present, conf = None, None
        if present is None:
            out[s] = (-1, 0.0)
            continue
        try:
            label = 1 if int(float(present)) == 1 else 0
        except (TypeError, ValueError):
            out[s] = (-1, 0.0)
            continue
        try:
            c = float(conf)
            c = min(max(c, 0.0), 1.0)
        except (TypeError, ValueError):
            c = 0.5
        out[s] = (label, c)
        found += 1
    return out if found else None


# --------------------------------------------------------------------------- #
# labellers
# --------------------------------------------------------------------------- #
class MockLabeller:
    """Deterministic fake labeller for testing the pipeline without a GPU.

    Produces well-formed JSON whose labels depend on simple keyword cues, so the
    end-to-end flow, resume logic and calibration code can be exercised in
    seconds before committing GPU hours.
    """

    name = "mock"
    CUES = {
        "anhedonia": ["no interest", "nothing matters", "enjoy"],
        "sleep_disturbance": ["sleep", "insomnia", "awake", "tired of lying"],
        "fatigue": ["exhausted", "no energy", "tired"],
        "appetite_change": ["eat", "appetite", "weight"],
        "psychomotor_change": ["restless", "slow", "fidget"],
        "nervousness": ["nervous", "anxious", "on edge"],
        "uncontrollable_worry": ["cant stop worrying", "can't stop worrying"],
        "excessive_worry": ["worry", "worried"],
        "trouble_relaxing": ["relax", "calm down"],
        "restlessness": ["restless", "pacing"],
        "fear_apprehension": ["afraid", "scared", "panic", "fear"],
    }

    _BODY = re.compile(r'"""(.*?)"""', re.S)

    def generate(self, prompts: list[str], symptoms: list[str]) -> list[str]:
        out = []
        for p in prompts:
            # Match against the post only. The prompt also carries the symptom
            # definitions, which contain the same cue words.
            m = self._BODY.search(p)
            low = (m.group(1) if m else p).lower()
            obj = {}
            for s in symptoms:
                hit = any(c in low for c in self.CUES.get(s, []))
                obj[s] = {"present": 1 if hit else 0,
                          "confidence": round(0.85 if hit else 0.7, 2)}
            out.append(json.dumps(obj))
        return out


@dataclass
class VLLMLabeller:
    """vLLM-backed labeller. Requires a GPU; untested on CPU-only machines."""

    model: str = DEFAULT_MODELS["quantised"]
    max_tokens: int = 700
    temperature: float = 0.0
    gpu_memory_utilization: float = 0.90
    max_model_len: int = 4096
    _llm = None
    _tok = None

    @property
    def name(self) -> str:
        return f"vllm:{self.model}"

    def load(self):
        from vllm import LLM
        from transformers import AutoTokenizer
        quant = "awq" if "AWQ" in self.model.upper() else None
        self._llm = LLM(model=self.model, quantization=quant, dtype="half",
                        gpu_memory_utilization=self.gpu_memory_utilization,
                        max_model_len=self.max_model_len, trust_remote_code=True)
        self._tok = AutoTokenizer.from_pretrained(self.model, trust_remote_code=True)
        return self

    def generate(self, prompts: list[str], symptoms: list[str]) -> list[str]:
        from vllm import SamplingParams
        if self._llm is None:
            self.load()
        chats = [self._tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": p}],
            tokenize=False, add_generation_prompt=True) for p in prompts]
        params = SamplingParams(temperature=self.temperature, max_tokens=self.max_tokens)
        outputs = self._llm.generate(chats, params)
        return [o.outputs[0].text for o in outputs]


# --------------------------------------------------------------------------- #
# labelling loop
# --------------------------------------------------------------------------- #
def run_labelling(df: pd.DataFrame, labeller, symptoms: list[str], out_jsonl: str,
                  onto: Ontology | None = None, batch_size: int = 32,
                  max_retries: int = 1, verbose: bool = True,
                  report_every: int = 10) -> dict:
    """Label ``df`` in batches, appending to ``out_jsonl`` so runs are resumable."""
    onto = onto or Ontology.load()
    os.makedirs(os.path.dirname(out_jsonl) or ".", exist_ok=True)

    done: set[str] = set()
    if os.path.exists(out_jsonl):
        with open(out_jsonl, encoding="utf-8") as fh:
            for line in fh:
                try:
                    done.add(json.loads(line)["post_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    todo = df[~df["post_id"].isin(done)].reset_index(drop=True)
    stats = {"already_done": len(done), "to_label": len(todo),
             "labelled": 0, "parse_failures": 0, "retried": 0}
    if verbose:
        print(f"{len(done):,} already labelled, {len(todo):,} to go")
    if todo.empty:
        return stats

    t0 = time.time()
    with open(out_jsonl, "a", encoding="utf-8") as sink:
        for start in range(0, len(todo), batch_size):
            batch = todo.iloc[start:start + batch_size]
            prompts = [build_prompt(t, symptoms, onto) for t in batch["text"]]
            raws = labeller.generate(prompts, symptoms)
            parsed = [parse_response(r, symptoms) for r in raws]

            for attempt in range(max_retries):
                bad = [i for i, p in enumerate(parsed) if p is None]
                if not bad:
                    break
                stats["retried"] += len(bad)
                retry = labeller.generate([prompts[i] for i in bad], symptoms)
                for i, r in zip(bad, retry):
                    parsed[i] = parse_response(r, symptoms)

            for (_, row), result in zip(batch.iterrows(), parsed):
                if result is None:
                    stats["parse_failures"] += 1
                    continue
                sink.write(json.dumps({
                    "post_id": row["post_id"],
                    "labels": {s: result[s][0] for s in symptoms},
                    "confidence": {s: result[s][1] for s in symptoms},
                    "labeller": getattr(labeller, "name", "unknown"),
                }) + "\n")
                stats["labelled"] += 1
            sink.flush()

            if verbose and (start // batch_size) % report_every == 0:
                seen = start + len(batch)
                rate = seen / max(time.time() - t0, 1e-6)
                left = (len(todo) - seen) / max(rate, 1e-6)
                print(f"  {seen:>7,}/{len(todo):,}  {rate:5.1f} posts/s  "
                      f"eta {left / 60:5.1f} min", flush=True)

    stats["elapsed_s"] = round(time.time() - t0, 1)
    stats["posts_per_second"] = round(stats["labelled"] / max(stats["elapsed_s"], 1e-6), 2)
    return stats


def load_labels(out_jsonl: str, symptoms: list[str]) -> pd.DataFrame:
    """Read the JSONL log into a wide frame of labels and confidences."""
    rows = []
    with open(out_jsonl, encoding="utf-8") as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            rec = {"post_id": r["post_id"], "labeller": r.get("labeller")}
            for s in symptoms:
                rec[f"wl_{s}"] = r["labels"].get(s, -1)
                rec[f"conf_{s}"] = r["confidence"].get(s, 0.0)
            rows.append(rec)
    return pd.DataFrame(rows).drop_duplicates("post_id", keep="last")


# --------------------------------------------------------------------------- #
# validation and calibration
# --------------------------------------------------------------------------- #
def validate_against_gold(weak: pd.DataFrame, gold: pd.DataFrame, symptoms: list[str],
                          onto: Ontology | None = None) -> pd.DataFrame:
    """Per-symptom agreement between LLM labels and gold labels.

    Run on the symptoms that do have gold labels. This is the evidence that the
    weak labeller is reliable enough to trust on the symptoms that do not.
    """
    onto = onto or Ontology.load()
    merged = gold.merge(weak, on="post_id", how="inner")
    rows = []
    for s in symptoms:
        gcol, wcol, ccol = onto.col(s), f"wl_{s}", f"conf_{s}"
        if gcol not in merged or wcol not in merged:
            continue
        sub = merged[(merged[gcol] >= 0) & (merged[wcol] >= 0)]
        if sub.empty:
            continue
        y, yhat = sub[gcol].to_numpy(), sub[wcol].to_numpy()
        tp = int(((y == 1) & (yhat == 1)).sum())
        fp = int(((y == 0) & (yhat == 1)).sum())
        fn = int(((y == 1) & (yhat == 0)).sum())
        tn = int(((y == 0) & (yhat == 0)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        acc = (tp + tn) / len(sub)
        po = acc
        pe = (((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp)) / (len(sub) ** 2))
        kappa = (po - pe) / (1 - pe) if pe < 1 else 0.0
        rows.append({"symptom": s, "n": len(sub), "gold_positive_rate": round(y.mean(), 3),
                     "llm_positive_rate": round(yhat.mean(), 3),
                     "precision": round(prec, 3), "recall": round(rec, 3),
                     "f1": round(f1, 3), "accuracy": round(acc, 3),
                     "cohen_kappa": round(kappa, 3),
                     "mean_confidence": round(float(sub[ccol].mean()), 3)})
    return pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)


def expected_calibration_error(conf: np.ndarray, correct: np.ndarray, bins: int = 10):
    """ECE plus the per-bin reliability table."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece, rows = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if not sel.any():
            continue
        acc, avg = correct[sel].mean(), conf[sel].mean()
        ece += (sel.sum() / len(conf)) * abs(acc - avg)
        rows.append({"bin": f"({lo:.1f}, {hi:.1f}]", "n": int(sel.sum()),
                     "mean_confidence": round(float(avg), 3),
                     "accuracy": round(float(acc), 3),
                     "gap": round(float(acc - avg), 3)})
    return float(ece), pd.DataFrame(rows)


def fit_confidence_scaler(weak: pd.DataFrame, gold: pd.DataFrame, symptoms: list[str],
                          onto: Ontology | None = None):
    """Fit a single temperature that maps raw confidence onto observed accuracy.

    Returns ``(temperature, before_ece, after_ece, reliability_table)``. Applied
    as ``p_calibrated = p ** temperature`` renormalised, which is monotone and
    needs only one parameter, so it is stable on a small gold set.
    """
    onto = onto or Ontology.load()
    merged = gold.merge(weak, on="post_id", how="inner")
    conf, correct = [], []
    for s in symptoms:
        gcol, wcol, ccol = onto.col(s), f"wl_{s}", f"conf_{s}"
        if gcol not in merged or wcol not in merged:
            continue
        sub = merged[(merged[gcol] >= 0) & (merged[wcol] >= 0)]
        conf.extend(sub[ccol].tolist())
        correct.extend((sub[gcol] == sub[wcol]).astype(float).tolist())
    if not conf:
        return None, None, None, pd.DataFrame()

    conf, correct = np.asarray(conf), np.asarray(correct)
    before, _ = expected_calibration_error(conf, correct)
    grid = np.linspace(0.25, 4.0, 76)
    best_t, best_ece = 1.0, before
    for t in grid:
        ece, _ = expected_calibration_error(np.clip(conf ** t, 0, 1), correct)
        if ece < best_ece:
            best_t, best_ece = float(t), ece
    _, table = expected_calibration_error(np.clip(conf ** best_t, 0, 1), correct)
    return best_t, before, best_ece, table


def save_weak_labels(weak: pd.DataFrame, sample: pd.DataFrame, out_dir: str,
                     symptoms: list[str], stats: dict, temperature: float | None = None,
                     validation: pd.DataFrame | None = None) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    merged = sample.merge(weak, on="post_id", how="inner")
    if temperature:
        for s in symptoms:
            merged[f"conf_{s}"] = np.clip(merged[f"conf_{s}"] ** temperature, 0, 1)
    path = os.path.join(out_dir, "weak_labels.parquet")
    merged.to_parquet(path, index=False, compression="zstd")

    paths = {"weak_labels": path}
    if validation is not None and not validation.empty:
        v = os.path.join(out_dir, "labeller_validation.csv")
        validation.to_csv(v, index=False)
        paths["validation"] = v

    manifest = {
        "target_symptoms": symptoms,
        "rows": int(len(merged)),
        "confidence_temperature": temperature,
        "run_stats": stats,
        "notes": [
            "Labels: 1 present, 0 absent, -1 unparsed. Confidence in [0, 1].",
            "Weight the concept loss by confidence; never treat -1 as a negative.",
            "Calibrated on the gold symptoms, then applied to the weak-only symptoms.",
        ],
    }
    m = os.path.join(out_dir, "weak_label_manifest.json")
    with open(m, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, default=str)
    paths["manifest"] = m
    return paths
