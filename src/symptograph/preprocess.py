"""Preprocessing and leakage-controlled splitting for SymptoGraph-LLM.

Turns seven heterogeneous mental-health corpora into one canonical table with a
shared symptom label space, then assigns splits so that no post, and no Reddit
author, can appear on both sides of a split.

Why this is not optional
------------------------
Measured on the downloaded data:

* ``DepSeverity`` is a re-annotation of the ``Dreaddit`` corpus - the same 3,526
  posts carry severity labels in one and binary stress labels in the other.
* ``LoST`` and ``IRF`` share 3,089 posts.
* ``IMHI`` re-packages the IRF and Dreaddit posts behind a ``"Post: "`` prefix.
* 23.3% of Reddit Mental Health authors (192,893 of 826,961) appear in more
  than one file.

Splitting each dataset independently would therefore leak training posts into
test sets. Every unique post gets one global ``post_id`` and exactly one split.

Label masking
-------------
Symptom columns use 1 (present), 0 (absent) and -1 (not annotated by this
source). Only non-negative entries should contribute to the concept loss,
otherwise unannotated symptoms are wrongly treated as negatives.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .jsonio import read_json, write_json

CONFIG = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "symptom_ontology.json")
DEDUP_PREFIX = 200   # characters of normalised text used to match near-duplicates
MIN_CHARS = 40       # posts shorter than this are dropped as uninformative
AUTHOR_SALT = "symptograph-v1"
PLACEHOLDER_AUTHORS = {"[deleted]", "[removed]", "automoderator", "none", "nan", ""}


# --------------------------------------------------------------------------- #
# text utilities
# --------------------------------------------------------------------------- #
_URL = re.compile(r"https?://\S+|www\.\S+")
_USER = re.compile(r"/?\bu/[A-Za-z0-9_-]+")
_SUB = re.compile(r"/?\br/[A-Za-z0-9_]+")
_IMHI_PREFIX = re.compile(r"^\s*Post:\s*")
_IMHI_TAIL = re.compile(r"\s*Question:.*$", re.S)
_ZW = re.compile(r"[\u200b-\u200f\u2028\u2029\ufeff]")
_WS = re.compile(r"\s+")
_NON_ALNUM = re.compile(r"[^a-z0-9 ]+")


def clean_text(text: object) -> str:
    """Normalise a post for modelling, keeping punctuation and casing."""
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    t = str(text)
    t = _IMHI_PREFIX.sub("", t)
    t = _IMHI_TAIL.sub("", t)
    t = t.replace("###", " ")            # DepressionEmo joins title and body with ###
    t = _ZW.sub("", t)
    t = _URL.sub(" ", t)
    t = _USER.sub(" ", t)
    t = _SUB.sub(" ", t)
    t = t.replace("&amp;", "&").replace("&gt;", ">").replace("&lt;", "<").replace("&#x200B;", " ")
    t = re.sub(r"[*_`>#]+", " ", t)      # markdown emphasis and quote markers
    return _WS.sub(" ", t).strip()


def dedup_key(text: object) -> str | None:
    """Stable key matching the same post across corpora, robust to truncation."""
    norm = _WS.sub(" ", _NON_ALNUM.sub(" ", clean_text(text).lower())).strip()
    if len(norm) < MIN_CHARS:
        return None
    return hashlib.md5(norm[:DEDUP_PREFIX].encode()).hexdigest()[:16]


def hash_author(author: object) -> str | None:
    """Pseudonymise a Reddit username. Returns None for bots and placeholders."""
    a = str(author).strip()
    if a.lower() in PLACEHOLDER_AUTHORS:
        return None
    return "a_" + hashlib.sha256((AUTHOR_SALT + a).encode()).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# ontology
# --------------------------------------------------------------------------- #
@dataclass
class Ontology:
    symptoms: list[str]
    label_map: dict
    exhaustive: dict
    subreddit_disorder: dict
    severity_scales: dict
    meta: list = field(default_factory=list)

    @classmethod
    def load(cls, path: str = CONFIG) -> "Ontology":
        cfg = read_json(path)
        return cls(
            symptoms=[s["name"] for s in cfg["symptoms"]],
            label_map=cfg["label_map"],
            exhaustive={k: v for k, v in cfg["exhaustive_sources"].items() if k != "comment"},
            subreddit_disorder=cfg["subreddit_disorder_map"],
            severity_scales=cfg["severity_scales"],
            meta=cfg["symptoms"],
        )

    def col(self, symptom: str) -> str:
        return f"sym_{symptom}"

    @property
    def columns(self) -> list[str]:
        return [self.col(s) for s in self.symptoms]


def _blank_symptoms(n: int, onto: Ontology) -> dict:
    """All symptoms unknown (-1) by default."""
    return {c: np.full(n, -1, dtype=np.int8) for c in onto.columns}


# --------------------------------------------------------------------------- #
# loaders - each returns the canonical schema
# --------------------------------------------------------------------------- #
def load_depressionemo(data: str, onto: Ontology) -> pd.DataFrame:
    rows = []
    for split in ("train", "val", "test"):
        path = os.path.join(data, "depressionemo", "Dataset", f"{split}.json")
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    r["_split"] = "val" if split == "val" else split
                    rows.append(r)
    df = pd.DataFrame(rows)
    out = pd.DataFrame({
        "text_raw": df["text"],
        "source": "depressionemo",
        "pool": "symptom_gold",
        "official_split": df["_split"],
        "date": pd.to_datetime(df["date"], errors="coerce"),
    })
    mapping = onto.label_map["depressionemo"]
    sym = _blank_symptoms(len(df), onto)
    for src_label, unified in mapping.items():          # exhaustive scheme -> 0 means absent
        sym[onto.col(unified)] = np.zeros(len(df), dtype=np.int8)
    for i, emotions in enumerate(df["emotions"]):
        for e in emotions or []:
            unified = mapping.get(e)
            if unified:
                sym[onto.col(unified)][i] = 1
    return pd.concat([out, pd.DataFrame(sym)], axis=1)


def load_depseverity_dreaddit(data: str, onto: Ontology) -> pd.DataFrame:
    """One pool: identical posts, severity labels from DepSeverity, stress from Dreaddit."""
    sev = pd.read_csv(os.path.join(data, "depseverity", "Reddit_depression_dataset.csv"))
    sev["key"] = sev["text"].map(dedup_key)
    sev = sev.dropna(subset=["key"]).drop_duplicates("key")
    scale = onto.severity_scales["severity_depression"]
    sev["severity_depression"] = sev["label"].map(scale).astype("Float32")

    frames = []
    for path in sorted(glob.glob(os.path.join(data, "dreaddit", "*.parquet"))):
        d = pd.read_parquet(path, columns=["text", "label", "subreddit", "social_timestamp"])
        d["official_split"] = "test" if "test" in os.path.basename(path) else "train"
        frames.append(d)
    if frames:
        dre = pd.concat(frames, ignore_index=True)
        dre["key"] = dre["text"].map(dedup_key)
        dre = dre.dropna(subset=["key"]).drop_duplicates("key")
    else:
        dre = pd.DataFrame(columns=["text", "label", "subreddit", "social_timestamp",
                                    "official_split", "key"])

    dre = dre.rename(columns={"text": "text_stress", "label": "stress_label"})
    merged = sev.rename(columns={"text": "text_sev"}).merge(
        dre[["key", "text_stress", "stress_label", "subreddit", "social_timestamp",
             "official_split"]],
        on="key", how="outer")
    out = pd.DataFrame({
        "text_raw": merged["text_sev"].combine_first(merged["text_stress"]),
        "source": "depseverity+dreaddit",
        "pool": "severity_stress",
        "official_split": merged["official_split"],
        "severity_depression": merged["severity_depression"],
        "stress": pd.to_numeric(merged["stress_label"], errors="coerce").astype("Float32"),
        "subreddit": merged["subreddit"],
        "date": pd.to_datetime(merged["social_timestamp"], unit="s", errors="coerce"),
    })
    return pd.concat([out.reset_index(drop=True),
                      pd.DataFrame(_blank_symptoms(len(out), onto))], axis=1)


def load_irf_lost(data: str, onto: Ontology) -> pd.DataFrame:
    """IRF and LoST share most of their posts, so they are merged on the dedup key."""
    irf_parts = []
    for split in ("train", "val", "test"):
        p = os.path.join(data, "irf", f"{split}_data.csv")
        if os.path.exists(p):
            d = pd.read_csv(p)
            d["official_split"] = split
            irf_parts.append(d)
    irf = pd.concat(irf_parts, ignore_index=True) if irf_parts else pd.DataFrame()
    if not irf.empty:
        irf["key"] = irf["text"].map(dedup_key)
        irf = irf.dropna(subset=["key"]).drop_duplicates("key")

    lost_path = os.path.join(data, "lost", "LoSTdataset.csv")
    lost = pd.read_csv(lost_path) if os.path.exists(lost_path) else pd.DataFrame()
    if not lost.empty:
        lost["key"] = lost["text"].map(dedup_key)
        lost = lost.dropna(subset=["key"]).drop_duplicates("key")

    keep_irf = ["key", "text", "belong", "burden", "belong_exp", "burden_exp", "official_split"]
    merged = (irf[[c for c in keep_irf if c in irf.columns]]
              .merge(lost[["key", "text", "self"]], on="key", how="outer", suffixes=("", "_lost")))

    out = pd.DataFrame({
        "text_raw": merged["text"].fillna(merged.get("text_lost")),
        "source": "irf+lost",
        "pool": "concept_spans",
        "official_split": merged.get("official_split"),
    })
    sym = _blank_symptoms(len(merged), onto)
    for src, unified in onto.label_map["irf"].items():
        if src in merged:
            v = pd.to_numeric(merged[src], errors="coerce")
            sym[onto.col(unified)] = np.where(v.isna(), -1, v.fillna(0)).astype(np.int8)
    if "self" in merged:
        v = pd.to_numeric(merged["self"], errors="coerce")
        sym[onto.col(onto.label_map["lost"]["self"])] = np.where(v.isna(), -1, v.fillna(0)).astype(np.int8)

    spans = []
    for _, r in merged.iterrows():
        d = {}
        for src_col, unified in (("belong_exp", "thwarted_belongingness"),
                                 ("burden_exp", "perceived_burdensomeness")):
            val = r.get(src_col)
            if isinstance(val, str) and val.strip():
                d[unified] = val.strip()
        spans.append(json.dumps(d) if d else None)
    out["spans"] = spans
    return pd.concat([out.reset_index(drop=True), pd.DataFrame(sym)], axis=1)


def load_cssrs(data: str, onto: Ontology) -> pd.DataFrame:
    path = os.path.join(data, "cssrs_reddit_500", "anonymized_dataset",
                        "500_Reddit_users_posts_labels.csv")
    df = pd.read_csv(path)
    scale = onto.severity_scales["severity_risk"]
    out = pd.DataFrame({
        "text_raw": df["Post"],
        "source": "cssrs",
        "pool": "risk_severity",
        "severity_risk": df["Label"].map(scale).astype("Float32"),
        "author_hash": df["User"].map(lambda u: "cssrs_" + str(u)),   # already anonymised
    })
    return pd.concat([out, pd.DataFrame(_blank_symptoms(len(out), onto))], axis=1)


# IMHI subsets that do not overlap the training pools, usable as external tests
IMHI_EXTERNAL = {"swmh", "CAMS", "t-sid", "loneliness", "DR"}


def load_imhi(data: str, onto: Ontology, external_only: bool = False) -> pd.DataFrame:
    """IMHI instruction data. ``external_only`` keeps just the non-overlapping subsets."""
    rows = []
    patterns = [
        (os.path.join(data, "imhi", "train_data", "complete_data", "*", "*.csv"), "train"),
        (os.path.join(data, "imhi", "test_data", "test_complete", "*.csv"), "test"),
    ]
    for pattern, split in patterns:
        for path in sorted(glob.glob(pattern)):
            subset = (os.path.basename(os.path.dirname(path)) if split == "train"
                      else os.path.splitext(os.path.basename(path))[0])
            if external_only and subset not in IMHI_EXTERNAL:
                continue
            d = pd.read_csv(path)
            if "post" not in d.columns:
                continue
            rows.append(pd.DataFrame({
                "text_raw": d["post"],
                "source": f"imhi:{subset}",
                "pool": "explanation_external" if subset in IMHI_EXTERNAL else "explanation",
                "official_split": split,
                "imhi_question": d.get("question"),
                "imhi_response": d.get("response"),
            }))
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    return pd.concat([out, pd.DataFrame(_blank_symptoms(len(out), onto))], axis=1)


def load_reddit_mh(data: str, onto: Ontology, subreddits=None, periods=None,
                   max_per_file: int | None = None) -> pd.DataFrame:
    """Reddit Mental Health Dataset: weak disorder labels plus the unlabelled pool."""
    files = sorted(glob.glob(os.path.join(data, "reddit_mental_health_slim", "*.csv.gz")))
    if not files:
        files = sorted(glob.glob(os.path.join(data, "reddit_mental_health", "*.csv")))
    rows = []
    for path in files:
        base = os.path.basename(path).replace("_features_tfidf_256.csv", "").replace(".csv.gz", "").replace(".csv", "")
        sub, _, period = base.rpartition("_")
        if subreddits and sub not in subreddits:
            continue
        if periods and period not in periods:
            continue
        d = pd.read_csv(path, compression="infer", usecols=["subreddit", "author", "date", "post"])
        if max_per_file:
            d = d.head(max_per_file)
        d["period"] = period
        rows.append(d)
    if not rows:
        return pd.DataFrame()
    df = pd.concat(rows, ignore_index=True)
    out = pd.DataFrame({
        "text_raw": df["post"],
        "source": "reddit_mental_health",
        "pool": "weak_unlabelled",
        "subreddit": df["subreddit"],
        "period": df["period"],
        "date": pd.to_datetime(df["date"], errors="coerce"),
        "author_hash": df["author"].map(hash_author),
        "disorder_weak": df["subreddit"].map(onto.subreddit_disorder),
    })
    return pd.concat([out, pd.DataFrame(_blank_symptoms(len(out), onto))], axis=1)


# --------------------------------------------------------------------------- #
# canonicalisation
# --------------------------------------------------------------------------- #
CANON = ["post_id", "source", "pool", "text", "text_raw", "n_chars", "n_words",
         "author_hash", "subreddit", "period", "date", "official_split",
         "disorder_weak", "severity_depression", "severity_risk", "stress",
         "spans", "imhi_question", "imhi_response", "split"]


def canonicalise(frames: list[pd.DataFrame], onto: Ontology) -> tuple[pd.DataFrame, dict]:
    """Clean, filter, deduplicate and merge labels across sources."""
    df = pd.concat([f for f in frames if f is not None and not f.empty], ignore_index=True)
    stats = {"loaded_rows": int(len(df))}

    df["text"] = df["text_raw"].map(clean_text)
    df["post_id"] = df["text_raw"].map(dedup_key)

    before = len(df)
    df = df[df["post_id"].notna()].copy()
    stats["dropped_short_or_empty"] = int(before - len(df))

    df["n_chars"] = df["text"].str.len()
    df["n_words"] = df["text"].str.split().map(len)

    for c in CANON:
        if c not in df.columns:
            df[c] = pd.NA

    sym_cols = onto.columns
    df[sym_cols] = df[sym_cols].fillna(-1).astype(np.int8)

    # Merge rows that are the same post: symptom labels take the most informative
    # value (1 > 0 > -1), other fields take the first non-null. Only the duplicated
    # post_ids need the expensive group-wise aggregation, which matters because the
    # Reddit pool contributes about 1.1M already-unique rows.
    stats["rows_before_merge"] = int(len(df))
    dup_mask = df["post_id"].duplicated(keep=False)
    unique_rows, dup_rows = df[~dup_mask], df[dup_mask]
    stats["rows_needing_merge"] = int(len(dup_rows))

    agg = {c: "first" for c in CANON if c != "post_id"}
    agg.update({c: "max" for c in sym_cols})
    agg["source"] = lambda s: "|".join(sorted(set(s)))
    agg["pool"] = lambda s: "|".join(sorted(set(s)))

    def strictest_split(values):
        """If any source holds a post in its test set, never train on it."""
        seen = {str(v) for v in values if pd.notna(v)}
        seen = {"val" if v in ("valid", "dev") else v for v in seen}
        for level in ("test", "val", "train"):
            if level in seen:
                return level
        return pd.NA

    agg["official_split"] = strictest_split
    for c in ("severity_depression", "severity_risk", "stress"):
        agg[c] = "max"
    if len(dup_rows):
        collapsed = dup_rows.groupby("post_id", as_index=False).agg(agg)
        merged = pd.concat([unique_rows[collapsed.columns], collapsed], ignore_index=True)
    else:
        merged = unique_rows.reset_index(drop=True)

    stats["unique_posts"] = int(len(merged))
    stats["duplicate_rows_merged"] = stats["rows_before_merge"] - stats["unique_posts"]
    return merged, stats


# --------------------------------------------------------------------------- #
# splitting
# --------------------------------------------------------------------------- #
def _bucket(value: str, salt: str) -> float:
    """Deterministic uniform value in [0, 1) for stable hash-based splitting."""
    h = hashlib.md5((salt + str(value)).encode()).hexdigest()[:8]
    return int(h, 16) / 0xFFFFFFFF


def _stratified_split(df: pd.DataFrame, label: str, seed: str,
                      val_frac: float, test_frac: float) -> pd.Series:
    """Deterministic hash split applied within each label value, for balance."""
    out = pd.Series(pd.NA, index=df.index, dtype="object")
    for value, grp in df.groupby(label):
        b = grp["post_id"].map(lambda p: _bucket(p, f"{seed}strat{label}{value}"))
        out.loc[grp.index] = np.where(b < test_frac, "test",
                                     np.where(b < test_frac + val_frac, "val", "train"))
    return out


# Pools re-split by our own stratified rule rather than the source's official split.
# Justified because the published splits for these corpora were stratified for a
# different target (Dreaddit for binary stress, not depression severity), which
# leaves too few mild/severe cases in test for a stable ordinal metric.
STRATIFY = {"severity_stress": "severity_depression", "risk_severity": "severity_risk"}


def assign_splits(df: pd.DataFrame, onto: Ontology, val_frac=0.15, test_frac=0.15,
                  seed: str = "symptograph", stratify: dict | None = None) -> pd.DataFrame:
    """Give every unique post exactly one split.

    Rules, in order of precedence:
      1. Posts in ``explanation_external`` pools are test-only.
      2. Pools in ``stratify`` are split stratified on their target label.
      3. Where the source provides an official split, keep it (strictest wins).
      4. Reddit posts with a known author are split author-disjoint.
      5. Everything else is split by a hash of ``post_id``.
    """
    df = df.copy()
    stratify = STRATIFY if stratify is None else stratify
    split = pd.Series(pd.NA, index=df.index, dtype="object")
    how = pd.Series(pd.NA, index=df.index, dtype="object")

    # 1. external evaluation sets
    ext = df["pool"].astype(str).str.contains("explanation_external")
    split[ext] = "test"
    how[ext] = "external_test_only"

    # 2. stratified pools
    pool_s = df["pool"].astype(str)
    for pool_key, label in stratify.items():
        if label not in df.columns:
            continue
        sel = pool_s.str.contains(pool_key, regex=False) & split.isna() & df[label].notna()
        if sel.any():
            split[sel] = _stratified_split(df.loc[sel], label, seed, val_frac, test_frac)
            how[sel] = f"stratified:{label}"

    # 3. official splits
    off = df["official_split"].notna() & split.isna()
    split[off] = df.loc[off, "official_split"].replace({"valid": "val", "dev": "val"})
    how[off] = "official"

    # 3. author-disjoint for the remaining Reddit rows
    auth = df["author_hash"].notna() & split.isna()
    if auth.any():
        b = df.loc[auth, "author_hash"].map(lambda a: _bucket(a, seed + "author"))
        split[auth] = np.where(b < test_frac, "test",
                               np.where(b < test_frac + val_frac, "val", "train"))
        how[auth] = "author_disjoint"

    # 4. hash of post_id
    rest = split.isna()
    if rest.any():
        b = df.loc[rest, "post_id"].map(lambda p: _bucket(p, seed + "post"))
        split[rest] = np.where(b < test_frac, "test",
                               np.where(b < test_frac + val_frac, "val", "train"))
        how[rest] = "post_hash"

    df["split"] = split.astype(str)
    df["split_strategy"] = how.astype(str)

    df = _enforce_author_disjoint(df)

    # Official splits can leave no val set; carve one from train by hash.
    for pool, grp in df.groupby(df["pool"].astype(str)):
        if "val" not in set(grp["split"]) and "train" in set(grp["split"]):
            tr = grp.index[grp["split"] == "train"]
            b = df.loc[tr, "post_id"].map(lambda p: _bucket(p, seed + "val" + pool))
            pick = tr[b < val_frac / (1 - test_frac)]
            df.loc[pick, "split"] = "val"
            df.loc[pick, "split_strategy"] = df.loc[pick, "split_strategy"] + "+val_carved"
    return df


_STRICTNESS = {"train": 0, "val": 1, "test": 2}


def _enforce_author_disjoint(df: pd.DataFrame) -> pd.DataFrame:
    """Repair authors whose posts landed in different splits.

    Needed because some Reddit Mental Health posts also appear in corpora that
    carry an official split (SWMH inside IMHI, for example), and that official
    split takes precedence over author-disjoint assignment. Every such author is
    moved wholesale to the strictest split they touch, so a user never has posts
    on both sides of an evaluation boundary.
    """
    known = df["author_hash"].notna()
    if not known.any():
        return df
    per_author = df.loc[known].groupby("author_hash")["split"].agg(
        lambda s: max(set(s), key=lambda v: _STRICTNESS.get(v, 0)))
    spanning = df.loc[known].groupby("author_hash")["split"].nunique()
    bad = spanning[spanning > 1].index
    if len(bad) == 0:
        return df
    sel = known & df["author_hash"].isin(bad)
    df.loc[sel, "split"] = df.loc[sel, "author_hash"].map(per_author)
    df.loc[sel, "split_strategy"] = df.loc[sel, "split_strategy"].astype(str) + "+author_repair"
    return df


def audit_leakage(df: pd.DataFrame) -> dict:
    """Confirm no post and no author crosses a split. Raises on failure."""
    report = {}
    per_post = df.groupby("post_id")["split"].nunique()
    report["posts_in_multiple_splits"] = int((per_post > 1).sum())

    known = df[df["author_hash"].notna()]
    per_author = known.groupby("author_hash")["split"].nunique()
    report["authors_in_multiple_splits"] = int((per_author > 1).sum())
    report["authors_total"] = int(len(per_author))

    report["split_sizes"] = df["split"].value_counts().to_dict()
    report["pools"] = df.groupby([df["pool"].astype(str), "split"]).size().unstack(fill_value=0).to_dict()

    problems = [k for k in ("posts_in_multiple_splits", "authors_in_multiple_splits") if report[k] > 0]
    if problems:
        raise AssertionError(f"leakage detected: { {k: report[k] for k in problems} }")
    return report


# --------------------------------------------------------------------------- #
# output
# --------------------------------------------------------------------------- #
def label_coverage(df: pd.DataFrame, onto: Ontology) -> pd.DataFrame:
    """Per-symptom count of annotated / positive rows, by supervision status."""
    rows = []
    for s, meta in zip(onto.symptoms, onto.meta):
        c = onto.col(s)
        known = (df[c] >= 0)
        rows.append({
            "symptom": s,
            "instrument": meta["instrument"],
            "item": meta["item"],
            "annotated": int(known.sum()),
            "positive": int((df[c] == 1).sum()),
            "supervision": "gold" if known.any() else "weak_only (needs N3)",
        })
    return pd.DataFrame(rows).sort_values(["supervision", "instrument", "item"],
                                          na_position="last").reset_index(drop=True)


def save(df: pd.DataFrame, onto: Ontology, out_dir: str, stats: dict, report: dict,
         weak_written: bool = False) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    cols = [c for c in CANON if c in df.columns] + ["split_strategy"] + onto.columns
    tidy = df[[c for c in dict.fromkeys(cols) if c in df.columns]]

    paths = {}
    big = tidy["pool"].astype(str).str.contains("weak_unlabelled")
    labelled, unlabelled = tidy[~big], tidy[big]
    p = os.path.join(out_dir, "labelled.parquet")
    labelled.to_parquet(p, index=False, compression="zstd")
    paths["labelled"] = p
    if len(unlabelled):
        p = os.path.join(out_dir, "weak_pool.parquet")
        unlabelled.to_parquet(p, index=False, compression="zstd")
        paths["weak_pool"] = p
    elif weak_written:
        paths["weak_pool"] = os.path.join(out_dir, "weak_pool.parquet")

    cov = label_coverage(tidy, onto)
    cov.to_csv(os.path.join(out_dir, "label_coverage.csv"), index=False)

    weak_rows = int(len(unlabelled))
    if not weak_rows and weak_written:
        weak_rows = int(stats.get("weak_pool", {}).get("rows_written", 0))

    manifest = {
        "schema_version": 1,
        "dedup_prefix_chars": DEDUP_PREFIX,
        "min_chars": MIN_CHARS,
        "symptoms": onto.symptoms,
        "counts": {"labelled_rows": int(len(labelled)), "weak_pool_rows": weak_rows},
        "preprocessing_stats": stats,
        "leakage_report": report,
        "files": paths,
        "notes": [
            "Symptom columns: 1 present, 0 absent, -1 not annotated by the source.",
            "DepSeverity and Dreaddit are the same posts; merged into pool severity_stress.",
            "LoST and IRF overlap heavily; merged into pool concept_spans.",
            "IMHI subsets swmh/CAMS/t-sid/loneliness/DR are external and test-only.",
        ],
    }
    write_json(os.path.join(out_dir, "manifest.json"), manifest)
    paths["manifest"] = os.path.join(out_dir, "manifest.json")
    paths["label_coverage"] = os.path.join(out_dir, "label_coverage.csv")
    return paths


WEAK_COLS = ["post_id", "source", "pool", "text", "n_chars", "n_words",
             "author_hash", "subreddit", "period", "date", "disorder_weak",
             "split", "split_strategy"]


def stream_weak_pool(data_dir: str, out_path: str, onto: Ontology,
                     labelled_keys: set, author_splits: dict,
                     subreddits=None, periods=None, max_per_file=None,
                     val_frac=0.15, test_frac=0.15, seed="symptograph",
                     verbose=True) -> dict:
    """Process the 1.1M-post Reddit pool one file at a time, writing incrementally.

    Holding the whole pool in memory peaks near 16 GB, which exceeds a free Colab
    runtime. Splits stay globally consistent without global state because author
    assignment is a deterministic hash, with explicit overrides for the authors
    that also appear in the labelled pools.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    files = sorted(glob.glob(os.path.join(data_dir, "reddit_mental_health_slim", "*.csv.gz")))
    if not files:
        files = sorted(glob.glob(os.path.join(data_dir, "reddit_mental_health", "*.csv")))

    seen: set[int] = set()
    writer = None
    stats = {"files": 0, "rows_read": 0, "dropped_short_or_empty": 0,
             "dropped_duplicate_of_labelled": 0, "dropped_duplicate_within_pool": 0,
             "rows_written": 0, "author_overrides_applied": 0}
    splits_count: dict[str, int] = {}
    try:
        for path in files:
            base = (os.path.basename(path).replace("_features_tfidf_256.csv", "")
                    .replace(".csv.gz", "").replace(".csv", ""))
            sub, _, period = base.rpartition("_")
            if (subreddits and sub not in subreddits) or (periods and period not in periods):
                continue
            d = pd.read_csv(path, compression="infer",
                            usecols=["subreddit", "author", "date", "post"])
            if max_per_file:
                d = d.head(max_per_file)
            stats["files"] += 1
            stats["rows_read"] += len(d)

            text = d["post"].map(clean_text)
            key = d["post"].map(dedup_key)
            ok = key.notna()
            stats["dropped_short_or_empty"] += int((~ok).sum())
            d, text, key = d[ok], text[ok], key[ok]

            in_labelled = key.map(lambda k: k in labelled_keys)
            stats["dropped_duplicate_of_labelled"] += int(in_labelled.sum())
            d, text, key = d[~in_labelled], text[~in_labelled], key[~in_labelled]

            as_int = key.map(lambda k: int(k, 16))
            fresh = ~as_int.isin(seen) & ~as_int.duplicated()
            stats["dropped_duplicate_within_pool"] += int((~fresh).sum())
            d, text, key, as_int = d[fresh], text[fresh], key[fresh], as_int[fresh]
            seen.update(as_int.tolist())
            if d.empty:
                continue

            author = d["author"].map(hash_author)
            override = author.map(lambda a: author_splits.get(a))
            stats["author_overrides_applied"] += int(override.notna().sum())
            bucket = author.map(
                lambda a: _bucket(a if a else "anon", seed + "author"))
            hashed = np.where(bucket < test_frac, "test",
                              np.where(bucket < test_frac + val_frac, "val", "train"))
            split = override.fillna(pd.Series(hashed, index=d.index))

            out = pd.DataFrame({
                "post_id": key.values,
                "source": "reddit_mental_health",
                "pool": "weak_unlabelled",
                "text": text.values,
                "n_chars": text.str.len().values,
                "n_words": text.str.split().map(len).values,
                "author_hash": author.values,
                "subreddit": d["subreddit"].values,
                "period": period,
                "date": pd.to_datetime(d["date"], errors="coerce").values,
                "disorder_weak": d["subreddit"].map(onto.subreddit_disorder).values,
                "split": split.values,
                "split_strategy": np.where(override.notna(), "author_override", "author_disjoint"),
            })
            for s, n in out["split"].value_counts().items():
                splits_count[s] = splits_count.get(s, 0) + int(n)
            stats["rows_written"] += len(out)

            table = pa.Table.from_pandas(out[WEAK_COLS], preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(out_path, table.schema, compression="zstd")
            writer.write_table(table)
            if verbose:
                print(f"  [{stats['files']:>3}] {base:38s} +{len(out):>6,} "
                      f"(total {stats['rows_written']:,})", flush=True)
    finally:
        if writer is not None:
            writer.close()

    stats["split_sizes"] = splits_count
    stats["unique_posts"] = stats["rows_written"]
    return stats


def run(data_dir: str, out_dir: str, include_weak_pool: bool = True,
        rmhd_subreddits=None, rmhd_periods=None, rmhd_max_per_file: int | None = None,
        imhi_external_only: bool = False, stream_weak: bool = True, verbose: bool = True):
    """Full pipeline. Returns (labelled dataframe, paths, stats, leakage report).

    The labelled corpora are processed in memory. The Reddit pool is streamed by
    default so the whole run fits in a free Colab runtime.
    """
    onto = Ontology.load()
    say = print if verbose else (lambda *a, **k: None)
    os.makedirs(out_dir, exist_ok=True)

    say("Loading labelled corpora")
    frames = []
    for name, fn in [
        ("DepressionEmo", lambda: load_depressionemo(data_dir, onto)),
        ("DepSeverity+Dreaddit", lambda: load_depseverity_dreaddit(data_dir, onto)),
        ("IRF+LoST", lambda: load_irf_lost(data_dir, onto)),
        ("CSSRS", lambda: load_cssrs(data_dir, onto)),
        ("IMHI", lambda: load_imhi(data_dir, onto, external_only=imhi_external_only)),
    ]:
        f = fn()
        say(f"  {name:22s} {len(f):>8,} rows")
        frames.append(f)

    if include_weak_pool and not stream_weak:
        f = load_reddit_mh(data_dir, onto, rmhd_subreddits, rmhd_periods, rmhd_max_per_file)
        say(f"  {'Reddit Mental Health':22s} {len(f):>8,} rows")
        frames.append(f)

    df, stats = canonicalise(frames, onto)
    df = assign_splits(df, onto)
    report = audit_leakage(df)
    say(f"\nLabelled: {len(df):,} unique posts "
        f"({stats['duplicate_rows_merged']:,} duplicate rows merged)")

    weak_stats = None
    if include_weak_pool and stream_weak:
        say("\nStreaming the Reddit weak pool")
        known = df[df["author_hash"].notna()]
        author_splits = known.groupby("author_hash")["split"].agg(
            lambda s: max(set(s), key=lambda v: _STRICTNESS.get(v, 0))).to_dict()
        weak_stats = stream_weak_pool(
            data_dir, os.path.join(out_dir, "weak_pool.parquet"), onto,
            labelled_keys=set(df["post_id"]), author_splits=author_splits,
            subreddits=rmhd_subreddits, periods=rmhd_periods,
            max_per_file=rmhd_max_per_file, verbose=verbose)
        stats["weak_pool"] = weak_stats

    paths = save(df, onto, out_dir, stats, report, weak_written=bool(weak_stats))
    return df, paths, stats, report
