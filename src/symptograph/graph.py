"""Symptom-disorder comorbidity graph for SymptoGraph-LLM (novelty N2).

Builds the heterogeneous graph whose nodes are the 22 unified symptoms plus the
disorder labels, with three relation types:

1. ``symptom -> disorder``  DSM-5-informed prior (``configs/dsm5_prior.json``).
2. ``symptom <-> symptom``  positive pointwise mutual information over
   co-annotated gold labels.
3. ``disorder <-> disorder`` PPMI over authors who post in more than one
   mental-health subreddit, which is an empirical comorbidity proxy.

Everything empirical is computed from the **training split only**, so no test
information reaches the graph prior.

Two honest limitations the report surfaces rather than hides:

* Symptom co-occurrence is only observable *within* a source's own label set.
  DepressionEmo co-annotates 8 symptoms, IRF 2 and LoST 1, and no open corpus
  annotates them jointly, so large parts of the symptom-symptom matrix are
  structurally unobservable. The observability mask records exactly which cells
  are supported by data.
* The disorder proxy is subreddit co-membership, not a clinical diagnosis.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .preprocess import Ontology

PRIOR_CONFIG = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "dsm5_prior.json")
NON_CLINICAL = {"control", "mixed"}


@dataclass
class SymptomGraph:
    symptoms: list[str]
    disorders: list[str]
    A_sd: np.ndarray            # symptoms x disorders, DSM-5 prior (0/1)
    A_ss: np.ndarray            # symptoms x symptoms, PPMI
    A_ss_mask: np.ndarray       # 1 where co-occurrence was observable
    A_ss_support: np.ndarray    # number of posts backing each cell
    A_dd: np.ndarray            # disorders x disorders, p(b|a) co-membership edge weight
    A_dd_ppmi: np.ndarray       # disorders x disorders, enrichment over multi-disorder authors
    A_dd_support: np.ndarray    # number of authors backing each cell
    node_features: np.ndarray | None = None
    feature_model: str | None = None

    @property
    def nodes(self) -> list[dict]:
        return ([{"name": s, "type": "symptom", "index": i} for i, s in enumerate(self.symptoms)]
                + [{"name": d, "type": "disorder", "index": len(self.symptoms) + i}
                   for i, d in enumerate(self.disorders)])

    def save(self, out_dir: str) -> dict:
        os.makedirs(out_dir, exist_ok=True)
        npz = os.path.join(out_dir, "graph.npz")
        arrays = dict(A_sd=self.A_sd, A_ss=self.A_ss, A_ss_mask=self.A_ss_mask,
                      A_ss_support=self.A_ss_support, A_dd=self.A_dd,
                      A_dd_ppmi=self.A_dd_ppmi, A_dd_support=self.A_dd_support)
        if self.node_features is not None:
            arrays["node_features"] = self.node_features
        np.savez_compressed(npz, **arrays)

        meta = os.path.join(out_dir, "graph_nodes.json")
        with open(meta, "w", encoding="utf-8") as fh:
            json.dump({"symptoms": self.symptoms, "disorders": self.disorders,
                       "nodes": self.nodes, "feature_model": self.feature_model,
                       "feature_dim": None if self.node_features is None
                                      else int(self.node_features.shape[1])},
                      fh, indent=2)
        return {"graph": npz, "nodes": meta}

    @classmethod
    def load(cls, out_dir: str) -> "SymptomGraph":
        z = np.load(os.path.join(out_dir, "graph.npz"))
        with open(os.path.join(out_dir, "graph_nodes.json"), encoding="utf-8") as fh:
            meta = json.load(fh)
        return cls(symptoms=meta["symptoms"], disorders=meta["disorders"],
                   A_sd=z["A_sd"], A_ss=z["A_ss"], A_ss_mask=z["A_ss_mask"],
                   A_ss_support=z["A_ss_support"], A_dd=z["A_dd"],
                   A_dd_ppmi=z["A_dd_ppmi"], A_dd_support=z["A_dd_support"],
                   node_features=z["node_features"] if "node_features" in z else None,
                   feature_model=meta.get("feature_model"))


# --------------------------------------------------------------------------- #
# association measure
# --------------------------------------------------------------------------- #
def ppmi_from_counts(joint: np.ndarray, totals: np.ndarray, n: int,
                     min_support: int = 10) -> np.ndarray:
    """Positive PMI from co-occurrence counts.

    ``joint[i, j]`` co-occurrences, ``totals[i]`` marginal counts, ``n`` samples.
    Cells with fewer than ``min_support`` co-occurrences are zeroed, because PMI
    is wildly unstable on small counts.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        p_joint = joint / max(n, 1)
        p_marg = totals / max(n, 1)
        expected = np.outer(p_marg, p_marg)
        pmi = np.log(np.divide(p_joint, expected, out=np.zeros_like(p_joint),
                               where=expected > 0),
                     out=np.zeros_like(p_joint), where=p_joint > 0)
    pmi[~np.isfinite(pmi)] = 0.0
    pmi[joint < min_support] = 0.0
    return np.maximum(pmi, 0.0)


# --------------------------------------------------------------------------- #
# edge builders
# --------------------------------------------------------------------------- #
def build_prior(onto: Ontology, disorders: list[str],
                prior_path: str = PRIOR_CONFIG) -> tuple[np.ndarray, dict]:
    with open(prior_path, encoding="utf-8") as fh:
        cfg = json.load(fh)
    edges = cfg["edges"]
    A = np.zeros((len(onto.symptoms), len(disorders)), dtype=np.float32)
    sym_ix = {s: i for i, s in enumerate(onto.symptoms)}
    unknown = []
    for j, d in enumerate(disorders):
        for s in edges.get(d, []):
            if s in sym_ix:
                A[sym_ix[s], j] = 1.0
            else:
                unknown.append((d, s))
    return A, {"unknown_symptoms": unknown, "provenance": cfg["provenance"],
               "disorders_without_prior": [d for d in disorders if not edges.get(d)]}


def build_symptom_cooccurrence(train: pd.DataFrame, onto: Ontology,
                               min_support: int = 10):
    """PPMI between symptoms, using only posts where both are annotated."""
    cols = onto.columns
    M = train[cols].to_numpy(dtype=np.int8)
    annotated = (M >= 0)
    positive = (M == 1)

    k = len(cols)
    joint = np.zeros((k, k), dtype=np.float64)
    both_annotated = np.zeros((k, k), dtype=np.float64)
    for i in range(k):
        ai = annotated[:, i]
        for j in range(k):
            pair = ai & annotated[:, j]
            n_pair = pair.sum()
            both_annotated[i, j] = n_pair
            if n_pair:
                joint[i, j] = (positive[pair, i] & positive[pair, j]).sum()

    # PMI needs a shared denominator, so normalise per pair by its own
    # observable population rather than a single global n.
    with np.errstate(divide="ignore", invalid="ignore"):
        p_joint = np.divide(joint, both_annotated, out=np.zeros_like(joint),
                            where=both_annotated > 0)
        marg = np.divide(np.diag(joint), np.diag(both_annotated),
                         out=np.zeros(k), where=np.diag(both_annotated) > 0)
        expected = np.outer(marg, marg)
        pmi = np.log(np.divide(p_joint, expected, out=np.zeros_like(p_joint),
                               where=expected > 0),
                     out=np.zeros_like(p_joint), where=p_joint > 0)
    pmi[~np.isfinite(pmi)] = 0.0
    pmi[joint < min_support] = 0.0
    A_ss = np.maximum(pmi, 0.0).astype(np.float32)
    np.fill_diagonal(A_ss, 0.0)
    mask = (both_annotated >= min_support).astype(np.float32)
    np.fill_diagonal(mask, 0.0)
    return A_ss, mask, both_annotated.astype(np.int32)


def _comembership_counts(authors: pd.Series, ix: dict) -> np.ndarray:
    k = len(ix)
    joint = np.zeros((k, k), dtype=np.float64)
    for groups in authors:
        idx = sorted({ix[g] for g in groups})
        for a in range(len(idx)):
            for b in range(a, len(idx)):
                joint[idx[a], idx[b]] += 1
                if a != b:
                    joint[idx[b], idx[a]] += 1
    return joint


def build_disorder_cooccurrence(weak_train: pd.DataFrame, disorders: list[str],
                                min_support: int = 10,
                                exclude_non_clinical: bool = True):
    """Disorder-disorder edges from authors who post in several subreddits.

    A clinical diagnosis is unavailable, so co-membership of mental-health
    subreddits by the same pseudonymised author is the comorbidity proxy.

    Two matrices are returned because PMI against an independence null does not
    work on this population. Every author in the table has at least one
    membership by construction, and authors average only 1.125 disorders, so
    observed co-membership is *below* the independence expectation for every
    pair (maximum lift 0.99) and PPMI would zero out the entire matrix.

    * ``cond[a, b] = p(b | a)``  conditional co-membership, used as the edge
      weight. Non-negative, dense and directly meaningful for message passing.
    * ``ppmi[a, b]``  enrichment computed only over authors with two or more
      disorders, where co-membership is the norm and the null is sensible. This
      is the interpretable view, and it recovers established comorbidities.
    """
    d = weak_train[weak_train["author_hash"].notna()][["author_hash", "disorder_weak"]]
    d = d[d["disorder_weak"].notna()]
    if exclude_non_clinical:
        d = d[~d["disorder_weak"].isin(NON_CLINICAL)]
    keep = [x for x in disorders if not (exclude_non_clinical and x in NON_CLINICAL)]
    ix = {x: i for i, x in enumerate(keep)}
    d = d[d["disorder_weak"].isin(ix)]

    pairs = d.drop_duplicates(["author_hash", "disorder_weak"])
    authors = pairs.groupby("author_hash")["disorder_weak"].apply(list)
    joint = _comembership_counts(authors, ix)
    totals = np.diag(joint).copy()

    with np.errstate(divide="ignore", invalid="ignore"):
        cond = np.divide(joint, totals[:, None],
                         out=np.zeros_like(joint), where=totals[:, None] > 0)
    np.fill_diagonal(cond, 0.0)
    cond[joint < min_support] = 0.0

    multi = authors[authors.map(len) > 1]
    joint_m = _comembership_counts(multi, ix)
    ppmi = ppmi_from_counts(joint_m, np.diag(joint_m).copy(), max(len(multi), 1),
                            min_support=min_support)
    np.fill_diagonal(ppmi, 0.0)

    # expand back to the full disorder list so indices line up with A_sd
    n = len(disorders)
    pos = {x: i for i, x in enumerate(disorders)}
    full_cond = np.zeros((n, n), dtype=np.float32)
    full_ppmi = np.zeros((n, n), dtype=np.float32)
    support = np.zeros((n, n), dtype=np.int32)
    for a, da in enumerate(keep):
        for b, db in enumerate(keep):
            full_cond[pos[da], pos[db]] = cond[a, b]
            full_ppmi[pos[da], pos[db]] = ppmi[a, b]
            support[pos[da], pos[db]] = int(joint[a, b])
    meta = {"authors_considered": int(len(authors)),
            "authors_multi_disorder": int(len(multi)),
            "mean_disorders_per_author": round(float(pairs.shape[0] / max(len(authors), 1)), 3),
            "disorders_used": keep,
            "edge_weight": "p(b|a) conditional co-membership",
            "enrichment": "PPMI over authors with >=2 disorders"}
    return full_cond, full_ppmi, support, meta


def build_node_features(onto: Ontology, disorders: list[str],
                        model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """Embed the symptom definitions and disorder names as initial node features.

    Returns ``(None, None)`` if sentence-transformers is unavailable, so the
    graph can still be built; the model notebook can fall back to learned
    embeddings.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None, None
    texts = [f"{m['name'].replace('_', ' ')}: {m['definition']}" for m in onto.meta]
    texts += [f"{d.replace('_', ' ')}: a mental health condition discussed online"
              for d in disorders]
    model = SentenceTransformer(model_name)
    feats = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(feats, dtype=np.float32), model_name


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def prior_support_report(graph: SymptomGraph) -> pd.DataFrame:
    """Which DSM-5 prior symptom pairs the training data actually supports.

    For each disorder, the prior implies its symptoms should co-occur. This
    checks the observable subset of those pairs, which is a sanity check on the
    hand-authored prior.
    """
    rows = []
    sym_ix = {s: i for i, s in enumerate(graph.symptoms)}
    for j, d in enumerate(graph.disorders):
        members = [s for s in graph.symptoms if graph.A_sd[sym_ix[s], j] > 0]
        pairs = [(a, b) for n, a in enumerate(members) for b in members[n + 1:]]
        observable = [(a, b) for a, b in pairs if graph.A_ss_mask[sym_ix[a], sym_ix[b]] > 0]
        supported = [(a, b) for a, b in observable if graph.A_ss[sym_ix[a], sym_ix[b]] > 0]
        rows.append({
            "disorder": d,
            "prior_symptoms": len(members),
            "implied_pairs": len(pairs),
            "observable_pairs": len(observable),
            "ppmi_positive": len(supported),
            "support_rate": round(len(supported) / len(observable), 3) if observable else None,
        })
    return pd.DataFrame(rows)


def top_edges(matrix: np.ndarray, rows: list[str], cols: list[str], k: int = 15,
              support: np.ndarray | None = None, symmetric: bool | None = None) -> pd.DataFrame:
    """Rank the strongest edges. Set ``symmetric=True`` to list each pair once."""
    out = []
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            sym = (rows is cols) if symmetric is None else symmetric
            if sym and j <= i:
                continue
            if matrix[i, j] > 0:
                rec = {"a": rows[i], "b": cols[j], "weight": round(float(matrix[i, j]), 3)}
                if support is not None:
                    rec["support"] = int(support[i, j])
                out.append(rec)
    cols_out = ["a", "b", "weight"] + (["support"] if support is not None else [])
    if not out:
        return pd.DataFrame(columns=cols_out)
    return (pd.DataFrame(out).sort_values("weight", ascending=False)
            .head(k).reset_index(drop=True))


def build(processed_dir: str, out_dir: str, min_support: int = 10,
          with_features: bool = True, verbose: bool = True):
    """Build and save the graph from the outputs of notebook 02."""
    say = print if verbose else (lambda *a, **k: None)
    onto = Ontology.load()

    labelled = pd.read_parquet(os.path.join(processed_dir, "labelled.parquet"))
    train = labelled[labelled["split"] == "train"]
    say(f"labelled train posts: {len(train):,}")

    weak_path = os.path.join(processed_dir, "weak_pool.parquet")
    weak_train = pd.DataFrame(columns=["author_hash", "disorder_weak"])
    if os.path.exists(weak_path):
        weak = pd.read_parquet(weak_path, columns=["author_hash", "disorder_weak", "split"])
        weak_train = weak[weak["split"] == "train"]
        say(f"weak-pool train posts: {len(weak_train):,}")

    disorders = sorted(set(d for d in weak_train.get("disorder_weak", pd.Series(dtype=object))
                           .dropna().unique()))
    if not disorders:
        with open(PRIOR_CONFIG, encoding="utf-8") as fh:
            disorders = sorted(json.load(fh)["edges"].keys())
    say(f"disorder nodes: {len(disorders)}")

    A_sd, prior_meta = build_prior(onto, disorders)
    A_ss, A_ss_mask, A_ss_support = build_symptom_cooccurrence(train, onto, min_support)
    A_dd, A_dd_ppmi, A_dd_support, dd_meta = build_disorder_cooccurrence(
        weak_train, disorders, min_support)

    feats, feat_model = (build_node_features(onto, disorders) if with_features else (None, None))
    if with_features and feats is None:
        say("sentence-transformers unavailable; node features skipped")

    graph = SymptomGraph(symptoms=onto.symptoms, disorders=disorders, A_sd=A_sd,
                         A_ss=A_ss, A_ss_mask=A_ss_mask, A_ss_support=A_ss_support,
                         A_dd=A_dd, A_dd_ppmi=A_dd_ppmi, A_dd_support=A_dd_support,
                         node_features=feats, feature_model=feat_model)
    paths = graph.save(out_dir)

    stats = {
        "symptom_nodes": len(onto.symptoms),
        "disorder_nodes": len(disorders),
        "prior_edges": int((A_sd > 0).sum()),
        "symptom_pairs_observable": int((A_ss_mask > 0).sum() // 2),
        "symptom_pairs_total": len(onto.symptoms) * (len(onto.symptoms) - 1) // 2,
        "symptom_edges_ppmi": int((A_ss > 0).sum() // 2),
        "disorder_edges_conditional": int((A_dd > 0).sum()),
        "disorder_pairs_enriched": int((A_dd_ppmi > 0).sum() // 2),
        "min_support": min_support,
        "prior": prior_meta,
        "disorder_cooccurrence": dd_meta,
        "node_feature_model": feat_model,
    }
    with open(os.path.join(out_dir, "graph_stats.json"), "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2, default=str)
    paths["stats"] = os.path.join(out_dir, "graph_stats.json")
    return graph, paths, stats
