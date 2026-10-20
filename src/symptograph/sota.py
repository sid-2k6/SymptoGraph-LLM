"""Three comparison models drawn from this project's own literature survey.

Each is registered into ``baselines.MODEL_REGISTRY``, so all of them inherit the
baseline harness unchanged: the same leakage-controlled splits, the same masked
multi-task loss, the same symptom ``pos_weight``, the same ordinal class
weights, the same validation-tuned per-symptom thresholds, the same checkpoint
selection rule, the same three seeds, the same ``history.csv``, and the same
per-example prediction dumps that ``significance.compare_models`` needs. Only
the representation differs, which is the only way a comparison answers a
question about architecture.

What these are, and what they are not
-------------------------------------
**They are re-implementations of the architecture each paper describes, on this
project's 22-symptom / 4-task label space, under this project's protocol.**

They are **not** reproductions of the published systems, and their numbers must
never be presented beside the numbers printed in those papers. Three reasons,
all of which belong in the paper's limitations:

1. The full method sections were not available: MDPI blocks automated access and
   no author code was found. Each architecture below was built from the
   abstract and the published description of its modules. Where a choice was
   underdetermined it is marked ``DEVIATION`` in the class docstring.
2. Every one of these papers targets a different label space - binary depression
   screening, or depression plus suicide risk - on a different corpus. Mapping
   them onto 22 symptoms, 4-level severity, 5-level risk and binary stress means
   giving them this project's ``SharedHeads``. That is deliberate: it isolates
   the representation. It also means these are not the models that produced the
   published scores.
3. Two of the three contain components that cannot be exercised on this data at
   all, noted per model.

If the PDFs become available, the ``DEVIATION`` notes are the list of things to
correct.

The three
---------
``sota_ens_stack``   after [11] Suksam, Kaewbuadee & Jareanpon (2026),
                     *Information* 17(6), 546. Contests N1: symptoms first, via
                     a transformer ensemble rather than a concept bottleneck.
``sota_postgraph``   after [18] Rehman et al. (2026), *Healthcare* 14(11), 1440.
                     Contests N2: a graph estimated from post semantics and
                     author relations rather than a DSM-5 prior.
``sota_prompt_ssl``  after [21] Hsieh et al. (2025), *Scientific Reports*.
                     Contests the multi-label setup itself, via prompt
                     perspectives plus perturbation consistency.

Excluded after checking: [10] Bao, Pérez & Parapar (2024) proposes transformer
architectures that *classify and explain* symptom markers, and its contribution
is the quality of the natural-language explanation. Stripped of generation it
reduces to the ``mentalroberta`` baseline already present, and the explanation
quality cannot be scored by this metric suite - the same reason N4 was left out
of the proposed ladder. Reporting it as a classification competitor would
misrepresent the paper.
"""
from __future__ import annotations

import math

from .baselines import register_model

# Prompt perspectives for sota_prompt_ssl. Two, not more, for memory; see the
# DEVIATION note on that model.
PROMPT_PERSPECTIVES = (
    "Which clinical symptoms does this post describe?",
    "How severe is the distress expressed in this post?",
)


# --------------------------------------------------------------------------- #
# [11] transformer ensemble, stacking
# --------------------------------------------------------------------------- #
@register_model("sota_ens_stack")
def build_ens_stack(cfg, onto, train_texts, SharedHeads, load_transformer):
    """Stacking ensemble of transformer base learners, after [11].

    The paper runs a two-phase design. Phase 1 performs symptom-level
    classification with transformer learning strategies - it lists Single,
    Voting, Stacking, Bagging and Boosting. Phase 2 aggregates the predicted
    symptoms over a 14-day observation window for DSM-5-oriented binary
    depression screening.

    Phase 1 is implemented here as **stacking**: two pretrained transformer base
    learners each produce symptom logits, and a meta-learner is fitted on the
    concatenation of those logits. The other three tasks read a fused
    representation through the shared heads.

    DEVIATION - two base learners, not the paper's full set of strategies.
      Three RoBERTa-base encoders at batch 32 x 192 tokens do not fit 24 GB, so
      the ensemble is two encoders with gradient checkpointing enabled on both.
      That keeps batch size identical to every other model in the comparison,
      which matters more than ensemble width.

    NOT APPLICABLE - Phase 2. DSM-5 screening over a 14-day window needs several
      posts per author within a window. In this project's gold-labelled split
      the median is one post per author (822,884 authors over 1,017,900 pool
      posts, mean 1.24), so the window cannot be formed. Phase 1 is therefore
      the whole of what is compared, and the paper's screening claim is out of
      scope here rather than evaluated and failed.
    """
    import torch
    import torch.nn as nn

    K = len(onto.symptoms)
    names = [cfg.transformer_name, cfg.transformer_fallback]

    class EnsembleStack(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoders = nn.ModuleList()
            self.loaded = []
            self.encoder_fallback_used = False
            for n in names:
                enc, tok, loaded, fell = load_transformer(cfg, n)
                self._tok = tok
                if hasattr(enc, "gradient_checkpointing_enable"):
                    # two encoders in one step would otherwise not fit alongside
                    # a batch of 32 at 192 tokens on a 24 GB card
                    enc.gradient_checkpointing_enable()
                    enc.config.use_cache = False
                self.encoders.append(enc)
                self.loaded.append(loaded)
                self.encoder_fallback_used = self.encoder_fallback_used or fell
            self.loaded_name = " + ".join(self.loaded)
            dims = [e.config.hidden_size for e in self.encoders]
            self.drop = nn.Dropout(cfg.dropout)
            # one base learner per encoder
            self.base = nn.ModuleList([nn.Linear(d, K) for d in dims])
            # meta-learner over the stacked base predictions
            self.meta = nn.Linear(K * len(self.encoders), K)
            self.fuse = nn.Linear(sum(dims), cfg.hidden_dim)
            self.heads = SharedHeads(cfg.hidden_dim)

        def forward(self, batch):
            pooled, base_logits = [], []
            for enc, head in zip(self.encoders, self.base):
                h = enc(input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"]).last_hidden_state
                m = batch["attention_mask"].unsqueeze(-1).float()
                p = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
                pooled.append(p)
                base_logits.append(head(self.drop(p)))
            out = self.heads(torch.relu(self.fuse(torch.cat(pooled, dim=-1))))
            # the stacked symptom prediction replaces the shared head's
            out["symptoms"] = self.meta(torch.cat(base_logits, dim=-1))
            out["base_symptom_logits"] = base_logits
            return out

    model = EnsembleStack()

    def helper(texts):
        return model._tok(list(texts), padding=True, truncation=True,
                          max_length=cfg.max_length, return_tensors="pt")

    return model, "hf", helper


# --------------------------------------------------------------------------- #
# [18] post graph from semantics and author relations
# --------------------------------------------------------------------------- #
@register_model("sota_postgraph")
def build_postgraph(cfg, onto, train_texts, SharedHeads, load_transformer):
    """Transformer plus a GNN over a post graph, after [18].

    The paper states the graph is estimated from semantic similarity between
    posts together with author relations, explicitly rather than from social
    interactions such as mentions or replies. Implemented here as: encode the
    posts, then for each batch build an adjacency from (a) cosine similarity
    between post embeddings, keeping the top ``k`` neighbours per post, and
    (b) an edge between any two posts sharing an author. Two message-passing
    layers run over that graph, each normalised and residual, before the heads.

    DEVIATION - the graph is built per mini-batch, not over the whole corpus.
      A corpus-level graph over 19,762 labelled posts cannot be held in the
      training loop alongside a transformer, and a transductive corpus graph
      would also let validation posts exchange messages with training posts,
      which would breach the leakage control this project enforces elsewhere.
      Per-batch construction keeps the split boundary intact. Expect weaker
      graph structure than the paper reports.

    NOT IMPLEMENTED - SHAP and LIME attribution, and the calibration analysis.
      These are post-hoc analyses of a trained model; they do not change any
      metric in this comparison.
    """
    import torch
    import torch.nn as nn

    class PostGraph(nn.Module):
        def __init__(self):
            super().__init__()
            enc, tok, loaded, fell = load_transformer(cfg, cfg.transformer_name)
            self._tok = tok
            self.encoder = enc
            self.loaded_name = loaded
            self.encoder_fallback_used = fell
            D = enc.config.hidden_size
            self.k = 8                      # neighbours kept per post
            self.drop = nn.Dropout(cfg.dropout)
            self.gnn = nn.ModuleList([nn.Linear(D, D) for _ in range(2)])
            self.norm = nn.ModuleList([nn.LayerNorm(D) for _ in range(2)])
            self.heads = SharedHeads(D)

        def adjacency(self, z, group):
            """Row-normalised adjacency from cosine kNN plus shared-author edges."""
            B = z.shape[0]
            zn = torch.nn.functional.normalize(z.float(), dim=-1)
            sim = zn @ zn.t()
            eye = torch.eye(B, device=z.device, dtype=torch.bool)
            sim = sim.masked_fill(eye, float("-inf"))
            k = min(self.k, max(B - 1, 1))
            A = torch.zeros(B, B, device=z.device)
            if B > 1:
                idx = sim.topk(k, dim=-1).indices
                A.scatter_(1, idx, 1.0)
                A = torch.maximum(A, A.t())          # symmetric
            same = (group.unsqueeze(0) == group.unsqueeze(1)) & (group.unsqueeze(0) >= 0)
            A = torch.maximum(A, same.float().masked_fill(eye, 0.0))
            A = A + torch.eye(B, device=z.device)    # self loop
            return A / A.sum(-1, keepdim=True).clamp(min=1.0)

        def forward(self, batch):
            h = self.encoder(input_ids=batch["input_ids"],
                             attention_mask=batch["attention_mask"]).last_hidden_state
            m = batch["attention_mask"].unsqueeze(-1).float()
            z = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
            A = self.adjacency(z, batch["group"])
            x = z
            for layer, norm in zip(self.gnn, self.norm):
                x = norm(torch.relu(layer(A @ x)) + x)
            out = self.heads(self.drop(x))
            out["graph_degree"] = A.sum(-1)
            return out

    model = PostGraph()

    def helper(texts):
        return model._tok(list(texts), padding=True, truncation=True,
                          max_length=cfg.max_length, return_tensors="pt")

    return model, "hf", helper


# --------------------------------------------------------------------------- #
# [21] prompt perspectives plus perturbation consistency
# --------------------------------------------------------------------------- #
@register_model("sota_prompt_ssl")
def build_prompt_ssl(cfg, onto, train_texts, SharedHeads, load_transformer):
    """Multi-perspective prompts and perturbation self-supervision, after [21].

    The paper describes a multi-task framework with two modules on a pretrained
    language-model backbone: a multi-perspective prompt design module, and a
    perturbation-based self-supervised learning module. Implemented here as:
    each post is encoded once per prompt perspective, the perspective
    representations are fused by learned attention, and a second forward pass
    under an independent dropout mask provides the perturbed view. The
    consistency term between the two views is returned as
    ``out["consistency"]`` and added to the loss by ``baselines.multitask_loss``
    only when the model exposes it.

    DEVIATION - two perspectives. Cost is linear in the number of perspectives
      because each is a separate encoder pass; two keeps the step within the
      same batch size as the rest of the comparison. The paper's perspective
      count was not recoverable.

    DEVIATION - the perturbation is dropout resampling, which is the standard
      cheap instantiation. The paper's perturbation operator was not
      recoverable; token masking or embedding noise would be the alternatives.
    """
    import torch
    import torch.nn as nn

    class PromptSSL(nn.Module):
        def __init__(self):
            super().__init__()
            enc, tok, loaded, fell = load_transformer(cfg, cfg.transformer_name)
            self._tok = tok
            self.encoder = enc
            self.loaded_name = loaded
            self.encoder_fallback_used = fell
            D = enc.config.hidden_size
            self.n_persp = len(PROMPT_PERSPECTIVES)
            self.drop = nn.Dropout(cfg.dropout)
            self.persp_attn = nn.Linear(D, 1)      # learned fusion over views
            self.heads = SharedHeads(D)

        def _encode(self, ids, mask):
            h = self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state
            m = mask.unsqueeze(-1).float()
            return (h * m).sum(1) / m.sum(1).clamp(min=1e-9)

        def _fuse(self, views):
            V = torch.stack(views, dim=1)                       # B,P,D
            w = torch.softmax(self.persp_attn(V), dim=1)         # B,P,1
            return (w * V).sum(1)

        def forward(self, batch):
            views = [self._encode(batch[f"input_ids_{i}"], batch[f"attention_mask_{i}"])
                     for i in range(self.n_persp)]
            z = self._fuse(views)
            out = self.heads(self.drop(z))
            if self.training:
                # perturbed view: same inputs, independent dropout
                z2 = self._fuse([self.drop(v) for v in views])
                out2 = self.heads(self.drop(z2))
                p = torch.sigmoid(out["symptoms"].float())
                q = torch.sigmoid(out2["symptoms"].float())
                eps = 1e-6
                # symmetric KL between the two views' symptom distributions
                kl = (p * ((p + eps) / (q + eps)).log()
                      + (1 - p) * ((1 - p + eps) / (1 - q + eps)).log())
                kl = kl + (q * ((q + eps) / (p + eps)).log()
                           + (1 - q) * ((1 - q + eps) / (1 - p + eps)).log())
                out["consistency"] = kl.mean()
            return out

    class PromptCollateHelper:
        """Tokenises each perspective separately; one tensor pair per view."""

        def __init__(self, tokenizer):
            self.tok = tokenizer
            self.n = len(PROMPT_PERSPECTIVES)

        def __call__(self, texts):
            enc = {}
            for i, prompt in enumerate(PROMPT_PERSPECTIVES):
                e = self.tok([f"{prompt} {t}" for t in texts], padding=True,
                             truncation=True, max_length=cfg.max_length,
                             return_tensors="pt")
                enc[f"input_ids_{i}"] = e["input_ids"]
                enc[f"attention_mask_{i}"] = e["attention_mask"]
            # the harness also expects the canonical pair; use the first view
            enc["input_ids"] = enc["input_ids_0"]
            enc["attention_mask"] = enc["attention_mask_0"]
            return enc

    model = PromptSSL()
    return model, "hf", PromptCollateHelper(model._tok)


SOTA_MODELS = ("sota_ens_stack", "sota_postgraph", "sota_prompt_ssl")

# Paper each model is after, for the results table and the citation list.
SOTA_SOURCES = {
    "sota_ens_stack": "[11] Suksam, Kaewbuadee & Jareanpon (2026), Information 17(6), 546",
    "sota_postgraph": "[18] Rehman et al. (2026), Healthcare 14(11), 1440",
    "sota_prompt_ssl": "[21] Hsieh et al. (2025), Scientific Reports",
}
