# Response to reviewer comments, round 2

All changes were applied **in place** to `docs/SymptoGraph-LLM_Paper_revised.docx`
by `scripts/revise_paper.py`, because that file carries hand edits made outside
the build pipeline and regenerating it would discard them.

| # | Comment | Status |
|---|---|---|
| 1 | Table 7 and Figure 3 stop at epoch 8 | **Done** |
| 2 | Give γ_max in Equation 11 | **Done** |
| 3 | Abstention vs max-probability baseline, plus a calibration metric | **Defined, not computed — blocked** |
| 4 | Figure 2 wiring wrong in two places | **Done** |
| 5 | Review criticises gaps this paper shares | **Done** |
| 6 | Unsupported statistics | **Done** |
| 7 | Ethics, data-availability, code-availability | **Done** |
| 8 | DOIs for [11], [18], [21] | **Checked — all correct** |
| 9 | Long sentences, "rather than" | **Done** |
| 10 | "LLM" in the title | **Role clarified; rename offered** |

---

### 1. Epochs 9 and 10

Table 7 now runs 1, 2, 3, 6, 8, 9, 10 (eight columns, sized to the 6.27 in text
block) and Figure 3 runs to epoch 10.

One thing the reviewer could not have known from the document: **seed 1 met the
early-stopping criterion at epoch 8**, so epochs 9 and 10 exist for two seeds
only. They are therefore two-seed means, not three-seed means, and mixing the
two silently would have been worse than stopping at 8. Both the table caption
and the figure caption say so, and in Figure 3 the tail is drawn with a shaded
band, hollow markers and a dashed line so the change in support is visible
rather than buried in a caption.

The added columns make the selection problem plainer, which is the point of the
comment: validation loss continues to 2.906 at epoch 10 while training risk QWK
sits at exactly 1.000 from epoch 7, and training loss passes below zero because
the learned-variance term contributes half the predicted log-variance.

### 2. γ_max

`γ_max = 10`, so weights are clipped to [0.1, 10]. Taken from the `cap`
argument of `class_weights()` in `src/symptograph/metrics.py` and now stated in
the Equation 11 paragraph.

### 3. Abstention baseline and calibration — blocked

Both comparisons are now **defined** in §4.2 and **reported as not computed**,
with the reason given.

The max-probability baseline and the expected calibration error both need
per-example predicted probabilities. Those were not retained by the training
runs; `performance_metrics.json` stores the aggregate risk-coverage curve and
nothing per post. Neither quantity can be recovered from what exists in the
repository, so inventing them was not an option.

§4.2 now states that the abstention result is evidence the variance head
**ranks its own reliability usefully**, and not evidence that it beats a
softmax-confidence baseline or that it is calibrated. That is the strongest
claim the retained data supports. The re-run that would settle both is listed
in §5.2.

### 4. Figure 2

Both defects were real and are fixed:

- the weak-label stream pointed into the variance head. It now runs to the
  symptom term of the objective, which is where its gradient actually goes;
- the symptom head sat downstream of the pooled vector. It is now fed by the
  refined concept logits, and the severity, risk and stress heads read those
  logits concatenated with the pooled vector.

The variance head's input is drawn as a stop-gradient edge from the pooling
stage. The heads row was moved directly beneath the graph row so these
connectors no longer cross other novelty regions. The caption now describes the
data flow explicitly, and `docs/diagrams/figure2_architecture.drawio` is
regenerated from the same declaration.

### 5. Criticisms this paper shares

Four were softened, each where the gap is genuinely shared:

- **[3] DepGLM** — unconstrained ordinal head. Now notes that the present work
  shares that property, pointing at §3.5.
- **[18] Rehman et al.** — no random-graph control. Now notes the omission is
  repeated here and recorded in §5.2.
- **[23]**, Table 1 gap column — "labeller bias neither estimated nor
  inverted". The noise channel was removed when the paper moved to the L0–L5
  ladder, so this gap is now shared; the cell says so.
- **[28]**, Table 1 gap column — "without supplying a paired procedure". No
  paired procedure is supplied here either; the cell says so.

[14] was checked and left alone: the gap attributed to it in the survey is the
absence of a decomposable output space, which this work does not share.

### 6. Unsupported statistics

- **69.3% keyword leakage** — removed. The point it supported (a model can
  match a disorder name in the text) is kept as a qualitative statement, with
  the absence of a keyword-masked control acknowledged.
- **132,850 authors / 327,866 posts** — removed, replaced by a qualitative
  statement about author concentration.
- **Development anecdotes** — the two carrying unsourced numbers are rewritten
  to argue from facts that are in the repository. The collapse of the ordinal
  heads is now justified by the 71.9% majority band from
  `processed/label_coverage.csv`; the threshold anecdote is now a statement
  about fixed cuts with no number attached.

### 7. Ethics, data and code availability

Added as §5.3, §5.4 and §5.5, covering licensing and the absence of
institutional review, pseudonymous handles and the single purpose they serve,
local-only weak labelling, intended use and the explicit statement that this is
not a screening instrument, the suicide-risk false-negative risk, what is and
is not redistributable, the retained run outputs, the missing per-example
predictions, and the released code and editable figure sources.

**One thing to flag:** the `5.3 Limitations` section added in the previous round
is **absent from the revised file**. It was not removed by this script. Since
the earlier review asked for it explicitly, I would restore it — say the word
and I will.

### 8. DOIs — all three are correct

Checked against Crossref. No change was needed, so none was made.

| Ref | DOI | Crossref record |
|---|---|---|
| [11] | `10.3390/info17060546` | *Information* **17**(6), 546, online 2 Jun 2026 ✓ |
| [18] | `10.3390/healthcare14111440` | *Healthcare* **14**(11), 1440, online 22 May 2026 ✓ |
| [21] | `10.1038/s41598-025-30873-x` | *Sci Rep* **16**(1), online 5 Dec 2025 ✓ |

On [21] specifically: **2025 with volume 16 is what the publisher records.**
*Scientific Reports* published it online in December 2025 into the 2026 volume,
so the citation is right and the oddity is real. A parenthetical note was added
so the next reader does not query it.

Related, since it looks like the same class of error: the title of **[28]**
reads "Large Language Mod". That typo is in the **publisher's own record**, so
reproducing it is correct citation practice. Left as is.

### 9. Register

"rather than" reduced from **70 to 36**. The formulaic cases were converted:
before a determiner or a gerund it became "instead of"; before a preposition it
became ", not". The remaining 36 are where the construction carries a genuine
contrast and reads better than the alternatives.

Nine sentences over 55 words were split at a semicolon, each only where both
halves stand alone.

### 10. "LLM" in the title

The system name is kept, and the limited role is now stated where the
framework is introduced: the language model **annotates training material
offline and is not part of the deployed model**, which is a fine-tuned encoder
of 125.8 M parameters.

This is the conservative choice, because "SymptoGraph-LLM" is the name used
throughout the repository, the prior review rounds and the commit history. If
you would rather drop it, **SymptoGraph** works everywhere with a single rename
and I will do it in one pass — but it should be one decision applied
consistently, not just a title change.

---

## Still blocked after this round

1. **Per-example predictions** — needed for comment 3, and for the paired
   significance tests §4.2 defines but does not report.
2. **Reference-model protocol parity** — the three reference models still ran
   at 8 epochs, batch 16, length 256 against 10/32/192, so Table 10 remains an
   uncontrolled comparison, as §4.1 states.
3. **`5.3 Limitations`** — missing from the revised file; recommend restoring.
