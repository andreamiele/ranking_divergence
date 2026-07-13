# Rank-histogram divergence metric exploration — summary

**Goal:** find the divergence between a model's next-token rank histogram and the
held-out real-text rank histogram (under a fixed gpt2-large reference) that best
**differentiates small differences in model performance**, while correlating well with
established distributional quality metrics and cleanly discriminating between models.

## Setup

- **Models:** DUO, MDLM, S-FLM (SFM), sampled with the official `s-flm` repo/checkpoints.
- **Grid:** temperature × NFE ∈ {8, 16, 32, 64, 128}. DUO and SFM were both extended down
  to t=0.2 (from an initial 0.6–1.4 matched grid) after the original SFM sweep showed no
  interior minimum in 0.6–1.4 — every divergence was monotonically increasing across the
  whole sampled range, i.e. the sweep never reached SFM's true cold optimum. With the
  extension, DUO and SFM both cover t=0.2–1.4 (245 configs each) and MDLM keeps its
  original wide range (150 configs, t=0.1–2.0). All three models are now included in every
  head-to-head comparison below.
- **Reference:** rank histogram of held-out OpenWebText (128 docs) scored with
  `gpt2-large`, identical across all models.
- **Binning:** candidate divergences operate on a coarsened histogram — **K linear head
  bins (ranks 1..K) + n_log log-spaced tail bins**, K≈20–30, n_log≈5–10, per the original
  email guidance (≤30 bins).

## What we correlate against (the "gold standards")

From *Hacking Generative Perplexity* (the paper's recommended distributional suite),
computed per config against the same held-out reference:

| Metric | Direction | What it measures |
|---|---|---|
| **MAUVE** | ↑ better | divergence frontier between generated/real text in embedding space |
| **GM** (Gradient Moment) | ↓ better | Fisher-score MMD (mean per-token NLL-gradient distance) |
| **DE** (Energy distance) | ↓ better | LM-free, handcrafted surface/discourse/entity features |
| **FMTyp-p** | ↑ better | Mahalanobis typicality p-value vs reference (0.5 = exchangeable) |

Also tracked (not gold standards, but useful diagnostics): **gen-PPL**, **unigram
entropy**, **Rep-1/2/3** (repetition rates).

## Divergence families tried

**Already-implemented baseline:**
- **Rank-Wasserstein** (optimal transport, log-rank ground cost) — the repo's original metric.
- OT with alternative ground costs: linear, sqrt, capped-log.

**Binned f-divergences** (on K-linear + n_log-log-tail bins):
- KL (forward), reverse-KL, Jensen-Shannon, symmetric-KL, **chi-square**, Hellinger,
  total-variation.
- Geometric OT × f-divergence mixtures.

**Novel / exotic families** (extended exploration, see Evaluation methodology below):
- **Rényi-α** and **Amari-α** divergences (continuous family generalizing KL/chi²/Hellinger).
- **Cressie-Read power-divergence family** (continuous λ, includes KL/chi²/Neyman-chi²/Hellinger as special cases).
- **Trimmed chi-square** (drop the top-k most-deviant bins before summing — a robustness variant).
- **Anderson-Darling** (tail-weighted CDF distance) and Cramér-von-Mises / KS statistic.
- **Bhattacharyya distance**.
- Symmetric chi², Neyman chi², triangular discrimination.
- Rank-power-weighted chi² (head/tail emphasis).
- Equal-reference-mass (quantile) binning × {chi², KL, JS, Hellinger}.
- Head-only chi² (no tail bins) and head/tail-split mixtures.
- **Max-ratio** (Rényi-∞ limit, `max_i(p_i/q_i)`) and **power-mean ratio divergence**
  (`(Σ p_i·(p_i/q_i)^t)^(1/t)`, interpolates toward max-ratio as t→∞).
- Top-m mean log-ratio (order-statistic focused on the m worst bins).
- **MMD** with an RBF kernel on log-rank support points.
- **Wasserstein-2** (quadratic ground cost, via quantile-function matching).
- Cheap tail/rank summaries: KS statistic, CDF-L2, CDF-Spearman, top-k mass difference,
  mean/median log-rank gap.

~180 distinct candidates evaluated in total (58 in the systematic grid search, 122+ in
LLM-agent-driven novel-form exploration).

## Evaluation methodology

A systematic search-and-evaluation harness scores every candidate on three axes:

- **`resolving_power`** (60% weight, headline): fraction of config pairs *close in true
  performance* (a composite of MAUVE/GM/DE/FMTyp-p) that the candidate orders correctly.
- **`alignment`** (25%): mean sign-corrected Spearman correlation with MAUVE/GM/DE/FMTyp-p.
- **`discrimination`** (15%): how cleanly the divergence separates DUO vs MDLM at each
  model's own best-per-NFE operating point (see methodology note below).

`research_score = 0.6·resolving_power + 0.25·alignment + 0.15·discrimination`.

**Methodology note — a bug hunt that changed the ranking twice.** An early version of
this harness declared `max-ratio`/`power-mean` the clear winners (+9% over chi²). Closer
inspection (triggered by these metrics visually failing to separate DUO/MDLM on a plot)
found three stacked measurement flaws, each fixed in turn:
1. DUO and MDLM were being compared on **mismatched temperature grids** — fixed by using
   the grid-matched `mdlm-sflm-duotemps-v1` resample.
2. "Discrimination" was a **whole-grid** measure, rewarding sensitivity to far-from-optimum
   extremes (which order-statistics like max-ratio are naturally hypersensitive to) rather
   than genuine separation at each model's achievable best — fixed by restricting to each
   model's near-optimum (best-per-NFE) point.
3. Even near-optimum, the comparison wasn't **blocked by NFE**, which penalized chi² for
   having real NFE-driven variance — fixed by blocking.

After all three fixes, max-ratio/power-mean's apparent edge disappeared — worse, they
**don't even reliably order DUO vs MDLM** (sign flips at NFE=32), while chi² and rank-W
are consistent across every NFE. Every subsequent finding below was validated against
this corrected harness, plus a standard trust-check battery per candidate: neighborhood
robustness (not a knife-edge), per-NFE sign consistency, correlation with chi² (same
family vs a different signal), and individual gold-metric breakdown (not cherry-picked to
one metric).

**A fourth bug, found later: the same whole-grid-vs-near-optimum flaw also inflated
`alignment`, not just `discrimination`.** Fix #2 above was applied to the discrimination
axis but not re-checked against alignment. Once SFM was added with its own full cold
range, re-running alignment (Spearman vs MAUVE/GM/DE/FMTyp-p) at each model's per-NFE
near-optimum point — rather than pooled across the whole temperature grid — showed that
**Rényi-α100's apparent lead was a whole-grid artifact**: on the whole grid it scores
~0.80, nearly identical to max-ratio and power-mean (~0.81), because α=100 makes it
behave almost like a Rényi-∞/max-ratio statistic (dominated by whichever single bin
deviates most). Across the whole grid this doesn't matter — cold, collapsed,
high-repetition configs are uniformly bad on every axis, so any tail-sensitive metric
"wins" on the trivial global trend. Restricted to near-optimum points, where the actual
discrimination has to happen, Rényi-100 collapses the same way max-ratio/power-mean did
(0.48 → 0.34 going from 2 to 3 models), while chi² and trimmed-chi² stay stable in both
scopes. See the alignment-scope table below.

## Results

### DUO-vs-MDLM discrimination at each model's near-optimum point (the trust-check table)

| Divergence | discrimination (η²) | consistent sign across all 5 NFEs? |
|---|---:|:---:|
| rank-Wasserstein | 0.850 | ✅ |
| chi² | 0.843 | ✅ |
| power-mean (ratio, t→∞ family) | 0.735 | ❌ flips at NFE=32 |
| max-ratio | 0.626 | ❌ flips at NFE=32 |

### Final leaderboard (research_score, corrected harness)

**Caveat added after the SFM re-check above: the `alignment` column here was computed
whole-grid, which we now know inflates tail-sensitive metrics (Rényi-α100 most of all —
see the alignment-scope table above). Treat Rényi-α100's rank in this table as
overstated; trimmed-chi² and chi² are the trustworthy top entries once alignment is
measured at near-optimum.**

| Divergence | research_score | resolving_power | alignment (whole-grid) | discrimination |
|---|---:|---:|---:|---:|
| ~~Rényi-α (α=100)~~ | ~~0.656~~ | 0.598 | 0.628 (inflated, see above) | 0.935 |
| **Trimmed chi² (drop top-2 bins)** | **0.646** | 0.597 | 0.586 | 0.931 |
| Anderson-Darling | 0.643 | 0.594 | 0.562 | 0.972 |
| Cressie-Read (λ=2) | 0.640 | 0.593 | 0.576 | 0.931 |
| Bhattacharyya | 0.637 | 0.595 | 0.557 | 0.936 |
| **chi²** (K25, n_log10) | 0.637 | 0.592 | 0.567 | 0.931 |
| rank-Wasserstein (log cost) | 0.637 | 0.593 | 0.544 | **0.966** |
| KL | 0.636 | 0.593 | 0.560 | 0.934 |
| Symmetric/Neyman chi², triangular, Amari-α, equal-mass binning | 0.634–0.637 | ~0.59 | ~0.55–0.56 | ~0.92–0.95 |
| MMD (RBF), Wasserstein-2 | ~0.63 | ~0.59 | ~0.54 | ~0.93–0.94 |
| Rank-weighted chi², head-tail split | 0.625–0.646 | ~0.59 | ~0.54–0.58 | ~0.75–0.96 |
| Power-mean, max-ratio, top-m (any order/direction) | 0.62–0.66 raw | — | — | **rejected: inconsistent NFE sign** |
| Head-only chi² (no tail bins) | 0.59–0.60 | 0.576 | 0.55 | 0.73–0.76 | clearly worse — tail bins matter |

### Alignment: whole-grid vs near-optimum, 2 models (DUO+MDLM) vs 3 models (+SFM)

`alignment` = mean sign-corrected Spearman correlation between the divergence and
MAUVE/GM/Energy-distance/FMTyp-p. "Near-optimum" restricts to each model's own
best-per-NFE point (5 NFEs × N models); "whole-grid" pools every sampled config.

| metric | whole-grid, 2-model | whole-grid, 3-model | near-optimum, 2-model | near-optimum, 3-model |
|---|---:|---:|---:|---:|
| max-ratio | 0.808 | 0.554 | 0.273 | 0.163 |
| power-mean | 0.807 | 0.551 | 0.270 | 0.240 |
| **Rényi-α100** | 0.805 | 0.553 | 0.476 | **0.341** |
| trimmed-chi² | 0.765 | 0.491 | 0.794 | **0.789** |
| Anderson-Darling | 0.731 | 0.443 | 0.555 | 0.445 |
| chi² | 0.744 | 0.466 | 0.821 | 0.764 |
| rank-Wasserstein | 0.450 | 0.211 | 0.579 | 0.539 |

Rényi-100 tracks max-ratio/power-mean almost exactly on the whole grid (all ~0.8) and
collapses the same way they do once restricted to near-optimum — confirming it inherited
their tail-sensitivity failure mode rather than being a genuine refinement of chi².
**Trimmed-chi² is the only metric whose near-optimum alignment is stable going from 2 to
3 models** (0.794 → 0.789); chi² is a close second but loses more ground (0.821 → 0.764).

### Original correlation table — rank-W vs KL vs chi², all metrics

Spearman correlation, pooled DUO+MDLM+SFM grid (established before the systematic
search below; motivated the whole investigation):

| metric | gen_ppl | unigram_entropy | mauve | gm | energy_distance | fmtyp_p | rep_1 | rep_2 | rep_3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rank_wasserstein | 0.935 | 0.917 | −0.512 | 0.824 | 0.110 | −0.178 | −0.924 | −0.924 | −0.923 |
| kl_bin20log5 | 0.821 | 0.804 | **−0.618** | 0.833 | **0.304** | **−0.344** | −0.810 | −0.810 | −0.810 |
| chi_square_bin20log5 | 0.744 | 0.729 | **−0.649** | 0.812 | **0.382** | **−0.414** | −0.734 | −0.735 | −0.734 |

This table is what first flagged **rank-Wasserstein as the weak link**: it correlates
almost perfectly with gen-PPL/entropy (i.e. it behaves like a repetition/collapse
detector) and is nearly *uncorrelated* with the LM-free Energy Distance (0.110) and
FMTyp-p (−0.178) — it has a well-documented **cold-collapse "hacking" failure mode**:
its optimum sits at a low-entropy, high-repetition regime that MAUVE/DE/FMTyp-p all
reject. chi²/KL don't share this pathology.

## Final recommendation

**Primary: trimmed chi-square, K=25, n_log=10, drop the top-2 most-deviant bins before
summing.**

```
chi2_i = (p_i - q_i)^2 / q_i
D_trimmed(p‖q) = sum(sorted(chi2)[:-2])   # drop the 2 largest per-bin terms
```

- Only metric whose near-optimum alignment with the gold standards (MAUVE/GM/DE/FMTyp-p)
  is stable going from 2 models (DUO+MDLM, 0.794) to 3 models (+SFM, 0.789) — every other
  candidate, including Rényi-α100, degrades by 0.03–0.13 when SFM is added (see the
  alignment-scope table above).
- research_score 0.646 in the whole-grid leaderboard, and the top trustworthy entry once
  Rényi-α100's inflated whole-grid alignment is discounted.
- Same discrimination robustness as chi² (η²=0.931, consistent sign across all 5 NFEs),
  without chi²'s larger near-optimum alignment drop.
- Tradeoff: one extra step (sort + drop top-k) vs plain chi², but no new hyperparameter
  beyond the trim count (fixed at 2, chosen to be small relative to the ~35-bin scheme).

**Simpler fallback: plain chi-square, K=25, n_log=10.**

- research_score 0.637, statistically tied with rank-Wasserstein on the (corrected)
  discrimination axis, without rank-Wasserstein's cold-collapse hacking vulnerability.
  Loses more near-optimum alignment than trimmed-chi² once SFM is added (0.821 → 0.764)
  but is still second-best of the metrics checked.
- Trivial closed form, no extra hyperparameters beyond the (already-standard) binning.
- Recommended if implementation simplicity should dominate the last few points of
  robustness.

**Downgraded after the SFM re-check: Rényi-α (α=100).** Originally the top pick on
whole-grid `research_score` (0.656), but its near-optimum alignment collapses on the
same axis as max-ratio/power-mean (0.476 → 0.341 going from 2 to 3 models) — see
"A fourth bug" above. It is not a genuine refinement of chi²; its earlier apparent edge
was a whole-grid tail-sensitivity artifact. Not recommended as primary; may still be
useful as a secondary diagnostic precisely because it IS sensitive to single-bin extreme
deviation, but should not be trusted as the headline metric.

**Not recommended:** rank-Wasserstein (cold-collapse hacking, weak alignment with
LM-free/typicality gold standards), max-ratio / power-mean / top-m in their raw
order-statistic form (fail the per-NFE sign-consistency check regardless of raw score —
they don't reliably tell DUO and MDLM apart, and now Rényi-α100 for the same reason).

## Where to find things

- `src/ranking_divergence/divergences.py` — the divergence library (binned f-divergences,
  alternative OT ground costs, tail/rank summaries).
- `examples/explore_divergences.py`, `plot_efficiency_summary.py`,
  `plot_metric_vs_temperature.py` — analysis and plotting tooling used to produce the
  tables and figures above.
- `outputs/diffusion_sweep_analysis/{duo-sflm-v1,mdlm-sflm-v1}/histograms/*.pt` +
  `reference_rank_histogram.pt` — the per-config rank histograms and held-out reference
  histograms underlying every result in this document, so they can be reproduced without
  re-scoring with `gpt2-large`.
