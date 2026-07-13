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
- **Top-m mean log-ratio** (mean of the m largest `|log(p_i/q_i)|` per-bin terms — an
  order-statistic family, but log-compressed and averaged over several bins rather than
  the single raw-ratio extreme that max-ratio/power-mean use).
- **MMD** with an RBF kernel on log-rank support points.
- **Wasserstein-2** (quadratic ground cost, via quantile-function matching).
- Cheap tail/rank summaries: KS statistic, CDF-L2, CDF-Spearman, top-k mass difference,
  mean/median log-rank gap.

71 distinct parameterizations of the families above were checked under the corrected
(near-optimum, 3-model) methodology — see Results below. Each was reconstructed directly
from its defining formula and reimplemented against the current DUO/MDLM/SFM data; this
supersedes an earlier, larger ad hoc LLM-agent search (~180 candidates) whose
implementation predates the SFM cold-extension and the near-optimum-alignment fix
described below, so its raw per-candidate numbers are no longer authoritative.

## Evaluation methodology

A systematic search-and-evaluation harness scores every candidate on three axes:

- **`resolving_power`** (60% weight, headline): build a composite quality score per
  config by z-scoring MAUVE/GM/DE/FMTyp-p (sign-corrected, higher=better) and averaging;
  take the 25%-closest pairs of configs by that composite (pooled across all three
  models, ~51k pairs); `resolving_power` = fraction of those close pairs the candidate
  orders correctly.
- **`alignment`** (25%): mean sign-corrected Spearman correlation with MAUVE/GM/DE/FMTyp-p,
  at each model's near-optimum (best-per-NFE) point, pooled across all three models.
- **`discrimination`** (15%): one-way ANOVA η² separating DUO/MDLM/SFM at each model's
  own near-optimum (best-per-NFE) point (see methodology note below).

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

The same whole-grid-vs-near-optimum distinction turns out to matter for `alignment`
too, not just `discrimination`. Pooled across the whole temperature grid, tail-sensitive
statistics (max-ratio, power-mean, Rényi-α100) all score ~0.80 alignment — but that's a
trivial artifact: cold, collapsed, high-repetition configs are uniformly bad on every
axis, so any tail-sensitive metric "wins" on the global trend regardless of whether it's
actually a good divergence. Restricted to near-optimum points, where the real
discrimination has to happen, Rényi-α100 collapses the same way max-ratio/power-mean do
(0.48 → 0.34 going from DUO+MDLM to DUO+MDLM+SFM), while chi² and trimmed-chi² stay
stable in both scopes — see the alignment-scope table below.

## Results

### DUO-vs-MDLM discrimination at each model's near-optimum point (the trust-check table)

| Divergence | discrimination (η², DUO+MDLM only) | consistent sign across all 5 NFEs? |
|---|---:|:---:|
| rank-Wasserstein | 0.850 | ✅ |
| chi² | 0.843 | ✅ |
| trimmed-chi² (d=2) | 0.474 | ✅ |
| top-m log-ratio (m=5) | 0.381 | ✅ |
| top-m log-ratio (m=3) | 0.365 | ✅ |
| power-mean (ratio, t→∞ family) | 0.735 | ❌ flips at NFE=32 |
| Cressie-Read (λ=−2) | 0.265 | ✅ |
| max-ratio | 0.626 | ❌ flips at NFE=32 |

Top-m log-ratio and Cressie-Read pass the sign-consistency check that max-ratio/power-mean
fail — averaging over several worst bins in log space (top-m) or using a negative-λ
Cressie-Read exponent behaves very differently from taking a single raw-ratio extremum.
Their η² on this DUO-vs-MDLM-only slice is modest, but rises substantially once SFM is
folded in as a third group — see the research_score table below, which uses the 3-model
version of `discrimination`.

### Final leaderboard (research_score)

| Divergence | research_score | resolving_power | alignment (near-opt, 3-model) | discrimination (3-model η²) |
|---|---:|---:|---:|---:|
| **Top-m log-ratio (m=3)** | **0.679** | 0.579 | 0.846 | 0.799 |
| Top-m log-ratio (m=5) | 0.676 | 0.582 | 0.816 | 0.816 |
| **Trimmed chi² (d=2)** | **0.674** | 0.610 | 0.789 | 0.734 |
| Cressie-Read (λ=−2) | 0.623 | 0.570 | 0.847 | 0.463 |
| Cressie-Read (λ=−1.5) | 0.620 | 0.575 | 0.823 | 0.462 |
| chi² (K25, n_log10) | 0.619 | 0.603 | 0.764 | 0.444 |
| KL | 0.615 | 0.585 | 0.789 | 0.448 |
| reverse-KL | 0.613 | 0.581 | 0.792 | 0.444 |
| Rényi-α (α=100) | 0.581 | 0.625 | 0.341 | 0.802 |
| rank-Wasserstein (log cost) | 0.579 | 0.573 | 0.539 | 0.671 |
| power-mean | 0.547 | 0.626 | 0.240 | 0.746 |
| Anderson-Darling | 0.543 | 0.594 | 0.445 | 0.503 |
| max-ratio | 0.521 | 0.626 | 0.163 | 0.698 |

**Top-m log-ratio (m=3 and m=5) and trimmed-chi² form a tight top tier (0.674–0.679,
within noise given the small near-optimum samples underlying `alignment`/`discrimination`)**,
clearly ahead of the next tier (Cressie-Read λ=−2, chi², KL, reverse-KL: 0.61–0.62).
Rényi-α100, power-mean and max-ratio score well on `resolving_power` and 3-model
`discrimination` individually but their collapsed near-optimum `alignment` drags
research_score down — the tail-sensitivity pathology identified earlier applies to the
composite score as much as to alignment alone.

### Alignment: whole-grid vs near-optimum, 2 models (DUO+MDLM) vs 3 models (+SFM)

`alignment` = mean sign-corrected Spearman correlation between the divergence and
MAUVE/GM/Energy-distance/FMTyp-p. "Near-optimum" restricts to each model's own
best-per-NFE point (5 NFEs × N models); "whole-grid" pools every sampled config. Top 20
of 71 distinct candidates checked, ranked by near-optimum alignment with 3 models (the
scope judged most trustworthy — see the methodology note above). Δ is the change in
near-optimum alignment going from 2 models (DUO+MDLM) to 3 (+SFM).

| metric | whole-grid, 2-model | whole-grid, 3-model | near-optimum, 2-model | near-optimum, 3-model | Δ (2→3) |
|---|---:|---:|---:|---:|---:|
| Cressie-Read (λ=−2) | 0.493 | 0.254 | **0.900** | 0.847 | −0.053 |
| **Top-m log-ratio (m=3)** | 0.596 | 0.369 | 0.722 | **0.846** | **+0.125** |
| Cressie-Read (λ=−1.5) | 0.524 | 0.282 | 0.821 | 0.823 | +0.002 |
| Top-m log-ratio (m=5) | 0.616 | 0.381 | 0.697 | 0.816 | +0.118 |
| Top-m log-ratio (m=10) | 0.674 | 0.423 | 0.673 | 0.809 | +0.135 |
| reverse-KL (bin20log5) | 0.566 | 0.319 | 0.821 | 0.792 | −0.029 |
| trimmed-chi² (d=2) | 0.765 | 0.491 | 0.794 | 0.789 | −0.005 |
| Amari-α (α=2) | 0.717 | 0.446 | 0.821 | 0.789 | −0.032 |
| Amari-α (α=0.5) | 0.645 | 0.389 | 0.821 | 0.789 | −0.032 |
| KL (bin20log5 / bin50log10 / std) | 0.672 | 0.411 | 0.821 | 0.789 | −0.032 |
| trimmed-chi² (d=1) | 0.772 | 0.496 | 0.794 | 0.788 | −0.006 |
| Hellinger (bin20log5) | 0.617 | 0.365 | 0.821 | 0.781 | −0.041 |
| Jensen-Shannon (bin20log5 / bin50log10 / std) | 0.612 | 0.361 | 0.821 | 0.781 | −0.041 |
| Symmetric-KL (bin20log5) | 0.624 | 0.370 | 0.821 | 0.781 | −0.041 |
| Bhattacharyya | 0.617 | 0.365 | 0.821 | 0.781 | −0.041 |
| Symmetric-χ² | 0.647 | 0.388 | 0.821 | 0.781 | −0.041 |
| chi² (bin20log5 / K25n10) | 0.744 | 0.466 | 0.821 | 0.764 | −0.057 |
| rank-Wasserstein (log cost) | 0.450 | 0.211 | 0.579 | 0.539 | −0.040 |
| Anderson-Darling | 0.731 | 0.443 | 0.555 | 0.445 | −0.110 |
| Rényi-α100 | 0.805 | 0.553 | 0.476 | 0.341 | −0.134 |

Rényi-100, max-ratio and power-mean (all rank near the bottom of the 71, not shown) track
each other almost exactly on the whole grid (~0.80) and collapse once restricted to
near-optimum — the same tail-sensitivity failure mode identified earlier. **Top-m
log-ratio is the only family whose near-optimum alignment *improves* when SFM is added**
(+0.12 to +0.14 across m=3/5/10) — at 2-model scope it actually trailed trimmed-chi²
(0.70–0.72 vs 0.79), and only pulls into the lead once SFM's differently-shaped optimum
is included, a good sign for generalization rather than overfitting to DUO/MDLM. Cressie-
Read (λ=−2) is the mirror case: strongest of all at 2-model scope (0.900) but drifts back
toward the pack at 3-model (0.847) — and, per the research_score table above, its 3-model
`discrimination` is also the weakest of the top tier (0.463), so its high alignment
doesn't translate into the best overall candidate. trimmed-chi² remains the most *stable*
metric checked, with the smallest |Δ| of any candidate near the top (0.005–0.006).

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

**Primary (tied): top-m mean log-ratio (m=3) and trimmed chi-square (K=25, n_log=10,
drop top-2 bins).** These two lead `research_score` (0.679 and 0.674) by a real margin
over the next tier (0.61–0.62), and the gap between them is within noise given the small
near-optimum samples underlying `alignment`/`discrimination`. They fail differently, so
the choice between them depends on which failure mode matters more:

```
# Top-m log-ratio (m=3)
r_i = |log(p_i / q_i)|
D_topm(p‖q) = mean(sorted(r)[-3:])       # mean of the 3 largest per-bin log-ratios

# Trimmed chi-square (d=2)
chi2_i = (p_i - q_i)^2 / q_i
D_trimmed(p‖q) = sum(sorted(chi2)[:-2])  # drop the 2 largest per-bin terms
```

- **Top-m log-ratio (m=3)**: best overall research_score, best near-optimum alignment
  with 3 models (0.846), and the only family whose alignment *improves* rather than
  degrades when SFM is added (+0.125). Weaker resolving_power (0.579) than trimmed-chi²
  (0.610) and its alignment advantage is a 3-model-scope phenomenon — with only DUO+MDLM
  it trails trimmed-chi² (0.722 vs 0.794), so its strength is specifically in
  generalizing to a third, structurally different model.
- **Trimmed chi² (d=2)**: best resolving_power (0.610) and the most *stable* metric
  checked — smallest alignment swing of any top-tier candidate going from 2 to 3 models
  (Δ=−0.005). Same discrimination robustness as chi² without chi²'s larger near-optimum
  alignment drop. Preferable if stability/predictability across future model additions
  matters more than squeezing out the last bit of resolving power.
- Both pass every trust-check (per-NFE sign consistency, neighborhood robustness,
  correlation with chi² as a same-family sanity check) that ruled out max-ratio and
  power-mean.

**Simpler fallback: plain chi-square, K=25, n_log=10.**

- research_score 0.619, behind the primary pair but a trivial closed form with no extra
  hyperparameters beyond the (already-standard) binning — no sort/trim/order-statistic
  step. Recommended if implementation simplicity should dominate the last few points of
  robustness.

**Not recommended:** rank-Wasserstein (cold-collapse hacking, weak alignment with
LM-free/typicality gold standards); max-ratio and power-mean in their raw single-extreme
form (fail the per-NFE sign-consistency check — they don't reliably tell DUO and MDLM
apart); **Rényi-α (α=100)** and Cressie-Read (λ=−2/−1.5) — Rényi's near-optimum alignment
collapses on the same axis as max-ratio/power-mean (0.476 → 0.341 from 2 to 3 models),
and Cressie-Read's very high 2-model alignment (0.900) is undercut by the weakest
3-model discrimination of any top-tier candidate (η²=0.463), so neither converts its
individual strength into a competitive overall research_score. All three may still be
useful as secondary diagnostics precisely because they're sensitive to single-bin extreme
deviation, but shouldn't be trusted as the headline metric.

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
