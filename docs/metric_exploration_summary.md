# Rank-histogram divergence metric exploration — summary

**Goal:** replace the repo's original metric — log-rank Wasserstein-1 between a model's
next-token rank histogram and a held-out real-text rank histogram (under a fixed
`gpt2-large` reference) — with a divergence that better **differentiates small
differences in model performance**, while correlating with established distributional
quality metrics and cleanly discriminating between models.

## Setup

- **Models:** DUO, MDLM, S-FLM (SFM), sampled with the official `s-flm` repo/checkpoints,
  across temperature × NFE ∈ {8, 16, 32, 64, 128}. DUO and SFM cover t=0.2–1.4 (245
  configs each); MDLM covers t=0.1–2.0 (150 configs). All three are used in every
  comparison below.
- **Reference:** rank histogram of held-out OpenWebText (128 docs), `gpt2-large`.
- **Binning:** every candidate divergence operates on a coarsened histogram — K linear
  head bins (ranks 1..K) + n_log log-spaced tail bins, K≈15–30, n_log≈5–15.
- **Gold standards** (from *Hacking Generative Perplexity*), computed per config:

  | Metric | Direction | What it measures |
  |---|---|---|
  | MAUVE | ↑ better | divergence frontier in gpt2-large embedding space |
  | GM (Gradient Moment) | ↓ better | Fisher-score MMD (per-token NLL-gradient distance) |
  | DE (Energy distance) | ↓ better | LM-free surface/discourse/entity features |
  | FMTyp-p | ↑ better | Mahalanobis typicality p-value vs reference |

  gen-PPL, unigram entropy, and Rep-1/2/3 are tracked as diagnostics, not gold standards
  — a metric that only tracks these can be hacked by cold, repetitive, low-entropy output
  (see below).

## Method

**Candidate families:** binned f-divergences (KL, reverse-KL, JS, symmetric-KL, chi²,
Hellinger, TV, symmetric/Neyman-χ², triangular, Bhattacharyya), Rényi-α and Amari-α,
Cressie-Read power-divergence (continuous λ), trimmed chi-square (drop the k most-deviant
bins), Anderson-Darling, max-ratio and power-mean ratio divergence, top-m mean log-ratio
(mean of the m largest per-bin `|log(p_i/q_i)|` terms), MMD (RBF kernel), Wasserstein-2,
equal-reference-mass binning, and alternative OT ground costs (linear/sqrt/capped-log) —
71 distinct parameterizations checked in total.

**Scoring.** Each candidate gets a `research_score = 0.6·resolving_power +
0.25·alignment + 0.15·discrimination`:
- `resolving_power`: build a composite quality score per config (z-scored, sign-corrected
  MAUVE/GM/DE/FMTyp-p, averaged); take the 25%-closest config pairs by that composite,
  pooled across all three models (~51k pairs); resolving_power = fraction ordered
  correctly by the candidate.
- `alignment`: mean sign-corrected Spearman correlation with MAUVE/GM/DE/FMTyp-p, at each
  model's near-optimum (best-per-NFE) point, pooled across all three models.
- `discrimination`: one-way ANOVA η² separating DUO/MDLM/SFM at their near-optimum points.

**The methodology bug that shaped every result below.** An early version of this harness
scored `alignment`/`discrimination` pooled across the *whole* temperature grid, and on
that basis declared max-ratio, power-mean, and later Rényi-α100 the clear winners. This
is a trap: cold, collapsed, high-repetition configs are uniformly bad on every axis at
once, so any statistic that's hypersensitive to extreme outlier bins "wins" on that
trivial global trend — regardless of whether it's a good divergence. Restricting to each
model's **near-optimum** point (the only regime that matters in practice) exposes this:
max-ratio and power-mean stop even reliably ordering DUO vs MDLM (sign flips at NFE=32),
and Rényi-α100's alignment collapses the same way (0.48 → 0.34 going from DUO+MDLM to
+SFM) despite topping the whole-grid leaderboard. Every result below uses the
near-optimum scope, plus a trust-check per candidate (per-NFE sign consistency,
neighborhood robustness, correlation with chi² as a same-family sanity check).

## Results

### Leaderboard (research_score)

| Divergence | research_score | resolving_power | alignment | discrimination |
|---|---:|---:|---:|---:|
| **Top-m log-ratio (m=3)** | **0.679** | 0.579 | 0.846 | 0.799 |
| Top-m log-ratio (m=5) | 0.676 | 0.582 | 0.816 | 0.816 |
| **Trimmed chi² (d=2)** | **0.674** | 0.610 | 0.789 | 0.734 |
| Cressie-Read (λ=−2) | 0.623 | 0.570 | 0.847 | 0.463 |
| chi² (K25, n_log10) | 0.619 | 0.603 | 0.764 | 0.444 |
| KL / reverse-KL | 0.613–0.615 | 0.58–0.59 | 0.79 | 0.44–0.45 |
| Rényi-α100 | 0.581 | 0.625 | 0.341 | 0.802 |
| rank-Wasserstein (log cost) | 0.579 | 0.573 | 0.539 | 0.671 |
| power-mean | 0.547 | 0.626 | 0.240 | 0.746 |
| Anderson-Darling | 0.543 | 0.594 | 0.445 | 0.503 |
| max-ratio | 0.521 | 0.626 | 0.163 | 0.698 |

**Top-m log-ratio (m=3/m=5) and trimmed-chi² form a tight top tier**, ahead of everything
else by a real margin. They fail differently: top-m log-ratio has the best alignment and
is the *only* candidate whose alignment **improves** when SFM is added (+0.12, vs
2-model-only DUO+MDLM where it actually trailed trimmed-chi²) — a genuine sign of
generalizing rather than overfitting DUO/MDLM's shared cold-collapse shape. Trimmed-chi²
has the best resolving_power and is the most *scope-stable* candidate checked (its
alignment barely moves, Δ=−0.005, going from 2 to 3 models). Cressie-Read(λ=−2) has the
single best alignment score (0.847) but the weakest discrimination in the top tier
(0.463) — a Pareto tradeoff, not a win. Rényi-α100/power-mean/max-ratio remain bottom-
tier: strong on resolving_power/discrimination individually, but their collapsed
near-optimum alignment (the tail-sensitivity artifact above) drags the composite down.

### The original correlation table that started this investigation

Spearman correlation, pooled DUO+MDLM+SFM, whole grid:

| metric | gen_ppl | unigram_entropy | mauve | gm | energy_distance | fmtyp_p |
|---|---:|---:|---:|---:|---:|---:|
| rank_wasserstein | 0.935 | 0.917 | −0.512 | 0.824 | 0.110 | −0.178 |
| KL (bin20log5) | 0.821 | 0.804 | −0.618 | 0.833 | 0.304 | −0.344 |
| chi² (bin20log5) | 0.744 | 0.729 | −0.649 | 0.812 | 0.382 | −0.414 |

rank-Wasserstein correlates almost perfectly with gen-PPL/entropy (behaves like a
repetition/collapse detector) and is nearly uncorrelated with the LM-free Energy Distance
(0.110) and FMTyp-p (−0.178) — a **cold-collapse hacking** failure mode: its optimum sits
at a low-entropy, high-repetition regime that MAUVE/DE/FMTyp-p all reject. This is what
motivated the whole search.

## Final recommendation

**Primary (tied): top-m mean log-ratio (m=3), or trimmed chi-square (K=25, n_log=10,
drop top-2 bins).**

```
# Top-m log-ratio (m=3)
r_i = |log(p_i / q_i)|
D_topm(p‖q) = mean(sorted(r)[-3:])       # mean of the 3 largest per-bin log-ratios

# Trimmed chi-square (d=2)
chi2_i = (p_i - q_i)^2 / q_i
D_trimmed(p‖q) = sum(sorted(chi2)[:-2])  # drop the 2 largest per-bin terms
```

Pick **top-m log-ratio** if generalizing to future/unseen model families matters most —
it's the only metric that gets *better*, not worse, when a third structurally-different
model is added. Pick **trimmed chi²** if stability and resolving power on the models at
hand matter most — it moves the least across every scope tested. Both clear every
trust-check that ruled out max-ratio and power-mean.

**Simpler fallback:** plain chi-square, K=25, n_log=10 (research_score 0.619) — a
trivial closed form, no sort/trim step, if implementation simplicity should dominate the
last few points of robustness.

**Not recommended:** rank-Wasserstein (cold-collapse hacking, weak alignment with
LM-free/typicality gold standards); max-ratio / power-mean in raw single-extreme form
(fail per-NFE sign consistency); Rényi-α100 and Cressie-Read (λ=−2/−1.5) (each wins one
axis individually but neither converts it into a competitive overall score — useful as a
secondary diagnostic, not as the headline metric).

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
