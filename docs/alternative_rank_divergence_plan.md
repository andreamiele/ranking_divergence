# Alternative rank-divergence study

Goal: compare alternatives to the existing log-rank Wasserstein metric on the same next-token rank histograms.

Planned/implemented components:

1. Persist per-config comparison histograms during scoring so GPT-2-large scoring is paid once. Diffusion sweeps write `histograms/<method>__<sample-file-stem>.pt`; parameter-free and OpenWebText analyses write `histograms/<name>.pt`; full runs also keep `reference_rank_histogram.pt`.
2. Compute alternative divergences offline from saved histograms: binned f-divergences, alternative Wasserstein ground costs, CDF/rank-tail summaries. The registry lives in `src/ranking_divergence/divergences.py`.
3. Run `examples/explore_divergences.py` on a scored run dir, optionally adding parameter-free baselines, to produce `divergence_metrics.csv`, correlation/discrimination/Pareto diagnostics, and `divergence_scorecard.csv`.
4. Rank candidate measures by a combined scorecard: correlation with existing metrics, method discrimination/V-curve behavior, and Pareto-front agreement with rank-Wasserstein.

The diffusion sample files are expected to be provided separately. Once copied here, run `examples/evaluate_diffusion_sweeps.py --sweep METHOD=DIR ...` first, then run `examples/explore_divergences.py --run-dir <scored-run> --baseline-run <optional-baselines>`.
