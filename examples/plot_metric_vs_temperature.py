"""Plot unigram-entropy / gen-PPL / a divergence vs temperature, one curve per NFE.

Reuses the 1x3 per-NFE-curve style of plot_diffusion_sweeps.plot_method_summary but lets
the third panel be any divergence column from a divergence_metrics.csv (rank_wasserstein,
chi_square_bin20log5, kl_bin20log5, ...). Produces one figure per (run, third-metric).
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Sequence

import numpy as np

LABELS = {
    "unigram_entropy": "Unigram entropy (nats)",
    "gen_ppl": "Generative perplexity",
    "rank_wasserstein": "Rank-Wasserstein",
    "chi_square_bin20log5": "Chi-square (20+5 bins)",
    "kl_bin20log5": "KL (20+5 bins)",
    "js_bin20log5": "Jensen-Shannon (20+5 bins)",
    "mauve": "MAUVE",
    "gm": "Gradient Moment",
    "energy_distance": "Energy distance D²_E",
    "fmtyp_p": "FMTyp-p",
    "chi2": "Chi-square (25+10 bins)",
    "max_ratio": "Max-ratio (autoresearch, 15+15 bins)",
    "powmean": "Power-mean t=300 (autoresearch, 15+15 bins)",
    "trimmed_chi2": "Trimmed chi-square (drop top-2, 25+10 bins)",
    "topm_logratio_m3": "Top-m log-ratio (m=3, 15+15 bins)",
}
# distance-like metrics span orders of magnitude -> log y; entropy/MAUVE/etc stay linear.
LOG_SCALE: set[str] = {"gen_ppl", "rank_wasserstein", "chi_square_bin20log5",
                       "kl_bin20log5", "js_bin20log5", "chi2", "max_ratio", "powmean",
                       "trimmed_chi2", "topm_logratio_m3"}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metrics-csv", type=Path, required=True)
    p.add_argument("--title", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--third-metric", default="rank_wasserstein")
    p.add_argument("--metrics", default=None,
                   help="Comma-separated panel list; overrides the default entropy/gen_ppl/third-metric.")
    p.add_argument("--dpi", type=int, default=200)
    return p.parse_args(argv)


def load_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    out = []
    for row in rows:
        try:
            row["nfe"] = int(float(row["nfe"]))
            row["temperature"] = float(row["temperature"])
        except (KeyError, ValueError):
            continue
        out.append(row)
    return out


def plot(rows: list[dict], metrics: list[str], title: str, output: Path, dpi: int) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(metrics), figsize=(4.5 * len(metrics), 4.4))
    axes = np.atleast_1d(axes)
    nfes = sorted({int(r["nfe"]) for r in rows})
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(nfes)))
    for ax, metric in zip(axes, metrics):
        for nfe, color in zip(nfes, colors):
            nfe_rows = sorted(
                (r for r in rows if int(r["nfe"]) == nfe and r.get(metric) not in (None, "")),
                key=lambda r: r["temperature"],
            )
            xs = [r["temperature"] for r in nfe_rows]
            ys = [float(r[metric]) for r in nfe_rows]
            if not xs:
                continue
            ax.plot(xs, ys, marker="o", markersize=2.5, linewidth=1.4, color=color, label=f"NFE {nfe}")
        ax.set_xlabel("Temperature")
        ax.set_ylabel(LABELS.get(metric, metric))
        ax.grid(alpha=0.25)
        if metric in LOG_SCALE:
            ax.set_yscale("log")
    axes[-1].legend(title="Sampling steps", fontsize=8)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    rows = load_rows(args.metrics_csv)
    metrics = ([m.strip() for m in args.metrics.split(",")] if args.metrics
               else ["unigram_entropy", "gen_ppl", args.third_metric])
    plot(rows, metrics, args.title, args.output, args.dpi)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
