"""Efficiency summary (all metrics) in the style of all_metrics_efficiency_summary.png.

For each method and NFE we select the temperature minimizing a chosen criterion
(default: rank_wasserstein -- "best rank-distance temperature", matching the original
figure) and plot every metric at that selected operating point, x=NFE, one line/method.

Panels = the four reference metrics (unigram_entropy, gen_ppl, MAUVE, GM) plus every
divergence in ranking_divergence.DIVERGENCES.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Sequence

import numpy as np

from ranking_divergence import DIVERGENCES

REFERENCE = ["unigram_entropy", "gen_ppl", "mauve", "gm", "energy_distance", "fmtyp_p",
             "rep_1", "rep_2", "rep_3", "rank_wasserstein"]
MAXIMIZE = {"mauve", "fmtyp_p"}  # higher is better; every other metric is minimized
LINEAR = {"unigram_entropy", "mauve", "fmtyp_p", "rep_1", "rep_2", "rep_3",
          "median_rank_gap", "cdf_spearman"}  # others log-scaled (distances span orders)
NFES = [8, 16, 32, 64, 128]
METHOD_STYLE = {"duo": ("#54a24b", "^"), "mdlm": ("#4c78a8", "o"), "sfm": ("#f58518", "D"),
                "candi": ("#e377c2", "s"), "ar": ("#e45756", "*")}
LABELS = {"unigram_entropy": "Unigram entropy (nats)", "gen_ppl": "Generative perplexity",
          "mauve": "MAUVE", "gm": "Gradient Moment", "energy_distance": "Energy distance D²_E",
          "fmtyp_p": "FMTyp-p", "rep_1": "Rep-1", "rep_2": "Rep-2", "rep_3": "Rep-3"}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", action="append", required=True, metavar="METHOD=CSV",
                   help="method=path/to/divergence_metrics.csv (repeatable)")
    p.add_argument("--baselines", type=Path, default=None,
                   help="CSV of parameter-free baselines (one row each) plotted at NFE 0")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dpi", type=int, default=200)
    return p.parse_args(argv)


def load(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for r in rows:
        r["nfe"] = int(float(r["nfe"]))
    return rows


def num(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def best_value(rows: list[dict], metric: str, nfe: int) -> float | None:
    """Per-metric optimum over temperature at this NFE (max for MAUVE, min otherwise)."""

    vals = [num(r.get(metric)) for r in rows if r["nfe"] == nfe]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return max(vals) if metric in MAXIMIZE else min(vals)


def main(argv: Sequence[str] | None = None) -> None:
    import matplotlib.pyplot as plt

    args = parse_args(argv)
    runs = {}
    for spec in args.run:
        method, path = spec.split("=", 1)
        runs[method] = load(Path(path))
    baselines = []
    if args.baselines is not None:
        with args.baselines.open(newline="", encoding="utf-8") as h:
            baselines = list(csv.DictReader(h))
    bl_styles = {"Mirror": ("X", "#b279a2"), "Periodic": ("P", "#ff9da6"),
                 "Phrase bank": ("v", "#9d755d"), "Top-k IID": ("h", "#bab0ac")}

    metrics = REFERENCE + list(DIVERGENCES)
    ncols = 5
    nrows = math.ceil(len(metrics) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.0 * nrows))
    axes = axes.flatten()

    for ax, metric in zip(axes, metrics):
        arrow = "↑" if metric in MAXIMIZE else "↓"
        for method, rows in runs.items():
            color, marker = METHOD_STYLE.get(method, ("#666666", "o"))
            xs, ys = [], []
            for i, n in enumerate(NFES):
                y = best_value(rows, metric, n)
                if y is None:
                    continue
                xs.append(i + 1)  # leave x=0 for parameter-free baselines
                ys.append(y)
            if xs:
                ax.plot(xs, ys, color=color, marker=marker, markersize=6, linewidth=1.6,
                        label=method.upper())
        for row in baselines:
            y = num(row.get(metric))
            if y is None:
                continue
            mk, col = bl_styles.get(row["method"], ("D", "#444444"))
            ax.scatter([0], [y], marker=mk, s=55, color=col, edgecolor="black", linewidth=0.4,
                       label=row["method"], zorder=4)
        ax.set_title(f"{LABELS.get(metric, metric)} ({arrow})", fontsize=9)
        ax.set_xticks(range(len(NFES) + 1))
        ax.set_xticklabels(["pf"] + [str(n) for n in NFES])
        ax.set_xlabel("NFE")
        ax.grid(alpha=0.25, which="both")
        if metric not in LINEAR:
            ax.set_yscale("log")
    for ax in axes[len(metrics):]:
        ax.set_visible(False)
    axes[0].legend(fontsize=8)
    fig.suptitle("Each metric at its OWN best temperature per NFE "
                 "(↑ = maximized, ↓ = minimized; baselines at NFE 0 = 'pf')", fontsize=13, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
