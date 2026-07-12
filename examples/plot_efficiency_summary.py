"""Efficiency summary: pick the lowest-rank-divergence temperature per method/NFE.

For each model and NFE we select the temperature that minimizes the chosen rank
divergence, then plot a small set of metrics (entropy / gen-PPL / rank-divergence) at
that operating point, x=NFE, one line per model. Parameter-free ("synthetic") baselines
are drawn at NFE 0. This mirrors docs/figures/all_metrics_efficiency_summary.png.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Sequence

import numpy as np

NFES = [8, 16, 32, 64, 128]
METHOD_STYLE = {"duo": ("#54a24b", "^", "DUO"), "mdlm": ("#4c78a8", "o", "MDLM"),
                "sfm": ("#f58518", "D", "SFM"), "candi": ("#e377c2", "s", "CANDI")}
BL_STYLE = {"Mirror": ("X", "#b279a2"), "Periodic": ("P", "#ff9da6"),
            "Phrase bank": ("v", "#9d755d"), "Top-k IID": ("h", "#bab0ac")}
LABELS = {"unigram_entropy": "Unigram entropy (nats)", "gen_ppl": "Generative perplexity",
          "rank_wasserstein": "Rank-Wasserstein", "chi_square_bin20log5": "Chi-square (20+5 bins)",
          "kl_bin20log5": "KL (20+5 bins)", "mauve": "MAUVE", "gm": "Gradient Moment"}
LOG = {"gen_ppl", "rank_wasserstein", "chi_square_bin20log5", "kl_bin20log5", "gm",
       "chi2", "max_ratio", "powmean", "trimmed_chi2", "anderson_darling", "renyi_a100"}
MAXIMIZE = {"mauve", "fmtyp_p"}  # higher is better -> select the max temperature instead of min


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", action="append", required=True, metavar="METHOD=CSV")
    p.add_argument("--baselines", type=Path, default=None)
    p.add_argument("--select", default="rank_wasserstein", help="rank divergence minimized to pick the temperature")
    p.add_argument("--metrics", default="unigram_entropy,gen_ppl,rank_wasserstein")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dpi", type=int, default=200)
    return p.parse_args(argv)


def num(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def load(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as h:
        rows = list(csv.DictReader(h))
    for r in rows:
        r["nfe"] = int(float(r["nfe"]))
    return rows


def best_rows(rows: list[dict], select: str) -> dict[int, dict]:
    maximize = select in MAXIMIZE
    best: dict[int, dict] = {}
    for r in rows:
        v = num(r.get(select))
        if v is None:
            continue
        n = r["nfe"]
        if n not in best or (v > num(best[n][select]) if maximize else v < num(best[n][select])):
            best[n] = r
    return best


def main(argv: Sequence[str] | None = None) -> None:
    import matplotlib.pyplot as plt

    args = parse_args(argv)
    selected = {spec.split("=", 1)[0]: best_rows(load(Path(spec.split("=", 1)[1])), args.select)
                for spec in args.run}
    baselines = list(csv.DictReader(args.baselines.open())) if args.baselines else []
    metrics = [m.strip() for m in args.metrics.split(",")]

    fig, axes = plt.subplots(1, len(metrics), figsize=(5.2 * len(metrics), 4.6))
    axes = np.atleast_1d(axes)
    for ax, metric in zip(axes, metrics):
        for method, best in selected.items():
            color, marker, label = METHOD_STYLE.get(method, ("#666", "o", method.upper()))
            xs = [i + 1 for i, n in enumerate(NFES) if n in best and num(best[n].get(metric)) is not None]
            ys = [num(best[n][metric]) for n in NFES if n in best and num(best[n].get(metric)) is not None]
            if xs:
                ax.plot(xs, ys, color=color, marker=marker, markersize=7, linewidth=1.7, label=label)
        for row in baselines:
            y = num(row.get(metric))
            if y is None:
                continue
            mk, col = BL_STYLE.get(row["method"], ("D", "#444"))
            ax.scatter([0], [y], marker=mk, s=70, color=col, edgecolor="black", linewidth=0.4,
                       label=row["method"], zorder=4)
        ax.set_title(LABELS.get(metric, metric), fontsize=11)
        ax.set_xticks(range(len(NFES) + 1))
        ax.set_xticklabels(["pf"] + [str(n) for n in NFES])
        ax.set_xlabel("NFE")
        ax.grid(alpha=0.25, which="both")
        if metric in LOG:
            ax.set_yscale("log")
    axes[0].legend(fontsize=8)
    direction = "highest" if args.select in MAXIMIZE else "lowest"
    fig.suptitle(f"Metrics at each method/NFE's {direction}-{args.select} temperature", fontsize=13)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
