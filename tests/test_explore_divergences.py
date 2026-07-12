import csv
import importlib.util
from pathlib import Path

import torch


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "examples" / "explore_divergences.py"
SPEC = importlib.util.spec_from_file_location("explore_divergences", SCRIPT_PATH)
assert SPEC is not None
explore_divergences = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(explore_divergences)


def test_explorer_matches_temperature_label_histograms_and_writes_scorecard(tmp_path):
    run_dir = tmp_path / "run"
    histogram_dir = run_dir / "histograms"
    histogram_dir.mkdir(parents=True)
    reference = torch.tensor([0.55, 0.25, 0.12, 0.05, 0.03], dtype=torch.float64)
    torch.save(reference, run_dir / "reference_rank_histogram.pt")

    rows = [
        {
            "method": "duo",
            "nfe": 8,
            "temperature": 0.5,
            "temperature_label": "0.500",
            "rank_wasserstein": 0.1,
            "gen_ppl": 20.0,
        },
        {
            "method": "duo",
            "nfe": 16,
            "temperature": 0.6,
            "temperature_label": "0.600",
            "rank_wasserstein": 0.06,
            "gen_ppl": 18.0,
        },
        {
            "method": "mdlm",
            "nfe": 8,
            "temperature": 0.7,
            "temperature_label": "0.700",
            "rank_wasserstein": 0.2,
            "gen_ppl": 30.0,
        },
    ]
    histograms = [
        torch.tensor([0.50, 0.27, 0.13, 0.07, 0.03], dtype=torch.float64),
        torch.tensor([0.54, 0.25, 0.12, 0.06, 0.03], dtype=torch.float64),
        torch.tensor([0.35, 0.25, 0.20, 0.12, 0.08], dtype=torch.float64),
    ]
    for row, histogram in zip(rows, histograms):
        key = f"{row['method']}__samples_steps{row['nfe']}_temp{row['temperature_label']}"
        torch.save(histogram / histogram.sum(), histogram_dir / f"{key}.pt")

    with (run_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    output_dir = tmp_path / "out"
    explore_divergences.main([
        "--run-dir",
        str(run_dir),
        "--output-dir",
        str(output_dir),
        "--skip-plots",
    ])

    scorecard = output_dir / "run" / "divergence_scorecard.csv"
    metrics = output_dir / "run" / "divergence_metrics.csv"
    assert scorecard.exists()
    assert metrics.exists()
    with metrics.open(newline="", encoding="utf-8") as handle:
        written = list(csv.DictReader(handle))
    assert written[0]["histogram_key"] == "duo__samples_steps8_temp0.500"
    assert "w1_log" in written[0]
