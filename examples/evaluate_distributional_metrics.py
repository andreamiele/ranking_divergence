"""Compute the distributional gold-standard metrics from docs/baselines_to_use.pdf
(MAUVE and Gradient Moment) per config on a scored sweep run.

For each generation config in ``<run-dir>/metrics.csv`` this loads the generated
texts (from the ``source_file`` column), and against a fixed held-out OpenWebText
reference computes:

* MAUVE (Pillutla et al., 2021) -- area under the divergence curve in gpt2-large
  terminal-token feature space (higher is better, in [0, 1]);
* Gradient Moment (Hoogeboom et al., 2026) -- squared L2 distance between the mean
  per-token NLL gradients of gpt2-large over generated vs reference text (lower is
  better). This is the Fisher-score MMD with the gpt2-large score feature map.

The reference features and reference mean gradient are computed once and reused.
Writes ``<run-dir>/distributional_metrics.csv`` keyed by ``method__<source stem>``
to match the histogram keys used by examples/explore_divergences.py.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ranking_divergence.data import OWT_HELDOUT_SPLIT, load_openwebtext_texts


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--scorer-model", default="gpt2-large")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-reference", type=int, default=256)
    parser.add_argument("--cache-dir", default=None, help="OWT cache dir; default reads run metadata.")
    parser.add_argument("--reference-split", default=OWT_HELDOUT_SPLIT)
    parser.add_argument("--mauve-max-tokens", type=int, default=1024)
    parser.add_argument("--mauve-batch-size", type=int, default=16)
    parser.add_argument("--gm-limit-samples", type=int, default=64, help="Gen samples per config for GM (cost control).")
    parser.add_argument("--gm-max-tokens", type=int, default=512)
    parser.add_argument("--methods", default=None, help="Comma-separated method filter, e.g. 'mdlm'.")
    parser.add_argument("--limit-configs", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


@torch.inference_mode()
def featurize(texts: Sequence[str], model, tokenizer, device: str, max_tokens: int, batch_size: int) -> np.ndarray:
    """MAUVE features: mean-pooled top-layer gpt2-large hidden states (per docs/baselines_to_use.pdf).

    Mean-pooling (vs the terminal token) avoids a length/cut-position artifact between
    fixed-length generated text and variable-length reference documents.
    """

    feats = []
    for start in range(0, len(texts), batch_size):
        batch = [t if t else tokenizer.eos_token for t in texts[start : start + batch_size]]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_tokens)
        input_ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        hidden = model(input_ids=input_ids, attention_mask=mask, output_hidden_states=True).hidden_states[-1]
        weights = mask.unsqueeze(-1).float()
        pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)
        feats.append(pooled.float().cpu().numpy())
    return np.concatenate(feats, axis=0)


def mean_nll_gradient(texts: Sequence[str], model, tokenizer, device: str, max_tokens: int) -> torch.Tensor:
    """Mean over texts of the gpt2-large per-token NLL gradient (flattened param vector)."""

    params = [p for p in model.parameters() if p.requires_grad]
    accum = torch.zeros(sum(p.numel() for p in params), dtype=torch.float64, device=device)
    used = 0
    for text in texts:
        enc = tokenizer(text or tokenizer.eos_token, return_tensors="pt", truncation=True, max_length=max_tokens)
        input_ids = enc["input_ids"].to(device)
        if input_ids.shape[1] < 2:
            continue
        model.zero_grad(set_to_none=True)
        out = model(input_ids=input_ids, labels=input_ids)
        out.loss.backward()  # loss is mean per-token NLL
        with torch.no_grad():
            flat = torch.cat([p.grad.reshape(-1).double() if p.grad is not None
                              else torch.zeros(p.numel(), dtype=torch.float64, device=device) for p in params])
            accum += flat
        used += 1
    model.zero_grad(set_to_none=True)
    return accum / max(used, 1)


def main(argv: Sequence[str] | None = None) -> None:
    import mauve

    args = parse_args(argv)
    out_path = args.run_dir / "distributional_metrics.csv"
    metadata = json.loads((args.run_dir / "metadata.json").read_text())
    cache_dir = args.cache_dir or metadata.get("cache_dir")

    rows = list(csv.DictReader((args.run_dir / "metrics.csv").open()))
    if args.methods:
        keep = {m.strip() for m in args.methods.split(",")}
        rows = [r for r in rows if r["method"] in keep]
    if args.limit_configs:
        rows = rows[: args.limit_configs]

    done: dict[str, dict] = {}
    if out_path.exists() and not args.force:
        done = {r["key"]: r for r in csv.DictReader(out_path.open())}
        print(f"Resuming: {len(done)} configs already computed.")

    print(f"Loading {args.scorer_model} on {args.device} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.scorer_model)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.scorer_model).to(args.device).eval()

    print(f"Loading {args.num_reference} held-out OWT reference docs ...")
    ref_texts = load_openwebtext_texts(split=args.reference_split, cache_dir=cache_dir, limit=args.num_reference)
    ref_feats = featurize(ref_texts, model, tokenizer, args.device, args.mauve_max_tokens, args.mauve_batch_size)
    print("Computing reference mean gradient (GM) ...")
    ref_grad = mean_nll_gradient(ref_texts[: args.gm_limit_samples], model, tokenizer, args.device, args.gm_max_tokens)

    results = list(done.values())
    for index, row in enumerate(rows, start=1):
        key = f"{row['method']}__{Path(row['source_file']).stem}"
        if key in done:
            continue
        texts = json.loads(Path(row["source_file"]).read_text())["generated_seqs"]
        gen_feats = featurize(texts, model, tokenizer, args.device, args.mauve_max_tokens, args.mauve_batch_size)
        mauve_out = mauve.compute_mauve(
            p_features=gen_feats, q_features=ref_feats, device_id=0, verbose=False, batch_size=args.mauve_batch_size
        )
        gen_grad = mean_nll_gradient(texts[: args.gm_limit_samples], model, tokenizer, args.device, args.gm_max_tokens)
        gm = float(((gen_grad - ref_grad) ** 2).sum().item())
        results.append({
            "key": key, "method": row["method"], "nfe": row["nfe"],
            "temperature_label": row["temperature_label"], "mauve": float(mauve_out.mauve), "gm": gm,
        })
        print(f"[{index}/{len(rows)}] {key}: MAUVE={mauve_out.mauve:.4f}  GM={gm:.4g}")
        # Checkpoint each config so a crash/timeout keeps progress.
        with out_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["key", "method", "nfe", "temperature_label", "mauve", "gm"])
            writer.writeheader()
            writer.writerows(results)
    print(f"Wrote {len(results)} configs to {out_path}")


if __name__ == "__main__":
    main()
