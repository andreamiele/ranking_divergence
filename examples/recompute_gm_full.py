"""Recompute Gradient Moment (GM) without subsampling and update distributional_metrics.csv.

GM = ||E_gen[grad NLL] - E_ref[grad NLL]||^2 (Fisher-score MMD). This uses ALL generated
samples per config and the full sequence length (no sample/token subsampling), and writes
only the ``gm`` column back, leaving the existing MAUVE values untouched.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ranking_divergence.data import OWT_HELDOUT_SPLIT, load_openwebtext_texts

def mean_nll_gradient_batched(texts, model, tokenizer, device, max_tokens, batch_size=8):
    """Gradient of the corpus-mean next-token NLL (token-weighted), batched.

    Since grad(mean) = mean(grads), one backward per batch (weighted by its valid-token
    count) yields the exact same mean gradient as a per-sample loop, far faster.
    """

    params = [p for p in model.parameters() if p.requires_grad]
    accum = torch.zeros(sum(p.numel() for p in params), dtype=torch.float64, device=device)
    total_tokens = 0
    for start in range(0, len(texts), batch_size):
        batch = [t or tokenizer.eos_token for t in texts[start:start + batch_size]]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_tokens)
        input_ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        if input_ids.shape[1] < 2:
            continue
        labels = input_ids.clone()
        labels[mask == 0] = -100
        n_tok = int((labels[:, 1:] != -100).sum().item())
        if n_tok == 0:
            continue
        model.zero_grad(set_to_none=True)
        model(input_ids=input_ids, attention_mask=mask, labels=labels).loss.backward()
        with torch.no_grad():
            flat = torch.cat([(p.grad.reshape(-1).double() if p.grad is not None
                               else torch.zeros(p.numel(), dtype=torch.float64, device=device)) for p in params])
            accum += flat * n_tok  # un-average so batches sum to the corpus total
        total_tokens += n_tok
    model.zero_grad(set_to_none=True)
    return accum / max(total_tokens, 1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--scorer-model", default="gpt2-large")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--num-reference", type=int, default=256)
    p.add_argument("--max-tokens", type=int, default=1024)  # full sequence
    p.add_argument("--cache-dir", default="/users/staff/dmi-dmi/miele0000/.cache/discrete_diffusion/owt")
    args = p.parse_args()

    dist_path = args.run_dir / "distributional_metrics.csv"
    existing = {r["key"]: r for r in csv.DictReader(dist_path.open())} if dist_path.exists() else {}
    metric_rows = list(csv.DictReader((args.run_dir / "metrics.csv").open()))

    tokenizer = AutoTokenizer.from_pretrained(args.scorer_model)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.scorer_model).to(args.device).eval()

    print(f"Reference gradient over {args.num_reference} docs (full, max_tokens={args.max_tokens})...")
    ref_texts = load_openwebtext_texts(split=OWT_HELDOUT_SPLIT, cache_dir=args.cache_dir, limit=args.num_reference)
    ref_grad = mean_nll_gradient_batched(ref_texts, model, tokenizer, args.device, args.max_tokens)

    rows = []
    for i, row in enumerate(metric_rows, 1):
        key = f"{row['method']}__{Path(row['source_file']).stem}"
        texts = json.loads(Path(row["source_file"]).read_text())["generated_seqs"]
        gen_grad = mean_nll_gradient_batched(texts, model, tokenizer, args.device, args.max_tokens)  # ALL samples
        gm = float(((gen_grad - ref_grad) ** 2).sum().item())
        out = dict(existing.get(key, {"key": key, "method": row["method"], "nfe": row["nfe"],
                                      "temperature_label": row["temperature_label"], "mauve": ""}))
        out["gm"] = gm
        rows.append(out)
        print(f"[{i}/{len(metric_rows)}] {key}: GM_full={gm:.4f}  (was {existing.get(key, {}).get('gm', '?')})")
        fields = ["key", "method", "nfe", "temperature_label", "mauve", "gm"]
        with dist_path.open("w", newline="") as h:
            w = csv.DictWriter(h, fieldnames=fields, extrasaction="ignore")
            w.writeheader(); w.writerows(rows)
    print(f"Updated GM (full) for {len(rows)} configs in {dist_path}")


if __name__ == "__main__":
    main()
