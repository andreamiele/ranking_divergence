"""Energy distance (DE) and Full-Mahalanobis Typicality p-value (FMTyp-p) per config.

These are the two LM-free / hypothesis-testing metrics from docs/baselines_to_use.pdf.
Both use a fixed handcrafted feature map psi(text) of surface / discourse / entity
statistics, compared against a held-out OpenWebText reference set.

* DE  (lower better):  D^2_E = 2 E||X-Y|| - E||X-X'|| - E||Y-Y'||  on z-scored features.
* FMTyp-p (higher better, ~0.5 = exchangeable with reference): mean over generated x of
  the fraction of reference points at least as atypical (Mahalanobis, Ledoit-Wolf cov).

Named-entity density uses spacy (en_core_web_sm) NER.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
from sklearn.covariance import LedoitWolf

from ranking_divergence.data import OWT_HELDOUT_SPLIT, load_openwebtext_texts

CONNECTIVES = {"however", "therefore", "moreover", "furthermore", "nevertheless", "consequently",
               "meanwhile", "although", "because", "thus", "hence", "whereas", "accordingly",
               "besides", "instead", "otherwise", "indeed", "similarly", "conversely", "subsequently"}
STOPWORDS = {"the", "a", "an", "and", "or", "but", "of", "to", "in", "is", "it", "that", "for",
             "on", "with", "as", "this", "was", "are", "be", "at", "by", "from", "he", "she", "they", "we"}
WORD_RE = re.compile(r"[A-Za-z']+")
SENT_RE = re.compile(r"[.!?]+")
_NLP = None
_NER_MAX_CHARS = 2000  # truncate long docs for NER speed; entity density is distributional


def _nlp():
    global _NLP
    if _NLP is None:
        disable = ["parser", "lemmatizer", "attribute_ruler"]
        try:
            import spacy
            _NLP = spacy.load("en_core_web_sm", disable=disable)
        except OSError:
            import en_core_web_sm
            _NLP = en_core_web_sm.load(disable=disable)
    return _NLP


def _surface(text: str) -> np.ndarray:
    """11-d surface/discourse feature vector (no NER)."""

    text = text or ""
    chars = len(text) or 1
    words = WORD_RE.findall(text)
    n = len(words) or 1
    lower = [w.lower() for w in words]
    lens = np.array([len(w) for w in words] or [0.0])
    sents = [s for s in SENT_RE.split(text) if s.strip()]
    cap = sum(1 for w in words if w[:1].isupper())
    uniq = len(set(lower))
    counts: dict[str, int] = {}
    for w in lower:
        counts[w] = counts.get(w, 0) + 1
    hapax = sum(1 for c in counts.values() if c == 1)
    return np.array([
        uniq / n,                                                  # type-token ratio
        float(lens.mean()),                                        # mean word length
        float(lens.std()),                                         # std word length
        n / max(len(sents), 1),                                    # mean words per sentence
        sum(c in ".,;:!?\"'()-" for c in text) / chars,            # punctuation density
        sum(c.isdigit() for c in text) / chars,                    # digit density
        text.count(",") / n,                                       # comma density
        cap / n,                                                   # capitalized-word ratio
        sum(w in CONNECTIVES for w in lower) / n,                  # discourse-connective freq
        sum(w in STOPWORDS for w in lower) / n,                    # stopword ratio
        hapax / max(uniq, 1),                                      # hapax ratio
    ], dtype=np.float64)


def _ner_density(texts) -> np.ndarray:
    """Named-entity density (entities per word) via spacy NER, batched."""

    truncated = [(t or "")[:_NER_MAX_CHARS] for t in texts]
    out = []
    for doc, original in zip(_nlp().pipe(truncated, batch_size=128), texts):
        n_words = len(WORD_RE.findall(original or "")) or 1
        out.append(len(doc.ents) / n_words)
    return np.asarray(out, dtype=np.float64).reshape(-1, 1)


def feature_matrix(texts) -> np.ndarray:
    surface = np.vstack([_surface(t) for t in texts])
    return np.hstack([surface, _ner_density(texts)])


def energy_distance(gen: np.ndarray, ref: np.ndarray, ref_self: float) -> float:
    from scipy.spatial.distance import cdist
    xy = cdist(gen, ref).mean()
    xx = cdist(gen, gen).mean()
    return float(2 * xy - xx - ref_self)


def fmtyp_p(gen_raw: np.ndarray, ref_m2: np.ndarray, mu: np.ndarray, prec: np.ndarray) -> float:
    d = gen_raw - mu
    gen_m2 = np.einsum("ij,jk,ik->i", d, prec, d)
    # fraction of reference at least as atypical, averaged over generated samples
    return float(np.mean([(ref_m2 >= m).mean() for m in gen_m2]))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--cache-dir", default="/users/staff/dmi-dmi/miele0000/.cache/discrete_diffusion/owt")
    p.add_argument("--num-reference", type=int, default=512)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    out_path = args.run_dir / "de_fmtyp_metrics.csv"
    rows_meta = list(csv.DictReader((args.run_dir / "metrics.csv").open()))

    print(f"Loading {args.num_reference} held-out OWT reference docs...")
    ref_texts = load_openwebtext_texts(split=OWT_HELDOUT_SPLIT, cache_dir=args.cache_dir, limit=args.num_reference)
    ref_raw = feature_matrix(ref_texts)
    # standardized features for DE
    mean, std = ref_raw.mean(0), ref_raw.std(0) + 1e-9
    ref_z = (ref_raw - mean) / std
    from scipy.spatial.distance import cdist
    ref_self = cdist(ref_z, ref_z).mean()
    # Mahalanobis setup for FMTyp-p (raw features)
    lw = LedoitWolf().fit(ref_raw)
    mu, prec = lw.location_, lw.precision_
    dref = ref_raw - mu
    ref_m2 = np.einsum("ij,jk,ik->i", dref, prec, dref)

    done = {}
    if out_path.exists() and not args.force:
        done = {r["key"]: r for r in csv.DictReader(out_path.open())}

    results = list(done.values())
    for i, row in enumerate(rows_meta, 1):
        key = f"{row['method']}__{Path(row['source_file']).stem}"
        if key in done:
            continue
        texts = json.loads(Path(row["source_file"]).read_text())["generated_seqs"]
        gen_raw = feature_matrix(texts)
        gen_z = (gen_raw - mean) / std
        de = energy_distance(gen_z, ref_z, ref_self)
        fp = fmtyp_p(gen_raw, ref_m2, mu, prec)
        results.append({"key": key, "method": row["method"], "nfe": row["nfe"],
                        "temperature_label": row["temperature_label"],
                        "energy_distance": de, "fmtyp_p": fp})
        if i % 25 == 0 or i == len(rows_meta):
            print(f"[{i}/{len(rows_meta)}] {key}: DE={de:.4f} FMTyp-p={fp:.4f}")
        with out_path.open("w", newline="") as h:
            w = csv.DictWriter(h, fieldnames=["key", "method", "nfe", "temperature_label",
                                              "energy_distance", "fmtyp_p"])
            w.writeheader(); w.writerows(results)
    print(f"Wrote {len(results)} configs to {out_path}")


if __name__ == "__main__":
    main()
