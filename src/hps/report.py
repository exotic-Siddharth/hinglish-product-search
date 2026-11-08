"""Markdown results tables for the README, from results/*.json and results/per_query/.

    python -m hps.report

Every table reads saved results; nothing is typed in by hand. Warnings go to stderr
when a per-query file is missing or two runs disagree on which queries a set holds.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from hps.evaluate import query_sets
from hps.metrics import ndcg_at_k

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

ROWS = [  # (results name, label)
    ("bm25", "BM25"),
    ("bm25-stem", "BM25 + plural stemming"),
    ("bm25-stem+rewrite", "Dictionary rewrite to English, then BM25 + stemming"),
    ("all-MiniLM-L6-v2", "MiniLM-L6, pre-trained"),
    ("paraphrase-multilingual-MiniLM-L12-v2", "Multilingual MiniLM-L12 (paraphrase), pre-trained"),
    ("multilingual-e5-small", "Multilingual E5-small (retrieval), pre-trained"),
    ("bi_en", "MiniLM-L6 fine-tuned on English ESCI"),
    ("bi_en+rewrite", "Dictionary rewrite, then the English fine-tuned model"),
    ("bi_hx", "MiniLM-L6 fine-tuned + Hinglish augmentation"),
    ("bi_hx+rewrite", "Dictionary rewrite, then the augmented model"),
    ("hybrid_bi_hx", "Stemmed BM25 + augmented model, rank fusion"),
    ("hybrid_bi_en+rewrite", "Rewrite, then stemmed BM25 + English fine-tuned model, fused"),
    ("hybrid_bi_hx+rewrite", "**Rewrite, then stemmed BM25 + augmented model, fused**"),
]
FREE_ROWS = ROWS + [("bi_hx+rr_hx_free", "Augmented model + fine-tuned reranker"),
                    ("hybrid_bi_hx+rr_hx+rewrite_free", "**Rewrite, fused retrieval, fine-tuned reranker**")]
RERANK_ROWS = [
    ("bi_hx_2k", "Augmented retriever"),
    ("bi_hx+ms-marco-MiniLM-L6-v2", "+ pre-trained cross-encoder (MS MARCO)"),
    ("bi_hx+rr_hx", "+ fine-tuned cross-encoder (+ Hinglish augmentation)"),
    ("bi_hx+rr_hx_int8", "+ fine-tuned cross-encoder, int8"),
    ("hybrid_bi_hx+rewrite_2k", "Rewrite, then stemmed BM25 + augmented retriever, fused"),
    ("hybrid_bi_hx+rr_hx+rewrite", "**+ fine-tuned cross-encoder**"),
    ("hybrid_bi_hx+rr_hx_int8+rewrite", "+ fine-tuned cross-encoder, int8 (the served path)"),
]
COMPARISONS = [  # (model A, model B, query set, metric, what A minus B measures)
    ("bi_hx", "bi_en", "hx", "ndcg@10", "Augmentation, on generated Hinglish"),
    ("bi_hx", "bi_en", "unseen", "ndcg@10", "Augmentation, on unseen Hindi words only"),
    ("bi_hx", "bi_en", "llm", "ndcg@10", "Augmentation, on free-form Hinglish"),
    ("bi_hx", "bi_en", "en", "ndcg@10", "Augmentation, cost on English"),
    ("bi_hx", "bi_en+rewrite", "hx", "ndcg@10", "Augmentation vs dictionary rewrite, generated Hinglish"),
    ("bi_hx", "bi_en+rewrite", "llm", "ndcg@10", "Augmentation vs dictionary rewrite, free-form Hinglish"),
    ("bi_hx", "bm25-stem", "en", "ndcg@10", "Fine-tuned retriever vs stemmed BM25, English"),
    ("bi_hx", "bm25-stem", "en", "condensed_ndcg@10", "Same, unjudged products removed"),
    ("bi_hx", "bm25-stem", "hx", "ndcg@10", "Fine-tuned retriever vs stemmed BM25, generated Hinglish"),
    ("bi_hx", "bm25-stem", "llm", "ndcg@10", "Fine-tuned retriever vs stemmed BM25, free-form Hinglish"),
    ("bi_hx", "multilingual-e5-small", "hx", "ndcg@10", "Fine-tuned retriever vs multilingual E5, generated Hinglish"),
    ("bi_hx+rr_hx", "bi_hx_2k", "en", "ndcg@10", "Fine-tuned reranker, English (2,000-query subset)"),
    ("bi_hx+rr_hx", "bi_hx_2k", "hx", "ndcg@10", "Fine-tuned reranker, generated Hinglish (subset)"),
    ("bi_hx+rr_hx", "bi_hx+ms-marco-MiniLM-L6-v2", "hx", "ndcg@10", "Fine-tuning the reranker, generated Hinglish"),
    ("bi_hx+rr_hx", "bi_hx+ms-marco-MiniLM-L6-v2", "en", "judged_ndcg@10", "Fine-tuning the reranker, judged list, English"),
    ("bi_hx+rr_hx", "bi_hx+ms-marco-MiniLM-L6-v2", "hx", "judged_ndcg@10", "Same, generated Hinglish"),
    ("bi_hx+ms-marco-MiniLM-L6-v2", "bi_hx_2k", "en", "judged_ndcg@10",
     "Pre-trained reranker vs no reranker, judged list, English"),
    ("bi_hx+rr_hx_int8", "bi_hx+rr_hx", "hx", "ndcg@10", "int8 vs full-precision reranker, generated Hinglish"),
    ("hybrid_bi_hx", "bi_hx", "en", "ndcg@10", "Rank fusion with stemmed BM25, English"),
    ("hybrid_bi_hx", "bi_hx", "hx", "ndcg@10", "Rank fusion with stemmed BM25, generated Hinglish"),
    ("bi_hx+rewrite", "bi_en+rewrite", "llm", "ndcg@10", "Augmentation when both use the rewrite, free-form"),
    ("hybrid_bi_hx+rewrite", "hybrid_bi_en+rewrite", "en", "ndcg@10", "Augmentation inside the fused pipeline, English"),
    ("hybrid_bi_hx+rewrite", "hybrid_bi_en+rewrite", "hx", "ndcg@10", "Same, generated Hinglish"),
    ("hybrid_bi_hx+rewrite", "hybrid_bi_en+rewrite", "unseen", "ndcg@10", "Same, unseen Hindi words"),
    ("hybrid_bi_hx+rewrite", "hybrid_bi_en+rewrite", "llm", "ndcg@10", "Same, free-form Hinglish"),
    ("hybrid_bi_hx+rewrite", "bm25-stem+rewrite", "hx", "ndcg@10", "Fused pipeline vs rewrite + BM25, generated Hinglish"),
    ("hybrid_bi_hx+rewrite", "bm25-stem+rewrite", "llm", "ndcg@10", "Fused pipeline vs rewrite + BM25, free-form"),
    ("hybrid_bi_hx+rewrite", "bm25-stem", "unseen", "ndcg@10", "Fused pipeline vs stemmed BM25, unseen Hindi words"),
    ("hybrid_bi_hx+rr_hx+rewrite", "hybrid_bi_hx+rewrite_2k", "en", "ndcg@10",
     "Reranker on the fused pipeline, English (subset)"),
    ("hybrid_bi_hx+rr_hx+rewrite", "hybrid_bi_hx+rewrite_2k", "hx", "ndcg@10",
     "Reranker on the fused pipeline, generated Hinglish (subset)"),
    ("hybrid_bi_hx+rr_hx+rewrite", "hybrid_bi_hx+rewrite_2k", "hx", "judged_ndcg@10",
     "Same, judged list"),
    ("hybrid_bi_hx+rr_hx+rewrite_free", "hybrid_bi_hx+rewrite", "llm", "ndcg@10",
     "Reranker on the fused pipeline, free-form"),
    ("hybrid_bi_hx+rr_hx+rewrite_free", "hybrid_bi_hx+rewrite", "llm", "condensed_ndcg@10",
     "Same, unjudged products removed"),
    ("hybrid_bi_hx+rr_hx_int8+rewrite", "hybrid_bi_hx+rr_hx+rewrite", "hx", "ndcg@10",
     "int8 vs full-precision reranker in the fused pipeline, generated Hinglish"),
]


def warn(msg):
    print(f"WARNING: {msg}", file=sys.stderr)


def load(name):
    path = RESULTS / f"{name}__test.json"
    return json.loads(path.read_text())["results"] if path.exists() else None


def check_consistency(names):
    """The same set must hold the same queries in every run it is compared across."""
    seen = {}
    for name in names:
        for set_name, r in (load(name) or {}).items():
            key = (set_name, r.get("limit"))
            if key in seen and seen[key][1] != r.get("queries_md5") and not r.get("rewrite"):
                warn(f"{set_name} (limit {r.get('limit')}) differs between {seen[key][0]} and {name}")
            seen.setdefault(key, (name, r.get("queries_md5")))


def cell(r, set_name, metric):
    return f"{r[set_name][metric]:.3f}" if r and set_name in r and metric in r[set_name] else "–"


def table(rows, cols):
    """cols: (header, set name, metric). Headers get the set's query count; a set no
    row has (the hand-written one, until it is filled in) gets no column."""
    counts = {}
    for name, _ in rows:
        for _, set_name, _ in cols:
            r = load(name)
            if r and set_name in r:
                counts.setdefault(set_name, r[set_name]["n_with_exact"])
    cols = [c for c in cols if c[1] in counts]
    head = " | ".join(f"{h} ({counts[s]:,})" for h, s, _ in cols)
    lines = [f"| Model | {head} |", "|---" * (len(cols) + 1) + "|"]
    for name, label in rows:
        r = load(name)
        if r is not None:
            lines.append(f"| {label} | " + " | ".join(cell(r, s, m) for _, s, m in cols) + " |")
    return "\n".join(lines)


def random_floor(set_name, k=None, draws=20):
    """Expected judged-list nDCG of a random order, by Monte Carlo with a fixed seed."""
    sets, qrels = query_sets()
    rng = np.random.default_rng(0)
    scores = []
    for q in sets[set_name]:
        items = list(qrels[q])
        scores += [ndcg_at_k([items[i] for i in rng.permutation(len(items))], qrels[q], k or len(items))
                   for _ in range(draws)]
    return float(np.mean(scores))


def paired_ci(a, b, query_set, metric, n=2000):
    """Mean of (a - b) over the same queries, with a bootstrap 95% interval."""
    files = [RESULTS / "per_query" / f"{m}__{query_set}.csv" for m in (a, b)]
    missing = [str(f.name) for f in files if not f.exists()]
    if missing:
        warn(f"missing per-query file(s) {missing}; comparison skipped")
        return None
    x, y = (pd.read_csv(f)[["query_id", metric]] for f in files)
    d = x.merge(y, on="query_id").dropna()
    if len(d) != len(x.dropna()) or len(d) != len(y.dropna()):
        warn(f"{a} and {b} cover different {query_set} queries; using the {len(d)} in common")
    diff = (d[f"{metric}_x"] - d[f"{metric}_y"]).to_numpy()
    means = diff[np.random.default_rng(0).integers(0, len(diff), (n, len(diff)))].mean(axis=1)
    return diff.mean(), *np.percentile(means, [2.5, 97.5]), len(diff)


def comparison_table():
    lines = ["| A minus B | Metric | Difference | 95% interval | Queries |", "|---|---|---|---|---|"]
    for a, b, query_set, metric, label in COMPARISONS:
        ci = paired_ci(a, b, query_set, metric)
        if ci:
            lines.append(f"| {label} | {metric} | {ci[0]:+.3f} | {ci[1]:+.3f} to {ci[2]:+.3f} | {ci[3]:,} |")
    return "\n".join(lines)


def latency_table():
    labels = {"bi_hx": "Retriever only", "bi_hx+rr_hx": "Retriever + reranker (top 50)",
              "bi_hx+rr_hx_int8": "Retriever + int8 reranker (top 50)",
              "hybrid_bi_hx+rr_hx_int8+rewrite": "Rewrite + fused retrieval + int8 reranker (served)"}
    lines = ["| Path | Encode p50 | Search p50 | Rerank p50 | Total p50 | Total p95 |", "|---|---|---|---|---|---|"]
    for name, label in labels.items():
        path = RESULTS / f"latency__{name}.json"
        if path.exists():
            d = json.loads(path.read_text())
            lines.append(f"| {label} | {d['encode_ms']['p50']:.0f} ms | {d['search_ms']['p50']:.0f} ms | "
                         f"{d['rerank_ms']['p50']:.0f} ms | {d['total_ms']['p50']:.0f} ms | {d['total_ms']['p95']:.0f} ms |")
    return "\n".join(lines)


def main():
    check_consistency([n for n, _ in FREE_ROWS + RERANK_ROWS])
    print("### Retrieval: nDCG@10 over 181,457 products\n")
    print(table(ROWS, [("English", "en", "ndcg@10"), ("Hinglish", "hx", "ndcg@10"),
                       ("same, in English", "hx_en", "ndcg@10"), ("Unseen words only", "unseen", "ndcg@10"),
                       ("same, in English", "unseen_en", "ndcg@10")]))
    print("\n### Recall: share of Exact products found\n")
    print(table(ROWS, [("English R@100", "en", "recall@100"), ("Hinglish R@100", "hx", "recall@100"),
                       ("Hinglish R@50", "hx", "recall@50"), ("Unseen R@100", "unseen", "recall@100")]))
    print("\n### Free-form Hinglish, not made by the generator\n")
    print(table(FREE_ROWS, [("Free-form Hinglish", "llm", "ndcg@10"), ("same, in English", "llm_en", "ndcg@10"),
                            ("Hand-written Hinglish", "hand", "ndcg@10"), ("same, in English", "hand_en", "ndcg@10")]))
    print("\n### Unjudged products: does the ranking of models depend on them?\n")
    print(table(ROWS, [("English nDCG@10", "en", "ndcg@10"), ("condensed", "en", "condensed_ndcg@10"),
                       ("judged@10", "en", "judged@10"), ("Hinglish nDCG@10", "hx", "ndcg@10"),
                       ("condensed", "hx", "condensed_ndcg@10"), ("judged@10", "hx", "judged@10")]))
    print("\n### Judged-list setting (ESCI Task 1 candidate lists, US only)\n")
    print(table(ROWS, [("English @10", "en", "judged_ndcg@10"), ("Hinglish @10", "hx", "judged_ndcg@10"),
                       ("English, full list", "en", "judged_ndcg"), ("Hinglish, full list", "hx", "judged_ndcg")]))
    print(f"| Random order | {random_floor('en', 10):.3f} | {random_floor('hx', 10):.3f} | "
          f"{random_floor('en'):.3f} | {random_floor('hx'):.3f} |")
    print("\n### Reranking (2,000-query test subset)\n")
    print(table(RERANK_ROWS, [("English", "en", "ndcg@10"), ("Hinglish", "hx", "ndcg@10"),
                              ("same, in English", "hx_en", "ndcg@10"), ("Unseen words only", "unseen", "ndcg@10"),
                              ("Judged list, Hinglish", "hx", "judged_ndcg@10")]))
    print("\n### Are the differences real? Paired bootstrap over queries\n")
    print(comparison_table())
    print("\n### Latency, one query at a time on a 4-core laptop CPU (300 Hinglish queries)\n")
    print(latency_table())


if __name__ == "__main__":
    main()
