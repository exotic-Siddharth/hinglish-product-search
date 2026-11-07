"""Reciprocal rank fusion (Cormack, Clarke and Buettcher, SIGIR 2009) of a lexical and a
dense ranking: a product scores the sum, over the systems that returned it, of
1 / (60 + its rank). Ranks, not scores, so BM25 and cosine scales never need calibrating."""
from collections import defaultdict

import numpy as np
from scipy.stats import rankdata

K = 60  # the paper's constant: it keeps one system's top rank from dominating


def fuse(runs, k):
    """runs: one (n_queries, depth) array of catalogue rows per system, best first, with -1
    for an empty slot. Returns (scores, rows), each (n_queries, k), best first, padded
    with -1. Equal scores keep the order of first appearance: the earlier system's."""
    n = len(runs[0])
    out_s, out_i = np.zeros((n, k), np.float32), np.full((n, k), -1, np.int64)
    for q in range(n):
        acc = defaultdict(float)
        for idx in runs:
            for rank, row in enumerate(idx[q], 1):
                if row >= 0:
                    acc[int(row)] += 1.0 / (K + rank)
        top = sorted(acc.items(), key=lambda kv: -kv[1])[:k]
        out_i[q, :len(top)] = [row for row, _ in top]
        out_s[q, :len(top)] = [s for _, s in top]
    return out_s, out_i


def fuse_scores(score_lists):
    """The same fusion over one fixed candidate list (the products judged for a query).
    Tied scores share their average rank, so a system with ties (BM25 gives every product
    without a query word 0) does not order them."""
    return sum(1.0 / (K + rankdata(-np.asarray(s, dtype=np.float64))) for s in score_lists)
