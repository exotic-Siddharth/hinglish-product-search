"""Single-query latency of each search path (one request at a time, CPU).

    python -m hps.latency                      # all four paths below
    python -m hps.latency --paths bi_hx,bi_hx+rr_hx_int8

The paths are measured in one process, interleaved query by query with the order
rotating, so background load on a shared machine slows every path alike and the
comparison between them stays fair. Names follow hps.evaluate.
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from hps.serve import Searcher

ROOT = Path(__file__).resolve().parents[2]
RETRIEVER, RERANKER = "../models/bi_hx", "../models/rr_hx"
PATHS = {  # results name: Searcher options
    "bi_hx": {},
    "bi_hx+rr_hx": {"reranker": RERANKER},
    "bi_hx+rr_hx_int8": {"reranker": RERANKER, "quantize": True},
    "hybrid_bi_hx+rr_hx_int8+rewrite": {"reranker": RERANKER, "quantize": True, "hybrid": True, "rewrite": True},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", default=",".join(PATHS), help="comma-separated names from PATHS")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--threads", type=int, default=4)
    args, _ = ap.parse_known_args()
    torch.set_num_threads(args.threads)

    q = pd.read_parquet(ROOT / "data" / "processed" / "queries.parquet")
    q = q[(q.split == "test") & (q.n_sub > 0)]
    queries = random.Random(0).sample(q.query_hx.tolist(), args.n)
    names = args.paths.split(",")
    searchers = {name: Searcher(RETRIEVER, device="cpu", **PATHS[name]) for name in names}
    for s in searchers.values():
        for x in queries[:10]:
            s.search(x)  # warm-up
    rows = {name: [] for name in names}
    for i, x in enumerate(queries):
        for name in names[i % len(names):] + names[:i % len(names)]:
            rows[name].append(searchers[name].search(x)[1])
    for name in names:
        df = pd.DataFrame(rows[name])
        df["total_ms"] = df.sum(axis=1)
        stats = {c: {"p50": float(np.percentile(df[c], 50)), "p95": float(np.percentile(df[c], 95))} for c in df}
        print(name, " ".join(f"{c} p50 {v['p50']:.1f} p95 {v['p95']:.1f}" for c, v in stats.items()))
        (ROOT / "results" / f"latency__{name}.json").write_text(json.dumps(
            {"device": "cpu", "threads": args.threads, "n": args.n, "interleaved_with": names, **stats}, indent=2))


if __name__ == "__main__":
    main()
