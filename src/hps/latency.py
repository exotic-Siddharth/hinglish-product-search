"""Single-query latency of the full search path (one request at a time, CPU).

    python -m hps.latency --retriever ../models/bi_hx --reranker ../models/rr_hx [--quantize]
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retriever", required=True)
    ap.add_argument("--reranker", default=None)
    ap.add_argument("--quantize", action="store_true")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    q = pd.read_parquet(ROOT / "data" / "processed" / "queries.parquet")
    q = q[(q.split == "test") & (q.n_sub > 0)]
    queries = random.Random(0).sample(q.query_hx.tolist(), args.n)
    s = Searcher(args.retriever, args.reranker, quantize=args.quantize)
    for x in queries[:10]:
        s.search(x)  # warm-up
    rows = [s.search(x)[1] for x in queries]
    df = pd.DataFrame(rows)
    df["total_ms"] = df.sum(axis=1)
    stats = {c: {"p50": float(np.percentile(df[c], 50)), "p95": float(np.percentile(df[c], 95))} for c in df}
    for c, v in stats.items():
        print(f"{c:10s} p50 {v['p50']:7.1f}  p95 {v['p95']:7.1f}")
    name = Path(args.retriever).name + (f"+{Path(args.reranker).name}" if args.reranker else "") \
        + ("_int8" if args.quantize else "")
    (ROOT / "results" / f"latency__{name}.json").write_text(json.dumps(
        {"threads": args.threads, "n": args.n, **stats}, indent=2))


if __name__ == "__main__":
    main()
