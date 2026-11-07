"""Evaluate a retriever (and optional reranker) on English and Hinglish queries.

    python -m hps.evaluate --retriever bm25
    python -m hps.evaluate --retriever bm25-stem --rewrite
    python -m hps.evaluate --retriever ../models/bi_hx --reranker ../models/rr_hx --limit 2000

Query sets, all built from the same ESCI test queries. Each Hinglish set has an
"_en" twin with the English originals of exactly the same queries, so a drop is
always measured like for like:
  en        every test query, in English
  hx        generator output with at least one Hindi word (grammar + lexicon + spelling)
  unseen    only held-out Hindi words swapped in: no grammar, no frames, no trained words
  llm       free-form Hinglish written by an LLM (data/llm_hinglish.csv)
  hand      free-form Hinglish written by a person (data/handwritten_hinglish.csv), once filled in

Metrics use the ESCI gains (E=1, S=0.1, C=0.01, I=0):
  retrieval over the 181,457-product corpus: nDCG@10 (unjudged products count as
      irrelevant), condensed nDCG@10 (unjudged products removed first), judged@10,
      Recall@50 (what the reranker sees), Recall@100, MRR@10
  judged list: order only the products judged for the query (the ESCI Task 1
      candidate lists, US only), scored at nDCG@10 and over the full list. Ties are
      broken at random with a fixed seed: the stored order of judgments is not
      label-neutral, and BM25 ties often (every judged product can score 0).
"""
import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from hps.bm25 import BM25
from hps.metrics import ndcg_at_k, retrieval_metrics
from hps.retrieval import (
    DenseIndex,
    corpus_embeddings,
    encode,
    eval_corpus,
    load_encoder,
    load_reranker,
)
from hps.rewrite import rewrite

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
FREE_FORM = (("llm", "llm_hinglish.csv", "llm_hinglish"), ("hand", "handwritten_hinglish.csv", "your_hinglish"))
HINGLISH_SETS = ("hx", "unseen", "llm", "hand")


def query_sets(split="test", limit=None):
    """{set name: {query_id: query text}} and {query_id: {product_id: label}}."""
    queries = pd.read_parquet(DATA / "queries.parquet")
    queries = queries[queries.split == split]
    if limit:
        queries = queries.sample(limit, random_state=0)
    qrels_df = pd.read_parquet(DATA / "qrels.parquet")
    qrels_df = qrels_df[qrels_df.query_id.isin(set(queries.query_id))]
    qrels = {q: dict(zip(g.product_id, g.label)) for q, g in qrels_df.groupby("query_id")}
    english = dict(zip(queries.query_id, queries["query"]))
    sets = {"en": english}
    for set_name, text_col, count_col in (("hx", "query_hx", "n_sub"), ("unseen", "query_unseen", "n_unseen")):
        chosen = queries[queries[count_col] > 0]
        sets[set_name] = dict(zip(chosen.query_id, chosen[text_col]))
    for set_name, file, col in FREE_FORM:
        path = ROOT / "data" / file
        if path.exists():
            df = pd.read_csv(path, dtype={col: str}).dropna(subset=[col])
            df = df[df.query_id.isin(qrels) & (df[col].str.strip() != "")]
            if len(df):
                sets[set_name] = dict(zip(df.query_id, df[col].str.strip()))
    for set_name in HINGLISH_SETS:
        if set_name in sets:
            sets[f"{set_name}_en"] = {q: english[q] for q in sets[set_name]}
    return sets, qrels


class Retriever:
    def __init__(self, name, products):
        self.name, self.texts = name, products["text"].tolist()
        if name in ("bm25", "bm25-stem"):
            self.bm25 = BM25(stem=name == "bm25-stem").fit(self.texts)
        else:
            self.model = load_encoder(name)
            self.emb = corpus_embeddings(self.model, name, self.texts)
            self.index = DenseIndex(self.emb)

    def search(self, queries, k=100):
        if self.name.startswith("bm25"):
            return self.bm25.search(queries, k)
        return self.index.search(encode(self.model, queries), k)

    def score(self, query, doc_idx):
        """Scores of specific catalogue rows for one query (judged setting)."""
        if self.name.startswith("bm25"):
            return self.bm25.score(query, doc_idx)
        return self.emb[doc_idx] @ encode(self.model, [query])[0]


def rerank(cross, queries, runs, texts, pid_index, k):
    out = {}
    for qid, ranked in runs.items():
        head = ranked[:k]
        s = cross.predict([(queries[qid], texts[pid_index[p]]) for p in head], batch_size=128)
        out[qid] = [head[i] for i in np.argsort(-s)] + ranked[k:]
    return out


def git_commit():
    out = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                         capture_output=True, text=True, check=False).stdout.strip()
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                          check=False)
    return head.stdout.strip() + ("+dirty" if out else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retriever", default="bm25", help="bm25, bm25-stem, or a sentence-transformers model")
    ap.add_argument("--reranker", default=None)
    ap.add_argument("--rerank-k", type=int, default=50)
    ap.add_argument("--quantize", action="store_true", help="int8 reranker")
    ap.add_argument("--rewrite", action="store_true", help="rewrite Hinglish queries to English first")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None, help="evaluate a random subset of queries")
    ap.add_argument("--name", default=None)
    ap.add_argument("--sets", default=None, help="comma-separated query sets to run (default: all)")
    args = ap.parse_args()
    name = args.name or (Path(args.retriever).name
                         + (f"+{Path(args.reranker).name}" if args.reranker else "")
                         + ("_int8" if args.quantize else "") + ("+rewrite" if args.rewrite else ""))

    sets, qrels = query_sets(args.split, args.limit)
    if args.sets:
        sets = {k: v for k, v in sets.items() if k in args.sets.split(",")}
    for set_name in [k for k, v in sets.items() if not v]:
        print(f"skipping {set_name}: no queries in this subset")
        del sets[set_name]
    products = eval_corpus()
    pids = products["product_id"].tolist()
    pid_index = {p: i for i, p in enumerate(pids)}
    t0 = time.time()
    ret = Retriever(args.retriever, products)
    print(f"index ready in {time.time() - t0:.0f}s")
    cross = load_reranker(args.reranker, args.quantize) if args.reranker else None

    results = {}
    per_query = ROOT / "results" / "per_query"
    per_query.mkdir(parents=True, exist_ok=True)
    for set_name, qs in sets.items():
        if args.rewrite and set_name in HINGLISH_SETS:
            qs = {q: rewrite(text) for q, text in qs.items()}
        qids = list(qs)
        _, idx = ret.search([qs[q] for q in qids], k=100)
        runs = {q: [pids[i] for i in row] for q, row in zip(qids, idx)}
        if cross:
            runs = rerank(cross, qs, runs, ret.texts, pid_index, args.rerank_k)

        rng = np.random.default_rng(0)
        rows = []
        for q in qids:
            items = list(qrels[q])
            items = [items[i] for i in rng.permutation(len(items))]  # random tie-breaking
            doc_rows = [pid_index[p] for p in items]
            s = (cross.predict([(qs[q], ret.texts[r]) for r in doc_rows], batch_size=128) if cross
                 else ret.score(qs[q], doc_rows))
            ranked = [items[i] for i in np.argsort(-np.asarray(s), kind="stable")]
            row = {"query_id": q, "judged_ndcg@10": ndcg_at_k(ranked, qrels[q], 10),
                   "judged_ndcg": ndcg_at_k(ranked, qrels[q], len(ranked))}
            row.update(retrieval_metrics(runs[q], qrels[q]) or {})
            rows.append(row)
        scores = pd.DataFrame(rows)
        scores.to_csv(per_query / f"{name}__{set_name}.csv", index=False)

        res = {m: float(scores[m].mean()) for m in scores.columns if m != "query_id"}
        res |= {"n_queries": len(qids), "n_with_exact": int(scores["ndcg@10"].notna().sum()),
                "limit": args.limit, "rewrite": args.rewrite and set_name in HINGLISH_SETS,
                "queries_md5": hashlib.md5(json.dumps(sorted(qs.items())).encode()).hexdigest()}
        results[set_name] = res
        print(f"{set_name:9s} n={len(qids):5d} " + "  ".join(
            f"{m}={res[m]:.4f}" for m in ("ndcg@10", "condensed_ndcg@10", "recall@100", "judged_ndcg@10")))

    # Merge into an existing results file, so a re-run of some sets keeps the others.
    # Every set carries its own n, limit and query hash, which report.py checks.
    path = ROOT / "results" / f"{name}__{args.split}.json"
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        results = json.loads(path.read_text())["results"] | results
    path.write_text(json.dumps({"retriever": args.retriever, "reranker": args.reranker, "split": args.split,
                                "commit": git_commit(), "results": results}, indent=2))


if __name__ == "__main__":
    main()
