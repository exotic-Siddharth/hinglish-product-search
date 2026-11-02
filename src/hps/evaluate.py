"""Evaluate a retriever (and optional reranker) on English and Hinglish queries.

    python -m hps.evaluate --retriever bm25
    python -m hps.evaluate --retriever sentence-transformers/all-MiniLM-L6-v2
    python -m hps.evaluate --retriever ../models/biencoder --reranker ../models/reranker

Query sets (all from the same held-out ESCI queries):
  en           original English query
  hx           Hinglish version, only queries the generator actually changed
  hx_heldout   subset of hx that contains Hindi words never used in training

Two settings:
  retrieval    search the whole 482k-product catalogue (nDCG@10, Recall@100, MRR@10)
  judged       order only the products judged for that query - the official
               ESCI Task 1 setting (nDCG@10 with E/S/C/I gains)
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from hps.bm25 import BM25
from hps.metrics import ndcg_at_k, summarise
from hps.retrieval import DenseIndex, corpus_embeddings, encode, eval_corpus, load_encoder, load_reranker

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"


def load(split, limit=None):
    queries = pd.read_parquet(DATA / "queries.parquet")
    queries = queries[queries.split == split]
    if limit:
        queries = queries.sample(limit, random_state=0)
    qrels_df = pd.read_parquet(DATA / "qrels.parquet")
    qrels_df = qrels_df[qrels_df.query_id.isin(set(queries.query_id))]
    qrels = {q: dict(zip(g.product_id, g.label)) for q, g in qrels_df.groupby("query_id")}
    products = eval_corpus()
    sets = {
        "en": dict(zip(queries.query_id, queries["query"])),
        "hx": dict(zip(queries.query_id[queries.n_sub > 0], queries.query_hx[queries.n_sub > 0])),
        "hx_heldout": dict(zip(queries.query_id[queries.n_heldout > 0], queries.query_hx[queries.n_heldout > 0])),
    }
    # Real Hinglish written by a person (data/handwritten_hinglish.csv), if filled in.
    hand_path = ROOT / "data" / "handwritten_hinglish.csv"
    if hand_path.exists():
        hand = pd.read_csv(hand_path, dtype={"your_hinglish": str}).dropna(subset=["your_hinglish"])
        hand = hand[hand.query_id.isin(qrels) & (hand.your_hinglish.str.strip() != "")]
        if len(hand):
            sets["hand"] = dict(zip(hand.query_id, hand.your_hinglish.str.strip()))
            sets["hand_en"] = dict(zip(hand.query_id, hand.english_query))
    return sets, qrels, products


class Retriever:
    def __init__(self, name, products):
        self.name, self.texts = name, products["text"].tolist()
        if name == "bm25":
            self.bm25 = BM25().fit(self.texts)
        else:
            self.model = load_encoder(name)
            self.emb = corpus_embeddings(self.model, name, self.texts)
            self.index = DenseIndex(self.emb)

    def search(self, queries, k=100):
        if self.name == "bm25":
            return self.bm25.search(queries, k)
        return self.index.search(encode(self.model, queries), k)

    def score(self, query, doc_idx):
        """Scores of specific catalogue rows for one query (judged setting)."""
        if self.name == "bm25":
            return self.bm25.score(query, doc_idx)
        return self.emb[doc_idx] @ encode(self.model, [query])[0]


def rerank(cross, queries, runs, texts, pid_index, k):
    out = {}
    for qid, ranked in runs.items():
        head = ranked[:k]
        s = cross.predict([(queries[qid], texts[pid_index[p]]) for p in head], batch_size=128)
        out[qid] = [head[i] for i in np.argsort(-s)] + ranked[k:]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retriever", default="bm25")
    ap.add_argument("--reranker", default=None)
    ap.add_argument("--rerank-k", type=int, default=50)
    ap.add_argument("--quantize", action="store_true", help="int8 reranker")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None, help="evaluate a random subset of queries")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    name = args.name or (Path(args.retriever).name
                         + (f"+{Path(args.reranker).name}" if args.reranker else "")
                         + ("_int8" if args.quantize else ""))

    sets, qrels, products = load(args.split, args.limit)
    pids = products["product_id"].tolist()
    pid_index = {p: i for i, p in enumerate(pids)}
    t0 = time.time()
    ret = Retriever(args.retriever, products)
    print(f"index ready in {time.time() - t0:.0f}s")
    cross = load_reranker(args.reranker, args.quantize) if args.reranker else None

    results = {}
    for set_name, qs in sets.items():
        qids = list(qs)
        _, idx = ret.search([qs[q] for q in qids], k=100)
        runs = {q: [pids[i] for i in row] for q, row in zip(qids, idx)}
        if cross:
            runs = rerank(cross, qs, runs, ret.texts, pid_index, args.rerank_k)
        res = summarise(runs, qrels)

        judged = []
        for q in qids:
            items = list(qrels[q])
            rows = [pid_index[p] for p in items]
            s = (cross.predict([(qs[q], ret.texts[r]) for r in rows], batch_size=128) if cross
                 else ret.score(qs[q], rows))
            judged.append(ndcg_at_k([items[i] for i in np.argsort(-s)], qrels[q], 10))
        res["judged_ndcg@10"] = float(np.mean(judged))
        results[set_name] = res
        print(f"{set_name:11s} " + "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                            for k, v in res.items()))

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / f"{name}__{args.split}.json").write_text(json.dumps(
        {"retriever": args.retriever, "reranker": args.reranker, "split": args.split,
         "limit": args.limit, "results": results}, indent=2))


if __name__ == "__main__":
    main()
