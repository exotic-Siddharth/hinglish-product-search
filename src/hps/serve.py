"""Search API and demo page.

    cd src
    python -m hps.serve --retriever ../models/bi_hx --reranker ../models/rr_hx --quantize --hybrid --rewrite
    # open http://127.0.0.1:8000  (or GET /search?q=mardon+ke+liye+joote)
"""
import argparse
import html
import time

import numpy as np

from hps.bm25 import BM25
from hps.fusion import fuse
from hps.retrieval import (
    DenseIndex,
    corpus_embeddings,
    encode,
    eval_corpus,
    load_encoder,
    load_reranker,
)
from hps.rewrite import rewrite as rewrite_query

FUSION_DEPTH = 100  # each retriever's top 100 goes into the fusion, as in hps.evaluate


class Searcher:
    """The search path: optionally rewrite known Hindi words to English, retrieve with the
    bi-encoder + FAISS (fused with stemmed BM25 when hybrid), then optionally rerank the
    top `rerank_k` with a cross-encoder."""

    def __init__(self, retriever, reranker=None, rerank_k=50, quantize=False, device=None, hybrid=False,
                 rewrite=False):
        self.products = eval_corpus()
        self.texts = self.products["text"].tolist()
        self.ids = self.products["product_id"].tolist()
        self.model = load_encoder(retriever, device=device)
        self.index = DenseIndex(corpus_embeddings(self.model, retriever, self.texts))
        self.bm25 = BM25(stem=True).fit(self.texts) if hybrid else None
        self.cross = load_reranker(reranker, quantize, device=device) if reranker else None
        self.rerank_k, self.rewrite = rerank_k, rewrite

    def search(self, query, k=10):
        t0 = time.perf_counter()
        query = rewrite_query(query) if self.rewrite else query
        q = encode(self.model, [query])
        t1 = time.perf_counter()
        depth = max(k, self.rerank_k if self.cross else k)
        if self.bm25 is None:
            scores, idx = self.index.search(q, depth)
        else:
            _, dense = self.index.search(q, FUSION_DEPTH)
            s, lexical = self.bm25.search([query], FUSION_DEPTH)
            scores, idx = fuse([np.where(s > 0, lexical, -1), dense], FUSION_DEPTH)
        scores, idx = scores[0][:depth], idx[0][:depth]
        scores, idx = scores[idx >= 0], idx[idx >= 0]
        t2 = time.perf_counter()
        if self.cross:
            scores = self.cross.predict([(query, self.texts[i]) for i in idx], batch_size=64)
            order = np.argsort(-scores)
            scores, idx = scores[order], idx[order]
        t3 = time.perf_counter()
        hits = [{"product_id": self.ids[i], "text": self.texts[i], "score": float(s)}
                for i, s in zip(idx[:k], scores[:k])]
        ms = {"encode_ms": (t1 - t0) * 1e3, "search_ms": (t2 - t1) * 1e3, "rerank_ms": (t3 - t2) * 1e3}
        return hits, ms


PAGE = """<!doctype html><meta charset="utf-8"><title>Hinglish product search</title>
<style>body{{font:15px system-ui;max-width:760px;margin:40px auto;padding:0 16px}}
input{{width:75%;font-size:16px;padding:8px}}li{{margin:8px 0}}small{{color:#666}}</style>
<h2>Hinglish product search</h2>
<form><input name="q" value="{q}" placeholder="e.g. bachon ke liye pani wali botal" autofocus> <button>Search</button></form>
<p><small>{timing}</small></p><ol>{items}</ol>"""


def make_app(searcher):
    from fastapi import FastAPI, Query
    from fastapi.responses import HTMLResponse

    app = FastAPI(title="Hinglish product search")

    @app.get("/search")
    def search(q: str = Query(min_length=1, max_length=200), k: int = Query(10, ge=1, le=50)):
        hits, ms = searcher.search(q, k)
        return {"query": q, "results": hits, "latency": ms}

    @app.get("/", response_class=HTMLResponse)
    def page(q: str = Query("", max_length=200)):
        if not q:
            return PAGE.format(q="", timing="", items="")
        hits, ms = searcher.search(q, 10)
        items = "".join(f"<li>{html.escape(h['text'])} <small>({h['score']:.3f})</small></li>" for h in hits)
        timing = " · ".join(f"{k} {v:.0f}" for k, v in ms.items())
        return PAGE.format(q=html.escape(q), timing=timing, items=items)

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retriever", default="../models/bi_hx")
    ap.add_argument("--reranker", default=None)
    ap.add_argument("--quantize", action="store_true")
    ap.add_argument("--hybrid", action="store_true", help="fuse the retriever with stemmed BM25")
    ap.add_argument("--rewrite", action="store_true", help="rewrite known Hindi words to English first")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    import uvicorn
    searcher = Searcher(args.retriever, args.reranker, quantize=args.quantize, hybrid=args.hybrid,
                        rewrite=args.rewrite)
    uvicorn.run(make_app(searcher), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
