"""Search API and demo page.

    cd src
    python -m hps.serve --retriever ../models/bi_hx --reranker ../models/rr_hx
    # open http://127.0.0.1:8000  (or GET /search?q=mardon+ke+liye+joote)
"""
import argparse
import html
import time

import numpy as np

from hps.retrieval import DenseIndex, corpus_embeddings, encode, eval_corpus, load_encoder, load_reranker


class Searcher:
    """Two-stage search: bi-encoder + FAISS over the catalogue, then an optional
    cross-encoder reranks the top `rerank_k`."""

    def __init__(self, retriever, reranker=None, rerank_k=50, quantize=False):
        self.products = eval_corpus()
        self.texts = self.products["text"].tolist()
        self.ids = self.products["product_id"].tolist()
        self.model = load_encoder(retriever)
        self.index = DenseIndex(corpus_embeddings(self.model, retriever, self.texts))
        self.cross = load_reranker(reranker, quantize) if reranker else None
        self.rerank_k = rerank_k

    def search(self, query, k=10):
        t0 = time.perf_counter()
        q = encode(self.model, [query])
        t1 = time.perf_counter()
        scores, idx = self.index.search(q, max(k, self.rerank_k if self.cross else k))
        scores, idx = scores[0], idx[0]
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
<form><input name="q" value="{q}" placeholder="e.g. mardon ke liye kale joote" autofocus> <button>Search</button></form>
<p><small>{timing}</small></p><ol>{items}</ol>"""


def make_app(searcher):
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse

    app = FastAPI(title="Hinglish product search")

    @app.get("/search")
    def search(q: str, k: int = 10):
        hits, ms = searcher.search(q, k)
        return {"query": q, "results": hits, "latency": ms}

    @app.get("/", response_class=HTMLResponse)
    def page(q: str = ""):
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
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    import uvicorn
    uvicorn.run(make_app(Searcher(args.retriever, args.reranker, quantize=args.quantize)),
                host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
