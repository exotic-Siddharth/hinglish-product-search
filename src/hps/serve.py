"""Search API and demo page.

    cd src
    python -m hps.serve --retriever ../models/bi_hx --reranker ../models/rr_hx --quantize --hybrid --rewrite
    # open http://127.0.0.1:8000  (or GET /search?q=mardon+ke+liye+joote)
"""
import argparse
import html
import time
from urllib.parse import quote_plus

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


EXAMPLES = ["gents ke liye kaala jacket", "bachon ke liye pani wali botal", "ladkiyon ke liye gulabi juta",
            "mardon ke liye joote"]

PAGE = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Hinglish product search</title>
<style>
*{{box-sizing:border-box}}
body{{font:15px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif;color:#1f2328;background:#f6f7f9;margin:0}}
main{{max-width:820px;margin:0 auto;padding:32px 20px 48px}}
h1{{font-size:24px;margin:0 0 4px}} .sub{{color:#59636e;margin:0 0 18px}}
form{{display:flex;gap:8px}}
input{{flex:1;font-size:17px;padding:11px 14px;border:1px solid #c9ced6;border-radius:8px;background:#fff}}
input:focus{{outline:2px solid #1c5cab;outline-offset:-1px;border-color:#1c5cab}}
button{{font-size:16px;padding:0 20px;border:0;border-radius:8px;background:#1c5cab;color:#fff;cursor:pointer}}
.try{{margin:12px 0 0;color:#59636e;font-size:14px}}
.try a{{display:inline-block;margin:4px 6px 0 0;padding:3px 10px;border:1px solid #c9ced6;border-radius:999px;
color:#1f2328;text-decoration:none;background:#fff}} .try a:hover{{border-color:#1c5cab;color:#1c5cab}}
.meta{{margin:22px 0 10px;font-size:14px;color:#59636e}} .meta b{{color:#1f2328}}
ol{{list-style:none;padding:0;margin:0}}
li{{display:flex;gap:12px;background:#fff;border:1px solid #e1e4e8;border-radius:8px;padding:11px 14px;margin:0 0 8px}}
.rank{{color:#59636e;min-width:20px;font-variant-numeric:tabular-nums}}
.title{{font-weight:500}} .tags{{margin-top:3px;font-size:13px;color:#59636e}}
.tags span{{margin-right:14px}}
footer{{margin-top:28px;font-size:13px;color:#59636e}} footer a{{color:#1c5cab}}
</style>
<main>
<h1>Hinglish product search</h1>
<p class="sub">Search {n_products} Amazon products the way you would type it: Hindi words in English letters, mixed with English.</p>
<form><input name="q" value="{q}" placeholder="e.g. bachon ke liye pani wali botal" aria-label="Search"
 spellcheck="false" autocomplete="off" autofocus>
<button>Search</button></form>
<p class="try">Try: {examples}</p>
{results}
<footer>{pipeline} · <a href="https://github.com/exotic-Siddharth/hinglish-product-search">How it works</a></footer>
</main></html>"""


def product_card(rank, text):
    """Product texts are 'title | brand: X | color: Y'; show the title and the attributes as tags."""
    title, *attrs = text.split(" | ")
    tags = "".join(f"<span>{html.escape(a)}</span>" for a in attrs)
    return (f'<li><span class="rank">{rank}</span><div><div class="title">{html.escape(title)}</div>'
            f'<div class="tags">{tags}</div></div></li>')


def make_app(searcher):
    from fastapi import FastAPI, Query
    from fastapi.responses import HTMLResponse

    app = FastAPI(title="Hinglish product search")
    stages = (["rewrite"] if searcher.rewrite else []) + (["BM25 + dense, fused"] if searcher.bm25 else ["dense"])
    stages += ["cross-encoder rerank"] if searcher.cross else []
    encode_label = "rewrite + encode" if searcher.rewrite else "encode"
    pipeline = "Pipeline: " + " → ".join(stages)
    examples = "".join(f'<a href="/?q={quote_plus(e)}">{html.escape(e)}</a>' for e in EXAMPLES)

    def rewritten(q):
        return rewrite_query(q) if searcher.rewrite else q

    @app.get("/search")
    def search(q: str = Query(min_length=1, max_length=200), k: int = Query(10, ge=1, le=50)):
        hits, ms = searcher.search(q, k)
        return {"query": q, "rewritten": rewritten(q), "results": hits, "latency": ms}

    @app.get("/", response_class=HTMLResponse)
    def page(q: str = Query("", max_length=200)):
        results = ""
        if q.strip():
            hits, ms = searcher.search(q, 10)
            as_ = rewritten(q)
            searched = f" · searched as <b>{html.escape(as_)}</b>" if as_ != " ".join(q.lower().split()) else ""
            meta = (f'<p class="meta"><b>{sum(ms.values()):.0f} ms</b> ({encode_label} {ms["encode_ms"]:.0f}, '
                    f'retrieve {ms["search_ms"]:.0f}, rerank {ms["rerank_ms"]:.0f}){searched}</p>')
            results = meta + "<ol>" + "".join(product_card(i, h["text"]) for i, h in enumerate(hits, 1)) + "</ol>"
        return PAGE.format(q=html.escape(q), n_products=f"{len(searcher.texts):,}", examples=examples,
                           results=results, pipeline=pipeline)

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
