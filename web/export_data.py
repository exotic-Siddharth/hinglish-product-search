"""Export what the in-browser demo needs, plus the Python results for web/parity.mjs.

    python web/export_data.py --out web/build        (after python -m hps.prepare)

The page searches a subset of the evaluation catalogue: every product judged for the example
queries, plus a random sample of the other judged products, so most of what it searches was
judged for other queries, as in the full evaluation. Writes products.json, examples.json,
rewrite.json and meta.json (served with the page) and reference.json (parity check only).
"""
import argparse
import json
import random
from pathlib import Path

import pandas as pd

from hps.bm25 import BM25
from hps.metrics import ndcg_at_k
from hps.rewrite import DROP, phrase_table, rewrite

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="web/build")
    ap.add_argument("--n-generated", type=int, default=400, help="generated Hinglish test queries to include")
    ap.add_argument("--n-products", type=int, default=40000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    queries = pd.read_parquet(DATA / "queries.parquet")
    qrels_df = pd.read_parquet(DATA / "qrels.parquet")
    products = pd.read_parquet(DATA / "products.parquet")
    qrels = {q: dict(zip(g.product_id, g.label)) for q, g in qrels_df.groupby("query_id")}

    # Same sets as hps.evaluate.query_sets("test"); a query needs an Exact product to be scored.
    test = queries[queries.split == "test"]
    english = dict(zip(test.query_id, test["query"]))
    has_exact = {q for q, r in qrels.items() if "E" in r.values()}
    generated = [(q, h) for q, h, n in zip(test.query_id, test.query_hx, test.n_sub) if n > 0 and q in has_exact]
    free = pd.read_csv(ROOT / "data" / "llm_hinglish.csv", dtype={"llm_hinglish": str}).dropna(subset=["llm_hinglish"])
    free = [(q, h.strip()) for q, h in zip(free.query_id, free.llm_hinglish) if q in english and q in has_exact and h.strip()]
    chosen = [("generated", q, h) for q, h in rng.sample(generated, min(args.n_generated, len(generated)))]
    chosen += [("free-form", q, h) for q, h in free]

    # Catalogue: every product judged for a chosen query, then other evaluation products at random.
    eval_ids = set(qrels_df.product_id[qrels_df.query_id.isin(set(queries.query_id[queries.split != "train"]))])
    must = {p for _, q, _ in chosen for p in qrels[q]}
    rest = sorted(eval_ids - must)
    ids = sorted(must | set(rng.sample(rest, max(0, min(len(rest), args.n_products - len(must))))))
    text_of = dict(zip(products.product_id, products.text))
    texts = [text_of[p] for p in ids]
    index = {p: i for i, p in enumerate(ids)}

    bm25 = BM25(stem=True).fit(texts)

    def top10(text):
        s, i = bm25.search([text], k=10)
        return [[int(j), round(float(x), 5)] for x, j in zip(s[0], i[0]) if x > 0]  # 0 = no word in common

    examples, reference = [], []
    for set_name, q, h in chosen:
        rel = qrels[q]
        r = rewrite(h)
        runs = {"hinglish": top10(h), "rewritten": top10(r), "english": top10(english[q])}
        ndcg = {k: round(ndcg_at_k([ids[j] for j, _ in v], rel, 10), 5) for k, v in runs.items()}
        examples.append({"set": set_name, "hinglish": h, "english": english[q],
                         "judged": sorted([index[p], lab] for p, lab in rel.items())})
        reference.append({"hinglish": h, "rewritten": r, "runs": runs, "ndcg": ndcg})

    def mean(set_name, key):
        vals = [x["ndcg"][key] for e, x in zip(examples, reference) if e["set"] == set_name]
        return round(sum(vals) / len(vals), 4)

    meta = {"products": len(texts), "evaluation_products": len(eval_ids), "examples": len(examples),
            "ndcg": {s: {k: mean(s, k) for k in ("hinglish", "rewritten", "english")} for s in ("generated", "free-form")},
            "counts": {s: sum(e["set"] == s for e in examples) for s in ("generated", "free-form")}}

    def dump(name, obj):
        (out / name).write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    dump("products.json", texts)
    dump("examples.json", examples)
    dump("rewrite.json", {"phrases": sorted([" ".join(k), v] for k, v in phrase_table().items()), "drop": sorted(DROP)})
    dump("meta.json", meta)
    dump("reference.json", reference)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
