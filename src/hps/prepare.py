"""Build the processed dataset from the raw ESCI parquet files.

    python -m hps.prepare            (run from src/)

Outputs in data/processed/:
  products.parquet  product_id, text                       (US catalogue, ~482k)
  qrels.parquet     query_id, product_id, label            (E/S/C/I judgments)
  queries.parquet   query_id, split, query, query_hx, n_sub, n_heldout
                    split is train / val / test; query_hx is the fixed Hinglish
                    version used for evaluation (val and test only)
"""
import random
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from hps.hinglish import Hinglishifier, is_heldout

ROOT = Path(__file__).resolve().parents[2]
RAW, OUT = ROOT / "data" / "raw", ROOT / "data" / "processed"
N_VAL = 1000


def product_text(df):
    """One line per product: title, then brand and colour when present."""
    text = df["product_title"].fillna("")
    for col, name in (("product_brand", "brand"), ("product_color", "color")):
        v = df[col].fillna("").str.strip()
        text = text + v.map(lambda s, name=name: f" | {name}: {s}" if s else "")
    return text.str.slice(0, 400)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ex = pd.read_parquet(RAW / "shopping_queries_dataset_examples.parquet")
    ex = ex[(ex.product_locale == "us") & (ex.small_version == 1)]

    queries = ex.groupby("query_id").agg(query=("query", "first"), split=("split", "first")).reset_index()
    train_ids = queries.loc[queries.split == "train", "query_id"].sample(N_VAL, random_state=0)
    queries.loc[queries.query_id.isin(train_ids), "split"] = "val"

    hx = Hinglishifier(words="all")
    rng = random.Random(0)
    rows = []
    for q, s in zip(queries["query"], queries["split"]):
        if s == "train":
            rows.append((None, 0, 0))
            continue
        h, n, used = hx(q, rng)
        rows.append((h, n, sum(map(is_heldout, used))))
    queries[["query_hx", "n_sub", "n_heldout"]] = pd.DataFrame(rows, index=queries.index)
    queries.to_parquet(OUT / "queries.parquet", index=False)

    qrels = ex[["query_id", "product_id", "esci_label"]].rename(columns={"esci_label": "label"})
    qrels.to_parquet(OUT / "qrels.parquet", index=False)

    cols = ["product_id", "product_title", "product_brand", "product_color"]
    prod = pq.read_table(RAW / "shopping_queries_dataset_products.parquet", columns=cols + ["product_locale"],
                         filters=[("product_locale", "=", "us")]).to_pandas()
    prod = prod[prod.product_id.isin(set(qrels.product_id))].drop_duplicates("product_id")
    pd.DataFrame({"product_id": prod.product_id, "text": product_text(prod)}).to_parquet(
        OUT / "products.parquet", index=False)

    ev = queries[queries.split != "train"]
    print(queries.split.value_counts().to_dict(), f"products={len(prod)} judgments={len(qrels)}")
    print(f"eval queries changed by Hinglish lexicon: {(ev.n_sub > 0).mean():.1%}, "
          f"containing a held-out word: {(ev.n_heldout > 0).mean():.1%}")


if __name__ == "__main__":
    main()
