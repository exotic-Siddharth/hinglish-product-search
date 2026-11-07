"""Build the processed dataset from the raw ESCI parquet files.

    python -m hps.prepare            (run from src/)

Outputs in data/processed/:
  products.parquet          product_id, text                (judged US products, ~482k)
  qrels.parquet             query_id, product_id, label     (E/S/C/I judgments)
  protected_phrases.txt     catalogue brand phrases the generator never touches
  queries.parquet           query_id, split, query, and for val/test queries:
      query_hx, n_sub       full generator (grammar + lexicon + spelling); n_sub
                            counts swaps whose inserted word is Hindi, not English
      n_heldout             how many of those swaps use a held-out word family
      query_unseen, n_unseen  only held-out words swapped: no grammar, no frames,
                            no trained words, so it isolates unseen vocabulary
and data/heldout_words.txt, the held-out English entries, for anyone auditing the split.
"""
import random
import re
from collections import Counter
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from hps.hinglish import (
    PARTICLES,
    PROTECTED_PATH,
    STOP,
    Hinglishifier,
    heldout_words,
    is_heldout,
    load_lexicon,
    phrase_key,
)

ROOT = Path(__file__).resolve().parents[2]
RAW, OUT = ROOT / "data" / "raw", ROOT / "data" / "processed"
N_VAL = 1000
MIN_BRAND_PRODUCTS = 10
ENGLISH_MIN_COUNT = 5


def product_text(df):
    """One line per product: title, then brand and colour when present."""
    text = df["product_title"].fillna("")
    for col, name in (("product_brand", "brand"), ("product_color", "color")):
        v = df[col].fillna("").str.strip()
        text = text + v.map(lambda s, name=name: f" | {name}: {s}" if s else "")
    return text.str.slice(0, 400)


def protected_phrases(brands, lexicon):
    """Multi-word brands with at least MIN_BRAND_PRODUCTS products that contain a word the
    generator would otherwise change (under armour, black and decker), each also without a
    leading "the" (north face), plus the hand-listed compounds in data/protected_compounds.txt."""
    counts = Counter(phrase_key(b) for b in brands.dropna())
    triggers = set(lexicon) | STOP
    phrases = {p for p, n in counts.items() if n >= MIN_BRAND_PRODUCTS and 2 <= len(p) <= 4 and set(p) & triggers}
    phrases |= {p[1:] for p in phrases if p[0] == "the" and len(p) > 2}
    compounds = (ROOT / "data" / "protected_compounds.txt").read_text(encoding="utf-8").splitlines()
    phrases |= {phrase_key(c) for c in compounds if c.strip() and not c.startswith("#")}
    return sorted(phrases)


def hindi_swaps(used, english):
    """Swaps that put a Hindi word in the query: ladies, gents or frock leave it English."""
    return [(k, v) for k, v in used if any(t not in english for t in v.split() if t not in PARTICLES)]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ex = pd.read_parquet(RAW / "shopping_queries_dataset_examples.parquet")
    ex = ex[(ex.product_locale == "us") & (ex.small_version == 1)]
    cols = ["product_id", "product_title", "product_brand", "product_color"]
    catalogue = pq.read_table(RAW / "shopping_queries_dataset_products.parquet", columns=cols + ["product_locale"],
                              filters=[("product_locale", "=", "us")]).to_pandas()

    lexicon = load_lexicon()
    phrases = protected_phrases(catalogue.product_brand, lexicon)
    PROTECTED_PATH.write_text("\n".join(" ".join(p) for p in phrases) + "\n", encoding="utf-8")
    (ROOT / "data" / "heldout_words.txt").write_text("\n".join(sorted(heldout_words())) + "\n", encoding="utf-8")

    queries = ex.groupby("query_id").agg(query=("query", "first"), split=("split", "first")).reset_index()
    train_ids = queries.loc[queries.split == "train", "query_id"].sample(N_VAL, random_state=0)
    queries.loc[queries.query_id.isin(train_ids), "split"] = "val"
    english = {w for w, n in Counter(w for q in queries["query"][queries.split == "train"]
                                     for w in re.findall(r"[a-z0-9]+", q.lower())).items()
               if n >= ENGLISH_MIN_COUNT}

    full = Hinglishifier(words="all")
    unseen = Hinglishifier(words="heldout", p_rule=0.0, p_frame=0.0)
    rng_full, rng_unseen = random.Random(0), random.Random(1)
    rows = []
    for q, s in zip(queries["query"], queries["split"]):
        if s == "train":
            rows.append((None, 0, 0, None, 0))
            continue
        h, _, used = full(q, rng_full)
        swaps = hindi_swaps(used, english)
        u, _, used_u = unseen(q, rng_unseen)
        rows.append((h, len(swaps), sum(is_heldout(k) for k, _ in swaps), u, len(hindi_swaps(used_u, english))))
    queries[["query_hx", "n_sub", "n_heldout", "query_unseen", "n_unseen"]] = pd.DataFrame(rows, index=queries.index)
    queries.to_parquet(OUT / "queries.parquet", index=False)

    qrels = ex[["query_id", "product_id", "esci_label"]].rename(columns={"esci_label": "label"})
    qrels.to_parquet(OUT / "qrels.parquet", index=False)

    prod = catalogue[catalogue.product_id.isin(set(qrels.product_id))].drop_duplicates("product_id")
    pd.DataFrame({"product_id": prod.product_id, "text": product_text(prod)}).to_parquet(
        OUT / "products.parquet", index=False)

    ev = queries[queries.split == "test"]
    print(queries.split.value_counts().to_dict(), f"products={len(prod)} judgments={len(qrels)}")
    print(f"protected brand phrases: {len(phrases)}; held-out entries: {len(heldout_words())} of {len(lexicon)}")
    print(f"test queries with a Hindi swap: {(ev.n_sub > 0).sum()} ({(ev.n_sub > 0).mean():.1%}), "
          f"with a held-out swap: {(ev.n_heldout > 0).sum()}, unseen-only set: {(ev.n_unseen > 0).sum()}")


if __name__ == "__main__":
    main()
