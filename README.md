# Hinglish Product Search

**Product search that still works when Indian shoppers type the way they speak:**
`mardon ke liye kale joote` instead of `black shoes for men`.

Hundreds of millions of people in India shop online in *Hinglish*: romanised Hindi
mixed with English, with no fixed spelling (`kala` / `kaala`, `pani` / `paani`).
Search engines trained on English queries break on these queries. This project:

1. **Measures the problem** on Amazon's public ESCI shopping dataset (600k judged
   query–product pairs, 30k queries).
2. **Builds a test bed**: a Hinglish query generator with a held-out vocabulary,
   so the evaluation includes Hindi words the model never saw in training.
3. **Fixes it** with a fine-tuned two-tower retriever plus a cross-encoder reranker,
   trained with Hinglish augmentation.
4. **Serves it**: FastAPI search API with a demo page, CPU latency benchmarks, and
   an int8-quantised reranker.

<!-- RESULTS -->

## How it works

```
query ──► bi-encoder (MiniLM-L6, fine-tuned) ──► FAISS top-100 over 181k products
                                                          │
                                         cross-encoder reranks the top 50
                                                          │
                                                      top-10 results
```

**Hinglish generator** ([src/hps/hinglish.py](src/hps/hinglish.py)). It works in three random layers:

| Layer | Example |
|---|---|
| Grammar: Hindi postpositions and word order | `shoes for men` → `men ke liye shoes` / `mardon ke liye ...` (moved to the front) |
| | `bottle without straw` → `bina straw ke bottle` · `under 50` → `50 se kam` · `set of 4` → `4 ka set` · `3 year old` → `3 saal ke` |
| Lexicon: 222 everyday words, several spellings each ([data/hinglish_lexicon.tsv](data/hinglish_lexicon.tsv)) | `black` → `kala`/`kaala`/`kaali` · `water` → `pani`/`paani` |
| Spelling noise: how romanisation varies between people | `aa→a`, `ee→i`, `w↔v`, `chh→ch`, ... |

Brand and technical words stay in English, as people actually type them. Ambiguous
words are left out of the lexicon on purpose: `glasses` could be drinkware or
spectacles, so it is never translated.

**Honest evaluation.** The lexicon is split 80/20 by a hash of the English word.
Training augmentation only uses the 80% part. `Hinglish, unseen words` in the
results table is the subset of test queries that contain held-out Hindi words, so
it measures generalisation rather than memorising a dictionary. On top of that, a
separate set of 150 **free-form** Hinglish queries (written independently, not
produced by the generator) checks that the gains are not just the model learning
the generator's templates.

**Retriever training** ([src/hps/train_biencoder.py](src/hps/train_biencoder.py)). This is a
plain PyTorch loop with in-batch softmax (InfoNCE). Each batch has one row per
query: (query, an *Exact* product, a product the annotators judged *Irrelevant* as
a hard negative). With `p = 0.5` the query is swapped for a fresh random Hinglish
version, so the model sees both forms of every query.

**Reranker** ([src/hps/train_reranker.py](src/hps/train_reranker.py)). This is a
cross-encoder (MiniLM-L6, MS MARCO checkpoint) trained with binary cross-entropy
against the ESCI gain of each label (Exact 1, Substitute 0.1, Complement 0.01,
Irrelevant 0), with the same augmentation.

## Data and metrics

* **ESCI** (Amazon, KDD Cup 2022, Apache-2.0), US locale, "small version".
  19,888 train / 1,000 validation (carved out of train) / 8,956 test queries.
* **Search corpus:** the 181,457 products judged for any validation or test
  query, the same for every model.
* **nDCG@10** uses the official ESCI gains (E = 1, S = 0.1, C = 0.01, I = 0).
  Unjudged products count as irrelevant, so absolute retrieval numbers are
  pessimistic, but they are comparable across models. **Recall@100** = share of
  Exact products in the top 100, which is what the reranker can work with.
* **Judged nDCG@10** is the official ESCI Task 1 setting: re-order only the ~20
  judged products of each query. A random order already scores 0.575 here.

## Run it

```bash
pip install -r requirements.txt

# data (~1.2 GB from github.com/amazon-science/esci-data)
mkdir -p data/raw && cd data/raw
for f in examples products; do
  curl -LO https://github.com/amazon-science/esci-data/raw/main/shopping_queries_dataset/shopping_queries_dataset_$f.parquet
done
cd ../..

bash scripts/run_experiments.sh        # every number in this README (~5 h on 4 CPU cores)
pytest -q                              # unit tests

cd src && python -m hps.serve --retriever ../models/bi_hx --reranker ../models/rr_hx
# → http://127.0.0.1:8000   or   GET /search?q=bachon+ke+liye+pani+ki+botal
```

## Project layout

```
src/hps/
  prepare.py          ESCI → processed parquet + fixed Hinglish eval queries
  hinglish.py         the Hinglish query generator
  bm25.py             BM25 baseline (sparse matrices, no search library)
  metrics.py          nDCG / Recall / MRR with ESCI gains
  retrieval.py        encoders, cached catalogue embeddings, FAISS index
  train_biencoder.py  retriever fine-tuning
  train_reranker.py   cross-encoder fine-tuning
  evaluate.py         all evaluation settings and query sets
  serve.py            FastAPI app + demo page
  latency.py          per-stage p50/p95 latency
  report.py           results tables for this README
data/
  hinglish_lexicon.tsv       222 entries
  llm_hinglish.csv           150 free-form Hinglish test queries (LLM-written)
  handwritten_hinglish.csv   template: the same 150 queries, for a native speaker
tests/test_core.py
```

<!-- LIMITATIONS -->
