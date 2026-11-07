# Hinglish Product Search

Product search for queries typed in **Hinglish**, romanised Hindi mixed with English
(`stocking stuffers bado ke liye 10 se kam` for *stocking stuffers for adults under 10*),
built and evaluated on Amazon's public ESCI shopping dataset.

Code-mixed queries are common in Indian e-commerce: Flipkart reports that around 21% of
the queries on its platform are code-mixed Hinglish ([Kulkarni et al., 2022](https://arxiv.org/abs/2208.03713)).
Spelling is not standardised (`safed` / `sufed`, `botal` / `bottal`) and Hindi word order
differs from English (`ladkon ke liye ghadi` is literally *boys for watch*). A retriever
trained on English queries has never seen either. This project measures how much that
costs, and what fixes it, without any real Hinglish training data.

**What is in the repo**

1. **A Hinglish query generator** with grammar rules, a 185-word lexicon with spelling
   variants, and catalogue brand protection. 25 Hindi word families are held out of
   training, so the test set can include Hindi words the model never saw.
2. **An evaluation that compares like with like**: every Hinglish query set is paired with
   the same queries in English, a separate set changes *only* unseen Hindi words, and
   123 free-form Hinglish queries were written without the generator.
3. **Nine retrieval systems**, from BM25 to a fine-tuned two-tower model, including a
   "translate back to English" baseline, plus three rerankers, with paired bootstrap
   intervals on every comparison that matters.
4. **A served model**: FastAPI search API and demo page, CPU latency per stage, and an
   int8-quantised reranker.

## Results

> The final evaluation (generator v2, after an audit of the first version) is running;
> the tables and findings land here on 8 November 2026. `results/v1/` holds the
> pre-audit numbers, which used a flawed held-out split and are not comparable.

## How it works

```
query ──► two-tower retriever (MiniLM-L6, fine-tuned) ──► FAISS: top 50 of 181,457 products
                                                                 │
                                           cross-encoder (MiniLM-L6) re-scores the 50
                                                                 │
                                                           top 10 results
```

### The Hinglish generator ([src/hps/hinglish.py](src/hps/hinglish.py))

Each English query goes through three random layers. Real outputs from the test set:

| Layer | English query | Generated Hinglish |
|---|---|---|
| Postpositions follow the noun | `14k gold bracelets for women` | `14k sona bracelets aurton ke liye` |
| `with` / `without` | `20 ft trampoline with enclosure` | `20 ft trampoline enclosure wali` |
| | `100 oz water bottles without words` | `100 oz paani botalen bina words` |
| Numbers and quantities | `stocking stuffers for adults under 10` | `stocking stuffers bado ke liye 10 se kam` |
| | `counter height table legs set of 4` | `counter height mej legs 4 ka set` |
| | `11 year old girl gifts` | `11 saal ke girl tohfe` |
| Negation goes after the excluded thing | `laptop not chromebook` | `laptop chromebook nahin` |
| Brands are never touched | `mens white under armour shirts` | `mardana sufed under armour shirts` |

* **Lexicon** ([data/hinglish_lexicon.tsv](data/hinglish_lexicon.tsv)): 185 shopping
  words, each with its common romanised spellings, plus spelling noise (`aa`→`a`,
  `ee`→`i`, `w`↔`v`, ...). Substitution is word by word, so words whose shopping sense
  differs from the Hindi one are left out on purpose: `ring` (ring light), `sheet`
  (sticker sheet), `pot` (instant pot), `up` (make up), `new` (new york).
* **Protected phrases**: 1,683 phrases are never translated and never trigger a grammar
  rule: multi-word catalogue brands with at least 10 products (`under armour`,
  `hot wheels`, `black and decker`) plus 18 hand-listed compounds (`rose gold`,
  `green screen`).
* **Held-out word families**: entries are grouped into families (shared Hindi words,
  inflections such as `mard` / `mardon`, spelling variants), and 25 entries in
  family-sized groups never appear in training augmentation
  ([data/heldout_words.txt](data/heldout_words.txt)). A unit test generates training
  augmentations over the whole lexicon and asserts that no held-out word can appear.
* **Measured quality**: reading 120 random generated test queries by hand, 11 (9%) no
  longer meant what the annotators judged. Seven were book, film or brand titles the
  brand list does not cover (`big nate` → `bada nate`); four were wrong senses
  (`deep v`, `egg white`, `soft body`, `cord` → `dori`, a string), all four fixed in a
  last round of edits, so titles are the main error left. The relevance labels belong
  to the English query, so these errors are label noise that lowers every model's
  Hinglish score. Two known quirks remain: `under 5 dollars` becomes
  `5 ke andar dollars` (the rule moves only the number), and `garden bed` can become
  `garden palang` (a bed to sleep on).

### Models

* **Retriever** ([src/hps/train_biencoder.py](src/hps/train_biencoder.py)): MiniLM-L6
  two-tower model, fine-tuned for 2 epochs with in-batch softmax (InfoNCE). Each row is a
  query, one of its Exact products, and a hard negative (Irrelevant, else Complement;
  33% of queries have neither). With probability 0.5 a query goes through the generator,
  using training word families only. Most queries contain no lexicon word, so about 19%
  of training rows end up with a Hindi word, and 30% of training queries get one at
  least once over the two epochs. The English-only model is the same run with
  probability 0, which makes the pair a clean ablation of augmentation.
* **Reranker** ([src/hps/train_reranker.py](src/hps/train_reranker.py)): MS MARCO
  MiniLM-L6 cross-encoder, fine-tuned on 40,000 sampled judgment pairs with binary
  cross-entropy against the ESCI gain (Exact 1, Substitute 0.1, Complement 0.01,
  Irrelevant 0), with the same augmentation.
* **Baselines**: BM25, with and without plural stemming; pre-trained MiniLM,
  multilingual MiniLM and multilingual E5; and a **dictionary rewrite** that maps
  Hinglish back to English using the training word families
  ([src/hps/rewrite.py](src/hps/rewrite.py)), then searches with BM25 or the English
  model. The rewrite inverts the generator's own dictionary, so it is favoured on the
  generated sets; the free-form set is its fair test.

## Evaluation design

* **Data**: ESCI ([Reddy et al., 2022](https://arxiv.org/abs/2206.06588)), US locale,
  "small version": 19,888 training queries, 1,000 validation queries carved out of
  training, 8,956 test queries. No test query string appears in training.
* **Search corpus**: the 181,457 products judged for any validation or test query, the
  same for every model. Most products a model retrieves were judged for *other* queries,
  so each result is reported two ways: unjudged products counted as irrelevant
  (nDCG@10), and unjudged products removed first (condensed nDCG@10).

| Query set | What it is | Test queries |
|---|---|---|
| `en` | every test query, in English | 8,956 |
| `hx` | generator output with at least one Hindi word | 3,585 |
| `unseen` | only held-out Hindi words swapped in: no grammar, no frames, no trained words | 419 |
| `llm` | free-form Hinglish written by an LLM, translating only what the query says ([data/README.md](data/README.md)) | 123 |
| `hand` | the same 123 queries written by a person (template, not yet filled in) | 0 |

Each Hinglish set has an `_en` twin with the English originals of exactly the same
queries, so "how much does Hinglish cost" is measured on the same queries.

* **Metrics**: nDCG@10 with the ESCI gains, Recall@50 (what the reranker sees),
  Recall@100, MRR@10, and the judged-list setting of ESCI Task 1 (re-order only the
  products judged for each query), at @10 and over the full list, with seeded random
  tie-breaking.
* **Significance**: paired bootstrap 95% intervals over queries for every comparison in
  the findings. At 123 queries the interval half-width is about 0.04 nDCG.

## Run it

```bash
pip install -r requirements.txt

# data: about 1.2 GB from github.com/amazon-science/esci-data
mkdir -p data/raw && cd data/raw
for f in examples products; do
  curl -LO https://github.com/amazon-science/esci-data/raw/main/shopping_queries_dataset/shopping_queries_dataset_$f.parquet
done
cd ../..

bash scripts/run_experiments.sh   # data, training, every evaluation, latency, results/report.md
pytest -q                         # unit tests, also run by CI on every push

cd src && python -m hps.serve --retriever ../models/bi_hx --reranker ../models/rr_hx
# http://127.0.0.1:8000, or GET /search?q=bachon+ke+liye+pani+ki+botal&k=10
```

Training is resumable: the retriever checkpoints every 50 steps and the reranker every
200, and a checkpoint is only resumed by a run with the same settings, lexicon, brand
list and generator code.

## Project layout

```
src/hps/
  prepare.py          ESCI -> processed parquet, brand phrases, Hinglish test sets
  hinglish.py         the Hinglish generator and the held-out word families
  rewrite.py          dictionary baseline: Hinglish back to English
  bm25.py             BM25, optionally with an S-stemmer (sparse matrices, no search library)
  metrics.py          nDCG, condensed nDCG, Recall, MRR with ESCI gains
  retrieval.py        encoders, weight-keyed embedding cache, FAISS index
  train_biencoder.py  retriever fine-tuning (resumable)
  train_reranker.py   cross-encoder fine-tuning (resumable)
  checkpoint.py       checkpoints with a run fingerprint and RNG state
  evaluate.py         query sets, metrics, per-query scores, provenance
  report.py           every README table, from saved results
  serve.py            FastAPI app and demo page
  latency.py          per-stage p50/p95 latency on the CPU
data/                 lexicon, protected compounds, held-out list, free-form sets (see data/README.md)
tests/test_core.py    metrics, generator rules, leak test, checkpoints, rewrite, stemmer
```

## Limitations

* **The Hinglish is synthetic or LLM-written.** No set here was typed by real shoppers:
  the generated sets carry about 9% meaning changes, and the free-form set was written
  by an LLM. The `hand` template is there for a native speaker to fill in.
* **A US catalogue stands in for an Indian one.** Prices are in dollars and the products
  are American, so this measures robustness to code-mixed queries, not Indian shopping.
* **Unjudged products.** Most retrieved products have no judgment for the query, so
  absolute retrieval numbers are pessimistic; condensed nDCG shows how much a comparison
  depends on that convention.
* **One architecture, one seed, CPU only.** Only MiniLM-L6 is fine-tuned, a single seed
  per model, on a 4-core laptop. The augmentation ablation does not separate grammar,
  lexicon and spelling noise.
* **The served model holds out 25 word families on purpose.** That is what makes the
  unseen-words test possible; a production model would train on every family.

## References

* Chandan K. Reddy, Lluís Màrquez, Fran Valero, Nikhil Rao, Hugo Zaragoza, Sambaran
  Bandyopadhyay, Arnab Biswas, Anlu Xing, Karthik Subbian. *Shopping Queries Dataset: A
  Large-Scale ESCI Benchmark for Improving Product Search.* arXiv:2206.06588, 2022.
* Mandar Kulkarni, Soumya Chennabasavaraj, Nikesh Garera. *Study of Encoder-Decoder
  Architectures for Code-Mix Search Query Translation.* arXiv:2208.03713, 2022.
* Tetsuya Sakai. *Alternatives to Bpref.* SIGIR 2007 (condensed lists for unjudged documents).
* Donna Harman. *How Effective Is Suffixing?* JASIS 42(1), 1991 (the S-stemmer).

## Licence

MIT, see [LICENSE](LICENSE). The ESCI query strings in `data/*.csv` are from the Shopping
Queries Dataset, Copyright Amazon.com, Inc. or its affiliates, under the Apache License
2.0 ([data/README.md](data/README.md)). ESCI itself is downloaded at runtime, not
redistributed.
