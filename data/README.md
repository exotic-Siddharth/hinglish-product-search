# Data in this folder

| File | What it is | Made by |
|---|---|---|
| `hinglish_lexicon.tsv` | 185 English shopping words with their romanised Hindi spellings | Hand-written for this project |
| `protected_compounds.txt` | 18 compounds that change meaning when translated word by word (rose gold, green screen) | Hand-written for this project |
| `heldout_words.txt` | The 25 lexicon entries whose Hindi word families never appear in training | Written by `src/hps/prepare.py`, from the lexicon |
| `llm_hinglish.csv` | 123 ESCI test queries with a free-form Hinglish version | Hinglish written by an LLM (Claude), see below |
| `handwritten_hinglish.csv` | The same 123 queries, with an empty column for a person to fill in | Empty template |

`raw/`, `processed/` and `cache/` are not in git. `prepare.py` rebuilds `processed/` from the
ESCI files in `raw/`, including `processed/protected_phrases.txt`: 1,683 phrases the generator
never touches, which are multi-word catalogue brands (under armour, hot wheels) plus the
compounds above.

## The free-form Hinglish set

The generated test sets come from the same generator the model trains with. That makes
them a good stress test and a weak realism test. `llm_hinglish.csv` is a second test that
does not come from the generator: an LLM (Claude) was asked to write each query the way an
Indian shopper might type it in Hinglish. The rules were:

* translate only what the English query says: no added product words, no fixed typos
  (`lazer tape` stays `lazer`), brands and titles unchanged;
* drop a query if nothing in it can be translated (model numbers, film and book titles) or if
  the original is not English (five ESCI "US" queries are Spanish).

That left 123 of 150 sampled queries. Of its 187 Hindi content words, 92 (49%) are not in
the generator's lexicon in any spelling (almari, gamla, palkein, jalpari, ketli). It does
share the common grammar words (ke liye, wala, bina) with the generator, because that is
how Hinglish is written. It is LLM-written, so it is not
evidence of how real users type. `handwritten_hinglish.csv` is there for a native speaker
to fill in; `evaluate.py` picks it up automatically as the `hand` set.

## Licence and attribution

The query strings in the `english_query` column of both CSVs come from the Shopping
Queries Dataset (ESCI), Copyright Amazon.com, Inc. or its affiliates, released under the
Apache License 2.0 (https://github.com/amazon-science/esci-data). Cite:

> Chandan K. Reddy, Lluís Màrquez, Fran Valero, Nikhil Rao, Hugo Zaragoza, Sambaran
> Bandyopadhyay, Arnab Biswas, Anlu Xing, Karthik Subbian. *Shopping Queries Dataset: A
> Large-Scale ESCI Benchmark for Improving Product Search.* arXiv:2206.06588, 2022.

Everything else in this folder is under the repository's MIT licence.
