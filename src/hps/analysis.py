"""Error analysis for the README: where Hinglish still costs, and what augmentation fixes.

    python -m hps.analysis              # tables, from saved per-query scores
    python -m hps.analysis --examples   # plus each model's top result for a few queries

The Hinglish gap of a query is its nDCG@10 in English minus its nDCG@10 in Hinglish,
same model, same judgments. To know which Hindi words a generated query got and
whether a grammar rule fired, this replays the generator exactly as prepare.py ran it
(same lexicon, brand list, seed and query order) and checks every replayed query
against data/processed/queries.parquet.
"""
import argparse
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from hps.hinglish import (
    ALWAYS_SEEN,
    FOLD,
    PARTICLES,
    Hinglishifier,
    is_heldout,
    load_lexicon,
)
from hps.prepare import ENGLISH_MIN_COUNT, hindi_swaps

ROOT = Path(__file__).resolve().parents[2]
PER_QUERY = ROOT / "results" / "per_query"
SYSTEMS = [("bm25-stem", "BM25 + stemming"), ("bi_en", "Fine-tuned, English only"),
           ("bi_hx", "Fine-tuned + augmentation"), ("bm25-stem+rewrite", "Rewrite, then BM25"),
           ("hybrid_bi_hx+rewrite", "Rewrite, then fused")]
FREE_SYSTEMS = SYSTEMS + [("hybrid_bi_hx+rr_hx+rewrite_free", "Rewrite, fused, reranked")]
GRAMMAR_WORDS = PARTICLES | ALWAYS_SEEN | {"ke", "liye", "ka", "ki", "wala", "wali", "bina", "se", "ko",
                                           "mein", "aur", "nahi", "wale"}
LABELS = {"E": "Exact", "S": "Substitute", "C": "Complement", "I": "Irrelevant"}


def per_query(name, set_name, metric="ndcg@10"):
    return pd.read_csv(PER_QUERY / f"{name}__{set_name}.csv", index_col="query_id")[metric]


def gaps(name, set_name):
    """English minus Hinglish nDCG@10 per query (queries without an Exact product drop out)."""
    return (per_query(name, f"{set_name}_en") - per_query(name, set_name)).dropna()


def english_vocab(queries):
    """prepare.py's English vocabulary: words seen at least ENGLISH_MIN_COUNT times in training queries."""
    counts = Counter(w for q in queries["query"][queries.split == "train"] for w in re.findall(r"[a-z0-9]+", q.lower()))
    return {w for w, n in counts.items() if n >= ENGLISH_MIN_COUNT}


def replay(queries, english):
    """{query_id: (English words swapped for Hindi, whether a grammar rule fired)} for every
    validation and test query, re-running prepare.py's generator call for call."""
    gen, rng, fired = Hinglishifier(words="all"), random.Random(0), []
    grammar = gen._grammar

    def traced(toks, r):  # every rule adds words the query did not have: ke liye, bina, se kam
        out = grammar(toks, r)
        fired.append(bool({t[0] for t in out} - {t[0] for t in toks}))
        return out

    gen._grammar = traced
    info = {}
    for qid, q, split, stored, n_sub in zip(queries.query_id, queries["query"], queries.split,
                                            queries.query_hx, queries.n_sub):
        if split == "train":
            continue
        fired.clear()
        text, _, used = gen(q, rng)
        words = [k for k, _ in hindi_swaps(used, english)]
        if text != stored or len(words) != n_sub:
            raise SystemExit(f"replay of query {qid} differs from queries.parquet: rerun hps.prepare")
        info[qid] = (words, fired[-1])  # the outermost call, which sees every rule
    return info


def gap_table(info):
    g = {name: gaps(name, "hx") for name, _ in SYSTEMS}
    ids = g["bi_hx"].index
    n_hindi = pd.Series({q: len(info[q][0]) for q in ids})
    rule = pd.Series({q: info[q][1] for q in ids})
    heldout = pd.Series({q: any(is_heldout(w) for w in info[q][0]) for q in ids})
    groups = [("All generated Hinglish", n_hindi > 0), ("1 Hindi word", n_hindi == 1),
              ("2 Hindi words", n_hindi == 2), ("3 or more", n_hindi >= 3),
              ("Hindi words, no grammar rule", ~rule), ("Hindi words and a grammar rule", rule),
              ("Trained Hindi words only", ~heldout), ("At least one held-out word", heldout)]
    lines = ["| Generated queries | Queries | " + " | ".join(label for _, label in SYSTEMS) + " |",
             "|---" * (len(SYSTEMS) + 2) + "|"]
    for label, mask in groups:
        sel = mask[mask].index
        lines.append(f"| {label} | {len(sel):,} | " + " | ".join(f"{g[n].reindex(sel).mean():.3f}" for n, _ in SYSTEMS)
                     + " |")
    return "\n".join(lines)


def word_table(info, lexicon, top=10, min_queries=20):
    g_en, g_hx = gaps("bi_en", "hx"), gaps("bi_hx", "hx")
    by_word = defaultdict(list)
    for q in g_hx.index:
        for w in set(info[q][0]):
            by_word[w].append(q)
    rows = [(w, len(q), g_en.reindex(q).mean(), g_hx.reindex(q).mean()) for w, q in by_word.items()
            if len(q) >= min_queries]
    head = ["| English word | Hindi (first spelling) | Queries | Fine-tuned, English only | Fine-tuned + augmentation |",
            "|---|---|---|---|---|"]
    out = []
    for title, key in (("Costliest for the augmented model", lambda r: -r[3]),
                       ("Most improved by augmentation", lambda r: r[3] - r[2])):
        lines = [f"**{title}** (words in at least {min_queries} queries; gap in nDCG@10)\n"] + head
        for w, n, a, b in sorted(rows, key=key)[:top]:
            lines.append(f"| {w}{' (held out)' if is_heldout(w) else ''} | {lexicon[w][0]} | {n} | {a:.3f} | {b:.3f} |")
        out.append("\n".join(lines))
    return "\n\n".join(out)


def fold(token):
    for a, b in FOLD:
        token = token.replace(a, b)
    return token.rstrip("y")


def free_form_table(english, lexicon):
    """The free-form set split by whether its Hindi words are in the generator's lexicon,
    in any spelling (the count behind data/README.md)."""
    d = pd.read_csv(ROOT / "data" / "llm_hinglish.csv")
    forms = {fold(t) for variants in lexicon.values() for v in variants for t in v.split()}
    kind, total, outside = {}, 0, 0
    for qid, text, eng in zip(d.query_id, d.llm_hinglish, d.english_query):
        hindi = [t for t in text.split() if not (t in set(eng.lower().split()) or t in english
                                                 or t in GRAMMAR_WORDS or any(c.isdigit() for c in t))]
        oov = [t for t in hindi if fold(t) not in forms]
        total, outside = total + len(hindi), outside + len(oov)
        kind[qid] = ("A Hindi word outside the lexicon" if oov else
                     "Every Hindi word in the lexicon" if hindi else "Grammar words only")
    g = {name: gaps(name, "llm") for name, _ in FREE_SYSTEMS}
    lines = [f"Hindi content words: {total}, outside the lexicon in any spelling: {outside} ({outside / total:.0%})\n",
             "| Free-form queries | Queries | " + " | ".join(label for _, label in FREE_SYSTEMS) + " |",
             "|---" * (len(FREE_SYSTEMS) + 2) + "|"]
    kinds = pd.Series(kind).reindex(g["bi_hx"].index)
    for label in ("Every Hindi word in the lexicon", "A Hindi word outside the lexicon", "Grammar words only"):
        sel = kinds[kinds == label].index
        lines.append(f"| {label} | {len(sel)} | " + " | ".join(f"{g[n].reindex(sel).mean():.3f}" for n, _ in FREE_SYSTEMS)
                     + " |")
    return "\n".join(lines)


def wins_losses():
    d = (per_query("bi_hx", "hx") - per_query("bi_en", "hx")).dropna()
    u = (per_query("bi_hx", "unseen") - per_query("bi_en", "unseen")).dropna()
    return (f"Augmented minus English-only model, per query: better by more than 0.1 nDCG@10 on {(d > 0.1).mean():.0%} "
            f"of generated Hinglish queries, worse by more than 0.1 on {(d < -0.1).mean():.0%}; "
            f"on the unseen-words set, {(u > 0.1).mean():.0%} and {(u < -0.1).mean():.0%}.")


def examples(n=3):
    """Top result of each model for a few generated queries, picked at random (seed 0)
    from two groups: augmentation gained more than 0.5 nDCG@10, and both models fail
    in Hinglish although the augmented one gets the English query right."""
    from hps.evaluate import Retriever, query_sets
    from hps.retrieval import eval_corpus

    sets, qrels = query_sets()
    products = eval_corpus()
    models = {"English only": "../models/bi_en", "Augmented": "../models/bi_hx"}  # the names evaluate.py caches under
    retrievers = {label: Retriever(name, products) for label, name in models.items()}
    en_hx, aug_hx = per_query("bi_en", "hx"), per_query("bi_hx", "hx")
    aug_en = per_query("bi_hx", "hx_en")
    rng = np.random.default_rng(0)
    groups = {"Augmentation fixes it": (aug_hx - en_hx)[lambda s: s > 0.5].index,
              "Still fails": aug_hx[(aug_hx == 0) & (en_hx == 0) & (aug_en >= 0.8)].index}
    lines = ["| | Query | English-only model: top result | Augmented model: top result |", "|---|---|---|---|"]
    for group, ids in groups.items():
        for q in sorted(rng.choice(list(ids), size=min(n, len(ids)), replace=False)):
            cells = []
            for r in retrievers.values():
                _, idx = r.search([sets["hx"][q]], k=1)
                row = products.iloc[int(idx[0][0])]
                title = row.text.split(" | ")[0]
                title = title if len(title) <= 60 else title[:57] + "..."
                cells.append(f"{title} ({LABELS.get(qrels[q].get(row.product_id), 'not judged')})")
            lines.append(f"| {group} | `{sets['hx'][q]}` ({sets['en'][q]}) | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", action="store_true", help="also run both retrievers on a few queries")
    args, _ = ap.parse_known_args()
    queries = pd.read_parquet(ROOT / "data" / "processed" / "queries.parquet")
    english, lexicon = english_vocab(queries), load_lexicon()
    info = replay(queries, english)
    print("### Where the Hinglish gap comes from (generated set: English minus Hinglish nDCG@10)\n")
    print(gap_table(info))
    print("\n### Which Hindi words cost the most\n")
    print(word_table(info, lexicon))
    print("\n### Free-form Hinglish: words the generator never had\n")
    print(free_form_table(english, lexicon))
    print("\n" + wins_losses())
    if args.examples:
        print("\n### Examples\n")
        print(examples())


if __name__ == "__main__":
    main()
