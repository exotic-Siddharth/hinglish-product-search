"""Dictionary baseline: rewrite a Hinglish query back into English, then search.

    rewrite("mardon ke liye kaala chashma")  ->  "men for black glasses"

Built only from the training word families, so held-out Hindi words stay unknown,
exactly as for the trained models. It inverts the generator's own dictionary and
spelling rules, so it is favoured on the generated query sets; the free-form
queries, written without the generator, are its fair test.
"""
import re
from functools import lru_cache
from itertools import product

from hps.hinglish import FRAMES, SPELLING, heldout_words, load_lexicon

# Words the grammar rules insert, mapped back to the English they stand for.
GRAMMAR = {
    ("ke", "liye"): "for", ("ke", "saath"): "with", ("wala",): "with", ("wali",): "with",
    ("bina",): "without", ("ke", "bina"): "without", ("se", "kam"): "under", ("ke", "andar"): "under",
    ("ke", "neeche"): "under", ("nahi",): "not", ("nahin",): "not", ("saal",): "year", ("mahine",): "month",
}
# Frame and particle words with no content of their own ("online" is English, so it stays).
DROP = {w for frame in FRAMES for w in frame.split() if w not in ("{}", "online")} | {"ka", "ki", "ke"}


def spellings(token):
    """The token and every form one spelling rule can make of it."""
    return {token} | {re.sub(pattern, repl, token, count=1) for pattern, repl in SPELLING}


@lru_cache(maxsize=1)
def phrase_table():
    """Hindi phrase (token tuple) -> English word, from training word families only."""
    held = heldout_words()
    table = dict(GRAMMAR)
    for english, variants in sorted(load_lexicon().items()):
        if english in held:
            continue
        for variant in variants:
            for form in product(*(sorted(spellings(t)) for t in variant.split())):
                table.setdefault(form, english)
    return table


def rewrite(query):
    table = phrase_table()
    longest = max(len(k) for k in table)
    toks, out, i = query.lower().split(), [], 0
    while i < len(toks):
        for n in range(min(longest, len(toks) - i), 0, -1):
            english = table.get(tuple(toks[i:i + n]))
            if english:
                out.append(english)
                i += n
                break
        else:
            if toks[i] not in DROP:
                out.append(toks[i])
            i += 1
    return " ".join(out)
