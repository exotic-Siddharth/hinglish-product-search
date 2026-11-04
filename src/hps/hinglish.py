"""Turn English shopping queries into code-mixed Hinglish (romanised Hindi + English).

    "running shoes for men under 50"  ->  "mardon ke liye running joote 50 se kam"
    "water bottle without straw"      ->  "paani botal bina straw ke"

Three layers, each random:
  1. grammar  - Hindi postpositions and word order ("for X" -> "X ke liye",
                "with X" -> "X wala", "under 50" -> "50 se kam", ...)
  2. lexicon  - common words swapped for romanised Hindi (data/hinglish_lexicon.tsv);
                brand and tech words stay English, as people actually type them
  3. spelling - romanisation varies between people (kala/kaala, pani/paani)

Lexicon entries are split into "train" and "heldout" so the test set can measure
how the model does on Hindi words it was never shown. The split is by word family,
not by entry: `big` and `large` both give `bada`, and `man` / `men` give `mard` /
`mardon`, so holding out one entry of a family would leak the rest through training.
"""
import hashlib
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

LEXICON_PATH = Path(__file__).resolve().parents[2] / "data" / "hinglish_lexicon.tsv"

# Romanisation variants people actually produce. Applied to Hindi words only.
SPELLING = [
    ("aa", "a"), ("ee", "i"), ("oo", "u"), ("w", "v"), ("v", "w"), ("chh", "ch"),
    ("kh", "k"), ("dh", "d"), ("bh", "b"), ("z", "j"),
    ("a$", "aa"), ("i$", "ee"), ("e$", "ey"), ("on$", "o"),
]
FRAMES = ["{} chahiye", "mujhe {} chahiye", "{} dikhao", "{} kharidna hai",
          "{} online", "accha {}", "{} kahan milega", "{} batao"]
NUM = re.compile(r"^\$?\d+(\.\d+)?$")
STOP = {"with", "without", "for", "and", "under"}

# Particles carry no meaning of their own, so they do not link two entries.
PARTICLES = {"ka", "ki", "ke", "wala", "wali", "se", "bina"}
# Words the grammar rules and frames produce whatever the lexicon split is. They
# are always seen in training, so a family containing one can never be held out.
ALWAYS_SEEN = {"liye", "saath", "kam", "andar", "neeche", "saal", "mahine"} | {
    w for frame in FRAMES for w in frame.split() if w != "{}"}
# Folds every spelling rule above, so all spellings of a word share one root.
FOLD = [("aa", "a"), ("ee", "i"), ("oo", "u"), ("chh", "ch"), ("kh", "k"),
        ("dh", "d"), ("bh", "b"), ("z", "j"), ("w", "v")]


def load_lexicon(path=LEXICON_PATH):
    lines = Path(path).read_text(encoding="utf-8").splitlines()[1:]
    return {en: variants.split("|") for en, variants in (l.split("\t") for l in lines if l.strip())}


def root(token):
    """First 4 letters after folding spelling: kaala/kala and mard/mardon share a root."""
    for spelling, folded in FOLD:
        token = token.replace(spelling, folded)
    return token[:4]


def entry_roots(variants):
    return {root(t) for v in variants for t in v.split() if t not in PARTICLES}


@lru_cache(maxsize=1)
def heldout_words(frac=0.2):
    """English entries whose whole Hindi word family is held out of training."""
    lexicon = load_lexicon()
    parent = {w: w for w in lexicon}

    def find(w):
        while parent[w] != w:
            parent[w] = parent[parent[w]]
            w = parent[w]
        return w

    owner = {}
    for word, variants in lexicon.items():
        for r in entry_roots(variants):
            parent[find(word)] = find(owner.setdefault(r, word))
    families = defaultdict(set)
    for word in lexicon:
        families[find(word)].add(word)

    seen_roots = {root(t) for t in ALWAYS_SEEN}
    held = set()
    for family in families.values():
        roots = set().union(*(entry_roots(lexicon[w]) for w in family))
        if not roots & seen_roots and int(hashlib.md5(min(family).encode()).hexdigest(), 16) % 1000 < frac * 1000:
            held |= family
    return frozenset(held)


def is_heldout(word):
    return word in heldout_words()


def normalise(tok):
    return tok.strip(".,!?;:\"()").replace("'", "")


class Hinglishifier:
    """words: which lexicon entries may be used - "train", "heldout" or "all"."""

    def __init__(self, lexicon=None, words="all", p_word=0.8, p_rule=0.8,
                 p_reorder=0.5, p_noise=0.25, p_frame=0.25):
        lexicon = lexicon or load_lexicon()
        keep = {"all": lambda w: True, "train": lambda w: not is_heldout(w),
                "heldout": is_heldout}[words]
        self.lex = {w: v for w, v in lexicon.items() if keep(w)}
        self.p_word, self.p_rule, self.p_reorder = p_word, p_rule, p_reorder
        self.p_noise, self.p_frame = p_noise, p_frame

    def _spell(self, word, rng):
        if rng.random() >= self.p_noise:
            return word
        pat, rep = SPELLING[rng.randrange(len(SPELLING))]
        return re.sub(pat, rep, word, count=1)

    def _span_end(self, toks, start):
        """End of the noun phrase that starts at `start` (exclusive)."""
        end = start
        while end < len(toks) and normalise(toks[end][0]) not in STOP:
            end += 1
        return end

    def _grammar(self, toks, rng):
        """toks: list of [text, translatable]. Returns a new list."""
        out, i = [], 0
        while i < len(toks):
            w = normalise(toks[i][0])
            nxt = [normalise(t[0]) for t in toks[i + 1:i + 3]]
            fire = rng.random() < self.p_rule
            if fire and NUM.match(w) and nxt[:1] and nxt[0] in ("year", "years", "month", "months") \
                    and nxt[1:2] == ["old"]:
                unit = "saal" if nxt[0].startswith("year") else "mahine"
                out += [[w, False], [unit, False], [rng.choice(["ke", "ka"]), False]]
                i += 3
            elif fire and w in ("set", "pack") and nxt[:1] == ["of"] and len(nxt) > 1 and NUM.match(nxt[1]):
                out += [[nxt[1], False], ["ka", False], [w, False]]
                i += 3
            elif fire and w == "under" and nxt[:1]:
                if NUM.match(nxt[0]):
                    out += [[nxt[0], False]] + [[t, False] for t in rng.choice([["se", "kam"], ["ke", "andar"]])]
                else:
                    out += [toks[i + 1], ["ke", False], ["neeche", False]]
                i += 2
            elif fire and w in ("for", "with", "without") and i + 1 < len(toks):
                end = self._span_end(toks, i + 1)
                obj = self._grammar(toks[i + 1:end], rng)
                if w == "for":
                    block = obj + [["ke", False], ["liye", False]]
                    if end == len(toks) and out and rng.random() < self.p_reorder:
                        out = block + out  # Hindi word order: purpose first
                    else:
                        out += block
                elif w == "with":
                    tail = rng.choice([["wala"], ["wali"], ["ke", "saath"]])
                    out += obj + [[t, False] for t in tail]
                else:
                    form = rng.randrange(3)
                    if form == 0:
                        out += [["bina", False]] + obj + [["ke", False]]
                    elif form == 1:
                        out += obj + [["ke", False], ["bina", False]]
                    else:
                        out += [["bina", False]] + obj
                i = end
            elif w in ("the", "a", "an") and rng.random() < 0.5:
                i += 1
            else:
                out.append(toks[i])
                i += 1
        return out

    def __call__(self, query, rng):
        """Returns (hinglish_query, n_substituted_words, used_words)."""
        toks = self._grammar([[t, True] for t in query.lower().split()], rng)
        words, used = [], []
        for text, translatable in toks:
            key = normalise(text)
            if translatable and key in self.lex and rng.random() < self.p_word:
                text = " ".join(self._spell(p, rng) for p in rng.choice(self.lex[key]).split())
                used.append(key)
            words.append(text)
        q = " ".join(words)
        if rng.random() < self.p_frame:
            q = rng.choice(FRAMES).format(q)
        return q, len(used), used
