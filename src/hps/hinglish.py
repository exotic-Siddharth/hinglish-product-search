"""Turn English shopping queries into code-mixed Hinglish (romanised Hindi + English).

Three layers, each random:
  1. grammar  - Hindi postpositions and word order ("for X" -> "X ke liye",
                "with X" -> "X wala", "under 50" -> "50 se kam", "A not B" -> "A B nahi")
  2. lexicon  - common words swapped for romanised Hindi (data/hinglish_lexicon.tsv).
                Swaps are word by word, so words with several shopping senses
                (ring, sheet, pot, up) are left out of the lexicon, and brand phrases
                from the catalogue (under armour, hot wheels) are never touched
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
STOP = {"with", "without", "for", "and", "under", "not"}
PERSON = {"men", "women", "mens", "womens", "man", "woman", "kids", "kid", "girls", "girl", "boys", "boy",
          "baby", "babies", "toddler", "toddlers", "adults", "adult", "teens", "teen", "infants", "infant",
          "dogs", "dog", "cats", "cat", "ladies", "seniors", "youth", "newborn", "newborns"}
NOT_PURPOSE = {"looking", "look", "search", "searching", "shopping"}
ATTRIBUTES = {"size", "heavy", "pack", "inch", "inches", "lb", "lbs", "oz", "set", "count"}
UNDER_PLACES = {"sink", "bed", "desk", "cabinet", "cabinets", "counter", "seat", "tree", "car", "couch",
                "sofa", "table", "stairs", "door", "bathroom", "kitchen"}
POSTPOSITIONS = {"ke", "ka", "ki", "se", "ko"}
PROTECTED_PATH = LEXICON_PATH.parent / "processed" / "protected_phrases.txt"

# Particles carry no meaning of their own, so they do not link two entries.
PARTICLES = {"ka", "ki", "ke", "wala", "wali", "se", "bina"}
# Words the grammar rules and frames produce whatever the lexicon split is. They
# are always seen in training, so a family containing one can never be held out.
ALWAYS_SEEN = {"liye", "saath", "kam", "andar", "neeche", "saal", "mahine", "nahi", "nahin"} | {
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


def phrase_key(text):
    """Token tuple used to match catalogue phrases: "Black+Decker" and "black & decker" agree."""
    return tuple(normalise(t) for t in text.lower().replace("+", " and ").replace("&", " and ").split())


def load_protected(path=PROTECTED_PATH):
    """Brand phrases from the catalogue (under armour, hot wheels, instant pot), written by
    prepare.py. Their words are never translated and never trigger a grammar rule."""
    path = Path(path)
    if not path.exists():
        return set()
    return {tuple(line.split()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


class Hinglishifier:
    """words: which lexicon entries may be used - "train", "heldout" or "all".
    protected: phrases to leave alone; None loads the catalogue brand list if prepare.py wrote one."""

    def __init__(self, lexicon=None, words="all", protected=None, p_word=0.8, p_rule=0.8,
                 p_reorder=0.5, p_noise=0.25, p_frame=0.25):
        lexicon = lexicon or load_lexicon()
        keep = {"all": lambda w: True, "train": lambda w: not is_heldout(w),
                "heldout": is_heldout}[words]
        self.lex = {w: v for w, v in lexicon.items() if keep(w)}
        self.protected = load_protected() if protected is None else set(protected)
        self.max_phrase = max((len(p) for p in self.protected), default=0)
        self.p_word, self.p_rule, self.p_reorder = p_word, p_rule, p_reorder
        self.p_noise, self.p_frame = p_noise, p_frame

    def _protect(self, toks):
        keys = [normalise(t[0]).replace("&", "and") for t in toks]
        for n in range(self.max_phrase, 1, -1):
            for i in range(len(toks) - n + 1):
                if tuple(keys[i:i + n]) in self.protected:
                    for t in toks[i:i + n]:
                        t[1] = False

    def _spell(self, word, rng):
        if rng.random() >= self.p_noise:
            return word
        pat, rep = SPELLING[rng.randrange(len(SPELLING))]
        return re.sub(pat, rep, word, count=1)

    def _span_end(self, toks, start, rule):
        """End (exclusive) of the phrase a rule word governs, starting at `start`."""
        end = start
        while end < len(toks) and normalise(toks[end][0]) not in STOP:
            end += 1
        if rule == "for":
            for j in range(start, end):
                if normalise(toks[j][0]) in PERSON:
                    while j + 1 < end and normalise(toks[j + 1][0]) in PERSON:
                        j += 1  # "for teen girls" is one purpose
                    return j + 1  # "for women waterproof": the purpose is "women"
        if rule in ("without", "not"):
            for j in range(start + 1, end):
                w = normalise(toks[j][0])
                if NUM.match(w) or w in ATTRIBUTES:
                    return j  # "without arms 300 lb": only "arms" is excluded
        return end

    def _grammar(self, toks, rng):
        """toks: list of [text, translatable]. Returns a new list. Only original,
        unprotected words (translatable) can trigger a rule."""
        out, i = [], 0
        while i < len(toks):
            w = normalise(toks[i][0])
            nxt = [normalise(t[0]) for t in toks[i + 1:i + 3]]
            fire = toks[i][1] and rng.random() < self.p_rule
            if fire and NUM.match(w) and nxt[:1] and nxt[0] in ("year", "years", "month", "months") \
                    and nxt[1:2] == ["old"]:
                unit = "saal" if nxt[0].startswith("year") else "mahine"
                out += [[w, False], [unit, False], [rng.choice(["ke", "ka"]), False]]
                i += 3
            elif fire and w in ("set", "pack") and nxt[:1] == ["of"] and len(nxt) > 1 and NUM.match(nxt[1]):
                out += [[nxt[1], False], ["ka", False], [w, False]]
                i += 3
            elif fire and w == "under" and nxt[:1] and NUM.match(nxt[0]):
                out += [[nxt[0], False]] + [[t, False] for t in rng.choice([["se", "kam"], ["ke", "andar"]])]
                i += 2
            elif fire and w == "under" and nxt[:1] and nxt[0] in UNDER_PLACES:
                out += [toks[i + 1], ["ke", False], ["neeche", False]]
                i += 2
            elif fire and w in ("for", "with", "without", "not") and i + 1 < len(toks):
                end = self._span_end(toks, i + 1, w)
                previous = normalise(out[-1][0]) if out else ""
                if end == i + 1 or (w == "for" and previous in NOT_PURPOSE):
                    out.append(toks[i])  # "looking for": a verb, not a purpose
                    i += 1
                    continue
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
                elif w == "without":
                    form = rng.randrange(3)
                    if form == 0:
                        out += [["bina", False]] + obj + [["ke", False]]
                    elif form == 1:
                        out += obj + [["ke", False], ["bina", False]]
                    else:
                        out += [["bina", False]] + obj
                else:  # Hindi negates after the excluded thing: "laptop chromebook nahi"
                    out += obj + [[rng.choice(["nahi", "nahin"]), False]]
                i = end
            elif w in ("the", "a", "an") and rng.random() < 0.5:
                i += 1
            else:
                out.append(toks[i])
                i += 1
        return out

    def __call__(self, query, rng):
        """Returns (hinglish_query, n_swaps, swaps); swaps are (english_word, hindi_variant)."""
        toks = [[t, True] for t in query.lower().split()]
        self._protect(toks)
        toks = self._grammar(toks, rng)
        words, used = [], []
        for idx, (text, translatable) in enumerate(toks):
            key = normalise(text)
            if translatable and key in self.lex and rng.random() < self.p_word:
                variant = rng.choice(self.lex[key])
                text = " ".join(self._spell(p, rng) for p in variant.split())
                used.append((key, variant))
                following = normalise(toks[idx + 1][0]) if idx + 1 < len(toks) else ""
                after_for = "for" in (normalise(t[0]) for t in toks[max(0, idx - 3):idx])
                if (key in PERSON and text.endswith(("on", "o")) and following not in POSTPOSITIONS
                        and not after_for):
                    text += " ke liye"  # an oblique plural needs a postposition: "hoodie ladkiyon ke liye"
            words.append(text)
        q = " ".join(words)
        if rng.random() < self.p_frame:
            q = rng.choice(FRAMES).format(q)
        return q, len(used), used
