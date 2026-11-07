"""Run from the project root:  pytest -q"""
import random
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hps.bm25 import BM25
from hps.hinglish import (
    ALWAYS_SEEN,
    PARTICLES,
    Hinglishifier,
    entry_roots,
    is_heldout,
    load_lexicon,
    root,
)
from hps.metrics import mrr_at_k, ndcg_at_k, recall_at_k


def test_ndcg_perfect_and_reversed():
    qrel = {"a": "E", "b": "S", "c": "I"}
    assert ndcg_at_k(["a", "b", "c"], qrel) == pytest.approx(1.0)
    worst = ndcg_at_k(["c", "b", "a"], qrel)
    expected = (0.1 / np.log2(3) + 1.0 / np.log2(4)) / (1.0 + 0.1 / np.log2(3))
    assert worst == pytest.approx(expected)


def test_unjudged_products_count_as_zero():
    qrel = {"a": "E"}
    assert ndcg_at_k(["x", "a"], qrel) == pytest.approx(1 / np.log2(3))


def test_recall_and_mrr():
    qrel = {"a": "E", "b": "E", "c": "S"}
    assert recall_at_k(["c", "a", "z"], qrel, k=2) == 0.5
    assert mrr_at_k(["c", "a"], qrel) == 0.5


def test_heldout_split_is_disjoint_and_about_20_percent():
    lex = load_lexicon()
    train = Hinglishifier(lex, words="train").lex
    held = Hinglishifier(lex, words="heldout").lex
    assert not set(train) & set(held)
    assert set(train) | set(held) == set(lex)
    assert 0.1 < len(held) / len(lex) < 0.3


def test_training_augmentation_never_uses_heldout_words():
    h = Hinglishifier(words="train", p_word=1.0)
    rng = random.Random(0)
    for q in ["black shoes for women", "small water bottle", "kids table and chairs"] * 50:
        _, _, used = h(q, rng)
        assert used and not any(is_heldout(word) for word, _ in used)


def test_no_heldout_hindi_word_is_reachable_from_training():
    """The "unseen words" results are only honest if this holds, for every
    word, spelling and inflection that training augmentation can produce."""
    lex = load_lexicon()
    held = {w for w in lex if is_heldout(w)}
    assert held
    held_roots = set().union(*(entry_roots(lex[w]) for w in held))
    held_tokens = {t for w in held for v in lex[w] for t in v.split() if t not in PARTICLES}

    assert not held_roots & {root(t) for t in ALWAYS_SEEN}
    for word, variants in lex.items():
        if word not in held:
            assert not entry_roots(variants) & held_roots, word

    h = Hinglishifier(words="train", p_word=1.0, p_rule=1.0, p_noise=1.0, p_frame=1.0)
    rng = random.Random(0)
    everything = " for ".join(lex) + " with " + " without ".join(lex)
    for _ in range(200):
        assert not set(h(everything, rng)[0].split()) & held_tokens


def test_word_families_stay_together():
    # big/large share a Hindi word, man/men are inflections, kaala/kala are spellings.
    for a, b in (("big", "large"), ("man", "men"), ("shoe", "shoes"), ("toy", "toys")):
        assert is_heldout(a) == is_heldout(b)
    assert root("kaala") == root("kala")
    assert root("mard") == root("mardon")


def test_grammar_rules():
    h = Hinglishifier(protected=set(), p_word=0.0, p_rule=1.0, p_reorder=0.0, p_frame=0.0)
    rng = random.Random(0)
    assert h("shoes for men", rng)[0] == "shoes men ke liye"
    assert h("set of 4 cups", rng)[0] == "4 ka set cups"
    out = h("bottle under 20", rng)[0]
    assert out in ("bottle 20 se kam", "bottle 20 ke andar")
    assert h("toys for 3 year old boys", rng)[0].startswith("toys 3 saal")
    assert h("organizer under sink", rng)[0] == "organizer sink ke neeche"


def test_rules_keep_the_meaning():
    h = Hinglishifier(protected=set(), p_word=0.0, p_rule=1.0, p_reorder=0.0, p_frame=0.0)
    rng = random.Random(0)
    # Hindi negates after the excluded item: "laptop not chromebook" must not say "not laptop".
    assert h("laptop not chromebook", rng)[0] in ("laptop chromebook nahi", "laptop chromebook nahin")
    # The purpose ends at the person; attributes after it stay with the product.
    assert h("boots for women waterproof", rng)[0] == "boots women ke liye waterproof"
    # "looking for" is a verb, and "under" only means beneath for a place.
    assert h("looking for red dress", rng)[0] == "looking for red dress"
    assert h("under layers for women", rng)[0] == "under layers women ke liye"
    # "without" excludes only the thing, not the size that follows.
    assert h("chairs without arms 300 lb", rng)[0].endswith("300 lb")


def test_brand_phrases_are_never_translated_or_parsed():
    lexicon = {"black": ["kala"], "shirt": ["kameez"]}
    h = Hinglishifier(lexicon, protected={("under", "armour"), ("black", "and", "decker")},
                      p_word=1.0, p_rule=1.0, p_frame=0.0, p_noise=0.0)
    rng = random.Random(0)
    assert h("under armour shirt", rng)[0] == "under armour kameez"
    assert h("black & decker drill", rng)[0] == "black & decker drill"
    assert h("black shirt", rng)[0] == "kala kameez"


def test_oblique_plural_gets_a_postposition():
    h = Hinglishifier({"girls": ["ladkiyon"]}, protected=set(), p_word=1.0, p_rule=1.0,
                      p_reorder=0.0, p_frame=0.0, p_noise=0.0)
    rng = random.Random(0)
    assert h("hoodie girls", rng)[0] == "hoodie ladkiyon ke liye"
    assert h("hoodie for girls", rng)[0] == "hoodie ladkiyon ke liye"


def test_generator_is_deterministic_given_seed():
    h = Hinglishifier()
    a = [h(q, random.Random(7))[0] for q in ["red dress for girls", "dog food without grain"]]
    b = [h(q, random.Random(7))[0] for q in ["red dress for girls", "dog food without grain"]]
    assert a == b


def test_bm25_ranks_matching_doc_first():
    docs = ["red cotton dress for girls", "stainless steel water bottle", "black leather wallet for men"]
    bm = BM25().fit(docs)
    _, idx = bm.search(["water bottle"], k=2)
    assert idx[0, 0] == 1
    assert bm.score("leather wallet", [0, 2]).argmax() == 1


def test_checkpoint_restores_model_optimizer_and_step(tmp_path):
    import torch

    from hps.checkpoint import load_ckpt, run_fingerprint, save_ckpt

    def build():
        m = torch.nn.Linear(3, 1)
        o = torch.optim.AdamW(m.parameters(), lr=0.1)
        return m, o, torch.optim.lr_scheduler.LambdaLR(o, lambda s: 1 / (s + 1))

    m, o, s = build()
    m(torch.ones(1, 3)).sum().backward()
    o.step()
    s.step()
    run = run_fingerprint({"lr": 0.1}, {"a", "b"})
    save_ckpt(tmp_path / "c.pt", m, o, s, run, step=7, log=[{"val": 0.5}])

    m2, o2, s2 = build()
    untouched = m2.weight.clone()
    other_run = run_fingerprint({"lr": 0.1}, {"a"})
    assert load_ckpt(tmp_path / "c.pt", m2, o2, s2, other_run) is None
    assert torch.equal(m2.weight, untouched)
    assert load_ckpt(tmp_path / "missing.pt", m2, o2, s2, run) is None

    state = load_ckpt(tmp_path / "c.pt", m2, o2, s2, run)
    assert state == {"step": 7, "log": [{"val": 0.5}]}
    assert torch.equal(m.weight, m2.weight)
    assert s2.last_epoch == 1
    assert o2.state_dict()["state"][0]["step"] == o.state_dict()["state"][0]["step"]
