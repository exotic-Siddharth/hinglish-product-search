"""Run from the project root:  pytest -q"""
import random
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hps.bm25 import BM25
from hps.hinglish import Hinglishifier, is_heldout, load_lexicon
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
        assert not any(map(is_heldout, used))


def test_grammar_rules():
    h = Hinglishifier(p_word=0.0, p_rule=1.0, p_reorder=0.0, p_frame=0.0)
    rng = random.Random(0)
    assert h("shoes for men", rng)[0] == "shoes men ke liye"
    assert h("set of 4 cups", rng)[0] == "4 ka set cups"
    out = h("bottle under 20", rng)[0]
    assert out in ("bottle 20 se kam", "bottle 20 ke andar")
    assert h("toys for 3 year old boys", rng)[0].startswith("toys 3 saal")


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
