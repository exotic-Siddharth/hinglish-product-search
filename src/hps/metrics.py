"""Ranking metrics.

ESCI labels are graded: Exact, Substitute, Complement, Irrelevant. We use the
gains from the KDD Cup 2022 ESCI challenge (Task 1): E=1.0, S=0.1, C=0.01, I=0.
"""
import numpy as np

GAINS = {"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0}


def dcg(gains, k):
    g = np.asarray(gains, dtype=float)[:k]
    return float(np.sum(g / np.log2(np.arange(2, g.size + 2))))


def ndcg_at_k(ranked_ids, qrel, k=10):
    """ranked_ids: product ids in ranked order. qrel: {product_id: label}.

    Unjudged products count as gain 0. The ideal ranking is built from all
    judged products of the query, so the score is comparable between a
    reranking setting (rank only judged items) and full retrieval.
    """
    gains = [GAINS.get(qrel.get(p, "I"), 0.0) for p in ranked_ids]
    ideal = sorted((GAINS[l] for l in qrel.values()), reverse=True)
    idcg = dcg(ideal, k)
    return dcg(gains, k) / idcg if idcg > 0 else 0.0


def recall_at_k(ranked_ids, qrel, k=100, label="E"):
    """Fraction of the query's `label` products found in the top k."""
    rel = {p for p, l in qrel.items() if l == label}
    if not rel:
        return float("nan")
    return len(rel.intersection(ranked_ids[:k])) / len(rel)


def mrr_at_k(ranked_ids, qrel, k=10, label="E"):
    for i, p in enumerate(ranked_ids[:k]):
        if qrel.get(p) == label:
            return 1.0 / (i + 1)
    return 0.0


def retrieval_metrics(ranked, qrel):
    """Metrics of one ranked list, or None if the query has no Exact product (its
    recall and MRR are undefined, so it is left out of every retrieval mean).

    Most retrieved products were judged for other queries, not this one, so two
    conventions are reported: ndcg@10 counts unjudged products as irrelevant, and
    condensed_ndcg@10 removes them first (Sakai 2007). judged@10 says how much of
    the top 10 is judged at all."""
    if "E" not in qrel.values():
        return None
    return {
        "ndcg@10": ndcg_at_k(ranked, qrel, 10),
        "condensed_ndcg@10": ndcg_at_k([p for p in ranked if p in qrel], qrel, 10),
        "judged@10": float(np.mean([p in qrel for p in ranked[:10]])),
        "recall@50": recall_at_k(ranked, qrel, 50),
        "recall@100": recall_at_k(ranked, qrel, 100),
        "mrr@10": mrr_at_k(ranked, qrel, 10),
    }
