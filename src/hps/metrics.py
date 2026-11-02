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


def summarise(runs, qrels, ks=(10,), recall_ks=(100,)):
    """runs: {query_id: ranked ids}. Returns mean metrics over queries that
    have at least one Exact product (others have undefined recall/MRR)."""
    out = {f"ndcg@{k}": [] for k in ks}
    out.update({f"recall@{k}": [] for k in recall_ks})
    out["mrr@10"] = []
    for qid, ranked in runs.items():
        qrel = qrels[qid]
        if "E" not in qrel.values():
            continue
        for k in ks:
            out[f"ndcg@{k}"].append(ndcg_at_k(ranked, qrel, k))
        for k in recall_ks:
            out[f"recall@{k}"].append(recall_at_k(ranked, qrel, k))
        out["mrr@10"].append(mrr_at_k(ranked, qrel, 10))
    return {m: float(np.mean(v)) for m, v in out.items()} | {"n_queries": len(out["mrr@10"])}
