"""Markdown results tables from results/*.json (pasted into the README).

    python -m hps.report
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

ROWS = [  # (results file stem, label)
    ("bm25__test", "BM25"),
    ("all-MiniLM-L6-v2__test", "MiniLM-L6, pre-trained (English)"),
    ("paraphrase-multilingual-MiniLM-L12-v2__test", "Multilingual MiniLM-L12, pre-trained"),
    ("bi_en__test", "MiniLM-L6 fine-tuned, English only"),
    ("bi_hx__test", "MiniLM-L6 fine-tuned, + Hinglish augmentation"),
]
RERANK_ROWS = [
    ("bi_hx_2k__test", "Fine-tuned retriever (+ Hinglish)"),
    ("bi_hx+rr_hx__test", "  + cross-encoder reranker (top 50)"),
    ("bi_hx+rr_hx_int8__test", "  + reranker, int8"),
]


def load(stem):
    p = ROOT / "results" / f"{stem}.json"
    return json.loads(p.read_text())["results"] if p.exists() else None


def table(rows, cols):
    head = "| Model | " + " | ".join(c[0] for c in cols) + " |"
    lines = [head, "|" + "---|" * (len(cols) + 1)]
    for stem, label in rows:
        r = load(stem)
        if r is None:
            continue
        cells = [f"{r[s][m]:.3f}" if s in r else "–" for _, s, m in cols]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    main_cols = [("English nDCG@10", "en", "ndcg@10"), ("Hinglish nDCG@10", "hx", "ndcg@10"),
                 ("Hinglish, unseen words", "hx_heldout", "ndcg@10"),
                 ("English R@100", "en", "recall@100"), ("Hinglish R@100", "hx", "recall@100")]
    free_cols = [("English (same 150)", "llm_en", "ndcg@10"), ("Free-form Hinglish", "llm", "ndcg@10"),
                 ("Hand-written Hinglish", "hand", "ndcg@10")]
    judged_cols = [("English", "en", "judged_ndcg@10"), ("Hinglish", "hx", "judged_ndcg@10")]
    print("### Retrieval over 181k products (8,956 test queries)\n")
    print(table(ROWS, main_cols))
    print("\n### Free-form Hinglish (150 test queries, not from the generator)\n")
    print(table(ROWS + [("bi_hx+rr_hx_free__test", "Fine-tuned retriever + reranker")], free_cols))
    print("\n### Official ESCI Task 1 setting: rank only the judged products (random order = 0.575)\n")
    print(table(ROWS, judged_cols))
    print("\n### Reranking (2,000-query test subset)\n")
    print(table(RERANK_ROWS, main_cols[:3] + [judged_cols[0], judged_cols[1]]))


if __name__ == "__main__":
    main()
