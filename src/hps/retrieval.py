"""Dense retrieval: encode the catalogue once (cached), search with FAISS."""
from pathlib import Path

import faiss
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "cache"


def eval_corpus():
    """Every product judged for any val or test query (181k). The same corpus is
    used for every model, so scores are comparable."""
    import pandas as pd
    data = ROOT / "data" / "processed"
    q = pd.read_parquet(data / "queries.parquet", columns=["query_id", "split"])
    r = pd.read_parquet(data / "qrels.parquet", columns=["query_id", "product_id"])
    ids = set(r.product_id[r.query_id.isin(set(q.query_id[q.split != "train"]))])
    p = pd.read_parquet(data / "products.parquet")
    return p[p.product_id.isin(ids)].reset_index(drop=True)


def load_reranker(path, quantize=False):
    """Cross-encoder; quantize=True swaps its Linear layers for int8 (CPU speed-up)."""
    from sentence_transformers import CrossEncoder
    cross = CrossEncoder(path, max_length=96)
    if quantize:
        import torch
        from torch.ao.quantization import quantize_dynamic
        cross.model = quantize_dynamic(cross.model, {torch.nn.Linear}, dtype=torch.qint8)
    return cross


def load_encoder(name, device=None):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(name, device=device)
    model.max_seq_length = 64
    return model


def encode(model, texts, batch_size=256, **kw):
    return model.encode(list(texts), batch_size=batch_size, normalize_embeddings=True,
                        convert_to_numpy=True, **kw).astype(np.float32)


def corpus_embeddings(model, name, texts):
    """Embeddings of the whole catalogue, cached under data/cache/."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / (name.replace("/", "__").replace("\\", "__").replace(":", "") + f"__{len(texts)}.npy")
    if path.exists():
        return np.load(path)
    emb = encode(model, texts, show_progress_bar=True)
    np.save(path, emb)
    return emb


class DenseIndex:
    def __init__(self, emb):
        self.index = faiss.IndexFlatIP(emb.shape[1])
        self.index.add(emb)

    def search(self, q_emb, k=100):
        return self.index.search(q_emb, k)
