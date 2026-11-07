"""Dense retrieval: encode the catalogue once (cached), search with FAISS."""
import hashlib
from pathlib import Path

import faiss
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "cache"
# Models trained with instruction prefixes mark queries and passages (intfloat/e5).
PREFIXES = {"e5": ("query: ", "passage: ")}


def eval_corpus():
    """Every product judged for any val or test query (181,457). Every model searches
    this same corpus; most of it was judged for other queries than the one searched."""
    import pandas as pd
    data = ROOT / "data" / "processed"
    q = pd.read_parquet(data / "queries.parquet", columns=["query_id", "split"])
    r = pd.read_parquet(data / "qrels.parquet", columns=["query_id", "product_id"])
    ids = set(r.product_id[r.query_id.isin(set(q.query_id[q.split != "train"]))])
    p = pd.read_parquet(data / "products.parquet")
    return p[p.product_id.isin(ids)].reset_index(drop=True)


def load_reranker(path, quantize=False, device=None):
    """Cross-encoder. quantize=True swaps its Linear layers for int8, a CPU-only kernel,
    so a quantised reranker always runs on the CPU."""
    from sentence_transformers import CrossEncoder
    cross = CrossEncoder(path, max_length=96, device="cpu" if quantize else device)
    if quantize:
        import torch
        from torch.ao.quantization import quantize_dynamic
        quantize_dynamic(cross.model, {torch.nn.Linear}, dtype=torch.qint8, inplace=True)
    return cross


def load_encoder(name, device=None):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(name, device=device)
    model.max_seq_length = 64
    model.hps_prefixes = next((p for key, p in PREFIXES.items() if key in Path(name).name.lower()), ("", ""))
    return model


def encode(model, texts, kind="query", batch_size=256, **kw):
    prefix = getattr(model, "hps_prefixes", ("", ""))[kind == "passage"]
    return model.encode([prefix + t for t in texts], batch_size=batch_size, normalize_embeddings=True,
                        convert_to_numpy=True, **kw).astype(np.float32)


def weights_tag(name):
    """A local model is retrained in place, so its cache key includes the size and
    modification time of its weight files: new weights never reuse old vectors."""
    path = Path(name)
    if not path.is_dir():
        return ""
    files = sorted(f for f in path.rglob("*") if f.suffix in (".safetensors", ".bin"))
    signature = "|".join(f"{f.relative_to(path)}:{f.stat().st_size}:{f.stat().st_mtime_ns}" for f in files)
    return "__" + hashlib.md5(signature.encode()).hexdigest()[:8]


def cache_path(name, n_texts):
    stem = name.replace("/", "__").replace("\\", "__").replace(":", "")
    return CACHE / f"{stem}{weights_tag(name)}__{n_texts}.npy"


def corpus_embeddings(model, name, texts):
    """Embeddings of the whole catalogue, cached under data/cache/."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = cache_path(name, len(texts))
    if path.exists():
        return np.load(path)
    emb = encode(model, texts, kind="passage", show_progress_bar=True)
    np.save(path, emb)
    return emb


class DenseIndex:
    def __init__(self, emb):
        self.index = faiss.IndexFlatIP(emb.shape[1])
        self.index.add(emb)

    def search(self, q_emb, k=100):
        return self.index.search(q_emb, k)
