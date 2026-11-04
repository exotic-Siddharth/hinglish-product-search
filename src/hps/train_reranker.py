"""Fine-tune a cross-encoder reranker on graded ESCI judgments.

    python -m hps.train_reranker --out ../models/rr_hx --p-hx 0.5

The cross-encoder reads (query, product) together, so it is far more accurate
than the bi-encoder but too slow to score the whole catalogue: it only reorders
the bi-encoder's top 50. Target = ESCI gain (E=1, S=0.1, C=0.01, I=0) with a
binary cross-entropy loss, so the score is a calibrated "how exact is this match".
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from hps.checkpoint import load_ckpt, run_fingerprint, save_ckpt
from hps.hinglish import Hinglishifier, heldout_words
from hps.metrics import GAINS, ndcg_at_k

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"


def pairs(split):
    q = pd.read_parquet(DATA / "queries.parquet")
    q = q[q.split == split]
    r = pd.read_parquet(DATA / "qrels.parquet").merge(q[["query_id", "query", "query_hx", "n_sub"]], on="query_id")
    text = dict(zip(*pd.read_parquet(DATA / "products.parquet")[["product_id", "text"]].values.T))
    r["text"] = r.product_id.map(text)
    return r


@torch.no_grad()
def score(model, tok, queries, texts, batch=128):
    out = []
    for i in range(0, len(queries), batch):
        enc = tok(queries[i:i + batch], texts[i:i + batch], truncation=True, max_length=96,
                  padding=True, return_tensors="pt")
        out.append(model(**enc).logits.squeeze(-1).float())
    return torch.cat(out).numpy()


def judged_ndcg(model, tok, val, hx=False):
    v = val[val.n_sub > 0] if hx else val
    s = score(model, tok, (v.query_hx if hx else v["query"]).tolist(), v.text.tolist())
    v = v.assign(s=s)
    return float(np.mean([ndcg_at_k(list(g.sort_values("s", ascending=False).product_id),
                                    dict(zip(g.product_id, g.label)), 10) for _, g in v.groupby("query_id")]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="cross-encoder/ms-marco-MiniLM-L6-v2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--p-hx", type=float, default=0.5)
    ap.add_argument("--pairs", type=int, default=80000, help="training pairs sampled from the train split")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--val-queries", type=int, default=300, help="val queries for the quick check")
    ap.add_argument("--ckpt-every", type=int, default=200, help="steps between resumable checkpoints")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForSequenceClassification.from_pretrained(args.base, num_labels=1)
    hx = Hinglishifier(words="train")

    train = pairs("train").sample(frac=1.0, random_state=args.seed).head(args.pairs)
    val = pairs("val")
    val = val[val.query_id.isin(val.query_id.drop_duplicates().sample(args.val_queries, random_state=0))]
    total = min(args.max_steps or 10**9, len(train) // args.batch)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    warmup = max(1, total // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / warmup, max(0.0, (total - s) / (total - warmup))))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / "ckpt.pt"
    start = 0
    fingerprint = run_fingerprint(vars(args), heldout_words())
    state = load_ckpt(ckpt, model, opt, sched, fingerprint)
    if state:
        start = state["step"]
        print(f"resumed from step {start}", flush=True)
    else:
        print(f"{len(train)} pairs, {total} steps; base val judged nDCG "
              f"en={judged_ndcg(model.eval(), tok, val):.4f}", flush=True)

    t0 = time.time()
    model.train()
    for step in range(start, total):
        rng = random.Random(args.seed * 10**6 + step)
        b = train.iloc[step * args.batch:(step + 1) * args.batch]
        queries = [hx(q, rng)[0] if rng.random() < args.p_hx else q for q in b["query"]]
        enc = tok(queries, b.text.tolist(), truncation=True, max_length=96, padding=True, return_tensors="pt")
        target = torch.tensor([GAINS[l] for l in b.label], dtype=torch.float32)
        loss = F.binary_cross_entropy_with_logits(model(**enc).logits.squeeze(-1), target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad()
        if (step + 1) % 100 == 0:
            print(f"step {step + 1}/{total} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)
        if (step + 1) % args.ckpt_every == 0:
            save_ckpt(ckpt, model, opt, sched, fingerprint, step=step + 1)

    model.eval()
    log = {"val_en": judged_ndcg(model, tok, val), "val_hx": judged_ndcg(model, tok, val, hx=True)}
    print(log, flush=True)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    (out / "train_log.json").write_text(json.dumps({"args": vars(args), "log": log}, indent=2))
    ckpt.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
