"""Fine-tune a two-tower (bi-encoder) retriever on ESCI, optionally with
Hinglish query augmentation.

    python -m hps.train_biencoder --out ../models/bi_en   --p-hx 0      (English only)
    python -m hps.train_biencoder --out ../models/bi_hx   --p-hx 0.5    (+ Hinglish)

Each epoch has one row per training query: the query, one of its Exact products,
and a hard negative when it has one (an Irrelevant product, else a Complement;
33% of queries have neither). Loss is in-batch softmax (InfoNCE): a query must
score its own product above every other product in the batch, hard negatives
included. One row per query per batch keeps hidden positives rare: about 0.1% of
in-batch negatives turn out to be judged Exact for that query.

With probability p-hx a row's query goes through the Hinglish generator, using only
the "train" word families (see hinglish.py), so held-out Hindi words stay unseen
until evaluation. Most queries contain no lexicon word, so in the default run
about 22% of training rows actually contain a Hindi word.
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

from hps import hinglish
from hps.checkpoint import load_ckpt, run_fingerprint, save_ckpt
from hps.hinglish import LEXICON_PATH, PROTECTED_PATH, Hinglishifier, heldout_words
from hps.metrics import ndcg_at_k
from hps.retrieval import encode, load_encoder

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"


def build_examples(split):
    q = pd.read_parquet(DATA / "queries.parquet")
    q = q[q.split == split]
    r = pd.read_parquet(DATA / "qrels.parquet")
    r = r[r.query_id.isin(set(q.query_id))]
    text = dict(zip(*pd.read_parquet(DATA / "products.parquet")[["product_id", "text"]].values.T))
    by_q = {qid: g for qid, g in r.groupby("query_id")}
    ex = []
    for qid, query in zip(q.query_id, q["query"]):
        g = by_q[qid]
        pos = [text[p] for p in g.product_id[g.label == "E"]] or [text[p] for p in g.product_id[g.label == "S"]]
        neg = [text[p] for p in g.product_id[g.label == "I"]] or [text[p] for p in g.product_id[g.label == "C"]]
        if pos:
            ex.append((query, pos, neg))
    return ex


def embed(model, texts, max_len):
    model.max_seq_length = max_len
    feats = model.tokenize(texts)
    feats = {k: v.to(model.device) if torch.is_tensor(v) else v for k, v in feats.items()}
    return F.normalize(model(feats)["sentence_embedding"], dim=-1)


def judged_ndcg(model, split="val", hx=False):
    """Mean nDCG@10 when ranking each query's judged products (cheap model check)."""
    q = pd.read_parquet(DATA / "queries.parquet")
    q = q[q.split == split]
    if hx:
        q = q[q.n_sub > 0]
    r = pd.read_parquet(DATA / "qrels.parquet")
    r = r[r.query_id.isin(set(q.query_id))]
    prods = pd.read_parquet(DATA / "products.parquet")
    prods = prods[prods.product_id.isin(set(r.product_id))]
    model.max_seq_length = 64
    pe = dict(zip(prods.product_id, encode(model, prods.text.tolist())))
    model.max_seq_length = 32
    qe = encode(model, (q.query_hx if hx else q["query"]).tolist())
    scores = []
    by_q = {qid: g for qid, g in r.groupby("query_id")}
    for qid, e in zip(q.query_id, qe):
        g = by_q[qid]
        s = np.stack([pe[p] for p in g.product_id]) @ e
        ranked = g.product_id.values[np.argsort(-s)]
        scores.append(ndcg_at_k(list(ranked), dict(zip(g.product_id, g.label)), 10))
    return float(np.mean(scores))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--p-hx", type=float, default=0.5, help="probability a query is Hinglish-augmented")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--scale", type=float, default=20.0, help="inverse temperature of the softmax")
    ap.add_argument("--max-steps", type=int, default=None, help="for smoke tests")
    ap.add_argument("--ckpt-every", type=int, default=50, help="steps between resumable checkpoints")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = load_encoder(args.base)
    hx = Hinglishifier(words="train")
    examples = build_examples("train")
    steps_per_epoch = len(examples) // args.batch
    total = min(args.max_steps or 10**9, steps_per_epoch * args.epochs)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    warmup = max(1, total // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / warmup, max(0.0, (total - s) / max(1, total - warmup))))
    print(f"{len(examples)} training queries, {steps_per_epoch} steps/epoch, {total} steps total")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / "ckpt.pt"
    log, step, t0 = [], 0, time.time()
    fingerprint = run_fingerprint(vars(args), heldout_words(), (LEXICON_PATH, PROTECTED_PATH, hinglish.__file__))
    state = load_ckpt(ckpt, model, opt, sched, fingerprint)
    if state:
        log, step = state["log"], state["step"]
        print(f"resumed from step {step}", flush=True)

    model.train()
    while step < total:
        # Order and augmentation depend only on (seed, step), and the checkpoint holds the
        # torch RNG for dropout, so a resumed run continues where the interrupted one stopped.
        epoch, b = divmod(step, steps_per_epoch)
        order = list(range(len(examples)))
        random.Random(args.seed * 1000 + epoch).shuffle(order)
        rng = random.Random(args.seed * 10**6 + step)
        batch = [examples[i] for i in order[b * args.batch:(b + 1) * args.batch]]
        queries = [hx(q, rng)[0] if rng.random() < args.p_hx else q for q, _, _ in batch]
        pos = [rng.choice(p) for _, p, _ in batch]
        neg = [rng.choice(n) for _, _, n in batch if n]
        qe = embed(model, queries, 32)
        de = embed(model, pos + neg, 64)
        logits = args.scale * qe @ de.T
        loss = F.cross_entropy(logits, torch.arange(len(batch), device=logits.device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad()
        step += 1
        if step % 50 == 0 or step == total:
            print(f"epoch {epoch} step {step}/{total} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)
        end_of_epoch = step % steps_per_epoch == 0 or step == total
        if end_of_epoch:
            model.eval()
            with torch.no_grad():
                v = {"epoch": epoch, "step": step, "val_en": judged_ndcg(model), "val_hx": judged_ndcg(model, hx=True)}
            log.append(v)
            print(v, flush=True)
            model.train()
        if end_of_epoch or step % args.ckpt_every == 0:
            save_ckpt(ckpt, model, opt, sched, fingerprint, step=step, log=log)

    model.eval()
    model.max_seq_length = 64  # the product length; judged_ndcg leaves it at the query length
    model.save(str(out))
    (out / "train_log.json").write_text(json.dumps({"args": vars(args), "log": log}, indent=2))
    ckpt.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
