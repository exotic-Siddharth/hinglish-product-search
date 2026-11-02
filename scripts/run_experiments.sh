#!/usr/bin/env bash
# Reproduces every number in the README results table.
# CPU-only (4 cores): ~5 hours. On a Colab T4 GPU: well under an hour.
set -euo pipefail
cd "$(dirname "$0")/../src"
mkdir -p ../logs
export PYTHONIOENCODING=utf-8

run() { echo ">>> $*"; python -m "$@" 2>&1 | tee "../logs/$(echo "$*" | tr ' /:' '___' | cut -c1-80).log" | grep -v "Batches\|Loading weights"; }

# 0. data
[ -f ../data/processed/queries.parquet ] || run hps.prepare

# 1. baselines (no training)
run hps.evaluate --retriever bm25
run hps.evaluate --retriever sentence-transformers/all-MiniLM-L6-v2
run hps.evaluate --retriever sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

# 2. fine-tuned bi-encoders: English-only vs + Hinglish augmentation (the ablation)
run hps.train_biencoder --out ../models/bi_en --p-hx 0   --epochs 2
run hps.train_biencoder --out ../models/bi_hx --p-hx 0.5 --epochs 2
run hps.evaluate --retriever ../models/bi_en
run hps.evaluate --retriever ../models/bi_hx

# 3. cross-encoder reranker on top of the best retriever (2,000-query subset: reranking is slow on CPU)
run hps.train_reranker --out ../models/rr_hx --p-hx 0.5 --pairs 40000
run hps.evaluate --retriever ../models/bi_hx --limit 2000 --name bi_hx_2k
run hps.evaluate --retriever ../models/bi_hx --reranker ../models/rr_hx --limit 2000
