#!/usr/bin/env bash
# Rebuilds every result behind the README tables, from the raw ESCI files in data/raw/.
# Measured on a 4-core laptop CPU (i7-1195G7), no GPU: see the README for stage times.
set -euo pipefail
cd "$(dirname "$0")/../src"
mkdir -p ../logs
export PYTHONIOENCODING=utf-8

run() {  # each step logs to logs/<step>.log; on failure, show the end of the log and stop
  local log="../logs/$(echo "$*" | tr ' /:' '___' | cut -c1-90).log"
  echo ">>> $*"
  if ! python -W ignore -m "$@" > "$log" 2>&1; then echo "FAILED: $*"; tail -n 20 "$log"; exit 1; fi
  tail -n 3 "$log"
}

# 0. data: queries, judgments, products, brand phrases, Hinglish test sets
run hps.prepare

# 1. training: English-only vs + Hinglish augmentation (the ablation), then the reranker
run hps.train_biencoder --out ../models/bi_en --p-hx 0   --epochs 2
run hps.train_biencoder --out ../models/bi_hx --p-hx 0.5 --epochs 2
run hps.train_reranker  --out ../models/rr_hx --p-hx 0.5 --pairs 40000

# 2. retrieval over the 181,457-product corpus
run hps.evaluate --retriever bm25
run hps.evaluate --retriever bm25-stem
run hps.evaluate --retriever bm25-stem --rewrite
run hps.evaluate --retriever sentence-transformers/all-MiniLM-L6-v2
run hps.evaluate --retriever sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
run hps.evaluate --retriever intfloat/multilingual-e5-small
run hps.evaluate --retriever ../models/bi_en
run hps.evaluate --retriever ../models/bi_en --rewrite
run hps.evaluate --retriever ../models/bi_hx

# 3. reranking, on a 2,000-query subset (cross-encoders are slow on a CPU)
SUB=en,hx,hx_en,unseen,unseen_en
run hps.evaluate --retriever ../models/bi_hx --limit 2000 --sets $SUB --name bi_hx_2k
run hps.evaluate --retriever ../models/bi_hx --reranker cross-encoder/ms-marco-MiniLM-L6-v2 --limit 2000 --sets $SUB
run hps.evaluate --retriever ../models/bi_hx --reranker ../models/rr_hx --limit 2000 --sets $SUB
run hps.evaluate --retriever ../models/bi_hx --reranker ../models/rr_hx --quantize --limit 2000 --sets $SUB
run hps.evaluate --retriever ../models/bi_hx --reranker ../models/rr_hx --sets llm,llm_en,hand,hand_en --name bi_hx+rr_hx_free

# 4. latency, one query at a time on the CPU
run hps.latency --retriever ../models/bi_hx
run hps.latency --retriever ../models/bi_hx --reranker ../models/rr_hx
run hps.latency --retriever ../models/bi_hx --reranker ../models/rr_hx --quantize

python -W ignore -m hps.report | tee ../results/report.md
python -W ignore -m hps.analysis --examples | tee ../results/analysis.md
python -W ignore -m hps.figure    # results/hinglish_gap.png, light and dark
