### Retrieval: nDCG@10 over 181,457 products

| Model | English (8,955) | Hinglish (3,584) | same, in English (3,584) | Unseen words only (418) | same, in English (418) |
|---|---|---|---|---|---|
| BM25 | 0.371 | 0.242 | 0.343 | 0.262 | 0.339 |
| BM25 + plural stemming | 0.396 | 0.260 | 0.376 | 0.279 | 0.371 |
| Dictionary rewrite to English, then BM25 + stemming | 0.396 | 0.360 | 0.376 | 0.279 | 0.371 |
| MiniLM-L6, pre-trained | 0.350 | 0.159 | 0.325 | 0.179 | 0.310 |
| Multilingual MiniLM-L12 (paraphrase), pre-trained | 0.220 | 0.087 | 0.223 | 0.102 | 0.209 |
| Multilingual E5-small (retrieval), pre-trained | 0.412 | 0.149 | 0.391 | 0.206 | 0.373 |
| MiniLM-L6 fine-tuned on English ESCI | 0.397 | 0.185 | 0.383 | 0.198 | 0.356 |
| Dictionary rewrite, then the English fine-tuned model | 0.397 | 0.355 | 0.383 | 0.198 | 0.356 |
| MiniLM-L6 fine-tuned + Hinglish augmentation | 0.396 | 0.241 | 0.378 | 0.213 | 0.358 |
| Dictionary rewrite, then the augmented model | 0.396 | 0.353 | 0.378 | 0.213 | 0.358 |
| Stemmed BM25 + augmented model, rank fusion | 0.430 | 0.287 | 0.408 | 0.274 | 0.395 |
| Rewrite, then stemmed BM25 + English fine-tuned model, fused | 0.433 | 0.393 | 0.414 | 0.269 | 0.397 |
| **Rewrite, then stemmed BM25 + augmented model, fused** | 0.430 | 0.389 | 0.408 | 0.274 | 0.395 |

### Recall: share of Exact products found

| Model | English R@100 (8,955) | Hinglish R@100 (3,584) | Hinglish R@50 (3,584) | Unseen R@100 (418) |
|---|---|---|---|---|
| BM25 | 0.617 | 0.481 | 0.403 | 0.496 |
| BM25 + plural stemming | 0.657 | 0.505 | 0.424 | 0.520 |
| Dictionary rewrite to English, then BM25 + stemming | 0.657 | 0.627 | 0.547 | 0.520 |
| MiniLM-L6, pre-trained | 0.641 | 0.362 | 0.291 | 0.395 |
| Multilingual MiniLM-L12 (paraphrase), pre-trained | 0.449 | 0.224 | 0.172 | 0.256 |
| Multilingual E5-small (retrieval), pre-trained | 0.677 | 0.299 | 0.249 | 0.405 |
| MiniLM-L6 fine-tuned on English ESCI | 0.691 | 0.409 | 0.332 | 0.447 |
| Dictionary rewrite, then the English fine-tuned model | 0.691 | 0.650 | 0.559 | 0.447 |
| MiniLM-L6 fine-tuned + Hinglish augmentation | 0.690 | 0.514 | 0.425 | 0.474 |
| Dictionary rewrite, then the augmented model | 0.690 | 0.650 | 0.558 | 0.474 |
| Stemmed BM25 + augmented model, rank fusion | 0.727 | 0.555 | 0.473 | 0.558 |
| Rewrite, then stemmed BM25 + English fine-tuned model, fused | 0.728 | 0.691 | 0.603 | 0.552 |
| **Rewrite, then stemmed BM25 + augmented model, fused** | 0.727 | 0.690 | 0.601 | 0.558 |

### Free-form Hinglish, not made by the generator

| Model | Free-form Hinglish (123) | same, in English (123) |
|---|---|---|
| BM25 | 0.201 | 0.383 |
| BM25 + plural stemming | 0.212 | 0.409 |
| Dictionary rewrite to English, then BM25 + stemming | 0.309 | 0.409 |
| MiniLM-L6, pre-trained | 0.165 | 0.383 |
| Multilingual MiniLM-L12 (paraphrase), pre-trained | 0.112 | 0.248 |
| Multilingual E5-small (retrieval), pre-trained | 0.127 | 0.414 |
| MiniLM-L6 fine-tuned on English ESCI | 0.182 | 0.434 |
| Dictionary rewrite, then the English fine-tuned model | 0.277 | 0.434 |
| MiniLM-L6 fine-tuned + Hinglish augmentation | 0.232 | 0.442 |
| Dictionary rewrite, then the augmented model | 0.304 | 0.442 |
| Stemmed BM25 + augmented model, rank fusion | 0.261 | 0.456 |
| Rewrite, then stemmed BM25 + English fine-tuned model, fused | 0.326 | 0.459 |
| **Rewrite, then stemmed BM25 + augmented model, fused** | 0.336 | 0.456 |
| Augmented model + fine-tuned reranker | 0.237 | 0.470 |
| **Rewrite, fused retrieval, fine-tuned reranker** | 0.306 | 0.467 |

### Unjudged products: does the ranking of models depend on them?

| Model | English nDCG@10 (8,955) | condensed (8,955) | judged@10 (8,955) | Hinglish nDCG@10 (3,584) | condensed (3,584) | judged@10 (3,584) |
|---|---|---|---|---|---|---|
| BM25 | 0.371 | 0.570 | 0.422 | 0.242 | 0.470 | 0.288 |
| BM25 + plural stemming | 0.396 | 0.595 | 0.450 | 0.260 | 0.487 | 0.305 |
| Dictionary rewrite to English, then BM25 + stemming | 0.396 | 0.595 | 0.450 | 0.360 | 0.586 | 0.413 |
| MiniLM-L6, pre-trained | 0.350 | 0.593 | 0.397 | 0.159 | 0.374 | 0.188 |
| Multilingual MiniLM-L12 (paraphrase), pre-trained | 0.220 | 0.462 | 0.252 | 0.087 | 0.257 | 0.104 |
| Multilingual E5-small (retrieval), pre-trained | 0.412 | 0.629 | 0.446 | 0.149 | 0.315 | 0.164 |
| MiniLM-L6 fine-tuned on English ESCI | 0.397 | 0.634 | 0.433 | 0.185 | 0.414 | 0.212 |
| Dictionary rewrite, then the English fine-tuned model | 0.397 | 0.634 | 0.433 | 0.355 | 0.612 | 0.393 |
| MiniLM-L6 fine-tuned + Hinglish augmentation | 0.396 | 0.632 | 0.433 | 0.241 | 0.502 | 0.279 |
| Dictionary rewrite, then the augmented model | 0.396 | 0.632 | 0.433 | 0.353 | 0.612 | 0.392 |
| Stemmed BM25 + augmented model, rank fusion | 0.430 | 0.647 | 0.478 | 0.287 | 0.527 | 0.331 |
| Rewrite, then stemmed BM25 + English fine-tuned model, fused | 0.433 | 0.648 | 0.481 | 0.393 | 0.633 | 0.440 |
| **Rewrite, then stemmed BM25 + augmented model, fused** | 0.430 | 0.647 | 0.478 | 0.389 | 0.631 | 0.438 |

### Judged-list setting (ESCI Task 1 candidate lists, US only)

| Model | English @10 (8,955) | Hinglish @10 (3,584) | English, full list (8,955) | Hinglish, full list (3,584) |
|---|---|---|---|---|
| BM25 | 0.698 | 0.680 | 0.823 | 0.814 |
| BM25 + plural stemming | 0.704 | 0.687 | 0.826 | 0.817 |
| Dictionary rewrite to English, then BM25 + stemming | 0.704 | 0.708 | 0.826 | 0.831 |
| MiniLM-L6, pre-trained | 0.711 | 0.680 | 0.829 | 0.812 |
| Multilingual MiniLM-L12 (paraphrase), pre-trained | 0.681 | 0.663 | 0.811 | 0.802 |
| Multilingual E5-small (retrieval), pre-trained | 0.730 | 0.694 | 0.842 | 0.820 |
| MiniLM-L6 fine-tuned on English ESCI | 0.733 | 0.693 | 0.843 | 0.820 |
| Dictionary rewrite, then the English fine-tuned model | 0.733 | 0.731 | 0.843 | 0.844 |
| MiniLM-L6 fine-tuned + Hinglish augmentation | 0.732 | 0.703 | 0.843 | 0.826 |
| Dictionary rewrite, then the augmented model | 0.732 | 0.730 | 0.843 | 0.843 |
| Stemmed BM25 + augmented model, rank fusion | 0.728 | 0.704 | 0.840 | 0.828 |
| Rewrite, then stemmed BM25 + English fine-tuned model, fused | 0.729 | 0.729 | 0.841 | 0.843 |
| **Rewrite, then stemmed BM25 + augmented model, fused** | 0.728 | 0.727 | 0.840 | 0.843 |
| Random order | 0.577 | 0.588 | 0.746 | 0.757 |

### Reranking (2,000-query test subset)

| Model | English (2,000) | Hinglish (798) | same, in English (798) | Unseen words only (105) | Judged list, Hinglish (798) |
|---|---|---|---|---|---|
| Augmented retriever | 0.397 | 0.245 | 0.389 | 0.222 | 0.709 |
| + pre-trained cross-encoder (MS MARCO) | 0.444 | 0.282 | 0.435 | 0.283 | 0.700 |
| + fine-tuned cross-encoder (+ Hinglish augmentation) | 0.444 | 0.272 | 0.441 | 0.253 | 0.717 |
| + fine-tuned cross-encoder, int8 | 0.439 | 0.268 | 0.436 | 0.250 | 0.716 |
| Rewrite, then stemmed BM25 + augmented retriever, fused | 0.430 | 0.393 | 0.417 | 0.274 | 0.729 |
| **+ fine-tuned cross-encoder** | 0.456 | 0.413 | 0.444 | 0.272 | 0.749 |
| + fine-tuned cross-encoder, int8 (the served path) | 0.453 | 0.409 | 0.439 | 0.279 | 0.746 |

### Are the differences real? Paired bootstrap over queries

| A minus B | Metric | Difference | 95% interval | Queries |
|---|---|---|---|---|
| Augmentation, on generated Hinglish | ndcg@10 | +0.056 | +0.052 to +0.061 | 3,584 |
| Augmentation, on unseen Hindi words only | ndcg@10 | +0.015 | +0.007 to +0.023 | 418 |
| Augmentation, on free-form Hinglish | ndcg@10 | +0.050 | +0.027 to +0.074 | 123 |
| Augmentation, cost on English | ndcg@10 | -0.001 | -0.002 to +0.001 | 8,955 |
| Augmentation vs dictionary rewrite, generated Hinglish | ndcg@10 | -0.113 | -0.121 to -0.106 | 3,584 |
| Augmentation vs dictionary rewrite, free-form Hinglish | ndcg@10 | -0.045 | -0.084 to -0.007 | 123 |
| Fine-tuned retriever vs stemmed BM25, English | ndcg@10 | -0.000 | -0.005 to +0.005 | 8,955 |
| Same, unjudged products removed | condensed_ndcg@10 | +0.037 | +0.032 to +0.042 | 8,955 |
| Fine-tuned retriever vs stemmed BM25, generated Hinglish | ndcg@10 | -0.019 | -0.026 to -0.012 | 3,584 |
| Fine-tuned retriever vs stemmed BM25, free-form Hinglish | ndcg@10 | +0.020 | -0.016 to +0.057 | 123 |
| Fine-tuned retriever vs multilingual E5, generated Hinglish | ndcg@10 | +0.092 | +0.085 to +0.100 | 3,584 |
| Fine-tuned reranker, English (2,000-query subset) | ndcg@10 | +0.046 | +0.038 to +0.055 | 2,000 |
| Fine-tuned reranker, generated Hinglish (subset) | ndcg@10 | +0.027 | +0.015 to +0.039 | 798 |
| Fine-tuning the reranker, generated Hinglish | ndcg@10 | -0.011 | -0.021 to -0.001 | 798 |
| Fine-tuning the reranker, judged list, English | judged_ndcg@10 | +0.025 | +0.020 to +0.031 | 2,000 |
| Same, generated Hinglish | judged_ndcg@10 | +0.017 | +0.008 to +0.026 | 798 |
| Pre-trained reranker vs no reranker, judged list, English | judged_ndcg@10 | -0.012 | -0.019 to -0.005 | 2,000 |
| int8 vs full-precision reranker, generated Hinglish | ndcg@10 | -0.004 | -0.008 to +0.000 | 798 |
| Rank fusion with stemmed BM25, English | ndcg@10 | +0.034 | +0.030 to +0.037 | 8,955 |
| Rank fusion with stemmed BM25, generated Hinglish | ndcg@10 | +0.045 | +0.040 to +0.050 | 3,584 |
| Augmentation when both use the rewrite, free-form | ndcg@10 | +0.027 | +0.008 to +0.046 | 123 |
| Augmentation inside the fused pipeline, English | ndcg@10 | -0.003 | -0.005 to -0.002 | 8,955 |
| Same, generated Hinglish | ndcg@10 | -0.004 | -0.006 to -0.002 | 3,584 |
| Same, unseen Hindi words | ndcg@10 | +0.005 | -0.002 to +0.013 | 418 |
| Same, free-form Hinglish | ndcg@10 | +0.010 | -0.005 to +0.026 | 123 |
| Fused pipeline vs rewrite + BM25, generated Hinglish | ndcg@10 | +0.029 | +0.024 to +0.033 | 3,584 |
| Fused pipeline vs rewrite + BM25, free-form | ndcg@10 | +0.027 | -0.002 to +0.056 | 123 |
| Fused pipeline vs stemmed BM25, unseen Hindi words | ndcg@10 | -0.005 | -0.018 to +0.009 | 418 |
| Reranker on the fused pipeline, English (subset) | ndcg@10 | +0.027 | +0.018 to +0.035 | 2,000 |
| Reranker on the fused pipeline, generated Hinglish (subset) | ndcg@10 | +0.020 | +0.007 to +0.033 | 798 |
| Same, judged list | judged_ndcg@10 | +0.020 | +0.010 to +0.031 | 798 |
| Reranker on the fused pipeline, free-form | ndcg@10 | -0.030 | -0.063 to +0.005 | 123 |
| Same, unjudged products removed | condensed_ndcg@10 | -0.006 | -0.025 to +0.013 | 123 |
| int8 vs full-precision reranker in the fused pipeline, generated Hinglish | ndcg@10 | -0.004 | -0.008 to +0.000 | 798 |

### Latency, one query at a time on a 4-core laptop CPU (300 Hinglish queries)

| Path | Encode p50 | Search p50 | Rerank p50 | Total p50 | Total p95 |
|---|---|---|---|---|---|
| Retriever only | 14 ms | 8 ms | 0 ms | 22 ms | 36 ms |
| Retriever + reranker (top 50) | 15 ms | 8 ms | 638 ms | 664 ms | 924 ms |
| Retriever + int8 reranker (top 50) | 14 ms | 9 ms | 328 ms | 351 ms | 553 ms |
| Rewrite + fused retrieval + int8 reranker (served) | 13 ms | 26 ms | 316 ms | 355 ms | 526 ms |
