### Where the Hinglish gap comes from (generated set: English minus Hinglish nDCG@10)

| Generated queries | Queries | BM25 + stemming | Fine-tuned, English only | Fine-tuned + augmentation | Rewrite, then BM25 | Rewrite, then fused |
|---|---|---|---|---|---|---|
| All generated Hinglish | 3,584 | 0.116 | 0.198 | 0.137 | 0.016 | 0.019 |
| 1 Hindi word | 2,566 | 0.088 | 0.169 | 0.112 | 0.012 | 0.016 |
| 2 Hindi words | 831 | 0.179 | 0.269 | 0.197 | 0.023 | 0.025 |
| 3 or more | 187 | 0.220 | 0.280 | 0.211 | 0.029 | 0.034 |
| Hindi words, no grammar rule | 2,722 | 0.119 | 0.206 | 0.142 | 0.017 | 0.018 |
| Hindi words and a grammar rule | 862 | 0.106 | 0.174 | 0.119 | 0.011 | 0.022 |
| Trained Hindi words only | 3,163 | 0.112 | 0.197 | 0.132 | 0.006 | 0.007 |
| At least one held-out word | 421 | 0.141 | 0.207 | 0.171 | 0.088 | 0.109 |

### Which Hindi words cost the most

**Costliest for the augmented model** (words in at least 20 queries; gap in nDCG@10)

| English word | Hindi (first spelling) | Queries | Fine-tuned, English only | Fine-tuned + augmentation |
|---|---|---|---|---|
| cat | billi | 26 | 0.393 | 0.370 |
| belt (held out) | patta | 21 | 0.362 | 0.320 |
| purple | baingani | 21 | 0.333 | 0.306 |
| socks | moze | 20 | 0.368 | 0.276 |
| toddler | chhote bachche | 35 | 0.369 | 0.271 |
| hat | topi | 30 | 0.409 | 0.265 |
| pants | patloon | 27 | 0.299 | 0.259 |
| rings (held out) | angoothiyan | 20 | 0.236 | 0.249 |
| boy | ladka | 24 | 0.252 | 0.238 |
| dog | kutta | 43 | 0.324 | 0.225 |

**Most improved by augmentation** (words in at least 20 queries; gap in nDCG@10)

| English word | Hindi (first spelling) | Queries | Fine-tuned, English only | Fine-tuned + augmentation |
|---|---|---|---|---|
| tall | lamba | 21 | 0.295 | 0.141 |
| hat | topi | 30 | 0.409 | 0.265 |
| costume | poshak | 49 | 0.361 | 0.218 |
| silver | chandi | 27 | 0.285 | 0.151 |
| mens | gents | 85 | 0.243 | 0.110 |
| tool (held out) | auzaar | 20 | 0.260 | 0.139 |
| dresses | frock | 23 | 0.274 | 0.159 |
| toy | khilona | 22 | 0.233 | 0.119 |
| adult | bade | 20 | 0.180 | 0.068 |
| oil | tel | 38 | 0.234 | 0.130 |

### Free-form Hinglish: words the generator never had

Hindi content words: 187, outside the lexicon in any spelling: 92 (49%)

| Free-form queries | Queries | BM25 + stemming | Fine-tuned, English only | Fine-tuned + augmentation | Rewrite, then BM25 | Rewrite, then fused | Rewrite, fused, reranked |
|---|---|---|---|---|---|---|---|
| Every Hindi word in the lexicon | 35 | 0.130 | 0.297 | 0.231 | 0.009 | 0.046 | 0.070 |
| A Hindi word outside the lexicon | 67 | 0.273 | 0.303 | 0.254 | 0.173 | 0.182 | 0.249 |
| Grammar words only | 21 | 0.065 | 0.014 | 0.037 | 0.020 | 0.046 | 0.032 |

Augmented minus English-only model, per query: better by more than 0.1 nDCG@10 on 24% of generated Hinglish queries, worse by more than 0.1 on 4%; on the unseen-words set, 14% and 6%.

### Examples

| | Query | English-only model: top result | Augmented model: top result |
|---|---|---|---|
| Augmentation fixes it | `ihome baahar ka` (ihome outdoor) | Kanun Karsisinda (not judged) | iHome iBT158 Smart Bluetooth Speaker - With Color Changin... (Exact) |
| Augmentation fixes it | `mardana scotch plaid flannel robe` (men's scotch plaid flannel robe) | Patirica Soft Knit Front Zipper Nightgown Robe for Women ... (not judged) | Alexander Del Rossa Men's Warm Cotton Flannel Robe with H... (Exact) |
| Augmentation fixes it | `rv kaala paani hose dikhao` (rv black water hose) | Hosea (not judged) | Camco 21012 RhinoEXTREME 20ft RV Sewer Hose Kit (Exact) |
| Still fails | `3.5 ft christmas pedh lights ke saath dikhao` (3.5 ft christmas tree with lights) | FAISHILAN Solar Flame Flickering Torch Solar Tiki Lights ... (not judged) | DEWENWILS C9 Blue Christmas Lights Outdoor, 33.3ft 50 LED... (not judged) |
| Still fails | `6mm gt2 rf fiber glass patta` (6mm gt2 rf fiber glass belt) | 2-Pack HPTech Tempered Glass For Samsung Galaxy S6 Screen... (not judged) | JBL GTO629 Premium 6.5-Inch Co-Axial Speaker - Set of 2 (not judged) |
| Still fails | `a duniya bina you` (a world without you) | Duniya Chor Ke Ja Tani - Single (not judged) | Duniya Chor Ke Ja Tani - Single (not judged) |
