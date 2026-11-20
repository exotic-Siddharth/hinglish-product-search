// The lexical path of Hinglish Product Search, ported to JavaScript so it runs in a browser:
// the dictionary rewrite (src/hps/rewrite.py) and stemmed BM25 (src/hps/bm25.py).
// web/parity.mjs checks it against the Python code on the demo's example queries.

// src/hps/bm25.py
const TOKEN = /[a-z0-9]+/g;

export function tokenize(text) {
  return text.toLowerCase().match(TOKEN) || [];
}

/** Harman's S-stemmer: only plural endings, so shoe and shoes match. */
export function sStem(w) {
  if (w.endsWith("ies") && !w.endsWith("eies") && !w.endsWith("aies")) return w.slice(0, -3) + "y";
  if (w.endsWith("es") && !w.endsWith("aes") && !w.endsWith("ees") && !w.endsWith("oes")) return w.slice(0, -1);
  if (w.endsWith("s") && !w.endsWith("us") && !w.endsWith("ss")) return w.slice(0, -1);
  return w;
}

export const tokenizeStemmed = (text) => tokenize(text).map(sStem);

/** Okapi BM25 with the same weights as hps.bm25.BM25(stem=True), stored as compact postings lists. */
export class BM25 {
  constructor(texts, { k1 = 1.2, b = 0.75 } = {}) {
    const vocab = new Map(), docTerms = [], docCounts = [], dl = new Float64Array(texts.length);
    texts.forEach((t, d) => {
      const counts = new Map();
      for (const tok of tokenizeStemmed(t)) {
        let id = vocab.get(tok);
        if (id === undefined) vocab.set(tok, (id = vocab.size));
        counts.set(id, (counts.get(id) || 0) + 1);
      }
      docTerms.push(Int32Array.from(counts.keys()));
      docCounts.push(Float64Array.from(counts.values()));
      for (const c of counts.values()) dl[d] += c;
    });
    const n = texts.length, avg = dl.reduce((s, x) => s + x, 0) / n;
    const df = new Int32Array(vocab.size);
    for (const terms of docTerms) for (const t of terms) df[t]++;
    const start = new Int32Array(vocab.size + 1);
    for (let t = 0; t < vocab.size; t++) start[t + 1] = start[t] + df[t];
    const docs = new Int32Array(start[vocab.size]), weights = new Float64Array(start[vocab.size]), fill = start.slice(0, -1);
    docTerms.forEach((terms, d) => {
      const norm = k1 * (1 - b + (b * dl[d]) / avg);
      terms.forEach((t, j) => {
        const tf = docCounts[d][j], idf = Math.log1p((n - df[t] + 0.5) / (df[t] + 0.5));
        docs[fill[t]] = d;
        weights[fill[t]++] = ((tf * (k1 + 1)) / (tf + norm)) * idf;
      });
    });
    Object.assign(this, { n, vocab, start, docs, weights });
  }

  /** Top k [docIndex, score] with score > 0 (a repeated query word counts once). */
  search(query, k = 10) {
    const scores = new Float64Array(this.n), hit = [];
    for (const tok of new Set(tokenizeStemmed(query))) {
      const t = this.vocab.get(tok);
      if (t === undefined) continue;
      for (let p = this.start[t]; p < this.start[t + 1]; p++) {
        if (scores[this.docs[p]] === 0) hit.push(this.docs[p]);
        scores[this.docs[p]] += this.weights[p];
      }
    }
    hit.sort((a, b) => scores[b] - scores[a] || a - b);
    return hit.slice(0, k).map((d) => [d, scores[d]]);
  }
}

// src/hps/rewrite.py
/** table: {phrases: [["ke liye", "for"], ...], drop: [...]} as written by web/export_data.py. */
export function makeRewriter(table) {
  const map = new Map(table.phrases), drop = new Set(table.drop);
  const longest = Math.max(...table.phrases.map(([k]) => k.split(" ").length));
  /** Greedy longest-match rewrite; steps say what happened to each word. */
  return function rewrite(query) {
    const toks = query.toLowerCase().split(/\s+/).filter(Boolean); // Python's str.split()
    const out = [], steps = [];
    let i = 0;
    while (i < toks.length) {
      let matched = false;
      for (let n = Math.min(longest, toks.length - i); n > 0; n--) {
        const phrase = toks.slice(i, i + n).join(" "), english = map.get(phrase);
        if (english) {
          out.push(english);
          steps.push({ from: phrase, to: english, kind: "rewritten" });
          i += n;
          matched = true;
          break;
        }
      }
      if (!matched) {
        const kept = !drop.has(toks[i]);
        if (kept) out.push(toks[i]);
        steps.push({ from: toks[i], to: kept ? toks[i] : null, kind: kept ? "kept" : "dropped" });
        i++;
      }
    }
    return { text: out.join(" "), steps };
  };
}

// src/hps/metrics.py: ESCI gains from the KDD Cup 2022 (Task 1)
export const GAINS = { E: 1.0, S: 0.1, C: 0.01, I: 0.0 };

/** nDCG@k of ranked doc indices against {docIndex: label}; unjudged products count as 0. */
export function ndcgAtK(ranked, judged, k = 10) {
  const dcg = (gains) => gains.slice(0, k).reduce((s, g, i) => s + g / Math.log2(i + 2), 0);
  const ideal = dcg(Object.values(judged).map((l) => GAINS[l]).sort((a, b) => b - a));
  return ideal > 0 ? dcg(ranked.map((d) => GAINS[judged[d]] ?? 0)) / ideal : 0;
}

/** "title | brand: X | color: Y" (src/hps/prepare.py product_text) -> parts for display. */
export function parseProduct(text) {
  let rest = text, color = "", brand = "";
  const c = rest.lastIndexOf(" | color: ");
  if (c >= 0) { color = rest.slice(c + 10); rest = rest.slice(0, c); }
  const b = rest.lastIndexOf(" | brand: ");
  if (b >= 0) { brand = rest.slice(b + 10); rest = rest.slice(0, b); }
  return { title: rest, brand, color };
}
