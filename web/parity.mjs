// Run the JavaScript port on the demo's example queries and compare it with the Python results in
// reference.json. Writes parity.json (shown on the page) and fails if the port drifts.
//   node web/parity.mjs web/build
import { readFile, writeFile } from "node:fs/promises";
import { BM25, makeRewriter, ndcgAtK } from "./hps.js";

const dir = process.argv[2] ?? "web/build";
const read = async (f) => JSON.parse(await readFile(`${dir}/${f}`, "utf8"));
const [products, examples, table, ref, meta] = await Promise.all(
  ["products.json", "examples.json", "rewrite.json", "reference.json", "meta.json"].map(read));

let t = Date.now();
const bm25 = new BM25(products);
const buildMs = Date.now() - t;
const rewrite = makeRewriter(table);

const sameScores = (js, py) => js.length === py.length && js.every(([, s], j) => Math.abs(s - py[j][1]) <= 1e-4 * Math.max(1, Math.abs(s)));
let rewriteSame = 0, listsSame = 0, idsSame = 0, lists = 0, searchMs = 0;
const ndcg = {};
const diffs = [];
t = Date.now();
examples.forEach((ex, i) => {
  const r = ref[i];
  const rw = rewrite(ex.hinglish).text;
  rewriteSame += rw === r.rewritten;
  if (rw !== r.rewritten && diffs.length < 5) diffs.push(`rewrite: ${ex.hinglish} -> js "${rw}" / python "${r.rewritten}"`);
  const judged = Object.fromEntries(ex.judged);
  for (const [key, text] of [["hinglish", ex.hinglish], ["rewritten", rw], ["english", ex.english]]) {
    const js = bm25.search(text, 10);
    lists++;
    listsSame += sameScores(js, r.runs[key]);
    idsSame += js.length === r.runs[key].length && js.every(([d], j) => d === r.runs[key][j][0]);
    ((ndcg[ex.set] ??= {})[key] ??= []).push(ndcgAtK(js.map(([d]) => d), judged));
  }
});
searchMs = (Date.now() - t) / lists;
const mean = (a) => Math.round((a.reduce((s, x) => s + x, 0) / a.length) * 10000) / 10000;
const browser = Object.fromEntries(Object.entries(ndcg).map(([s, v]) => [s, Object.fromEntries(Object.entries(v).map(([k, a]) => [k, mean(a)]))]));
const report = {
  examples: examples.length, products: products.length,
  same_rewrite: Math.round((rewriteSame / examples.length) * 1000) / 1000,
  same_top10_scores: Math.round((listsSame / lists) * 1000) / 1000,
  same_top10_order: Math.round((idsSame / lists) * 1000) / 1000,
  ndcg: { python: meta.ndcg, browser },
  node_build_ms: buildMs, node_search_ms: Math.round(searchMs * 10) / 10,
  built: new Date().toISOString().slice(0, 10),
};
await writeFile(`${dir}/parity.json`, JSON.stringify(report, null, 1));
console.log(JSON.stringify(report, null, 1));
if (diffs.length) console.log(diffs.join("\n"));

const problems = [];
if (report.same_rewrite < 1) problems.push("the rewrite differs from Python");
if (report.same_top10_scores < 0.99) problems.push("BM25 top-10 differs from Python too often");
for (const s of Object.keys(meta.ndcg)) for (const k of Object.keys(meta.ndcg[s])) {
  if (Math.abs(meta.ndcg[s][k] - browser[s][k]) > 0.005) problems.push(`nDCG@10 ${s}/${k} differs by more than 0.005`);
}
if (problems.length) { console.error("PARITY CHECK FAILED: " + problems.join("; ")); process.exit(1); }
console.log("parity check passed");
