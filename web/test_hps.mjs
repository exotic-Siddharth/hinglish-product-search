// Unit tests for the JavaScript port; expected values were produced by the Python code (src/hps).
import assert from "node:assert/strict";
import { test } from "node:test";
import { BM25, makeRewriter, ndcgAtK, parseProduct, sStem, tokenize } from "./hps.js";

test("tokenizer and S-stemmer match hps.bm25", () => {
  assert.deepEqual(tokenize("Men's T-Shirt, 2-Pack (XL) | brand: Hanes | color: Café-Noir 100%"),
    ["men", "s", "t", "shirt", "2", "pack", "xl", "brand", "hanes", "color", "caf", "noir", "100"]);
  const stems = { shoes: "shoe", dresses: "dresse", glasses: "glasse", boxes: "boxe", batteries: "battery",
    series: "sery", bus: "bus", class: "class", heroes: "heroe", toys: "toy", flies: "fly", eies: "eie", status: "status" };
  for (const [w, s] of Object.entries(stems)) assert.equal(sStem(w), s, w);
});

test("rewrite matches hps.rewrite on a small table", () => {
  const rewrite = makeRewriter({
    phrases: [["ke liye", "for"], ["mardon", "men"], ["kaala", "black"], ["kala", "black"], ["se kam", "under"], ["bado", "adult"]],
    drop: ["chahiye", "ka", "ke", "ki", "mujhe"],
  });
  assert.equal(rewrite("mardon ke liye kaala chashma").text, "men for black chashma");
  assert.equal(rewrite("  Mujhe  KAALA jacket chahiye ").text, "black jacket");
  assert.equal(rewrite("stocking stuffers bado ke liye 10 se kam").text, "stocking stuffers adult for 10 under");
  assert.deepEqual(rewrite("ka pack").steps.map((s) => s.kind), ["dropped", "kept"]);
});

test("BM25 scores and nDCG match Python", () => {
  const bm = new BM25(["red shoes for men", "blue shoe", "red dress | brand: Acme", "kids water bottle"]);
  const top = bm.search("red shoes shoes", 10);
  assert.deepEqual(top.map(([d]) => d), [0, 1, 2]);
  assert.ok(Math.abs(top[0][1] - 1.2667098) < 1e-6 && Math.abs(top[1][1] - 0.8225731) < 1e-6 && Math.abs(top[2][1] - 0.6333549) < 1e-6);
  assert.deepEqual(bm.search("laptop", 10), []);
  const judged = { 0: "E", 5: "S", 9: "C", 12: "I" };
  assert.ok(Math.abs(ndcgAtK([0, 1, 5, 9, 12, 3], judged) - 0.9870927) < 1e-6);
  assert.ok(Math.abs(ndcgAtK([9, 5, 0], judged) - 0.5365572) < 1e-6);
  assert.equal(ndcgAtK([], judged), 0);
});

test("product text is split back into title, brand and colour", () => {
  assert.deepEqual(parseProduct("Water Bottle | brand: Acme | color: Blue"), { title: "Water Bottle", brand: "Acme", color: "Blue" });
  assert.deepEqual(parseProduct("Water Bottle | color: Blue"), { title: "Water Bottle", brand: "", color: "Blue" });
});
