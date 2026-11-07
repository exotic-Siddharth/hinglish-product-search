"""The README figure: each model's nDCG@10 on English queries and on the same queries in
Hinglish, for the three Hinglish test sets, in a light and a dark version.

    python -m hps.figure    # writes results/hinglish_gap.png and results/hinglish_gap_dark.png
"""
import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

ROOT = Path(__file__).resolve().parents[2]
SYSTEMS = [("bm25-stem", "BM25 + stemming"), ("multilingual-e5-small", "Multilingual E5, pre-trained"),
           ("bi_en", "Fine-tuned, English only"), ("bi_hx", "Fine-tuned + Hinglish augmentation"),
           ("bm25-stem+rewrite", "Rewrite to English, then BM25"),
           ("hybrid_bi_hx+rewrite", "Rewrite, then BM25 + augmented, fused")]
FOCUS = "hybrid_bi_hx+rewrite"
SETS = [("hx", "Generated Hinglish"), ("unseen", "Only unseen Hindi words"), ("llm", "Free-form, LLM-written")]
THEMES = {  # one hue in two validated shades (light: steps 250/550, dark: 600/250), recessive chrome
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781", "grid": "#e1e0d9",
              "rule": "#c3c2b7", "english": "#86b6ef", "hinglish": "#1c5cab"},
    "dark": {"surface": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781", "grid": "#2c2c2a",
             "rule": "#383835", "english": "#184f95", "hinglish": "#86b6ef"},
}


def load(results):
    out = {}
    for name, _ in SYSTEMS:
        path = results / f"{name}__test.json"
        if path.exists():
            out[name] = json.loads(path.read_text())["results"]
    return out


def draw(res, theme, path):
    t = THEMES[theme]
    plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 9.5})
    fig, axes = plt.subplots(1, len(SETS), sharey=True, figsize=(11, 4.1), facecolor=t["surface"])
    names = [n for n, _ in SYSTEMS if n in res]
    labels = dict(SYSTEMS)
    ys = {n: len(names) - i for i, n in enumerate(names)}
    values = [res[n][k]["ndcg@10"] for n in names for s, _ in SETS for k in (s, f"{s}_en") if k in res[n]]
    lo = math.floor((min(values) - 0.02) * 20) / 20  # pad, then round out to 0.05
    hi = math.ceil((max(values) + 0.02) * 20) / 20
    for ax, (s, title) in zip(axes, SETS):
        ax.set_facecolor(t["surface"])
        count = next((res[n][s]["n_with_exact"] for n in names if s in res[n]), None)
        ax.set_title(f"{title}\n{count:,} queries" if count else title, loc="left", color=t["ink"], fontsize=10)
        for n in names:
            if s not in res[n] or f"{s}_en" not in res[n]:
                continue
            hx, en, y = res[n][s]["ndcg@10"], res[n][f"{s}_en"]["ndcg@10"], ys[n]
            ax.plot([hx, en], [y, y], color=t["rule"], lw=2, solid_capstyle="round", zorder=1)
            for x, key in ((en, "english"), (hx, "hinglish")):
                ax.scatter([x], [y], s=64, color=t[key], edgecolors=t["surface"], linewidths=1.5, zorder=2)
            if n == FOCUS:  # direct labels on the one row the story is about
                left, right = sorted([(hx, "hinglish"), (en, "english")])
                ax.annotate(f"{left[0]:.3f}", (left[0], y), xytext=(-7, 0), textcoords="offset points",
                            ha="right", va="center", color=t["ink"], fontsize=8.5)
                ax.annotate(f"{right[0]:.3f}", (right[0], y), xytext=(7, 0), textcoords="offset points",
                            ha="left", va="center", color=t["ink"], fontsize=8.5)
        ax.set_xlim(lo, hi)
        ax.set_ylim(0.4, len(names) + 0.6)
        ax.xaxis.set_major_locator(MultipleLocator(0.1))  # labels every 0.1, hairlines every 0.05
        ax.xaxis.set_minor_locator(MultipleLocator(0.05))
        ax.grid(axis="x", which="both", color=t["grid"], lw=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(which="both", colors=t["muted"], length=0, labelsize=8.5)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(t["rule"])
    axes[0].set_yticks([ys[n] for n in names], [labels[n] for n in names])
    for tick, n in zip(axes[0].get_yticklabels(), names):
        tick.set_color(t["ink"] if n == FOCUS else t["ink2"])
        tick.set_fontweight("bold" if n == FOCUS else "normal")
    axes[1].set_xlabel("nDCG@10, searching 181,457 products", color=t["ink2"])
    handles = [Line2D([], [], ls="", marker="o", ms=8, color=t[k], mec=t["surface"], mew=1.5, label=lab)
               for k, lab in (("english", "English query"), ("hinglish", "Same query in Hinglish"))]
    fig.legend(handles=handles, loc="upper right", ncol=2, frameon=False, labelcolor=t["ink2"], fontsize=9,
               bbox_to_anchor=(0.995, 1.0))
    fig.suptitle("nDCG@10 on English queries and on the same queries in Hinglish", x=0.01, ha="left",
                 color=t["ink"], fontsize=11.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=200, facecolor=t["surface"])
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results"))
    args, _ = ap.parse_known_args()
    results = Path(args.results)
    res = load(results)
    for theme, suffix in (("light", ""), ("dark", "_dark")):
        draw(res, theme, results / f"hinglish_gap{suffix}.png")
    print("wrote", results / "hinglish_gap.png", "and the dark version")


if __name__ == "__main__":
    main()
