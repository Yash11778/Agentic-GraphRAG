"""
Generate the architecture diagram for the submission (required deliverable).
Output: architecture_diagram.png in the project root.

    python scripts/generate_architecture_diagram.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

ROOT = Path(__file__).parent.parent.resolve()
OUT = ROOT / "architecture_diagram.png"

BG = "#0f172a"
CARD = "#1e293b"
BORDER = "#334155"
BLUE = "#3b82f6"
GREEN = "#22c55e"
ORANGE = "#f59e0b"
PURPLE = "#a855f7"
TEXT = "#e2e8f0"
SUBTEXT = "#94a3b8"


def box(ax, xy, w, h, text, sub=None, color=BLUE, fontsize=11):
    x, y = xy
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.6, edgecolor=color, facecolor=CARD, zorder=2,
    ))
    ax.text(x + w / 2, y + h / 2 + (0.06 if sub else 0), text,
            ha="center", va="center", color=TEXT, fontsize=fontsize, fontweight="bold", zorder=3)
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.14, sub,
                ha="center", va="center", color=SUBTEXT, fontsize=fontsize - 2.5, zorder=3)


def arrow(ax, p1, p2, color=BORDER, style="-"):
    ax.add_patch(FancyArrowPatch(
        p1, p2, arrowstyle="-|>", mutation_scale=14, linewidth=1.4,
        color=color, linestyle=style, zorder=1,
    ))


def main():
    fig, ax = plt.subplots(figsize=(14, 9))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis("off")

    ax.text(7, 8.6, "GraphRAG vs Basic RAG vs LLM-only — Legal Citation Graph",
            ha="center", color=TEXT, fontsize=16, fontweight="bold")
    ax.text(7, 8.25, "63,632 U.S. court opinions · 117.5M tokens · 9,632 real citation edges",
            ha="center", color=SUBTEXT, fontsize=10.5)

    # Dataset
    box(ax, (0.4, 6.9), 3.2, 1.0, "Pile of Law corpus", "eyecite + courts-db enrichment", color=GREEN)
    box(ax, (0.4, 5.5), 3.2, 1.0, "dataset_100m_enriched.jsonl", "63,632 opinions, 117.5M tokens", color=GREEN)
    arrow(ax, (2.0, 6.9), (2.0, 6.5))

    # Split into three data stores
    arrow(ax, (2.0, 5.5), (2.0, 5.0))
    box(ax, (0.4, 3.9), 3.2, 1.0, "FAISS index", "500,959 chunks · MiniLM", color=ORANGE)
    box(ax, (0.4, 2.5), 3.2, 1.3, "TigerGraph LegalGraph",
        "LegalCase --CITES--> LegalCase\nLegalCase --DECIDED_BY--> Court", color=PURPLE, fontsize=10)
    arrow(ax, (1.4, 5.0), (2.0, 4.9))
    arrow(ax, (2.6, 5.0), (2.0, 3.8))

    # Three pipelines
    px = 5.0
    box(ax, (px, 6.9), 3.2, 1.0, "Pipeline 1 — LLM-only", "Raw Gemini, no retrieval", color=BLUE)
    box(ax, (px, 5.3), 3.2, 1.0, "Pipeline 2 — Basic RAG", "FAISS top-k -> Gemini", color=ORANGE)
    box(ax, (px, 3.7), 3.2, 1.3, "Pipeline 3 — GraphRAG",
        "resolve case IDs -> citation_multihop\n_retrieve (CITES traversal) -> Gemini", color=PURPLE, fontsize=10)

    arrow(ax, (3.6, 4.4), (5.0, 6.9), color=ORANGE)
    arrow(ax, (3.6, 3.15), (5.0, 4.35), color=PURPLE)

    # Eval layer
    ex = 9.4
    box(ax, (ex, 5.3), 3.4, 1.0, "LLM-as-a-Judge", "HuggingFace InferenceClient\n(strict PASS/FAIL, Gemini fallback)", color=GREEN, fontsize=10)
    box(ax, (ex, 3.9), 3.4, 1.0, "BERTScore", "evaluate.load, rescale_with_baseline", color=GREEN)
    for y in (7.4, 5.8, 4.3):
        arrow(ax, (8.2, y), (9.4, 5.7))
        arrow(ax, (8.2, y), (9.4, 4.3))

    # Dashboard
    box(ax, (5.0, 1.6), 7.8, 1.4, "Comparison Dashboard (api/app.py + React)",
        "1 question in -> 3 answers + tokens/latency/cost/judge/BERTScore, side by side", color=BLUE, fontsize=11)
    for y in (7.4, 5.8, 4.3):
        arrow(ax, (6.6, y), (7.5, 3.0), color=BORDER)
    arrow(ax, (11.0, 5.3), (10.0, 3.0), color=BORDER)

    legend = [
        Line2D([0], [0], color=GREEN, lw=3, label="Data / evaluation"),
        Line2D([0], [0], color=ORANGE, lw=3, label="Basic RAG path"),
        Line2D([0], [0], color=PURPLE, lw=3, label="GraphRAG path"),
        Line2D([0], [0], color=BLUE, lw=3, label="LLM-only / dashboard"),
    ]
    ax.legend(handles=legend, loc="lower left", facecolor=CARD, edgecolor=BORDER,
              labelcolor=TEXT, fontsize=9, framealpha=0.9)

    plt.tight_layout()
    fig.savefig(OUT, facecolor=BG, dpi=180)
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
