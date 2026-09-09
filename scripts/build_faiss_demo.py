"""
Build a SMALL, demo-focused FAISS index from the full index.

Why: the full index (rag_index.faiss ~681 MB + chunks.pkl ~494 MB) cannot load on a
512 MB free-tier host — it OOMs. But Basic RAG's per-query token count and answer depend
only on the top-k chunks it retrieves, NOT on the total index size. So we pre-query the
full index with every demo question, collect the union of their top chunks, and rebuild a
tiny index from just those. The deployed dashboard then returns the EXACT same chunks (and
therefore the same answers + token numbers) it would from the full index — but in ~1-2 MB
that loads instantly with no OOM. The full benchmark report is still run locally on the
full index; this is purely the deploy artifact for the live demo.

Usage:
    python scripts/build_faiss_demo.py
"""

import json
import pickle
from pathlib import Path

import faiss
import numpy as np
from fastembed import TextEmbedding

ROOT       = Path(__file__).parent.parent.resolve()
FULL_FAISS = ROOT / "data/chunks/rag_index.faiss"
FULL_PKL   = ROOT / "data/chunks/chunks.pkl"
OUT_FAISS  = ROOT / "data/chunks/rag_index_demo.faiss"
OUT_PKL    = ROOT / "data/chunks/chunks_demo.pkl"
QA_FILE    = ROOT / "data/qa/qa_pairs_legal_multihop.json"

TOP_K = 30   # chunks to pull per question (Basic RAG retrieves top-8 at query time)
DIM   = 384


def load_questions() -> list[str]:
    qs: list[str] = []
    with open(QA_FILE, encoding="utf-8") as f:
        for item in json.load(f):
            q = item.get("question", "").strip()
            if q:
                qs.append(q)
    # A few extra broad legal phrasings users might type, so retrieval stays robust.
    qs += [
        "guilty plea withdrawal", "ineffective assistance of counsel",
        "Batson challenge peremptory", "waiver of right to appeal",
        "unpreserved for appellate review", "prior conviction impeachment",
        "agency defense plea allocution", "collateral attack divorce decree",
        "light most favorable to plaintiff", "jury instruction verdict sheet",
    ]
    return list(dict.fromkeys(qs))


def embed(embedder, texts: list[str]) -> np.ndarray:
    embs = np.array(list(embedder.embed(texts)), dtype=np.float32)
    faiss.normalize_L2(embs)
    return embs


def main():
    print("Loading full index (mmap) + chunks …", flush=True)
    index  = faiss.read_index(str(FULL_FAISS), faiss.IO_FLAG_MMAP)
    with open(FULL_PKL, "rb") as f:
        chunks = pickle.load(f)
    print(f"Full index: {index.ntotal:,} vectors, {len(chunks):,} chunks", flush=True)

    questions = load_questions()
    print(f"Querying with {len(questions)} demo questions (top-{TOP_K} each) …", flush=True)

    embedder = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
    q_embs   = embed(embedder, questions)
    _, idxs  = index.search(q_embs, TOP_K)

    keep = sorted({int(i) for row in idxs for i in row if 0 <= i < len(chunks)})
    print(f"Unique chunks kept: {len(keep):,}", flush=True)

    # Reconstruct the kept vectors from the full index (it's IndexFlatIP, so vectors
    # are stored verbatim and reconstruct() returns them exactly — no re-embedding).
    vecs = np.vstack([index.reconstruct(i) for i in keep]).astype(np.float32)
    small = faiss.IndexFlatIP(DIM)
    small.add(vecs)

    small_chunks = [chunks[i] for i in keep]

    OUT_FAISS.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(small, str(OUT_FAISS))
    with open(OUT_PKL, "wb") as f:
        pickle.dump(small_chunks, f)

    faiss_mb = OUT_FAISS.stat().st_size / 1_048_576
    pkl_mb   = OUT_PKL.stat().st_size / 1_048_576
    print(f"\nWrote {OUT_FAISS.name}: {small.ntotal:,} vectors ({faiss_mb:.1f} MB)", flush=True)
    print(f"Wrote {OUT_PKL.name}:   {len(small_chunks):,} chunks ({pkl_mb:.1f} MB)", flush=True)
    print("\nThese two files are small enough to commit and load with zero OOM.", flush=True)


if __name__ == "__main__":
    main()
