"""
Build FAISS index from chunks.jsonl for Basic RAG pipeline.
Uses fastembed (all-MiniLM-L6-v2) — same model as pipeline2_rag.py.
Streams chunks to avoid loading all into RAM; adds to FAISS incrementally.
Saves a checkpoint every --checkpoint-every batches so a crash resumes mid-way.

Usage:
    python scripts/build_faiss.py
    python scripts/build_faiss.py --chunks data/chunks/chunks.jsonl --batch-size 512 --embed-batch-size 8
"""

import argparse
import json
import pickle
from pathlib import Path

import faiss
import numpy as np
from fastembed import TextEmbedding
from tqdm import tqdm

DIM = 384


def count_lines(path: str) -> int:
    with open(path, "rb") as f:
        return sum(1 for _ in f)


def iter_batches(path: str, batch_size: int, skip_chunks: int = 0):
    skipped = 0
    batch = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if skipped < skip_chunks:
                skipped += 1
                continue
            batch.append(json.loads(line))
            if len(batch) == batch_size:
                yield batch
                batch = []
    if batch:
        yield batch


def save_checkpoint(index, batches_done: int, chunks_done: int,
                    index_path: str, progress_path: str):
    faiss.write_index(index, index_path + ".ckpt")
    with open(progress_path, "w") as f:
        json.dump({"batches_done": batches_done, "chunks_done": chunks_done}, f)


def load_checkpoint(index_path: str, progress_path: str):
    ckpt = Path(index_path + ".ckpt")
    prog = Path(progress_path)
    if ckpt.exists() and prog.exists():
        with open(prog) as f:
            p = json.load(f)
        index = faiss.read_index(str(ckpt))
        print(f"Resuming from checkpoint: {p['batches_done']} batches, "
              f"{p['chunks_done']:,} chunks already embedded")
        return index, p["batches_done"], p["chunks_done"]
    return None, 0, 0


def main(chunks_path: str, index_path: str, pkl_path: str,
         batch_size: int, embed_batch_size: int, checkpoint_every: int):
    Path(index_path).parent.mkdir(parents=True, exist_ok=True)
    progress_path = index_path + ".progress.json"

    total = count_lines(chunks_path)
    print(f"Total chunks     : {total:,}")
    print(f"Outer batch size : {batch_size}")
    print(f"Embed batch size : {embed_batch_size}")

    # Try to resume from checkpoint
    index, batches_done, chunks_done = load_checkpoint(index_path, progress_path)
    if index is None:
        index = faiss.IndexFlatIP(DIM)

    embedder = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")

    remaining = total - chunks_done
    n_batches_remaining = (remaining + batch_size - 1) // batch_size
    n_batches_total = (total + batch_size - 1) // batch_size

    print(f"Batches total    : {n_batches_total}  (remaining: {n_batches_remaining})")

    # Pass 1: build FAISS index
    with tqdm(total=n_batches_total, initial=batches_done, desc="Embedding") as pbar:
        for batch in iter_batches(chunks_path, batch_size, skip_chunks=chunks_done):
            texts = [c["text"] for c in batch]
            embs = np.array(
                list(embedder.embed(texts, batch_size=embed_batch_size)),
                dtype=np.float32,
            )
            faiss.normalize_L2(embs)
            index.add(embs)
            batches_done += 1
            chunks_done += len(batch)
            pbar.update(1)

            if batches_done % checkpoint_every == 0:
                save_checkpoint(index, batches_done, chunks_done,
                                index_path, progress_path)

    faiss.write_index(index, index_path)
    print(f"FAISS written: {index.ntotal:,} vectors -> {index_path}")

    # Clean up checkpoint files
    for p in [index_path + ".ckpt", progress_path]:
        try:
            Path(p).unlink()
        except FileNotFoundError:
            pass

    # Pass 2: write pkl by re-reading chunks
    print("Writing chunks.pkl...")
    all_chunks = []
    with open(chunks_path, encoding="utf-8") as f:
        for line in tqdm(f, total=total, desc="Loading chunks"):
            line = line.strip()
            if line:
                all_chunks.append(json.loads(line))
    with open(pkl_path, "wb") as f:
        pickle.dump(all_chunks, f)

    print(f"Index vectors : {index.ntotal:,}")
    print(f"FAISS index   : {index_path}")
    print(f"Chunks pickle : {pkl_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", default="data/chunks/chunks.jsonl")
    parser.add_argument("--index", default="data/chunks/rag_index.faiss")
    parser.add_argument("--pkl", default="data/chunks/chunks.pkl")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--embed-batch-size", type=int, default=8,
                        help="Batch size passed to fastembed (controls ONNX memory)")
    parser.add_argument("--checkpoint-every", type=int, default=50,
                        help="Save FAISS checkpoint every N outer batches")
    args = parser.parse_args()
    main(args.chunks, args.index, args.pkl,
         args.batch_size, args.embed_batch_size, args.checkpoint_every)
