"""corpus.jsonl → chunks + embeddings, written locally and loaded into TigerGraph.

    python agentic/ingest/build_vectors.py              # build locally
    python agentic/ingest/build_vectors.py --load       # build, then load
    python agentic/ingest/build_vectors.py --load-only  # load what is on disk

The local artefacts (`data/vectors/chunks.jsonl` and `embeddings.npy`) are the
source of truth, exactly as `data/graph/*.jsonl` is for the graph: the local
backend reads them directly and the loader ships the same numbers to Savanna, so
similarity results cannot depend on which backend answered.

Embedding runs on a local ONNX model, so the vector index costs no API calls and
is reproducible on any machine -- a judge can rebuild it without a key.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

from agentic.config import CORPUS, VECTOR_DIR
from agentic.graph_schema import (
    CHUNK_OVERLAP,
    CHUNK_TOKENS,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
)
from agentic.llm import ENCODING

CHUNKS_FILE = VECTOR_DIR / "chunks.jsonl"
EMBEDDINGS_FILE = VECTOR_DIR / "embeddings.npy"

# Partial progress. Embedding the whole corpus takes about an hour on a laptop
# CPU, and losing that to an OOM kill or a closed terminal is worth one file:
# the run checkpoints here and picks up where it stopped.
PARTIAL_FILE = VECTOR_DIR / "embeddings.partial.npy"
PARTIAL_MARK = VECTOR_DIR / "embeddings.partial.count"

# Vectors make the payload large, so batches are small: 200 chunks is roughly a
# megabyte of floats, which the REST endpoint accepts comfortably.
LOAD_BATCH = 200
# ONNX allocates activation buffers proportional to the batch and a 400-token
# window is long: at 1,000 per outer batch the run peaked above 5GB and was killed
# on a 14GB machine. Throughput was measured flat from 32 to 128 per inner batch,
# so these are set for memory, not speed -- the CPU is the bottleneck either way.
EMBED_BATCH = 256


def chunk_document(doc: dict) -> list[dict]:
    """Split one document into overlapping token windows.

    The title is prepended to the text that gets embedded but is not stored as
    part of the chunk. Event articles repeat their title in almost no sentence,
    so a chunk from the middle of one is otherwise unattributable: with the title
    in the embedding, a question naming the event can reach any of its chunks,
    while the stored text stays exactly what the corpus said.
    """
    tokens = ENCODING.encode(doc["text"], disallowed_special=())
    step = CHUNK_TOKENS - CHUNK_OVERLAP
    chunks = []
    for ordinal, start in enumerate(range(0, max(len(tokens), 1), step)):
        window = tokens[start:start + CHUNK_TOKENS]
        if not window:
            break
        text = ENCODING.decode(window)
        chunks.append({
            "id": f"{doc['doc_id']}:{ordinal}",
            "doc_id": doc["doc_id"],
            "ordinal": ordinal,
            "text": text,
            "embed_text": f"{doc['title']}\n\n{text}",
        })
        if start + CHUNK_TOKENS >= len(tokens):
            break
    return chunks


def build() -> tuple[list[dict], np.ndarray]:
    from fastembed import TextEmbedding

    chunks: list[dict] = []
    with CORPUS.open(encoding="utf-8") as f:
        for line in f:
            chunks.extend(chunk_document(json.loads(line)))
    print(f"{len(chunks)} chunks from the corpus")

    print(f"embedding with {EMBEDDING_MODEL} (local, CPU) ...", flush=True)
    started = time.perf_counter()
    model = TextEmbedding(model_name=EMBEDDING_MODEL)

    # Embedded in slices with progress rather than one exhausted generator: the
    # run takes minutes, and a silent process that might be wedged is impossible
    # to distinguish from one that is simply slow.
    vectors, first = _resume(len(chunks))
    if first:
        print(f"  resuming at {first}", flush=True)
    for start in range(first, len(chunks), EMBED_BATCH):
        window = chunks[start:start + EMBED_BATCH]
        vectors[start:start + len(window)] = np.array(
            list(model.embed([c["embed_text"] for c in window], batch_size=32)),
            dtype=np.float32,
        )
        done = start + len(window)
        if done % (EMBED_BATCH * 4) == 0 or done == len(chunks):
            _checkpoint(vectors, done)
            rate = (done - first) / max(time.perf_counter() - started, 1e-6)
            print(f"  {done}/{len(chunks)}  ({rate:.1f}/s, "
                  f"~{(len(chunks) - done) / max(rate, 1e-6) / 60:.0f} min left)", flush=True)
    print(f"  {vectors.shape} in {time.perf_counter() - started:.0f}s")
    if vectors.shape[1] != EMBEDDING_DIM:
        raise SystemExit(
            f"model returned {vectors.shape[1]} dimensions, schema declares "
            f"{EMBEDDING_DIM}; update EMBEDDING_DIM and reload the schema"
        )

    VECTOR_DIR.mkdir(parents=True, exist_ok=True)
    with CHUNKS_FILE.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            row = {k: v for k, v in chunk.items() if k != "embed_text"}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    np.save(EMBEDDINGS_FILE, vectors)
    PARTIAL_FILE.unlink(missing_ok=True)
    PARTIAL_MARK.unlink(missing_ok=True)
    print(f"wrote {CHUNKS_FILE} and {EMBEDDINGS_FILE}")
    return chunks, vectors


def _resume(total: int) -> tuple[np.ndarray, int]:
    """The checkpointed array and the index to continue from.

    The checkpoint is only trusted when it was taken over a corpus of the same
    size; any change to chunking invalidates it, and silently resuming onto
    differently-shaped vectors would corrupt the index in a way nothing downstream
    could detect.
    """
    VECTOR_DIR.mkdir(parents=True, exist_ok=True)
    if PARTIAL_FILE.exists() and PARTIAL_MARK.exists():
        saved = np.load(PARTIAL_FILE)
        done = int(PARTIAL_MARK.read_text().strip())
        if saved.shape == (total, EMBEDDING_DIM) and 0 < done <= total:
            return saved, done
    return np.zeros((total, EMBEDDING_DIM), dtype=np.float32), 0


def _checkpoint(vectors: np.ndarray, done: int) -> None:
    np.save(PARTIAL_FILE, vectors)
    PARTIAL_MARK.write_text(str(done))


def read_local() -> tuple[list[dict], np.ndarray]:
    """The chunks and vectors as written by `build`."""
    if not CHUNKS_FILE.exists() or not EMBEDDINGS_FILE.exists():
        raise SystemExit("no local vectors; run: python agentic/ingest/build_vectors.py")
    chunks = [json.loads(line) for line in CHUNKS_FILE.open(encoding="utf-8")]
    return chunks, np.load(EMBEDDINGS_FILE)


def load(chunks: list[dict], vectors: np.ndarray) -> None:
    from agentic.ingest.load_tigergraph import connect

    conn = connect()
    print(f"loading {len(chunks)} chunks into TigerGraph ...")
    started = time.perf_counter()
    total = 0
    for i in range(0, len(chunks), LOAD_BATCH):
        batch = [
            (c["id"], {
                "doc_id": c["doc_id"],
                "ordinal": c["ordinal"],
                "text": c["text"],
                "emb": vectors[i + j].tolist(),
            })
            for j, c in enumerate(chunks[i:i + LOAD_BATCH])
        ]
        total += conn.upsertVertices("Chunk", batch)
    print(f"  upserted in {time.perf_counter() - started:.0f}s")

    deadline = time.time() + 300
    while conn.getVertexCount("Chunk") < len(chunks) and time.time() < deadline:
        time.sleep(10)
    count = conn.getVertexCount("Chunk")
    print(f"  Chunk vertices in graph: {count} (expected {len(chunks)})")
    if count != len(chunks):
        raise SystemExit("chunk load incomplete")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--load", action="store_true", help="load into TigerGraph after building")
    ap.add_argument("--load-only", action="store_true", help="load the artefacts already on disk")
    args = ap.parse_args()

    if args.load_only:
        chunks, vectors = read_local()
    else:
        chunks, vectors = build()
    if args.load or args.load_only:
        load(chunks, vectors)


if __name__ == "__main__":
    main()
