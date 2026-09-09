"""
Preprocess dataset JSONL → chunked JSONL for FAISS + TigerGraph ingestion.

Reads from INPUT_PATH, writes to OUTPUT_PATH.
Set INPUT to dataset_100m_enriched.jsonl for Round 2.

Usage:
    python scripts/preprocess.py
    python scripts/preprocess.py --input data/raw/dataset_100m_enriched.jsonl --output data/chunks/chunks.jsonl
"""

import argparse
import json
import os
import re
from pathlib import Path

import tiktoken
from tqdm import tqdm

enc = tiktoken.get_encoding("cl100k_base")


def clean_text(text: str) -> str:
    text = re.sub(r"==+[^=]+=+", "", text)
    text = re.sub(r"\[\[([^\]|]*\|)?([^\]]*)\]\]", r"\2", text)
    text = re.sub(r"\{\{[^}]*\}\}", "", text)  # remove template markup
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text: str, max_tok: int = 256, overlap: int = 32) -> list[str]:
    tokens = enc.encode(text)
    step = max_tok - overlap
    chunks = []
    for start in range(0, len(tokens), step):
        end = min(start + max_tok, len(tokens))
        chunks.append(enc.decode(tokens[start:end]))
        if end == len(tokens):
            break
    return chunks


def main(input_path: str, output_path: str, max_tok: int, overlap: int):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Seek past any leading null bytes (can occur from partial/resumed downloads)
    start_offset = 0
    with open(input_path, "rb") as fb:
        chunk = fb.read(1 << 20)  # read 1 MB
        while chunk:
            idx = chunk.find(b'{')
            if idx != -1:
                start_offset += idx
                break
            start_offset += len(chunk)
            chunk = fb.read(1 << 20)
    if start_offset > 0:
        print(f"Skipping {start_offset:,} leading null bytes")

    # Count non-empty lines for progress bar
    total_lines = 0
    with open(input_path, "rb") as fb:
        fb.seek(start_offset)
        for ln in fb:
            if ln.strip():
                total_lines += 1
    print(f"Input: {input_path} ({total_lines:,} articles)")

    total_chunks = 0
    with open(input_path, "rb") as fb_raw, \
         open(output_path, "w", encoding="utf-8") as fout:
        fb_raw.seek(start_offset)
        fin = (ln.decode("utf-8", errors="replace") for ln in fb_raw)
        for line in tqdm(fin, total=total_lines, desc="Chunking", unit="docs"):
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            cleaned = clean_text(doc["text"])
            if not cleaned:
                continue
            chunks = chunk_text(cleaned, max_tok=max_tok, overlap=overlap)
            for j, chunk in enumerate(chunks):
                record = {
                    "id": f"{doc['id']}_c{j}",
                    "text": chunk,
                    "source": doc["title"],
                    "doc_id": doc["id"],
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_chunks += 1

    print(f"Total chunks: {total_chunks:,}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/dataset_100m_enriched.jsonl")
    parser.add_argument("--output", default="data/chunks/chunks.jsonl")
    parser.add_argument("--max-tok", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=32)
    args = parser.parse_args()
    main(args.input, args.output, args.max_tok, args.overlap)
