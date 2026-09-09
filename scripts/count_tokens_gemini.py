"""
Count tokens in dataset_100m_enriched.jsonl using Gemini's count_tokens API.
Required by hackathon judges to document official token count.

Usage:
    python scripts/count_tokens_gemini.py
    python scripts/count_tokens_gemini.py --input data/raw/dataset_100m_enriched.jsonl
"""

import argparse
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

ROOT = Path(__file__).parent.parent.resolve()


def main(input_path: str, model: str, batch_size: int):
    from google import genai

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set in .env")

    client = genai.Client(api_key=api_key)

    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Dataset not found: {input_file}")

    print(f"Counting tokens in: {input_file}")
    print(f"Model: {model} | Batch size: {batch_size} articles per call\n")

    # Seek past leading null bytes then stream articles — avoids loading all into RAM
    start_offset = 0
    with open(input_file, "rb") as fb:
        buf = fb.read(1 << 20)
        while buf:
            idx = buf.find(b'{')
            if idx != -1:
                start_offset += idx
                break
            start_offset += len(buf)
            buf = fb.read(1 << 20)

    total_tokens = 0
    total_articles = 0
    errors = 0
    batch: list[str] = []
    batch_idx = 0

    def flush_batch(b: list[str]) -> int:
        nonlocal errors
        combined = "\n\n".join(b)
        try:
            resp = client.models.count_tokens(model=model, contents=combined)
            return resp.total_tokens
        except Exception as e:
            print(f"  Batch {batch_idx} error: {e} — using tiktoken estimate")
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            errors += 1
            time.sleep(2)
            # encode per-text to avoid OOM on large batches
            return sum(len(enc.encode(t)) for t in b)

    with open(input_file, "rb") as fb:
        fb.seek(start_offset)
        for raw_line in fb:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            d = json.loads(raw_line.decode("utf-8", errors="replace"))
            batch.append(d.get("text", ""))
            total_articles += 1

            if len(batch) == batch_size:
                total_tokens += flush_batch(batch)
                batch_idx += 1
                batch = []
                if batch_idx % 10 == 0:
                    print(f"  {total_articles:,} articles — {total_tokens:,} tokens", flush=True)
                    time.sleep(0.5)

    if batch:
        total_tokens += flush_batch(batch)
        batch_idx += 1

    print(f"\n{'='*50}")
    print(f"Official Gemini token count : {total_tokens:,}")
    print(f"In millions                 : {total_tokens/1e6:.2f}M")
    print(f"Articles counted            : {total_articles:,}")
    print(f"Model used                  : {model}")
    print(f"API errors (fell back)      : {errors}")
    print(f"{'='*50}")

    result_file = ROOT / "data" / "token_count_official.json"
    with open(result_file, "w") as f:
        json.dump({
            "total_tokens": total_tokens,
            "total_articles": total_articles,
            "model": model,
            "input_file": str(input_file),
            "errors": errors,
        }, f, indent=2)
    print(f"\nSaved to: {result_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/dataset_100m_enriched.jsonl")
    parser.add_argument("--model", default=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
    parser.add_argument("--batch-size", type=int, default=50,
                        help="Articles per count_tokens API call (reduce if hitting limits)")
    args = parser.parse_args()
    main(args.input, args.model, args.batch_size)
