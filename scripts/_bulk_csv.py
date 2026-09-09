"""Shared helper for streaming CourtListener's public bulk-data bz2 CSV files
without loading them into memory. No auth required (anonymous S3 GET).

Correctness note: opinion/cluster text fields can contain embedded newlines
inside quoted CSV fields. We must NOT split the decompressed byte stream on
'\\n' and hand chunks to csv.reader independently -- that truncates multi-line
quoted fields. Instead we feed csv.reader a *line iterator* (one physical line
per yield); csv.reader pulls additional physical lines from that iterator on
its own when it detects it's still inside an open quote, which correctly
reassembles multi-line fields. This only works if csv.reader drives the
iteration itself (as done here), not if we pre-split into a list of "records".
"""
import csv
import bz2
import re
import time

import requests

BUCKET = "https://com-courtlistener-storage.s3-us-west-2.amazonaws.com"
csv.field_size_limit(10_000_000)  # opinion plain_text can be long

MAX_RETRIES = 8


def latest_bulk_file(prefix: str) -> str:
    resp = requests.get(f"{BUCKET}/", params={
        "list-type": "2", "prefix": f"bulk-data/{prefix}", "max-keys": "1000",
    }, timeout=30)
    resp.raise_for_status()
    keys = re.findall(r"<Key>([^<]+)</Key>", resp.text)
    if not keys:
        raise RuntimeError(f"No bulk-data files found for prefix {prefix!r}")
    return sorted(keys)[-1]


def _iter_decompressed_lines(url: str):
    """Streams and decompresses a bz2 file, transparently resuming via HTTP
    Range requests if the connection drops mid-download (these files are
    multi-GB and long-lived streams over consumer networks WILL drop
    occasionally -- without resume, a hiccup near the end wastes the entire
    download). The bz2 decompressor and partial-line buffer are kept across
    retries so no already-processed data is lost or reprocessed.
    """
    decompressor = bz2.BZ2Decompressor()
    buffer = ""
    bytes_read = 0
    attempt = 0

    while True:
        headers = {"Range": f"bytes={bytes_read}-"} if bytes_read else {}
        try:
            r = requests.get(url, stream=True, timeout=120, headers=headers)
            r.raise_for_status()
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                bytes_read += len(chunk)
                buffer += decompressor.decompress(chunk).decode("utf-8", errors="replace")
                *lines, buffer = buffer.split("\n")
                for line in lines:
                    yield line
                attempt = 0  # reset backoff after any successful progress
            break  # stream finished cleanly
        except (requests.exceptions.ChunkedEncodingError,
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout) as e:
            attempt += 1
            if attempt > MAX_RETRIES:
                raise RuntimeError(f"Giving up after {MAX_RETRIES} retries at byte {bytes_read}: {e}") from e
            wait = min(60, 3 * (2 ** attempt))
            print(f"  [connection dropped at byte {bytes_read:,}] retrying in {wait}s "
                  f"(attempt {attempt}/{MAX_RETRIES})...", flush=True)
            time.sleep(wait)

    if buffer:
        yield buffer


def stream_csv_dicts(url: str):
    """Yields each row of a bulk CSV as a dict, streaming — never holds the
    full (decompressed, multi-GB) file in memory.

    Malformed rows (CourtListener's export occasionally has a stray unescaped
    newline or unbalanced quote in free-text fields like plain_text) are
    skipped rather than crashing the whole multi-hour download -- a handful
    of dropped rows out of millions is an acceptable trade for resilience.
    """
    reader = csv.DictReader(_iter_decompressed_lines(url))
    it = iter(reader)
    skipped = 0
    while True:
        try:
            row = next(it)
        except StopIteration:
            break
        except csv.Error:
            skipped += 1
            if skipped <= 20 or skipped % 500 == 0:
                print(f"  [skipping malformed CSV row #{skipped}]", flush=True)
            continue
        yield row
