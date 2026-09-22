"""Where a result row came from: which backend answered it, which model, and when.

Result files are appended to across sessions, and the bundle or summary written
from one used to take its `backend` label from whatever `GRAPH_BACKEND` was set
to at packaging time. That is how a submission bundle came to say `local` over
fifty answers every one of whose tool calls had gone to Savanna: the rows were
right, the label was read from the wrong place.

The rows are the record. Each one is stamped when it is written, every
file-level label is derived from the rows, and a file whose rows disagree is
refused rather than labelled with a guess. Nothing here touches an answer.
"""
from __future__ import annotations

import time
from collections.abc import Iterable

FIELDS = ("backend", "model")
UNKNOWN = "unknown"


class ProvenanceError(RuntimeError):
    """Rows that cannot honestly share one label."""


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def stamp(row: dict, backend: str, model: str) -> dict:
    """Label one result row with where and when it was produced."""
    row["backend"] = backend
    row["model"] = model
    row["run_at"] = now()
    return row


def of_rows(rows: Iterable[dict]) -> dict:
    """The one backend and model every row carries, and when the last was written.

    Raises rather than picking a winner when the rows disagree: a benchmark file
    that mixes backends cannot be summarised under one label, and a submission
    bundle that mixes them is not one run.
    """
    rows = list(rows)
    out: dict = {}
    for field in FIELDS:
        values = sorted({str(r.get(field) or UNKNOWN) for r in rows})
        if len(values) > 1:
            raise ProvenanceError(
                f"rows carry more than one {field} ({', '.join(values)}); a file that "
                f"mixes runs cannot be labelled honestly. Start it again with --fresh."
            )
        out[field] = values[0] if values else UNKNOWN
    stamps = [r.get("run_at") for r in rows]
    # Rows written before the stamp existed have no time; the file then has none
    # rather than a guessed one.
    out["run_finished_at"] = max(stamps) if rows and all(stamps) else None
    return out


def check_resumable(done: Iterable[dict], backend: str, model: str) -> None:
    """Refuse to append this run's rows to a file that holds another run's.

    Called before a resume. Rows from before the stamp existed are `unknown`
    and do not block, since there is nothing to contradict.
    """
    done = list(done)
    if not done:
        return
    have = of_rows(done)
    for field, want in (("backend", backend), ("model", model)):
        if have[field] not in (UNKNOWN, want):
            raise ProvenanceError(
                f"this file was written with {field}={have[field]} and the current run "
                f"would use {want}. Set it to match, write elsewhere with --out, or "
                f"start again with --fresh."
            )
