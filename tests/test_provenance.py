"""Result-file provenance: labels come from the rows, never from the environment.

The defect these pin: a submission bundle repackaged under `GRAPH_BACKEND=local`
was labelled `local` over rows every one of which had been answered by Savanna.
"""
from __future__ import annotations

import json

import pytest

from eval2 import run_hidden, run_public
from eval2.provenance import ProvenanceError, check_resumable, of_rows, stamp


def row(qid="h1", backend="tigergraph", model="m", run_at="2026-09-15T14:49:47+0530"):
    return {"qid": qid, "status": "ok", "answer": "x", "backend": backend,
            "model": model, "run_at": run_at, "tokens": {"total": 10}, "trace": [{}, {}]}


def test_stamp_records_backend_model_and_time():
    r = stamp({"qid": "q"}, "tigergraph", "m")
    assert (r["backend"], r["model"]) == ("tigergraph", "m")
    assert r["run_at"][:2] == "20" and "T" in r["run_at"]


def test_of_rows_reads_one_label_and_the_last_write():
    got = of_rows([row(run_at="2026-09-15T14:00:00+0530"), row(run_at="2026-09-15T14:49:47+0530")])
    assert got == {"backend": "tigergraph", "model": "m",
                   "run_finished_at": "2026-09-15T14:49:47+0530"}


def test_of_rows_refuses_a_mix_rather_than_choosing():
    with pytest.raises(ProvenanceError, match="more than one backend"):
        of_rows([row(backend="tigergraph"), row(backend="local")])


def test_of_rows_is_unknown_not_guessed_for_unstamped_rows():
    got = of_rows([{"qid": "old"}])
    assert got == {"backend": "unknown", "model": "unknown", "run_finished_at": None}
    # One unstamped row and the file has no finish time, not a partial one.
    assert of_rows([row(), {"qid": "old", "backend": "tigergraph", "model": "m"}])["run_finished_at"] is None


def test_check_resumable_blocks_the_other_backend_only():
    check_resumable([], "local", "m")
    check_resumable([{"qid": "old"}], "local", "m")          # unstamped rows do not block
    check_resumable([row()], "tigergraph", "m")
    with pytest.raises(ProvenanceError, match="backend=tigergraph"):
        check_resumable([row()], "local", "m")
    with pytest.raises(ProvenanceError, match="model=m"):
        check_resumable([row()], "tigergraph", "other")


def test_bundle_is_labelled_from_its_rows_not_the_environment(monkeypatch):
    monkeypatch.setenv("GRAPH_BACKEND", "local")
    built = run_hidden.bundle([row("h2", run_at="2026-09-15T14:49:47+0530"),
                               row("h1", run_at="2026-09-15T14:40:00+0530")])
    assert built["backend"] == "tigergraph" and built["model"] == "m"
    assert built["run_finished_at"] == "2026-09-15T14:49:47+0530"
    assert built["packaged_at"] > built["run_finished_at"]
    assert [a["qid"] for a in built["answers"]] == ["h1", "h2"]
    # Rows predating multi-pipeline support carry no `pipeline`; they are read
    # as the agent's, because that is the only pipeline that could have written
    # them. Totals are per pipeline -- one average over all three would describe
    # no system that exists.
    assert built["pipelines"] == ["agentic"]
    assert built["totals"]["agentic"] == {
        "questions": 2, "answered": 2, "refused": 0, "total_tokens": 20,
        "avg_tokens": 10.0, "avg_context_tokens": 0.0, "avg_steps": 2.0,
        "avg_latency_s": 0.0, "avg_throttled_s": 0.0}


def test_bundle_keeps_the_three_pipelines_apart():
    """The organisers asked for every question answered by every pipeline.

    A single average over the three would hide the comparison that is the whole
    point, so totals are per pipeline and each question's rows sort together in
    the order the comparison is argued.
    """
    rows = [{**row(qid), "pipeline": name, "tokens": {"total": tokens, "context": tokens}}
            for qid in ("h1", "h2")
            for name, tokens in (("rag", 5000), ("graphrag", 6000), ("agentic", 400))]
    built = run_hidden.bundle(rows)

    assert built["pipelines"] == ["rag", "graphrag", "agentic"]
    assert [t["questions"] for t in built["totals"].values()] == [2, 2, 2]
    assert built["totals"]["agentic"]["avg_tokens"] == 400.0
    assert built["totals"]["graphrag"]["avg_tokens"] == 6000.0
    # Rows for one question stay adjacent, cheapest baseline first.
    assert [(a["qid"], a["pipeline"]) for a in built["answers"]][:3] == [
        ("h1", "rag"), ("h1", "graphrag"), ("h1", "agentic")]


def test_bundle_refuses_rows_from_two_backends():
    with pytest.raises(ProvenanceError):
        run_hidden.bundle([row("h1"), row("h2", backend="local")])


def test_summary_is_labelled_from_its_rows(monkeypatch, tmp_path):
    monkeypatch.setenv("GRAPH_BACKEND", "tigergraph")
    monkeypatch.setattr(run_public, "SUMMARY_FILE", tmp_path / "summary.json")
    scored = {"qtype": "lookup", "exact": True, "lenient": True, "status": "ok",
              "total_tokens": 100, "context_tokens": 5, "steps": 1,
              "grounding_f1": 1.0, "wall_clock_s": 1.0, "active_s": 1.0}
    rows = [stamp({"qid": "p1", "pipeline": p, **scored}, "local", "m")
            for p in ("rag", "graphrag", "agentic")]
    payload = run_public.write_summary(rows)
    on_disk = json.loads((tmp_path / "summary.json").read_text())
    assert on_disk == payload
    assert (payload["backend"], payload["model"], payload["n_rows"]) == ("local", "m", 3)
    assert payload["run_finished_at"] == max(r["run_at"] for r in rows)
    assert set(payload["summary"]) == {"rag", "graphrag", "agentic"}
