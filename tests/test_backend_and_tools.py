"""The local backend and the tool layer over it.

These pin the behaviours the agent's accuracy turned out to hinge on: numeric
predicates arriving as strings, token-wise `contains`, absent values, reverse
traversal, the empty-aggregate status and the field check.
"""
from __future__ import annotations

import numpy as np
import pytest

from agentic.tools.backend import LocalGraphBackend

# ── find ─────────────────────────────────────────────────────────────────────

def test_find_numeric_predicate_as_string_matches(backend):
    as_int = backend.find("Event", [("games_id", "eq", "2014-Winter"), ("competitors", "gt", 63)])
    as_str = backend.find("Event", [("games_id", "eq", "2014-Winter"), ("competitors", "gt", "63")])
    assert [v["id"] for v in as_int] == [v["id"] for v in as_str] == ["E4"]


def test_find_absent_int_never_matches_a_comparison(backend):
    ids = {v["id"] for v in backend.find("Event", [("competitors", "gt", 0)])}
    assert "E5" not in ids            # competitors is None there
    assert {v["id"] for v in backend.find("Event", [("competitors", "exists", False)])} == {"E5"}


def test_find_string_eq_is_folded(backend):
    assert [v["id"] for v in backend.find("Event", [("sport", "eq", "ALPINE SKIING"),
                                                    ("year", "eq", 2014)])] == ["E4", "E5"]
    assert backend.find("Athlete", [("name", "eq", "roland kokeny")])[0]["id"] == "roland-kokeny"


def test_contains_matches_every_token_anywhere(backend):
    # The corpus writes the venue with a joined word; the question spaces it.
    hit = backend.find("Event", [("venue", "contains", "Xiaohaituo Bobsleigh and Luge Track Beijing")])
    assert [v["id"] for v in hit] == ["E6"]
    # Plain substring still works, and tokens in another order still match.
    assert len(backend.find("Event", [("venue", "contains", "ExCeL")])) == 2
    assert [v["id"] for v in backend.find("Event", [("date_raw", "contains", "February 13 2022")])] == ["E6"]
    # A token that is nowhere in the value does not match.
    assert backend.find("Event", [("venue", "contains", "ExCeL Arena")]) == []


def test_find_rejects_unknown_operator(backend):
    with pytest.raises(ValueError):
        backend.find("Event", [("venue", "like", "x")])


# ── aggregate / neighbours / vocabulary ──────────────────────────────────────

def test_aggregate_counts_all_but_reduces_only_present_values(backend):
    agg = backend.aggregate("Event", [("games_id", "eq", "2014-Winter")], "competitors")
    assert agg["matched"] == 2
    assert agg["matched_ids"] == ["E4", "E5"]
    assert (agg["max"], agg["min"], agg["sum"]) == (65, 65, 65)
    empty = backend.aggregate("Event", [("year", "eq", 1800)], "competitors")
    assert empty == {"matched": 0, "matched_ids": [], "sum": None, "max": None, "min": None}


def test_aggregate_takes_the_corpus_value_however_implausible(backend):
    agg = backend.aggregate("Event", [("sport", "eq", "Fencing")], "competitors")
    assert agg["max"] == 41000000


def test_neighbors_out_in_both(backend):
    out = backend.neighbors("Event", "E1", ["WON_GOLD"], "out")
    assert [r["vertex"]["id"] for r in out] == ["yana-shemyakina"]
    assert out[0]["edge"]["noc"] == "UKR"
    incoming = backend.neighbors("Venue", "excel-exhibition-centre", ["HELD_AT"], "in")
    assert sorted(r["vertex"]["id"] for r in incoming) == ["E1", "E2"]
    assert all(r["direction"] == "in" for r in incoming)
    both = backend.neighbors("Games", "2012-Summer", None, "both")
    assert {r["edge"]["type"] for r in both} == {"AT_GAMES", "PRECEDED_BY"}


def test_vocabulary_and_count(backend):
    assert backend.vocabulary("Sport", "name") == ["Alpine skiing", "Fencing"]
    assert backend.count("Event") == 7
    assert backend.count("Nothing") == 0


def test_vector_search_returns_true_top_k_in_order(graph_dir):
    b = LocalGraphBackend(graph_dir)
    b._chunks = [{"id": f"c{i}", "doc_id": f"d{i}", "text": ""} for i in range(4)]
    raw = np.array([[1, 0], [0, 1], [0.6, 0.8], [0, 0]], dtype=np.float32)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    b._vectors = raw / np.where(norms == 0, 1, norms)
    hits = b.vector_search([0, 1], 2)
    assert [h["chunk"]["id"] for h in hits] == ["c1", "c2"]
    assert hits[0]["score"] == pytest.approx(1.0)
    assert hits[1]["score"] == pytest.approx(0.8)
    assert len(b.vector_search([1, 0], 10)) == 4


# ── tools ─────────────────────────────────────────────────────────────────────

def test_graph_filter_rejects_a_field_the_type_lacks(tools):
    r = tools.graph_filter("Games", [("title", "contains", "2008")])
    assert r.status == "error"
    assert "Games has no field 'title'" in r.error
    assert "id, name, year, season" in r.error
    r = tools.graph_filter("Nope", [])
    assert r.status == "error" and "unknown vertex type" in r.error


def test_graph_aggregate_zero_is_empty_not_ok(tools):
    r = tools.graph_aggregate("Event", [("sport", "eq", "Curling")])
    assert r.status == "empty"
    assert r.data["matched"] == 0
    r = tools.graph_aggregate("Event", [("sport", "eq", "Fencing")], "competitors")
    assert r.status == "ok"
    assert "matched_ids" not in r.data
    assert r.doc_ids == []          # E1/E3 ids do not start with Q; only corpus ids are cited


def test_graph_filter_cites_event_documents_only(tools):
    r = tools.graph_filter("Event", [("venue", "contains", "ExCeL")])
    assert sorted(r.doc_ids) == ["E1", "E2"]
    r = tools.graph_filter("Athlete", [])
    assert r.doc_ids == []


def test_entity_link_finds_longest_phrases_by_words(tools):
    r = tools.entity_link("Who won at the ExCeL Exhibition Centre, in fencing, at the 2012 Summer Olympics?")
    found = {(h["type"], h["id"]) for h in r.data}
    assert ("Venue", "excel-exhibition-centre") in found
    assert ("Sport", "fencing") in found
    assert ("Games", "2012-Summer") in found
    assert r.data[0]["type"] == "Venue"          # longest name first
    assert r.latency_ms < 500


def test_entity_link_matches_whole_words_only(tools):
    r = tools.entity_link("every event at the venue")   # "UKR"/"fencing" must not appear
    assert r.data == [] or all(h["type"] != "Nation" for h in r.data)


def test_medal_lookup_and_edition_step(tools):
    r = tools.medal_lookup("E1", "gold")
    assert r.data["athletes"][0]["name"] == "Yana Shemyakina"
    assert r.data["raw"] == "Yana Shemyakina"
    assert r.doc_ids == ["E1"]
    assert tools.medal_lookup("E1", "platinum").status == "error"
    step = tools.edition_step("2012-Summer", "previous")
    assert step.data["games"]["id"] == "2008-Summer"
    assert tools.edition_step("2008-Summer", "previous").status == "empty"
    assert tools.edition_step("2008-Summer", "next").data["games"]["id"] == "2012-Summer"
