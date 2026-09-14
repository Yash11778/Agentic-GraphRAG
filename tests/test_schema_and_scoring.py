"""graph_schema helpers and the scorer: the two places a wrong normalisation
would silently change every number in the benchmark."""
from __future__ import annotations

from agentic.graph_schema import coerce_value, fold, squash, unknown_fields
from eval2.score import (
    _median,
    _verdict,
    cost_of_agency,
    exact_match,
    grounding,
    lenient_match,
    normalise,
)

# ── fold / squash / coerce ──────────────────────────────────────────────────

def test_fold_is_accent_case_and_dash_insensitive():
    assert fold("Kökény") == fold("KOKENY") == "kokeny"
    assert fold("Men's K-2 – 1000") == "men's k-2 - 1000"
    assert fold("  two   spaces ") == "two spaces"
    assert fold(None) == ""


def test_fold_never_changes_letters():
    assert fold("Massu") != fold("Massú") or fold("Massú") == "massu"
    assert fold("London") != fold("Londo")


def test_squash_removes_separators_only():
    assert squash("Dani King, Laura Trott, Joanna Rowsell") == squash("Dani KingLaura TrottJoanna Rowsell")
    assert squash("Erik Lesser, Daniel Böhm") == "eriklesserdanielbohm"
    assert squash("24") != squash("42")


def test_coerce_value_turns_numeric_strings_into_ints_for_int_fields():
    assert coerce_value("competitors", "63") == 63
    assert coerce_value("competitors", " 1,200 ") == 1200
    assert coerce_value("year", 2012.0) == 2012
    assert coerce_value("competitors", "many") == "many"
    # String fields are never coerced: a venue called "2012" stays a string.
    assert coerce_value("venue", "2012") == "2012"


def test_unknown_fields_reports_only_missing_ones():
    assert unknown_fields("Games", ["title", "year"]) == ["title"]
    assert unknown_fields("Event", ["venue", "id"]) == []
    # An unknown type is not this function's business.
    assert unknown_fields("Nope", ["x"]) == []


# ── scorer ──────────────────────────────────────────────────────────────────

def test_normalise_strips_what_no_reader_would_call_a_difference():
    assert normalise("The Men's Marathon.") == "men's marathon"
    assert normalise("  Four ") == "4"
    assert normalise("Athletics – Men's 100 m") == normalise("Athletics - Men's 100 m")


def test_normalise_keeps_accents_and_apostrophe_variants():
    # Deliberately strict: the organisers' scorer is unknown, so the public
    # number must not be flattered by a normalisation they may not apply. The
    # agent handles apostrophe variants at the source, via the verbatim check.
    assert normalise("Massú") != normalise("Massu")
    assert normalise("Men’s épée") != normalise("Men's épée")


def test_exact_and_lenient_match():
    assert exact_match("4", ["4"])
    assert exact_match("four", ["4"])
    assert not exact_match("4 events", ["4"])
    assert lenient_match("The answer is 4", ["4"])
    assert not lenient_match("", ["4"])
    assert not exact_match("Dani King, Laura Trott", ["Dani KingLaura Trott"])


def test_grounding_precision_recall_f1():
    g = grounding(["a", "b", "x"], ["a", "b", "c", "d"])
    assert g["precision"] == round(2 / 3, 4)
    assert g["recall"] == 0.5
    assert g["cited"] == 3
    assert grounding([], ["a"])["f1"] == 0.0
    assert grounding(["a"], [])["f1"] == 0.0


def test_verdict_labels():
    assert _verdict(0.5, 300) == "worth the cost"
    assert _verdict(0.5, -300) == "strictly better"
    assert _verdict(-0.1, -300) == "worse and cheaper"
    assert _verdict(-0.1, 300) == "strictly worse"
    assert _verdict(0.0, 300) == "overkill"
    assert _verdict(0.0, -300) == "no gain, cheaper"


def test_cost_of_agency_per_qtype():
    summary = {
        "graphrag": {"by_qtype": {"lookup": {"exact": 0.8, "avg_total_tokens": 5000}}},
        "agentic": {"by_qtype": {"lookup": {"exact": 0.9, "avg_total_tokens": 4000},
                                 "extra": {"exact": 1.0, "avg_total_tokens": 1}}},
    }
    out = cost_of_agency(summary)
    assert set(out) == {"lookup"}
    assert out["lookup"]["delta_tokens"] == -1000
    assert out["lookup"]["verdict"] == "strictly better"
    assert out["lookup"]["exact_per_1k_tokens"] is None
    assert cost_of_agency({"agentic": {}}) == {}


def test_median():
    assert _median([]) == 0.0
    assert _median([3, 1, 2]) == 2
    assert _median([1, 2, 3, 4]) == 2.5
