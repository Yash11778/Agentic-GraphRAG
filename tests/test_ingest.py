"""Deterministic ingestion (LOCKED-4): the parser is the only place a corpus
fact becomes a graph fact, so each rule it applies is pinned here."""
from __future__ import annotations

from agentic.ingest.build_graph import parse_dates, slug, split_names
from agentic.ingest.parse_infobox import (
    is_olympic_event,
    normalize_event_name,
    parse_count,
    parse_games,
    parse_infobox,
    parse_title,
)

TENNIS_DOC = """[Infobox tennis tournament event]
  name: Women's doubles
  year: 2004

[Infobox Olympic event]
  event: Women's doubles
  games: 2004 Summer
  venue: Athens Olympic Tennis Centre, Athens
  dates: 15 to 22 August 2004
  competitors: 64
  nations: 24
  gold: Li TingSun Tiantian
  goldNOC: CHN

The women's doubles tennis event was held ...
  not_a_field: because the infobox ended at the blank line
"""


def test_olympic_infobox_is_found_below_a_leading_infobox():
    assert is_olympic_event(TENNIS_DOC)
    fields = parse_infobox(TENNIS_DOC)
    assert fields["venue"] == "Athens Olympic Tennis Centre, Athens"
    assert fields["gold"] == "Li TingSun Tiantian"
    assert "not_a_field" not in fields


def test_non_olympic_document_parses_its_leading_block_only():
    text = "[Infobox film]\n  director: Someone\n\nProse."
    assert not is_olympic_event(text)
    assert parse_infobox(text) == {"director": "Someone"}
    assert parse_infobox("No infobox at all") == {}


def test_parse_title_splits_on_the_first_separator_only():
    parts = parse_title("Canoeing at the 2012 Summer Olympics – Men's K-2 1000 metres")
    assert parts == {"sport": "Canoeing", "year": "2012", "season": "Summer",
                     "event_name": "Men's K-2 1000 metres", "games_id": "2012-Summer"}
    assert parse_title("Athletics at the 2010 Summer Youth Olympics – Boys' 100 metres")["sport"] == "Athletics"
    assert parse_title("Not an event title") is None


def test_parse_count_keeps_units_and_never_imputes():
    assert parse_count("87") == (87, None)
    assert parse_count("23 teams") == (23, "teams")
    assert parse_count("32 (16 pairs)") == (32, "(16 pairs)")
    assert parse_count("1,024") == (1024, None)
    assert parse_count(None) == (None, None)
    assert parse_count("unknown") == (None, None)


def test_parse_games():
    assert parse_games("2012 Summer") == {"games_id": "2012-Summer", "year": "2012", "season": "Summer"}
    assert parse_games("XXX Olympiad") is None
    assert parse_games(None) is None


def test_normalize_event_name_unifies_digit_unit_spacing():
    assert normalize_event_name("Men's 68kg") == normalize_event_name("men's 68 kg")
    assert normalize_event_name("Women's 10 kilometre") == "women's 10 kilometre"


def test_parse_dates_keeps_raw_and_extracts_bounds():
    out = parse_dates({"dates": "4–5 August 2012"})
    assert out["date_raw"] == "4–5 August 2012"
    assert (out["date_start"], out["date_end"], out["date_month"]) == (4, 5, "August")
    assert parse_dates({}) == {"date_raw": None, "date_start": None, "date_end": None, "date_month": None}


def test_split_names_respects_surname_prefixes():
    assert split_names("Rudolf DombiRoland Kökény") == ["Rudolf Dombi", "Roland Kökény"]
    assert split_names("Rosannagh MacLennan") == ["Rosannagh MacLennan"]
    assert split_names("A One, B Two and C Three") == ["A One", "B Two", "C Three"]


def test_slug_is_stable_and_ascii():
    assert slug("Roland Kökény") == "roland-kkny"
    assert slug("  ExCeL Exhibition Centre ") == "excel-exhibition-centre"
    assert slug("***") == "unknown"
