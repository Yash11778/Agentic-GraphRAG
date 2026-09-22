"""Shared fixtures: a seven-event graph small enough to reason about by hand.

Every test here runs without a network, a model, or the real corpus. The tiny
graph exercises the same code paths as the 2,187-event one -- folded string
matching, integer predicates with absent values, reverse traversal, the edition
chain -- which is what makes it a useful oracle for the tool layer.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agentic.tools.backend import LocalGraphBackend  # noqa: E402
from agentic.tools.tools import Tools  # noqa: E402

VERTICES = [
    {"type": "Games", "id": "2012-Summer", "name": "2012 Summer Olympics", "year": 2012, "season": "Summer"},
    {"type": "Games", "id": "2008-Summer", "name": "2008 Summer Olympics", "year": 2008, "season": "Summer"},
    {"type": "Games", "id": "2014-Winter", "name": "2014 Winter Olympics", "year": 2014, "season": "Winter"},
    {"type": "Sport", "id": "fencing", "name": "Fencing"},
    {"type": "Sport", "id": "alpine-skiing", "name": "Alpine skiing"},
    {"type": "Venue", "id": "excel-exhibition-centre", "name": "ExCeL Exhibition Centre"},
    {"type": "Venue", "id": "xiaohaituo", "name": "Xiaohaituo Bobsleigh and Luge TrackBeijing"},
    {"type": "Athlete", "id": "yana-shemyakina", "name": "Yana Shemyakina"},
    {"type": "Athlete", "id": "roland-kokeny", "name": "Roland Kökény"},
    {"type": "Nation", "id": "UKR", "code": "UKR"},
    {"type": "Event", "id": "E1", "title": "Fencing at the 2012 Summer Olympics – Women's épée",
     "event_name": "Women's épée", "sport": "Fencing", "games_id": "2012-Summer", "year": 2012,
     "season": "Summer", "venue": "ExCeL Exhibition Centre", "date_raw": "30 July",
     "competitors": 37, "nations": 23, "gold_raw": "Yana Shemyakina", "silver_raw": "Britta Heidemann",
     "bronze_raw": "Sun Yujie", "url": "http://x", "approx_tokens": 1000},
    {"type": "Event", "id": "E2", "title": "Judo at the 2012 Summer Olympics – Women's 57 kg",
     "event_name": "Women's 57 kg", "sport": "Judo", "games_id": "2012-Summer", "year": 2012,
     "season": "Summer", "venue": "ExCeL Exhibition Centre", "date_raw": "30 July 2012",
     "competitors": 24, "nations": 24, "gold_raw": "Kaori Matsumoto"},
    {"type": "Event", "id": "E3", "title": "Fencing at the 2008 Summer Olympics – Men's épée",
     "event_name": "Men's épée", "sport": "Fencing", "games_id": "2008-Summer", "year": 2008,
     "season": "Summer", "venue": "Fencing Hall", "date_raw": "10 August 2008",
     "competitors": 41000000, "nations": 23, "gold_raw": "Matteo Tagliariol"},
    {"type": "Event", "id": "E4", "title": "Alpine skiing at the 2014 Winter Olympics – Men's downhill",
     "event_name": "Men's downhill", "sport": "Alpine skiing", "games_id": "2014-Winter", "year": 2014,
     "season": "Winter", "venue": "Rosa Khutor", "date_raw": "9 February 2014",
     "competitors": 65, "nations": 27, "gold_raw": "Matthias Mayer"},
    {"type": "Event", "id": "E5", "title": "Alpine skiing at the 2014 Winter Olympics – Women's slalom",
     "event_name": "Women's slalom", "sport": "Alpine skiing", "games_id": "2014-Winter", "year": 2014,
     "season": "Winter", "venue": "Rosa Khutor", "date_raw": "21 February 2014",
     "competitors": None, "nations": None, "gold_raw": "Mikaela Shiffrin"},
    {"type": "Event", "id": "E6", "title": "Bobsleigh at the 2022 Winter Olympics – Women's monobob",
     "event_name": "Women's monobob", "sport": "Bobsleigh", "games_id": "2022-Winter", "year": 2022,
     "season": "Winter", "venue": "Xiaohaituo Bobsleigh and Luge TrackBeijing",
     "date_raw": "13, 14 February 2022", "competitors": 20, "nations": 12,
     "gold_raw": "Kaillie Humphries"},
    {"type": "Event", "id": "E7", "title": "Cycling at the 2012 Summer Olympics – Women's team pursuit",
     "event_name": "Women's team pursuit", "sport": "Cycling", "games_id": "2012-Summer", "year": 2012,
     "season": "Summer", "venue": "London Velopark", "date_raw": "3 to 4 August",
     "competitors": 30, "nations": 10, "gold_raw": "Dani KingLaura TrottJoanna Rowsell"},
    {"type": "Document", "id": "D1", "title": "Some film", "url": "", "infobox": "film", "approx_tokens": 5},
]

EDGES = [
    {"type": "AT_GAMES", "from_type": "Event", "from": "E1", "to_type": "Games", "to": "2012-Summer"},
    {"type": "AT_GAMES", "from_type": "Event", "from": "E2", "to_type": "Games", "to": "2012-Summer"},
    {"type": "AT_GAMES", "from_type": "Event", "from": "E3", "to_type": "Games", "to": "2008-Summer"},
    {"type": "IN_SPORT", "from_type": "Event", "from": "E1", "to_type": "Sport", "to": "fencing"},
    {"type": "IN_SPORT", "from_type": "Event", "from": "E3", "to_type": "Sport", "to": "fencing"},
    {"type": "HELD_AT", "from_type": "Event", "from": "E1", "to_type": "Venue", "to": "excel-exhibition-centre"},
    {"type": "HELD_AT", "from_type": "Event", "from": "E2", "to_type": "Venue", "to": "excel-exhibition-centre"},
    {"type": "WON_GOLD", "from_type": "Event", "from": "E1", "to_type": "Athlete", "to": "yana-shemyakina",
     "medal": "gold", "noc": "UKR"},
    {"type": "REPRESENTS", "from_type": "Athlete", "from": "yana-shemyakina", "to_type": "Nation", "to": "UKR",
     "event_id": "E1", "medal": "gold"},
    {"type": "PRECEDED_BY", "from_type": "Games", "from": "2012-Summer", "to_type": "Games", "to": "2008-Summer",
     "gap_years": 4},
    {"type": "PRECEDED_BY", "from_type": "Event", "from": "E1", "to_type": "Event", "to": "E3", "gap_years": 4},
]


@pytest.fixture(scope="session")
def graph_dir(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("graph")
    with (path / "vertices.jsonl").open("w", encoding="utf-8") as f:
        for v in VERTICES:
            f.write(json.dumps(v, ensure_ascii=False) + "\n")
    with (path / "edges.jsonl").open("w", encoding="utf-8") as f:
        for e in EDGES:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    return path


@pytest.fixture(scope="session")
def backend(graph_dir) -> LocalGraphBackend:
    return LocalGraphBackend(graph_dir)


@pytest.fixture(scope="session")
def tools(backend) -> Tools:
    return Tools(backend)
