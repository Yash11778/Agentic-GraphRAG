"""Facts about the graph that the loader, both backends and the tools must agree on.

Kept in one module because every one of these is a parity hazard: if the loader
and the TigerGraph backend disagree about which fields are integers, or about how
a string is folded before comparison, the two backends answer the same question
differently and the benchmark stops measuring control flow (LOCKED-3).
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

# ── Absence ──────────────────────────────────────────────────────────────────
# TigerGraph has no NULL for a primitive attribute. The loader writes these
# sentinels and the TigerGraph backend maps them back to None, so an absent value
# stays absent rather than becoming a number that predicates can match.
ABSENT_INT = -1
ABSENT_STR = ""

INT_FIELDS = frozenset({
    "year", "competitors", "nations", "date_start", "date_end", "approx_tokens",
    "ordinal", "gap_years",
})

# ── Folding ──────────────────────────────────────────────────────────────────
# Athlete and venue names carry diacritics a question may or may not reproduce
# ("Kökény" vs "Kokeny"). The local backend folds both sides at query time. GSQL's
# lower() handles case but not accents, so the folded form is precomputed at load
# time into a companion attribute and string predicates compare against that.
# Same function, same result, either backend.
FOLD_FIELDS: dict[str, tuple[str, ...]] = {
    "Event": (
        "title", "event_name", "sport", "season", "venue", "date_month",
        "gold_noc", "win_value", "event_key", "title_event_key", "games_id",
        "date_raw", "gold_raw", "silver_raw", "bronze_raw", "competitors_unit",
    ),
    "Games": ("name", "season"),
    "Sport": ("name",),
    "Venue": ("name",),
    "Nation": ("code",),
    "Athlete": ("name",),
    "Document": ("title",),
}

FOLD_SUFFIX = "_fold"

_COMBINING = re.compile(r"\s+")


@lru_cache(maxsize=262_144)
def fold(text: str | None) -> str:
    """Accent- and case-insensitive comparison key.

    Folding makes lookup robust without loosening it into fuzzy matching: two
    strings fold to the same key only when they differ in case, accents, dash
    style or whitespace, never in their letters.

    Cached because the local backend folds every stored value on every scan:
    a filter over 2,187 events folded some 7,000 strings per call, and the same
    7,000 strings the call before. With the cache a scan is a dictionary hit
    per vertex.
    """
    if text is None:
        return ""
    s = unicodedata.normalize("NFKD", str(text))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.replace("–", "-").replace("—", "-").replace("’", "'")
    return _COMBINING.sub(" ", s.lower()).strip()


_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def squash(text: str | None) -> str:
    """The fold with every separator removed: "Dani King, Laura Trott" and
    "Dani KingLaura Trott" squash to the same key.

    Used to recognise an answer that is the corpus's own string with the
    punctuation changed, so it can be handed back verbatim. Two strings that
    squash equal differ only in separators, case or accents -- never in a letter
    or a digit -- which is what makes snapping to the stored form safe.
    """
    return _NON_ALNUM.sub("", fold(text))


def coerce_value(field: str, value):
    """A predicate value in the type the field actually holds.

    Models write numbers as strings about as often as not ("63" for a
    competitor threshold), and a string compared against an integer attribute
    matches nothing. Both backends call this so the same predicate means the
    same thing wherever it runs; the TigerGraph backend would otherwise coerce
    and the local one would not, which is a parity failure in the shape of a
    wrong aggregate.
    """
    if field in INT_FIELDS and isinstance(value, str):
        stripped = value.strip().replace(",", "")
        if re.fullmatch(r"-?\d+", stripped):
            return int(stripped)
    if field in INT_FIELDS and isinstance(value, float) and value.is_integer():
        return int(value)
    return value


# ── Fields per vertex type ───────────────────────────────────────────────────
# What a predicate may name, per type. The tool layer checks a filter against
# this before running it, so a plan that asks a Games vertex for its `title`
# comes back as an error naming the fields that exist, instead of an empty
# result the planner then tries to "loosen" for two more steps. `id` is always
# valid. Derived from build_graph.py; a field added there is added here.
VERTEX_FIELDS: dict[str, tuple[str, ...]] = {
    "Event": (
        "id", "title", "event_name", "sport", "games_id", "year", "season", "venue",
        "competitors", "competitors_unit", "nations", "date_raw", "date_start",
        "date_end", "date_month", "gold_raw", "silver_raw", "bronze_raw", "gold_noc",
        "win_value", "event_key", "title_event_key", "url", "approx_tokens",
    ),
    "Games": ("id", "name", "year", "season"),
    "Sport": ("id", "name"),
    "Venue": ("id", "name"),
    "Athlete": ("id", "name"),
    "Nation": ("id", "code"),
    "Document": ("id", "title", "url", "infobox", "approx_tokens"),
}

# The subset worth describing to the planner. The rest are keys and bookkeeping.
PLANNER_FIELDS: dict[str, str] = {
    "Event": ("title, event_name, sport, games_id (e.g. 2012-Summer), year, season, "
              "venue, date_raw, competitors, nations, gold_raw, silver_raw, "
              "bronze_raw, gold_noc, win_value"),
    "Games": "id (e.g. 2012-Summer), name, year, season",
    "Sport": "id, name",
    "Venue": "id, name",
    "Athlete": "id, name",
    "Nation": "id, code",
}


def unknown_fields(vtype: str, fields) -> list[str]:
    """Fields a predicate names that the vertex type does not have."""
    known = VERTEX_FIELDS.get(vtype)
    if known is None:
        return []
    return [f for f in fields if f not in known]


def folded_field(vtype: str, field: str) -> str | None:
    """The companion attribute holding the folded form, if one was loaded."""
    if field in FOLD_FIELDS.get(vtype, ()):
        return field + FOLD_SUFFIX
    return None


# ── Vectors ─────────────────────────────────────────────────────────────────
# A local ONNX model: no API key, no per-call cost, and the same vectors on every
# machine, which matters because the benchmark must be reproducible by a judge.
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384

# Chunk size is a retrieval-quality choice, not a storage one. 400 tokens holds a
# complete infobox plus the opening prose of an event article, which is where the
# answerable facts are; the overlap keeps a fact that straddles a boundary
# retrievable from either side.
CHUNK_TOKENS = 400
CHUNK_OVERLAP = 60


# How many matched ids an aggregate returns as citable evidence. Sorted before
# capping so both backends return the same subset of a large match.
AGGREGATE_ID_CAP = 100


# ── Edges ────────────────────────────────────────────────────────────────────
# Which attributes each edge type actually carries. The GSQL neighbour query
# returns one flat row shape for every edge type, so the backend uses this to
# drop the attributes an edge does not have -- otherwise a WON_GOLD edge would
# come back carrying an empty `event_id` that the local backend never produces,
# and the two backends would stop comparing equal.
EDGE_ATTRS: dict[str, tuple[str, ...]] = {
    "AT_GAMES": (),
    "IN_SPORT": (),
    "HELD_AT": (),
    "WON_GOLD": ("medal", "noc"),
    "WON_SILVER": ("medal", "noc"),
    "WON_BRONZE": ("medal", "noc"),
    "REPRESENTS": ("event_id", "medal"),
    "PRECEDED_BY": ("gap_years",),
}

REVERSE_PREFIX = "rev_"


def reverse_edge(edge_type: str) -> str:
    """The declared reverse of an edge type. Walking a reverse edge outward is
    how an "in" traversal is expressed; see the note in schema.gsql."""
    return REVERSE_PREFIX + edge_type
