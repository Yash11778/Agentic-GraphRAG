"""Deterministic infobox + title parsing (LOCKED-4 in docs/ARCHITECTURE.md).

The corpus renders each article's infobox as an indented key: value block at the
top of `text`, terminated by the first blank line. Olympic event articles carry a
`[Infobox Olympic event]` marker; everything else is a distractor for this
benchmark and is parsed only far enough to be chunked and embedded.

No LLM is involved here, and nothing is imputed: a field that is absent stays
absent so that downstream coverage numbers stay honest.
"""
from __future__ import annotations

import re

OLYMPIC_MARKER = "[Infobox Olympic event]"

_FIELD_RE = re.compile(r"^\s+([A-Za-z_][A-Za-z0-9_]*):\s*(.*)$")
_INFOBOX_RE = re.compile(r"^\[Infobox ([^\]]+)\]")

# "Sport at the 2012 Summer Olympics – Men's K-2 1000 metres". The dash is an
# en dash (U+2013) in every title that has one; a few also use a hyphen, so both
# are accepted. Split on the FIRST separator only -- event names contain dashes
# of their own ("K-2", "Men's 20 kilometres walk - final").
_TITLE_RE = re.compile(
    r"^(?P<sport>.+?)\s+at\s+the\s+(?P<year>\d{4})\s+(?P<season>Summer|Winter)"
    r"\s+(?:Youth\s+)?Olympics\s*[–—-]\s*(?P<event>.+)$"
)

# competitors is usually a bare int, but ~23 team events read "23 teams".
_COUNT_RE = re.compile(r"^\s*(\d[\d,]*)\s*(.*)$")


def infobox_type(text: str) -> str | None:
    """The infobox kind declared on the first line, or None if there is none."""
    m = _INFOBOX_RE.match(text)
    return m.group(1) if m else None


def is_olympic_event(text: str) -> bool:
    """Whether the document carries an Olympic event infobox anywhere.

    Not `startswith`. Twenty-five tennis articles open with a second infobox --
    `[Infobox tennis tournament event]` -- and carry the Olympic one below it. A
    leading-block-only test drops every one of them into the distractor pile,
    taking their venues, dates and medals with them.
    """
    return OLYMPIC_MARKER in text


def parse_infobox(text: str) -> dict[str, str]:
    """Key/value pairs from the Olympic infobox block.

    Starts at the Olympic marker wherever it appears, and stops at the first
    blank line after it: that is where the infobox ends and prose begins, and
    prose lines are never indented `key: value` pairs. Documents with no Olympic
    marker are parsed from their leading block instead, which is all a distractor
    needs.
    """
    fields: dict[str, str] = {}
    if OLYMPIC_MARKER in text:
        text = text[text.index(OLYMPIC_MARKER):]
    lines = text.split("\n")
    if not lines or not _INFOBOX_RE.match(lines[0]):
        return fields
    for line in lines[1:]:
        if not line.strip():
            break
        m = _FIELD_RE.match(line)
        if m:
            key, value = m.group(1), m.group(2).strip()
            if value:
                fields[key] = value
    return fields


def parse_count(raw: str | None) -> tuple[int | None, str | None]:
    """'87' -> (87, None);  '23 teams' -> (23, 'teams');  missing -> (None, None).

    The unit is kept rather than dropped: a count of teams is not a count of
    competitors, and an aggregation that silently mixes them would be wrong in a
    way no test would catch.
    """
    if not raw:
        return None, None
    m = _COUNT_RE.match(raw)
    if not m:
        return None, None
    value = int(m.group(1).replace(",", ""))
    unit = m.group(2).strip().lower() or None
    return value, unit


def parse_title(title: str) -> dict[str, str] | None:
    """Split an Olympic event title into sport, games and event name."""
    m = _TITLE_RE.match(title)
    if not m:
        return None
    return {
        "sport": m.group("sport").strip(),
        "year": m.group("year"),
        "season": m.group("season"),
        "event_name": m.group("event").strip(),
        "games_id": f"{m.group('year')}-{m.group('season')}",
    }


def parse_games(raw: str | None) -> dict[str, str] | None:
    """Infobox `games: "2012 Summer"` -> Games vertex fields."""
    if not raw:
        return None
    m = re.match(r"^(\d{4})\s+(Summer|Winter)", raw.strip())
    if not m:
        return None
    return {"games_id": f"{m.group(1)}-{m.group(2)}",
            "year": m.group(1), "season": m.group(2)}


# Surname prefixes that carry an internal capital ("MacLennan", "McDonald",
# "O'Brien"). Without these, the team-name splitter cuts people in half.
_NAME_PREFIXES = ("mac", "mc", "o'", "d'", "fitz", "de", "van", "von", "le",
                  "la", "du", "da", "di", "del", "der", "st", "ter")


def normalize_event_name(name: str) -> str:
    """Comparison key for an event name.

    Infoboxes write "Men's 68kg" while questions write "men's 68 kg"; both must
    reach the same event. Digit/unit spacing is normalised, punctuation dropped.
    """
    import unicodedata
    s = unicodedata.normalize("NFKD", name or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().replace("\u2013", "-").replace("\u2014", "-").replace("\u2019", "'")
    s = re.sub(r"(\d)\s*(kg|m|km|metres|meters|mm)\b", r"\1 \2", s)
    s = re.sub(r"[^a-z0-9'+\- ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()
