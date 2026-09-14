"""The evidence ledger: what the investigation holds, and whether it is complete.

Deduplicates evidence by document and, separately, tracks completeness of any set
the agent enumerated. LOCKED-5 forbids answering from an incomplete evidence set,
and "incomplete" only means something against a specific claim.

Completeness is keyed by the predicate signature, not by a global count. That
distinction is the whole correctness of the gate: `graph_aggregate` counting 238
events in 2012 says nothing about whether a later one-row lookup for the largest
of them is complete, and a gate that compared those two numbers would block every
superlative answer. A set is incomplete only when the agent enumerated *the same*
predicates the database counted and came back with fewer rows.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agentic.llm import count_tokens
from agentic.tools.tools import Evidence


def signature(vertex_type: str, predicates: Any) -> str:
    """A stable key for one filter, so a count and a listing can be compared.

    Normalised through JSON with sorted parts because the model will not write
    the same predicates in the same order twice, and an ordering difference must
    not read as a different question.
    """
    rows = []
    for item in (predicates or []):
        if isinstance(item, dict):
            field_name = item.get("field")
            operator = item.get("operator") or item.get("op")
            value = item.get("value")
        else:
            parts = list(item)[:3] + [None, None, None]
            field_name, operator, value = parts[0], parts[1], parts[2]
        rows.append([str(field_name).lower(), str(operator).lower(),
                     str(value).lower()])
    rows.sort()
    return json.dumps([vertex_type.lower(), rows], separators=(",", ":"))


@dataclass
class EvidenceLedger:
    """Accumulated evidence, deduplicated by document id."""

    items: dict[str, Evidence] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)

    # signature → how many the database says exist / how many were listed.
    expectations: dict[str, int] = field(default_factory=dict)
    enumerations: dict[str, int] = field(default_factory=dict)
    expectation_source: str = ""

    def add(self, evidence: list[Evidence]) -> int:
        """Add evidence, returning how many documents were genuinely new.

        The return value is what tells the orchestrator a step made no progress,
        which is one of the five stop criteria. A step that re-retrieves what is
        already held has cost tokens and added nothing, and the loop must be able
        to notice rather than repeat it until the budget runs out.
        """
        added = 0
        for item in evidence:
            if not item.doc_id or item.doc_id in self.items:
                continue
            self.items[item.doc_id] = item
            self.order.append(item.doc_id)
            added += 1
        return added

    def expect(self, key: str, count: int, source: str) -> None:
        """Record the database's own count for one filter.

        Only ever set from `graph_aggregate`, never from the model's estimate:
        the point of the gate is that the database decides what complete means.
        """
        self.expectations[key] = count
        self.expectation_source = source

    def enumerated(self, key: str, count: int) -> None:
        """Record how many rows a listing of the same filter returned."""
        self.enumerations[key] = max(count, self.enumerations.get(key, 0))

    @property
    def incomplete(self) -> tuple[str, int, int] | None:
        """The first enumeration that fell short of its counted size."""
        for key, expected in self.expectations.items():
            listed = self.enumerations.get(key)
            if listed is not None and listed < expected:
                return key, expected, listed
        return None

    @property
    def complete(self) -> bool | None:
        """True, False, or None when no claim can be checked either way."""
        comparable = [k for k in self.expectations if k in self.enumerations]
        if not comparable:
            return None
        return self.incomplete is None

    @property
    def expected_count(self) -> int | None:
        """The largest counted set, for reporting. Not used by the gate."""
        return max(self.expectations.values()) if self.expectations else None

    @property
    def doc_ids(self) -> list[str]:
        return list(self.order)

    def snippets(self, limit: int | None = None) -> list[tuple[str, str]]:
        """Labelled blocks for the generation context."""
        ids = self.order if limit is None else self.order[:limit]
        return [(self.items[d].doc_id, self.items[d].snippet) for d in ids]

    def summary(self) -> str:
        """A compact description of the ledger for the orchestrator's prompt.

        Deliberately short: the planner needs to know what it holds and whether a
        set is short, not to re-read every snippet on every turn. Re-sending the
        full evidence each step is the easiest way to make an agent expensive for
        no accuracy gain.
        """
        parts = []
        if self.order:
            head = ", ".join(self.order[:10])
            more = f" and {len(self.order) - 10} more" if len(self.order) > 10 else ""
            parts.append(f"{len(self.order)} documents: {head}{more}.")
        else:
            parts.append("No documents retrieved yet.")
        short = self.incomplete
        if short:
            _, expected, listed = short
            parts.append(f"One enumerated set is short: {listed} of {expected}.")
        elif self.expectations:
            counted = ", ".join(str(v) for v in self.expectations.values())
            parts.append(f"Counts computed in the database: {counted}.")
        return " ".join(parts)

    @property
    def tokens(self) -> int:
        return sum(count_tokens(item.snippet) for item in self.items.values())
