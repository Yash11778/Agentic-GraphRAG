"""Pipeline 2 — GraphRAG. Link entities, traverse a fixed path, generate.

The same tools the agent has, driven by a fixed sequence instead of a plan: link
the question's entities, expand each one hop, pull the documents that turns up,
generate. No step here depends on what an earlier step found, and that is the
whole point of the comparison -- it isolates "having a graph" from "deciding how
to use one", which is what Pipeline 3 adds.

Vector search stays in the path as a fallback for questions that name no entity
the graph knows, so this pipeline is never strictly worse than Pipeline 1.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agentic.graph_schema import fold
from agentic.llm import LLMClient
from agentic.pipelines.base import (
    PipelineResult,
    TraceStep,
    finish,
    generate_answer,
    render_context,
)
from agentic.pipelines.rag import CONTEXT_BUDGET
from agentic.tools.tools import Evidence, Tools

# One hop from each linked entity. Two hops from a Games vertex reaches every
# event of that edition and drowns the context; the fixed pipeline has no way to
# decide when that is worth doing, which is exactly the limitation being shown.
EDGE_TYPES = ("HELD_AT", "AT_GAMES", "IN_SPORT", "WON_GOLD", "WON_SILVER",
              "WON_BRONZE", "PRECEDED_BY")
MAX_EVENTS = 8
FALLBACK_K = 8

# Words that appear in nearly every event title and so separate nothing.
_STOPWORDS = frozenset(
    "the a an at of in on and or to for what which who whom how many much "
    "did was were is are according provided corpus olympics olympic games "
    "summer winter event events medal gold silver bronze won win".split()
)


def rank_candidates(question: str, candidates: dict[str, dict]) -> list[str]:
    """Order candidate events by how well they match the question.

    Without this the pipeline is not a baseline, it is a coin toss: linking
    "Shooting" and "2004 Summer" reaches 193 and 115 events respectively, and
    taking the first eight in traversal order retrieves eight arbitrary
    documents. Two deterministic signals decide the order -- how many of the
    linked entities reached the same event, and how much of the question's
    vocabulary its title shares. No model is involved, so the pipeline stays a
    fixed traversal.
    """
    words = {w for w in fold(question).replace("-", " ").split() if w not in _STOPWORDS}

    def score(entry: dict) -> tuple[int, int, str]:
        title = fold(entry["vertex"].get("title") or "")
        title_words = {w for w in title.replace("-", " ").split() if w not in _STOPWORDS}
        return (entry["reached_by"], len(words & title_words), entry["vertex"]["id"])

    return [vid for vid, _ in
            sorted(candidates.items(), key=lambda kv: score(kv[1]), reverse=True)]


def run(question: str, tools: Tools, llm: LLMClient, qid: str = "") -> PipelineResult:
    started = time.perf_counter()
    result = PipelineResult(qid=qid, question=question, pipeline="graphrag",
                            answer="", status="error")
    evidence: list[Evidence] = []
    blocks: list[tuple[str, str]] = []
    step = 0

    links = tools.entity_link(question)
    step += 1
    result.tool_calls += 1
    result.trace.append(TraceStep(
        step_no=step, agent="entity_linker", tool="entity_link",
        retrieval_method="lexical", args_digest="question", status=links.status,
        latency_ms=links.latency_ms, new_evidence_count=len(links.data or []),
    ))

    candidates: dict[str, dict] = {}
    for entity in (links.data or []):
        neighbours = tools.graph_neighbors(entity["type"], entity["id"],
                                           EDGE_TYPES, "both")
        step += 1
        result.tool_calls += 1
        result.trace.append(TraceStep(
            step_no=step, agent="graph_traverser", tool="graph_neighbors",
            retrieval_method="graph", args_digest=f"{entity['type']}:{entity['id']}",
            status=neighbours.status, latency_ms=neighbours.latency_ms,
            new_evidence_count=len(neighbours.doc_ids),
        ))
        evidence.extend(neighbours.evidence)
        for row in (neighbours.data or []):
            vertex = row["vertex"]
            if vertex.get("type") != "Event":
                continue
            entry = candidates.setdefault(vertex["id"], {"vertex": vertex, "reached_by": 0})
            entry["reached_by"] += 1

    # Entities alone answer nothing; the events they point at carry the facts.
    event_ids = rank_candidates(question, candidates)
    if event_ids:
        docs = tools.doc_fetch(event_ids[:MAX_EVENTS])
        step += 1
        result.tool_calls += 1
        result.trace.append(TraceStep(
            step_no=step, agent="doc_retriever", tool="doc_fetch",
            retrieval_method="document",
            args_digest=f"top {len(event_ids[:MAX_EVENTS])} of {len(event_ids)} candidates",
            status=docs.status, latency_ms=docs.latency_ms,
            new_evidence_count=len(docs.doc_ids),
        ))
        evidence.extend(docs.evidence)
        blocks += [(f"{d['doc_id']} {d['title']}", d["text"]) for d in (docs.data or [])]

    if not blocks:
        hits = tools.vector_search(question, k=FALLBACK_K)
        step += 1
        result.tool_calls += 1
        result.trace.append(TraceStep(
            step_no=step, agent="similarity_searcher", tool="vector_search",
            retrieval_method="vector", args_digest=f"k={FALLBACK_K}",
            status=hits.status, latency_ms=hits.latency_ms,
            chunks_returned=len(hits.data or []),
            new_evidence_count=len(hits.doc_ids), strategy_change=True,
            note="no linked entity reached a document",
        ))
        evidence.extend(hits.evidence)
        blocks += [(f"{c['doc_id']} score={c['score']}", c["text"])
                   for c in (hits.data or [])]

    if not blocks:
        result.status = "insufficient_evidence"
        result.stop_reason = "no_evidence_found"
        result.wall_clock_s = round(time.perf_counter() - started, 3)
        return result

    context, context_tokens = render_context(blocks, CONTEXT_BUDGET)
    result.context_tokens = context_tokens
    answer, usage = generate_answer(llm, question, context)
    result.stop_reason = "fixed_traversal_complete"
    return finish(result, answer, usage, started, evidence)
