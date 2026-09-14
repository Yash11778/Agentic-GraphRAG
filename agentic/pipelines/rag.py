"""Pipeline 1 — RAG. Retrieve similar chunks, generate, stop.

One retrieval step, no graph, no planning. This is the baseline the other two are
measured against, and it is deliberately a good-faith implementation: same
embedding model, same generation budget, a k large enough to be fair. Where it
fails it should fail for a structural reason -- a question whose answer is spread
across dozens of documents cannot be assembled from a top-k window -- not because
the baseline was built to lose.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agentic.llm import LLMClient
from agentic.pipelines.base import (
    PipelineResult,
    TraceStep,
    finish,
    generate_answer,
    render_context,
)
from agentic.tools.tools import Tools

DEFAULT_K = 12

# The context ceiling every pipeline shares. Large enough for a dozen chunks,
# small enough that a pipeline cannot "win" by pasting the corpus in.
CONTEXT_BUDGET = 6000


def run(question: str, tools: Tools, llm: LLMClient, qid: str = "",
        k: int = DEFAULT_K) -> PipelineResult:
    started = time.perf_counter()
    result = PipelineResult(qid=qid, question=question, pipeline="rag",
                            answer="", status="error")

    hits = tools.vector_search(question, k=k)
    result.tool_calls += 1
    result.trace.append(TraceStep(
        step_no=1, agent="similarity_searcher", tool="vector_search",
        retrieval_method="vector", args_digest=f"k={k}", status=hits.status,
        latency_ms=hits.latency_ms, chunks_returned=len(hits.data or []),
        new_evidence_count=len(hits.doc_ids),
    ))

    if not hits.ok:
        result.status = "error" if hits.status == "error" else "insufficient_evidence"
        result.error = hits.error
        result.stop_reason = "retrieval_empty"
        result.wall_clock_s = round(time.perf_counter() - started, 3)
        return result

    blocks = [(f"{chunk['doc_id']} score={chunk['score']}", chunk["text"])
              for chunk in hits.data]
    context, context_tokens = render_context(blocks, CONTEXT_BUDGET)
    result.context_tokens = context_tokens

    answer, usage = generate_answer(llm, question, context)
    result.stop_reason = "single_retrieval"
    return finish(result, answer, usage, started, hits.evidence)
