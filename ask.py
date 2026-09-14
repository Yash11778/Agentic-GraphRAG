"""Ask one question and watch the investigation happen.

    python ask.py "How many alpine skiing events at the 2014 Winter Olympics had more than 63 competitors?"
    python ask.py --all "Who won the gold medal in the event held at Eton Dorney on 1 August 2012?"
    python ask.py --pipeline rag "..."

The demo entry point. `--all` runs the same question through RAG, GraphRAG and
the agent and prints the three answers side by side with their cost, which is the
whole benchmark in one screen. The agent's steps are printed as they happen, from
the same trace the benchmark records -- nothing here is narrated after the fact.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from agentic.config import graph_backend_name  # noqa: E402
from agentic.llm import LLMClient  # noqa: E402
from agentic.pipelines import agentic, graphrag, rag  # noqa: E402
from agentic.pipelines.base import PipelineResult, TraceStep  # noqa: E402
from agentic.tools.tools import Tools  # noqa: E402

PIPELINES = {"rag": rag, "graphrag": graphrag, "agentic": agentic}
BOLD, DIM, GREEN, YELLOW, RED, RESET = "\033[1m", "\033[2m", "\033[32m", "\033[33m", "\033[31m", "\033[0m"


def show_step(step: TraceStep) -> None:
    """One line per step, printed the moment the step is recorded."""
    colour = {"ok": GREEN, "empty": YELLOW, "error": RED, "incomplete": YELLOW}.get(step.status, "")
    label = "plan" if step.tool == "plan" else f"{step.agent} → {step.tool}"
    extra = []
    if step.context_tokens:
        extra.append(f"{step.context_tokens} ctx tok")
    if step.new_evidence_count:
        extra.append(f"+{step.new_evidence_count} docs")
    if step.strategy_change:
        extra.append("strategy change")
    print(f"  {DIM}{step.step_no:>2}{RESET} {label:<36} {colour}{step.status:<10}{RESET}"
          f" {step.latency_ms:>6.0f} ms  {DIM}{' · '.join(extra)}{RESET}")
    if step.args_digest and step.tool != "plan":
        print(f"     {DIM}{step.args_digest[:150]}{RESET}")
    if step.note:
        print(f"     {DIM}{step.note[:150]}{RESET}")


def run_one(name: str, question: str, tools: Tools, llm: LLMClient, live: bool) -> PipelineResult:
    print(f"\n{BOLD}{name}{RESET}")
    started = time.perf_counter()
    if name == "agentic":
        result = agentic.run(question, tools, llm, observer=show_step if live else None)
    else:
        result = PIPELINES[name].run(question, tools, llm)
        for step in result.trace:
            show_step(step)
    elapsed = time.perf_counter() - started
    mark = GREEN if result.status == "ok" else YELLOW
    print(f"\n  {mark}{BOLD}answer:{RESET} {result.answer}")
    print(f"  {DIM}{result.total_tokens} tokens ({result.context_tokens} retrieved context) · "
          f"{result.llm_calls} model calls · {result.tool_calls} tool calls · "
          f"{elapsed - result.throttled_s:.1f}s"
          + (f" (+{result.throttled_s:.0f}s rate-limited)" if result.throttled_s else "")
          + (f" · route {result.route}" if name == "agentic" else "")
          + (f" · stopped: {result.stop_reason}" if result.stop_reason else "")
          + f" · cites {len(result.citations)} docs{RESET}")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("question")
    ap.add_argument("--pipeline", choices=list(PIPELINES), default="agentic")
    ap.add_argument("--all", action="store_true", help="run all three pipelines")
    args = ap.parse_args()

    tools, llm = Tools(), LLMClient()
    print(f"{DIM}backend={graph_backend_name()} model={llm.settings.model}{RESET}")
    print(f"{BOLD}Q:{RESET} {args.question}")

    names = list(PIPELINES) if args.all else [args.pipeline]
    results = [run_one(name, args.question, tools, llm, live=True) for name in names]

    if len(results) > 1:
        print(f"\n{BOLD}{'pipeline':<10}{'tokens':>8}{'context':>9}{'calls':>7}   answer{RESET}")
        for name, result in zip(names, results, strict=True):
            print(f"{name:<10}{result.total_tokens:>8}{result.context_tokens:>9}"
                  f"{result.llm_calls:>7}   {result.answer[:70]}")


if __name__ == "__main__":
    main()
