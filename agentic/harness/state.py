"""InvestigationState: everything one agentic run knows about itself.

Holds the question, the plan, the evidence ledger, the budget, the trace, and the
open gap -- the single sentence describing what is still missing. The gap is what
makes the loop agentic rather than iterative: every step is chosen against it, and
the run ends when it closes or provably cannot.

Stop criteria live here too, in one ordered function, so that "why did it stop"
has exactly one answer and it is recorded rather than reconstructed.
"""
from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agentic.harness.budget import Budget
from agentic.harness.evidence import EvidenceLedger
from agentic.pipelines.base import TraceStep

# Stop reasons, in the order they are checked. The first three are decisions the
# agent makes; the last two are failures it admits (LOCKED-6).
SUFFICIENT = "sufficient_evidence"
COVERAGE_COMPLETE = "coverage_complete"
NO_PROGRESS = "no_progress"
BUDGET_EXHAUSTED = "budget_exhausted"
UNRECOVERABLE = "unrecoverable"

FAILURE_REASONS = (NO_PROGRESS, BUDGET_EXHAUSTED, UNRECOVERABLE)


@dataclass
class InvestigationState:
    question: str
    qid: str = ""
    plan: str = ""
    gap: str = ""
    route: str = "agentic"
    ledger: EvidenceLedger = field(default_factory=EvidenceLedger)
    budget: Budget = field(default_factory=Budget)
    trace: list[TraceStep] = field(default_factory=list)
    # What the tools actually returned, kept for the answer that is written when
    # the loop ends without the model volunteering one. Evidence snippets alone
    # are too thin for that: a graph tool's snippet names a document, while the
    # answer usually lives in the values the call returned.
    findings: list[tuple[str, str]] = field(default_factory=list)

    # Set by the evaluator when it judges the evidence sufficient.
    sufficient: bool = False
    # Consecutive steps that moved nothing forward.
    barren_steps: int = 0
    # Calls already made, so that repeating one counts as standing still.
    seen_calls: set[str] = field(default_factory=set)
    # Tool calls that returned error or empty, in a row.
    failed_steps: int = 0
    strategy_changes: int = 0

    input_tokens: int = 0
    output_tokens: int = 0
    llm_calls: int = 0
    # Seconds spent waiting on the provider's rate limit. Reported apart from
    # wall clock so latency numbers describe the system and not the free tier.
    throttled_s: float = 0.0

    # Called with every step as it is recorded, so a caller can show an
    # investigation while it runs. Optional; the benchmark passes nothing.
    observer: Callable[[TraceStep], None] | None = None

    def record(self, step: TraceStep) -> None:
        """Log a step and update the progress and failure counters.

        Progress is not the same as new documents. `graph_aggregate` cites
        nothing and is still the most informative call in the system, so a step
        counts as progress when it returned usable data -- unless it is a call
        already made, which is standing still however much data it returns.
        """
        self.trace.append(step)
        self.budget.spend(steps=1, tokens=step.context_tokens + step.output_tokens)

        # Only retrieval attempts can stall an investigation. A rejected
        # generation or a coverage push-back is bookkeeping, not a failed attempt
        # to find something, and counting them would end runs that are still
        # making their first real try.
        if step.retrieval_method != "none":
            call = f"{step.tool}:{step.args_digest}"
            repeated = call in self.seen_calls
            self.seen_calls.add(call)
            moved = (step.new_evidence_count or step.produced_data) and not repeated
            self.barren_steps = 0 if moved else self.barren_steps + 1
            # Same rule for the failure counter: a rejected generation is not a
            # retrieval that came back empty, and counting it made one malformed
            # call plus two empty filters read as "every tool failed".
            if step.status in ("error", "empty"):
                self.failed_steps += 1
            else:
                self.failed_steps = 0
        if step.strategy_change:
            self.strategy_changes += 1
        if self.observer is not None:
            self.observer(step)

    def spend_llm(self, usage) -> None:
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens
        self.llm_calls += usage.calls
        self.budget.spend(steps=0, tokens=usage.total_tokens)
        throttled = getattr(usage, "throttled_ms", 0.0) / 1000
        self.throttled_s += throttled
        self.budget.credit_time(throttled)

    def stop_reason(self) -> str | None:
        """Why the investigation should end, or None to keep going.

        Order matters. Coverage is checked before the budget so that a run which
        has provably complete evidence stops for the right reason even on its last
        step, and `no_progress` is checked before the budget so that a stalled run
        is reported as stalled rather than as merely expensive.
        """
        if self.sufficient:
            return SUFFICIENT
        if self.ledger.complete is True:
            return COVERAGE_COMPLETE
        # A short enumeration is deliberately not a stop reason: the run should
        # keep retrieving while budget remains, and the checks below decide when
        # that stops being possible.
        if self.failed_steps >= 3:
            return UNRECOVERABLE
        if self.barren_steps >= 2:
            return NO_PROGRESS
        if self.budget.exhausted:
            return BUDGET_EXHAUSTED
        return None

    @property
    def failed(self) -> bool:
        """True when the run ended without the evidence it needed.

        A failed run answers `insufficient_evidence` and scores as wrong. It is
        never quietly downgraded into a best-effort guess, which is the whole of
        LOCKED-6: the benchmark is only worth reading if its failures are visible.
        """
        return self.stop_reason() in FAILURE_REASONS

    def situation(self) -> str:
        """The state description the orchestrator reasons over each step.

        Compact on purpose. The planner needs the question, what is held, what is
        missing, and how much budget is left -- not a transcript. Replaying the
        whole history every step is what makes naive agents cost five times what
        they should for no accuracy gain.
        """
        lines = [
            f"Question: {self.question}",
            f"Evidence: {self.ledger.summary()}",
            f"Still needed: {self.gap or 'not yet determined'}",
            f"Steps used: {self.budget.steps}/{self.budget.max_steps}",
        ]
        if self.budget.pressured:
            lines.append("Budget is nearly spent: answer now if the evidence allows it.")
        if self.trace:
            recent = self.trace[-3:]
            lines.append("Recent steps: " + "; ".join(
                f"{s.tool}({s.args_digest}) -> {s.status}, +{s.new_evidence_count} docs"
                for s in recent
            ))
        return "\n".join(lines)
