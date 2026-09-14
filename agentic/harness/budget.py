"""Step, token and wall-clock caps for one investigation.

Two levels, and the difference between them is the point. The soft cap tells the
orchestrator to start closing the investigation down while it can still answer;
the hard cap stops it. An agent that only has a hard cap spends its last step on
retrieval and then has nothing left to answer with, which reads as a reasoning
failure when it is really an accounting one.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class Budget:
    max_steps: int = 8
    max_tokens: int = 20_000
    max_seconds: float = 150.0

    # Fractions at which the orchestrator is told to wrap up.
    soft_fraction: float = 0.75

    steps: int = 0
    tokens: int = 0
    started: float = field(default_factory=time.perf_counter)

    def spend(self, steps: int = 1, tokens: int = 0) -> None:
        self.steps += steps
        self.tokens += tokens

    def credit_time(self, seconds: float) -> None:
        """Give back time the agent did not spend.

        Provider back-off and a suspended machine both show up as wall clock the
        investigation never used. Charging them to the budget ended runs after a
        single tool call with `budget_exhausted`, which reads as an agent failure
        and is not one.
        """
        if seconds > 0:
            self.started += seconds

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.started

    @property
    def exhausted(self) -> bool:
        return (self.steps >= self.max_steps
                or self.tokens >= self.max_tokens
                or self.elapsed >= self.max_seconds)

    @property
    def pressured(self) -> bool:
        """Past the soft cap: answer with what is in hand if it is enough."""
        return (self.steps >= self.max_steps * self.soft_fraction
                or self.tokens >= self.max_tokens * self.soft_fraction
                or self.elapsed >= self.max_seconds * self.soft_fraction)

    def reason(self) -> str:
        """Which cap was hit, for the trace. Never guessed after the fact."""
        if self.steps >= self.max_steps:
            return f"step cap {self.max_steps}"
        if self.tokens >= self.max_tokens:
            return f"token cap {self.max_tokens}"
        if self.elapsed >= self.max_seconds:
            return f"time cap {self.max_seconds}s"
        return ""

    def as_dict(self) -> dict[str, float | int]:
        return {"steps": self.steps, "max_steps": self.max_steps,
                "tokens": self.tokens, "max_tokens": self.max_tokens,
                "elapsed_s": round(self.elapsed, 2)}
