"""Scoring: accuracy, grounding, completeness and token cost.

Exact match against the provided gold string is the primary metric and is not
overridable. An LLM judge is available as a secondary read on free-text answers,
but a judge that can turn a wrong string into a pass is a judge that can flatter
the system it is scoring, and the whole benchmark rests on that not happening.

Normalisation is deliberately narrow. It removes differences no reader would
call a different answer -- case, surrounding punctuation, unicode space and dash
variants, articles, digits written as words -- and nothing else. It never
normalises accents away, because "Massu" and "Massú" are different spellings of a
name the corpus writes one way.
"""
from __future__ import annotations

import re
import unicodedata

# Narrow no-break space and friends: the model copies these out of the corpus,
# and an answer that differs from gold only by an invisible character is the same
# answer.
_SPACES = dict.fromkeys(map(ord, "       "), " ")
_DASHES = dict.fromkeys(map(ord, "‐‑‒–—―"), "-")

_ARTICLES = re.compile(r"\b(the|a|an)\b")
_PUNCT_EDGE = re.compile(r"^[\s\"'“”‘’.,;:()\[\]]+|[\s\"'“”‘’.,;:()\[\]]+$")

_NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "eleven": "11", "twelve": "12", "thirteen": "13", "fourteen": "14",
    "fifteen": "15", "sixteen": "16", "seventeen": "17", "eighteen": "18",
    "nineteen": "19", "twenty": "20",
}


def normalise(text: str) -> str:
    """The comparison form of an answer."""
    if text is None:
        return ""
    s = unicodedata.normalize("NFC", str(text))
    s = s.translate(_SPACES).translate(_DASHES)
    s = s.lower()
    s = _PUNCT_EDGE.sub("", s)
    s = _ARTICLES.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return _NUMBER_WORDS.get(s, s)


def exact_match(answer: str, gold: list[str]) -> bool:
    """Strict: the answer equals one of the accepted gold strings."""
    got = normalise(answer)
    return any(got == normalise(g) for g in gold)


def lenient_match(answer: str, gold: list[str]) -> bool:
    """Lenient: the gold string appears in the answer, or the reverse.

    Reported alongside strict, never instead of it. It catches an answer that is
    right but wrapped in a sentence the pipeline was told not to write, which is
    a prompt-following failure rather than a retrieval one, and the two are worth
    telling apart.
    """
    got = normalise(answer)
    if not got:
        return False
    return any(g and (normalise(g) in got or got in normalise(g))
               for g in (normalise(x) for x in gold))


def grounding(citations: list[str], gold_docs: list[str]) -> dict[str, float]:
    """Precision, recall and F1 of cited documents against the gold set.

    This is the evidence-quality score and it is objective: the public set gives
    the gold document ids, so nothing here depends on a model's opinion.
    """
    cited = {c for c in citations if c}
    gold = {g for g in gold_docs if g}
    if not gold:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "cited": len(cited)}
    hit = len(cited & gold)
    precision = hit / len(cited) if cited else 0.0
    recall = hit / len(gold)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4), "cited": len(cited)}


def score_row(result: dict, question: dict) -> dict:
    """One pipeline's result on one question, scored."""
    gold = question.get("answer") or []
    gold_docs = question.get("gold_doc_ids") or []
    row = {
        "qid": question["qid"],
        "qtype": question.get("qtype", ""),
        "pipeline": result["pipeline"],
        "answer": result["answer"],
        "gold": gold[0] if gold else "",
        "status": result["status"],
        "route": result.get("route", ""),
        "stop_reason": result.get("stop_reason", ""),
        "exact": exact_match(result["answer"], gold) if gold else None,
        "lenient": lenient_match(result["answer"], gold) if gold else None,
        "context_tokens": result.get("context_tokens", 0),
        "input_tokens": result.get("input_tokens", 0),
        "output_tokens": result.get("output_tokens", 0),
        "total_tokens": result.get("total_tokens", 0),
        "llm_calls": result.get("llm_calls", 0),
        "tool_calls": result.get("tool_calls", 0),
        "steps": len(result.get("trace") or []),
        "wall_clock_s": result.get("wall_clock_s", 0.0),
        # Wall clock less time slept on the provider's rate limit: the latency
        # of the system rather than of the free tier it ran on.
        "active_s": round(max(result.get("wall_clock_s", 0.0)
                              - result.get("throttled_s", 0.0), 0.0), 3),
        "coverage_expected": result.get("coverage_expected"),
        "coverage_actual": result.get("coverage_actual"),
    }
    row.update({f"grounding_{k}": v
                for k, v in grounding(result.get("citations") or [], gold_docs).items()})
    return row


def summarise(rows: list[dict]) -> dict[str, dict]:
    """Per-pipeline totals, and per-pipeline-per-qtype breakdowns."""
    summary: dict[str, dict] = {}
    for pipeline in sorted({r["pipeline"] for r in rows}):
        subset = [r for r in rows if r["pipeline"] == pipeline]
        summary[pipeline] = _aggregate(subset)
        by_type: dict[str, dict] = {}
        for qtype in sorted({r["qtype"] for r in subset}):
            by_type[qtype] = _aggregate([r for r in subset if r["qtype"] == qtype])
        summary[pipeline]["by_qtype"] = by_type
    return summary


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def _aggregate(rows: list[dict]) -> dict:
    n = len(rows) or 1
    exact = sum(1 for r in rows if r["exact"])
    # Older result rows predate `active_s`; wall clock is the honest fallback.
    active = [r.get("active_s", r["wall_clock_s"]) for r in rows]
    return {
        "n": len(rows),
        "exact": round(exact / n, 4),
        "lenient": round(sum(1 for r in rows if r["lenient"]) / n, 4),
        "insufficient": round(
            sum(1 for r in rows if r["status"] == "insufficient_evidence") / n, 4),
        "avg_total_tokens": round(sum(r["total_tokens"] for r in rows) / n, 1),
        "avg_context_tokens": round(sum(r["context_tokens"] for r in rows) / n, 1),
        "avg_steps": round(sum(r["steps"] for r in rows) / n, 2),
        "avg_grounding_f1": round(sum(r["grounding_f1"] for r in rows) / n, 4),
        "avg_wall_clock_s": round(sum(r["wall_clock_s"] for r in rows) / n, 2),
        # The median is the latency figure to quote. A mean over a run that
        # slept on rate limits reports the provider's queue, not the system.
        "median_wall_clock_s": round(_median([r["wall_clock_s"] for r in rows]), 2),
        "median_active_s": round(_median(active), 2),
    }


def cost_of_agency(summary: dict[str, dict], baseline: str = "graphrag",
                   agent: str = "agentic") -> dict[str, dict]:
    """Accuracy gained per 1,000 extra tokens, per question type.

    This is the number the hackathon's headline question reduces to: where the
    agent earns its cost, where it breaks even, and where it is simply more
    expensive. A negative value is a real finding and is reported as one.
    """
    if baseline not in summary or agent not in summary:
        return {}
    out: dict[str, dict] = {}
    base_types = summary[baseline]["by_qtype"]
    agent_types = summary[agent]["by_qtype"]
    for qtype in sorted(set(base_types) | set(agent_types)):
        base = base_types.get(qtype)
        theirs = agent_types.get(qtype)
        if not base or not theirs:
            continue
        delta_accuracy = theirs["exact"] - base["exact"]
        delta_tokens = theirs["avg_total_tokens"] - base["avg_total_tokens"]
        out[qtype] = {
            "delta_exact": round(delta_accuracy, 4),
            "delta_tokens": round(delta_tokens, 1),
            "exact_per_1k_tokens": (round(delta_accuracy / (delta_tokens / 1000), 4)
                                    if delta_tokens > 1 else None),
            "verdict": _verdict(delta_accuracy, delta_tokens),
        }
    return out


def _verdict(delta_accuracy: float, delta_tokens: float) -> str:
    """Plain-language reading of one question type.

    The ratio alone is unreadable when both terms improve: accuracy up and tokens
    down divides to a negative number that looks like a regression and is the
    opposite. These four labels are what the chart and the write-up actually say.
    """
    cheaper = delta_tokens < 0
    if delta_accuracy > 0.02:
        return "strictly better" if cheaper else "worth the cost"
    if delta_accuracy < -0.02:
        return "worse and cheaper" if cheaper else "strictly worse"
    return "no gain, cheaper" if cheaper else "overkill"
