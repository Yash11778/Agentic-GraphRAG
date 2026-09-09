"""
Offline benchmark: run all three pipelines on the real multi-hop QA set and report
pass rate, tokens, latency, token reduction and BERTScore -- broken down by hop count.

Every GraphRAG answer comes from a live TigerGraph query; there is no snapshot and no
curated subset. Set BASIC_RAG_FULL=1 to benchmark Basic RAG against the full 500k-chunk
index (both pipelines then draw on the same 117.5M-token corpus).

    python eval/evaluate.py
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).parent.parent.resolve()
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from pipelines.pipeline1_llm import pipeline1
from pipelines.pipeline2_rag import pipeline2
from pipelines.pipeline3_graphrag import pipeline3
from eval.judge import llm_judge_with_source, compute_bertscore

QA_FILE = ROOT / "data/qa/qa_pairs_legal_multihop.json"
OUT_FILE = ROOT / "eval/results/eval_results.csv"

PIPELINES = [("llm_only", pipeline1), ("basic_rag", pipeline2), ("graphrag", pipeline3)]


def main():
    with open(QA_FILE, encoding="utf-8") as f:
        qa_pairs = json.load(f)
    print(f"Loaded {len(qa_pairs)} questions from {QA_FILE}")

    rows = []
    for i, qa in enumerate(tqdm(qa_pairs, desc="Evaluating")):
        question, ground_truth = qa["question"], qa["answer"]
        for name, fn in PIPELINES:
            try:
                result = fn(question)
            except Exception as e:
                print(f"  [{name}] pipeline ERROR: {e}", flush=True)
                result = {"answer": "", "total_tokens": 0, "latency_s": 0, "cost_usd": 0}
            # One retry after a pause: gemini_generate already retries transient 503s
            # internally, so reaching here twice means the API is genuinely down for
            # this row. Recorded as source='error' (surfaced in the summary), never
            # silently mixed into real verdicts.
            judge, judge_source = "FAIL", "error"
            for judge_attempt in range(2):
                try:
                    judge, judge_source = llm_judge_with_source(question, ground_truth, result["answer"])
                    break
                except Exception as e:
                    print(f"  [{name}] judge ERROR (attempt {judge_attempt + 1}/2): {e}", flush=True)
                    if judge_attempt == 0:
                        time.sleep(15)
            rows.append({
                "qid": i,
                "hop_count": qa.get("hop_count", 0),
                "pipeline": name,
                "judge": judge,
                "judge_source": judge_source,
                "total_tokens": result["total_tokens"],
                "latency_s": result["latency_s"],
                "cost_usd": result.get("cost_usd", 0),
                "answer": result["answer"],
                "ground_truth": ground_truth,
                "question": question,
            })

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_FILE, index=False)
    print(f"\nSaved: {OUT_FILE}")
    print_summary(df)


def print_summary(df: pd.DataFrame):
    print("\n=== EVALUATION SUMMARY ===")
    print("\nJudge source:")
    print(df["judge_source"].value_counts().to_string())

    def line(sub, name):
        pct = (sub["judge"] == "PASS").mean() * 100
        return (f"  [{name}] pass={pct:.1f}%  avg_tokens={sub['total_tokens'].mean():.0f}  "
                f"avg_latency={sub['latency_s'].mean():.2f}s")

    print("\nOverall:")
    for name, _ in PIPELINES:
        print(line(df[df["pipeline"] == name], name))

    print("\nBy hop count:")
    for hop in sorted(df["hop_count"].unique()):
        print(f"  hop_count={hop}:")
        for name, _ in PIPELINES:
            sub = df[(df["hop_count"] == hop) & (df["pipeline"] == name)]
            if len(sub):
                print("  " + line(sub, name))

    rag_avg = df[df["pipeline"] == "basic_rag"]["total_tokens"].mean()
    gr = df[df["pipeline"] == "graphrag"]
    reduction = (1 - gr["total_tokens"].mean() / rag_avg) * 100 if rag_avg else 0.0

    print("\nComputing BERTScore (all pipelines)...")
    bs = {}
    for name, _ in PIPELINES:
        sub = df[df["pipeline"] == name]
        bs[name] = compute_bertscore(sub["answer"].fillna("").tolist(),
                                     sub["ground_truth"].tolist())
        if bs[name].get("error"):
            # A broken metric must be loud: a silent 0.0 looks like a terrible score
            # instead of a failed computation.
            print(f"WARNING: BERTScore FAILED for {name}: {bs[name]['error']}")
        else:
            print(f"  [{name}] raw={bs[name]['raw_f1']}  rescaled={bs[name]['rescaled_f1']}")
    bs = bs["graphrag"]

    gr_pass = (gr["judge"] == "PASS").mean() * 100
    rag_pass = (df[df["pipeline"] == "basic_rag"]["judge"] == "PASS").mean() * 100
    print(f"\nFINAL: basic_rag_pass={rag_pass:.1f}%  graphrag_pass={gr_pass:.1f}%  "
          f"token_reduction={reduction:.1f}%  bertscore_rescaled={bs['rescaled_f1']}")


if __name__ == "__main__":
    main()
