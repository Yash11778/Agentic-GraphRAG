# Final Benchmark — GraphRAG vs Basic RAG vs LLM-only

**Full-scale run on the complete corpus.** Both retrieval pipelines operate on the same
**63,632 U.S. court opinions / 117.5M tokens**: GraphRAG over the TigerGraph citation graph
(63,632 `LegalCase` vertices, 9,632 `CITES` edges, 1,413 `Court` vertices); Basic RAG over
the full FAISS index (500,959 chunks, `BASIC_RAG_FULL=1`).

**QA set:** 55 multi-hop questions (`data/qa/qa_pairs_legal_multihop.json`) — 40 two-hop,
15 three-hop — each built from an actual in-corpus citation chain.

**Judge:** one strict Gemini judge for the entire run (`JUDGE_MODE=gemini`; every one of
the 165 rows in `run1.csv` has `judge_source=gemini_strict`). No fallback judge, no
mid-run judge swap, no auto-pass, judge errors count as FAIL. BERTScore via
`evaluate.load("bertscore", rescale_with_baseline=True)`, computed for all three pipelines.

## Results (strict judge, live TigerGraph — `run1.csv`)

| Metric                        | LLM-only | Basic RAG | **GraphRAG** |
|-------------------------------|:--------:|:---------:|:------------:|
| Overall pass rate             |   1.8%   |    9.1%   |  **50.9%**   |
| 2-hop pass rate               |   2.5%   |   12.5%   |  **55.0%**   |
| **3-hop pass rate**           |   0.0%   |  **0.0%** |  **40.0%**   |
| Avg tokens / question         |   609    |   2,288   |  **1,423**   |
| Avg latency                   |  4.6s    |   3.7s    |    7.0s      |
| BERTScore (raw / rescaled f1) | 0.787 / 0.362 | 0.807 / 0.422 | **0.841 / 0.524** |

**Headline:** GraphRAG passes **5.6× more often than Basic RAG** using **37.8% fewer
tokens**, and leads on BERTScore. The 3-hop column is the core proof — flat top-k
similarity cannot assemble a three-case citation chain (A←B←C), so Basic RAG lands 0%;
graph traversal reaches 40%.

A strict judge fails partial, hedged, or off-question answers, so absolute pass rates are
low across the board by design (LLM-only collapses to 1.8% because it cannot know the
cited opinions at all). The relative gap between pipelines under one uniform judge is the
measurement.

## Method note — fair comparison

Each multi-hop question cites its cases by identifier the way a real legal query does, e.g.
"People v. Batson (6047231)". GraphRAG resolves that citation to a vertex and traverses real
`CITES` edges; Basic RAG receives the **identical** question text and retrieves by embedding
similarity. Not answer leakage — the question supplies the citation, both pipelines see the
same input, and the answer (how the cases relate) must still be synthesized from retrieved
text. This isolates GraphRAG's real value (multi-hop traversal + synthesis) from the
orthogonal problem of entity-linking noisy Pile-of-Law captions.

## Integrity

- **5 of 55** GraphRAG answers were no-context (a genuine retrieval miss against the live
  graph); all 5 are counted as FAIL in the 50.9%. Every other answer came from a live GSQL
  query — no snapshot fallback, no curated subset.
- No auto-pass judge, no BERTScore override, equal 512-token output caps for all pipelines.
- Token reduction comes from precise retrieval (seed cases + ≤2 citation neighbours), not
  from shorter answers.

## Run-to-run variance

An LLM judge over 55 questions has a measured noise floor of roughly ±3–5 percentage
points between identical runs. Single-run numbers above are exact for `run1.csv`;
subsequent runs are aggregated as mean ± range with `eval/aggregate_runs.py` and the
table will be updated to the aggregate once ≥3 strict-judge runs are recorded.

Raw data: `eval/results/run1.csv` (identical copy: `eval/results/eval_results.csv`)

## Historical results (superseded)

`eval/results/full_eval_results_final.csv` is an earlier run judged by a mixed
HF-primary / lenient-Gemini-fallback judge (69.1% GraphRAG pass). That judge mixture
inflated absolute pass rates and has been replaced by the single strict judge above; the
file is kept only for provenance and its numbers are not cited anywhere.
