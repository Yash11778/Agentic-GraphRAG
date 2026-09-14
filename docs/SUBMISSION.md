# Submission pack — what to hand in and what to say

## Checklist

| Item | Where | Status |
|---|---|---|
| Source repository | this repo (`git push` after committing) | commit pending |
| Hidden-50 answers with traces | `data/results/submission.json` — final build, TigerGraph backend, 49/50 answered | done |
| Public benchmark, 3 pipelines × 100 | `data/results/public.jsonl`, `public_summary.json` — agentic 96%, final build | done |
| Architecture diagram | `docs/diagrams/architecture.mmd` (+ `.png` if rendered), `docs/diagrams/index.html` | done |
| Agent loop diagram | `docs/diagrams/agent_loop.mmd` | done |
| Graph schema diagram | `docs/diagrams/graph_schema.mmd` | done |
| Demo video | script in `docs/DEMO_SCRIPT.md` | to record |
| Dashboard | `frontend/` (static; `npm run build`, deploys on Vercel via `vercel.json`) | done |
| Write-up | `README.md` + `docs/ARCHITECTURE.md` | done |
| Tests | `python -m pytest` (61 tests, no network) | passing |

## Ordering that fits the free-tier quota

Groq's daily token limit is per organisation, so the order matters:

1. `python eval2/run_hidden.py --fresh` — 50 questions, ~230k tokens. **The deliverable.**
   Do this first, on `GRAPH_BACKEND=tigergraph`.
2. `python eval2/run_public.py --pipelines agentic --redo` — 100 questions, ~450k tokens,
   two to three days of quota unless a teammate's key is added to `GROQ_API_KEYS`.
   The baselines do not need rerunning: nothing in their path changed.
3. `python eval2/report.py && (cd frontend && npm run build)`.

Both have run on the final build: public agentic 96%, hidden 49/50 answered. The
result files, `report.json` and the README table all reflect them.

## The five sentences to lead with

1. **Same tools, same budget, three control flows** — so a delta is attributable to
   agency and nothing else (LOCKED-3, LOCKED-7).
2. **The model never counts.** Aggregation and superlative run as one GSQL query;
   the agent supplies predicates, TigerGraph supplies the number (LOCKED-2).
3. **No answer from an incomplete set.** The coverage gate compares what the agent
   listed against what the database counted, for the same predicates, and pushes
   back (LOCKED-5).
4. **Failure is a status, not a guess.** `insufficient_evidence` scores as wrong;
   the refusal rate is on the dashboard next to the accuracy (LOCKED-6).
5. **Agency is routed, not assumed.** A one-document question takes a fast path on
   a tighter budget and escalates only when that fails — the hackathon's question,
   built into the product.

## Judging criteria → where the evidence is

| Likely criterion | Evidence |
|---|---|
| Agentic effectiveness | per-step trace (`agent`, `tool`, `retrieval_method`, `strategy_change`, `stop_reason`) in every result row; trace explorer in the dashboard |
| Use of TigerGraph | `agentic/ingest/schema.gsql`, `agentic/tools/queries.gsql`; `eval2/validate_backends.py` 35/35 parity (rerun 2026-09-12 after the predicate changes); vector index on `Chunk.emb` |
| Accuracy | strict exact match, per qtype; grounding P/R/F1 against gold doc ids |
| Token efficiency | context / input / output split per pipeline; "where the tokens go" chart |
| When is agency overkill | cost-of-agency chart, per qtype, against both baselines; the `lookup` exception stated plainly; fast-path routing |
| Engineering quality | deterministic ingestion with measured coverage; two backends behind one interface; 61 unit tests; ruff-clean; resumable runners |
| Honesty | variance note in README; refusals counted as wrong; pre-fix vs post-fix numbers labelled |

## What changed in the final pass (for the Q&A)

The fixed build removed six defects found by reading the failed traces, none of which
was a question-type handler:

- numeric thresholds arriving as strings matched nothing in the local backend
  (a parity bug: TigerGraph coerced, local did not) — now coerced in one shared place;
- a filter over a field the vertex type lacks returned empty instead of an error,
  and the planner spent its budget loosening it — now an error naming the real fields;
- a count of zero came back as `ok` — now `empty`, with the recovery hint;
- tool results were rendered as full vertex JSON, so a 700-token window held four
  rows and the planner learned to ask for five — now compact rows, fourteen fit;
- `contains` was a raw substring test, and the corpus writes some venue names with
  joined words — now every token must occur, in both backends identically;
- an answer that was the corpus's string with separators changed scored as wrong —
  now a deterministic verbatim check hands back the stored string and records that
  it did.

Plus: a malformed tool call no longer counts toward `unrecoverable`; the entity
linker went from ~300 ms to under 1 ms; local filters from ~15 ms to ~2 ms; latency
is reported net of rate-limit sleeps, as a median.
