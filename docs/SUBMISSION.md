# Submission pack — what to hand in and what to say

## Checklist

| Item | Where | Status |
|---|---|---|
| Source repository | this repo — push to `origin/main` before the deadline | pending |
| Hidden-50 answers with traces | `data/results/submission.json` — 150 rows: every one of the 50 held-out questions answered by all three pipelines with its tokens, latency and trace, as the organisers asked. Agent 50/50 answered, RAG and GraphRAG 42/50. `backend: tigergraph` is derived from the rows, each stamped at run time | done |
| Public benchmark, 3 pipelines × 100 | `data/results/public.jsonl`, `public_summary.json` — all 300 rows on TigerGraph, final build: agentic 99%, RAG 52%, GraphRAG 47% | done |
| Public set on the local backend | `data/results/public_local.jsonl` — the same 300 in memory (98 / 53 / 48), a parity check | done |
| Architecture, agent-loop and schema diagrams | `docs/diagrams/` — tracked in the repo (`.mmd` sources and rendered `.png`), since the architecture diagram is a required deliverable; the README also carries the text diagram | done |
| Demo video | recorded from `docs/DEMO_SCRIPT.md`, which stays out of the repo (gitignored); the walkthrough is the video itself | to record |
| Dashboard | `frontend/` (static; `npm run build`, deploys on Vercel via `vercel.json`) | done |
| Write-up | `README.md` + `docs/ARCHITECTURE.md` | done |
| Tests | `python -m pytest` (73 tests, no network) | passing |

## Reproducing the numbers

With the credentials in `.env` rotated by the client, each run is minutes, not days:

1. `GRAPH_BACKEND=tigergraph python eval2/run_hidden.py --fresh` — 50 questions
   through all three pipelines, 150 rows, ~737k tokens. **The deliverable.**
   Savanna must be resumed first. Budget an hour rather than the agent's own
   five minutes: GraphRAG sends a ~5k-token context on every question and draws
   the provider's rate limiting, and the 2026-09-22 run met Groq timeouts that
   cost 48 min for 100 baseline rows. The run resumes, and rows that error are
   not recorded as done, so re-running the command picks them up.
2. `GRAPH_BACKEND=tigergraph python eval2/run_public.py --fresh` — 300 runs on
   Savanna, ~1.4M tokens. The agent takes about 7 min; the baselines took 37 min
   for 200 rows on 2026-09-25.
3. `python eval2/run_public.py --out data/results/public_local.jsonl --fresh` —
   the same 300 on the local backend, a parity check beside the headline file.
4. `python eval2/report.py && (cd frontend && npm run build)`.

All have run on the final build: the agent's public rows and the hidden set on
2026-09-15 and 2026-09-22, the public baselines on TigerGraph on 2026-09-25.
Public agentic 99% on TigerGraph (98% local), hidden 50/50 answered. The result
files, `report.json` and the README tables all reflect them. The rule throughout: a headline number is replaced only by
a full re-run, never by re-running failures.

Derived files are rebuilt from the rows without running anything:
`python eval2/run_hidden.py --package` (the bundle), `python eval2/run_public.py
--summarise` (the summary), `python eval2/report.py` (the dashboard feed). Their
`backend` and `model` labels come from the rows, so a rebuild on a laptop with
`GRAPH_BACKEND=local` cannot mislabel a TigerGraph run — which is exactly what
happened once (ARCHITECTURE §11b, "Provenance is on the row").

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
| Use of TigerGraph | `agentic/ingest/schema.gsql`, `agentic/tools/queries.gsql`; `eval2/validate_backends.py` 38/38 parity (rerun 2026-09-15 after the whole-number `contains` change); vector index on `Chunk.emb`; all 300 public runs scored on TigerGraph, its own vector search included (agent 99/100); the local re-run in `data/results/public_local.jsonl` agrees on 98 of the agent's 100 answers |
| Accuracy | strict exact match, per qtype; grounding P/R/F1 against gold doc ids |
| Token efficiency | context / input / output split per pipeline; "where the tokens go" chart |
| When is agency overkill | cost-of-agency chart, per qtype, against both baselines; the `lookup` exception stated plainly; fast-path routing |
| Engineering quality | deterministic ingestion with measured coverage; two backends behind one interface; 73 unit tests; ruff-clean; resumable runners; every result row stamped with its backend and model |
| Honesty | variance note in README; refusals counted as wrong; pre-fix vs post-fix numbers labelled; a token-saving prompt rewrite that cost accuracy was measured, reverted and recorded (ARCHITECTURE §11b) |

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

## If asked "why not fewer tokens?"

About 85% of the agent's tokens are the orchestrator prompt and tool schemas, resent
on every call; retrieved context is under 150 tokens. A terser rewrite of both was
tried on 2026-09-15: 8.5% fewer tokens, two public points and one hidden answer
lost over full re-runs. Results outrank tokens, so it was reverted and recorded.
The remaining lever, merging the planning call into the first orchestrator turn,
would save roughly 400 tokens a question and was left alone.

## The last pass (2026-09-15)

A last pass read the remaining four failed traces and found two more defects,
again general and again unit-tested: a number inside a `contains` value matched
inside other numbers ("6" in "2016", so a date filter returned every event at a
venue that year), and the verbatim check could replace an answer that named both
the medallist and the event with the event's title. Both backends were changed
together (a new `s_number` GSQL op) and re-verified at 38/38 parity; the trace now
records what the model wrote whenever the verbatim check fires.
