# Agentic GraphRAG over the Olympics corpus

Submission for the **TigerGraph Agentic GraphRAG Hackathon, Round 1**.

Three pipelines answer the same 100 questions over one corpus, so the benchmark can
answer the question the hackathon actually asks: **when does an agentic, multi-step
investigation beat a single GraphRAG or RAG retrieval, and when is it just more
expensive?**

| Pipeline | Control flow |
|---|---|
| **P1 RAG** | vector search → generate |
| **P2 GraphRAG** | entity link → fixed traversal → generate |
| **P3 Agentic GraphRAG** | orchestrator plans, selects tools, evaluates evidence, decides when to stop |

All three call the **same** retrieval tools and run under the **same** generation
budget. They differ only in control flow, which is the only way a token or accuracy
delta can be attributed to agency rather than to one pipeline having better plumbing.

---

## The corpus and why it is hard

2,951 Wikipedia documents, ~5.47M tokens. 2,162 are Olympic event articles with
machine-parseable infoboxes; 789 are films, people and companies that no question
touches — distractors that punish naive similarity search.

The 150 questions split into five types, and half the hidden set is aggregation or
superlative:

| qtype | public | hidden | what it structurally requires |
|---|---:|---:|---|
| aggregation | 21 | 15 | enumerate a complete set, filter numerically, count |
| multi_hop | 28 | 10 | reverse lookup by venue and date, then a medal edge |
| temporal | 22 | 8 | step the Games edition chain, then a medal edge |
| lookup | 19 | 7 | one document |
| superlative | 10 | 10 | enumerate a complete set, take the argmax |

One aggregation question spans 43 gold documents. Top-k similarity retrieval cannot
answer it at any k that fits a context window. That gap is the benchmark's story.

---

## Architecture

Full specification, including the locked design decisions and the record of what
changed while building, is in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

```
corpus.jsonl
     │
     ▼  deterministic infobox parsing, zero LLM
typed property graph  ──────▶  TigerGraph (GSQL) + Vector DB
     │
     ▼
shared tool layer:  vector_search · doc_fetch · entity_link · graph_neighbors
                    graph_filter · graph_aggregate · edition_step · medal_lookup
     │
     ├── P1 RAG ──────────┐
     ├── P2 GraphRAG ─────┤──▶ evaluation ──▶ dashboard
     └── P3 Agentic ──────┘
```

Four decisions carry most of the design:

- **Tools are general primitives, never question-type handlers.** There is no
  `answer_aggregation_question()`. There is `graph_filter` and `graph_aggregate`, and
  the orchestrator composes them. The composition *is* the agentic behaviour.
- **The model never does arithmetic or set enumeration.** Counting, comparison and
  argmax run in the database. This is both more accurate than generative counting and
  far cheaper in tokens.
- **No answer is emitted from an incomplete evidence set.** For enumeration questions
  the agent asks the graph for the expected cardinality and refuses to answer until
  its retrieved set matches. Stopping is a justified decision, not a step counter
  running out.
- **Failure is reported, never papered over.** A retrieval miss is
  `status="insufficient_evidence"` and scores as wrong.
- **The corpus is the answer, down to the punctuation.** When the model's answer
  is a cited document's own string with the separators changed, or names an
  event informally, a deterministic check hands back the stored string and
  records that it did so in the trace. It never changes a letter or a digit.

---

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env      # then fill in the four values
```

`.env` needs a Groq API key and a TigerGraph Savanna workspace:

| Variable | Where it comes from |
|---|---|
| `GROQ_API_KEY` | console.groq.com |
| `TG_HOST` | Savanna workspace → Connection Details |
| `TG_PASSWORD` | Savanna → Database Secrets (a secret, not your login password) |
| `TG_GRAPH` | the database name in your workspace |

`GRAPH_BACKEND` selects where the tool layer reads from. `local` uses the generated
JSONL and needs no network, which is how the agent is developed while the Savanna
workspace is suspended. `tigergraph` is the scored path. Both implement one
interface and are checked against each other, so the choice cannot change answers.

## Running it

```bash
# 1. corpus → typed property graph (deterministic, no LLM, ~20s)
.venv/bin/python agentic/ingest/build_graph.py

# 2. schema, GSQL queries and bulk load into TigerGraph (~3 min, idempotent)
.venv/bin/python agentic/ingest/load_tigergraph.py

# 3. chunk + embed the corpus, load the vector index (~1 hour on CPU, resumable)
.venv/bin/python agentic/ingest/build_vectors.py --load

# 4. checks
.venv/bin/python eval2/validate_backends.py                 # local vs TigerGraph
.venv/bin/python eval2/validate_tools.py --backend tigergraph
```

## Try one question

```bash
python ask.py "How many alpine skiing events at the 2014 Winter Olympics had more than 63 competitors?"
python ask.py --all "Who won the gold medal in the event held at ExCeL Exhibition Centre on 30 July at the 2012 Summer Olympics?"
```

`ask.py` prints the agent's steps as they happen, from the same trace the benchmark
records, and `--all` puts the three pipelines' answers and costs side by side.

## Tests

```bash
.venv/bin/python -m pytest        # 61 tests, no network, under a second
.venv/bin/ruff check .            # lint, configured in pyproject.toml
```

The tests run the tool layer over a seven-event graph and pin the behaviours the
agent's accuracy turned out to hinge on: numeric thresholds arriving as strings,
token-wise `contains`, absent values, the empty-aggregate status, the field check,
the coverage gate, the stop-reason order and the verbatim check.

## Benchmarking

```bash
.venv/bin/python eval2/run_public.py     # 100 questions x 3 pipelines, resumable
.venv/bin/python eval2/run_hidden.py     # 50 held-out questions -> submission bundle
.venv/bin/python eval2/report.py         # results -> frontend/public/report.json

cd frontend && npm install && npm run dev    # dashboard
```

Both runners resume: results append per row, and a rerun skips what is already
there. On a rate-limited free tier that matters, and `GROQ_API_KEYS` accepts
comma-separated fallback credentials that are rotated when one runs low.
`--qids` selects named questions, `--out` writes a side file so the headline
results are untouched, and `--compact` drops superseded rows from a file that has
been through several sessions.

---

## Ingestion is deterministic

Infoboxes are structured key-value text and titles follow
`Sport at the YEAR Season Olympics – Event`, so parsing is rule-based: zero LLM cost
and no extraction error to audit. Field coverage over the 2,162 Olympic documents:

| field | coverage |
|---|---|
| games, event, gold | 100% |
| date | 98.9% |
| competitors | 98.5% |
| nations | 98.4% |
| next | 97.6% |
| venue | 96.3% |
| prev | 93.4% |

**Nothing is imputed.** A field the corpus omits stays absent, end to end: the
loader writes an explicit sentinel and the TigerGraph backend maps it back to
absent, so a numeric predicate can never silently match a missing value.

## Validation before the agent

Two checks run before any pipeline exists, each isolating one failure mode.

`eval2/validate_schema.py` asks whether the schema can express every public answer,
given the gold documents. It measures the schema, not retrieval.

`eval2/validate_tools.py` asks whether the graph primitives can *reach* those answers
from the question text alone, with no gold ids. Current result:

```
qtype           total  solved  wrong  unreached
aggregation        21      21      0          0
lookup             19      19      0          0
multi_hop          28      27      0          1
superlative        10      10      0          0
temporal           22      22      0          0
ALL               100      99      0          1

tool-layer ceiling: 99.0%
```

The same script run with `--backend tigergraph` returns the same 99.0%, which is
how the ceiling is shown to be a property of the graph rather than of the local
JSONL files.

This is a ceiling, not a score. It uses regex predicate extraction as a stand-in for
the orchestrator's planner and is a test harness only — it is never part of a
pipeline, and no question-type handler exists on the scored path.

---

## Results

100 public questions, all three pipelines, one model and one generation budget
(`openai/gpt-oss-120b`, temperature 0, 1024 output tokens). Reproduce with
`python eval2/run_public.py`. Latency is the median per question, net of time
slept on the provider's rate limit.

| Pipeline | Exact match | Tokens / question | Context tokens | Grounding F1 | Steps | Median s | Refused |
|---|---:|---:|---:|---:|---:|---:|---:|
| P1 RAG | 53% | 4,900 | 4,658 | 0.360 | 1.0 | 1.0 | 13% |
| P2 GraphRAG | 48% | 5,412 | 5,205 | 0.096 | 3.6 | 1.6 | 19% |
| **P3 Agentic** | **96%** | **4,389** | **188** | **0.751** | 2.3 | 2.3 | 3% |

**The agent is both more accurate and cheaper.** It does not pay for its
reasoning with context: counting and comparison run in the database, so it puts
under 200 tokens of retrieved text in front of the model where the baselines put
about 5,000. Its extra second per question is two model calls, not retrieval:
tool calls take 2–15 ms.

### Where the difference comes from

| qtype | n | RAG | GraphRAG | Agentic | tokens R / G / A | agent context |
|---|---:|---:|---:|---:|---:|---:|
| lookup | 19 | 100% | 89% | 95% | 4,663 / 5,384 / 4,289 | 205 |
| temporal | 22 | 86% | 68% | **100%** | 4,670 / 5,394 / 4,069 | 204 |
| multi_hop | 28 | 46% | 54% | **89%** | 5,077 / 5,241 / 4,368 | 308 |
| aggregation | 21 | 10% | 5% | **100%** | 5,066 / 5,565 / 3,859 | 6 |
| superlative | 10 | 0% | 0% | **100%** | 5,016 / 5,658 / 6,453 | 164 |

Aggregation and superlative are the benchmark's point. One aggregation question
spans 43 gold documents; no top-k window answers it, and both baselines sit near
zero. The agent asks the database for the count instead: six tokens of context
per aggregation question, and every one of them right.

### When does the agent earn its cost?

Against GraphRAG, per question type:

| qtype | Δ accuracy | Δ tokens | verdict |
|---|---:|---:|---|
| aggregation | +95% | -1,706 | strictly better |
| superlative | +100% | +796 | worth the cost |
| multi_hop | +36% | -873 | strictly better |
| temporal | +32% | -1,326 | strictly better |
| lookup | +5% | -1,095 | strictly better |

Against RAG the picture is the same except for `lookup`, where RAG is 5 points
better and the agent is only slightly cheaper: **for a question one document
answers, the agent is overkill**, and the routing layer exists because of it.
45 of 100 questions took the fast path.

### What the four misses are

Three are refusals (`INSUFFICIENT_EVIDENCE`, scored as wrong by design). Two of
those are venue-and-date questions where two events share the venue and the
exact date string, so the graph cannot separate them and the agent declines to
guess; the third is a title lookup the planner never phrased as a filter the
graph could match. The fourth is a venue with six events on one day where the
agent chose the wrong one. The tool-layer ceiling from the question text alone is
99%, so the agent is within three points of what the graph can express.

**Variance.** Temperature is 0, but tool-calling is not fully deterministic: the
build before the final pass scored 86% and a re-run of its failures moved
individual question types by up to ten points. Single-run numbers should be read
with that in mind; the per-type margins over the baselines are far larger than
the variance.

### The final pass

Reading the failed traces of the 86% build found six defects, none of them a
question-type handler: numeric thresholds arriving as strings matched nothing (a
backend parity bug), a filter on a field the vertex type lacks returned empty
instead of an error, a count of zero was reported as `ok`, tool results were
rendered so verbosely that four rows filled the window, `contains` could not
reach venue names the corpus writes with joined words, and an answer that was the
corpus's string with the separators changed scored as wrong. Each is described in
[docs/ARCHITECTURE.md §11b](docs/ARCHITECTURE.md), each has a unit test, and the
TigerGraph backend agrees with the local one on all 35 parity cases after the
change. The verbatim check fired on 13 of the 100 questions and is recorded as a
step in each of those traces.

### Hidden set

The 50 held-out questions were run on the final build against TigerGraph
(`python eval2/run_hidden.py`, `GRAPH_BACKEND=tigergraph`): 49 answered, 1
declined, 4,476 tokens and 2.4 steps per question, 12 on the fast path. The
bundle with full traces is `data/results/submission.json`.

## Status

| Layer | State |
|---|---|
| Ingestion, property graph | done, coverage measured |
| TigerGraph schema, GSQL primitives, load | done, counts verified |
| Both backends | done, 35/35 parity checks including vector search and the final predicate semantics |
| Vector index | done, 19,832 chunks, local ONNX embeddings |
| Three pipelines | done |
| Agent harness, orchestrator, coverage gate, routing | done |
| Public benchmark | done, 300 runs on the final build |
| Hidden-50 bundle | done, `data/results/submission.json`: 49/50 answered on TigerGraph, 1 refused |
| Dashboard | done, static, reads `frontend/public/report.json` |
| Unit tests | done, 61, `python -m pytest` |
| Demo video | to record |

## Attribution

Corpus text is derived from English Wikipedia, licensed
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), and is redistributed
here unmodified under `hackathon-resources/` as provided by the organisers.
