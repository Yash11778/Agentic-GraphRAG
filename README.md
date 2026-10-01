# Agentic GraphRAG over the Olympics corpus

Submission for the **TigerGraph Agentic GraphRAG Hackathon, Round 1**.

**Live dashboard:** [huggingface.co/spaces/Yash-1903/agentic-graphrag](https://huggingface.co/spaces/Yash-1903/agentic-graphrag): results, cost of agency, the held-out
set, and every question's answers and agent trace.

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

**At a glance** (final build, every number from a full run on TigerGraph; details
under [Results](#results)):

| | RAG | GraphRAG | **Agentic GraphRAG** |
|---|---:|---:|---:|
| Exact match, 100 public questions | 52% | 47% | **99%** |
| Tokens per question | 4,957 | 5,484 | **4,284** |
| Retrieved context per question | 4,655 | 5,206 | **140** |
| Declined to answer | 14% | 20% | 0% |
| Hidden 50, answered | 42 / 50 | 42 / 50 | **50 / 50** |

The agent is the most accurate pipeline and also the cheapest, because counting and
comparison run inside TigerGraph instead of in the model's context window. The one
place it is overkill is a question a single document answers: there plain RAG is
just as accurate, and the agent routes those to a short path.

---

## The corpus and why it is hard

2,951 Wikipedia documents, ~5.47M tokens. 2,187 are Olympic event articles with
machine-parseable infoboxes; 764 are films, people and companies that no question
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
typed property graph  ──────▶  TigerGraph (GSQL + native vector index)
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
records, and `--all` puts the three pipelines' answers and costs side by side. With
`GRAPH_BACKEND=tigergraph` in `.env` every step runs against Savanna.

## Tests

```bash
.venv/bin/python -m pytest        # 73 tests, no network, under a second
.venv/bin/ruff check .            # lint, configured in pyproject.toml
```

The tests run the tool layer over a seven-event graph and pin the behaviours the
agent's accuracy turned out to hinge on: numeric thresholds arriving as strings,
token-wise `contains`, absent values, the empty-aggregate status, the field check,
the coverage gate, the stop-reason order, the verbatim check, whole-number
matching in `contains`, and the provenance labels on result files.

## Benchmarking

```bash
.venv/bin/python eval2/run_public.py     # 100 questions x 3 pipelines, resumable
.venv/bin/python eval2/run_hidden.py     # 50 held-out x 3 pipelines -> submission bundle
.venv/bin/python eval2/report.py         # results -> frontend/public/report.json

cd frontend && npm install && npm run dev    # dashboard
```

The dashboard is static and reads the committed `report.json`: a three-way results
view, the cost-of-agency chart against either baseline, the held-out set, and a
question explorer that replays any of the 100 investigations step by step.

Both runners resume: results append per row, and a rerun skips what is already
there. On a rate-limited free tier that matters, and `GROQ_API_KEYS` accepts
comma-separated fallback credentials that are rotated when one runs low.
`--qids` selects named questions, `--out` writes a side file so the headline
results are untouched, and `--compact` drops superseded rows from a file that has
been through several sessions.

**Every result row records where it came from.** A row is stamped with the
backend, the model and the time it was written, and every file-level label (the
`backend` in `submission.json` and `public_summary.json`, the provenance block in
`report.json`) is derived from the rows rather than read from the environment. A
resume onto a file written against the other backend is refused, and a file whose
rows disagree cannot be summarised or bundled at all. `run_hidden.py --package`
and `run_public.py --summarise` rebuild the derived files from the rows without
running anything.

---

## Ingestion is deterministic

Infoboxes are structured key-value text and titles follow
`Sport at the YEAR Season Olympics – Event`, so parsing is rule-based: zero LLM cost
and no extraction error to audit. Field coverage over the Olympic event documents
(measured on the 2,162 that parsed first; 25 tennis articles that open with a
tournament infobox were recovered later and are in the graph):

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
(`openai/gpt-oss-120b`, temperature 0, 1024 output tokens), all against
TigerGraph (`GRAPH_BACKEND=tigergraph`): every tool call is answered by the
installed GSQL queries and the vector index on Savanna. Reproduce with
`GRAPH_BACKEND=tigergraph python eval2/run_public.py`. Latency is the median per
question, net of time slept on the provider's rate limit.

| Pipeline | Exact match | Tokens / question | Context tokens | Grounding F1 | Steps | Median s | Refused |
|---|---:|---:|---:|---:|---:|---:|---:|
| P1 RAG | 52% | 4,957 | 4,655 | 0.359 | 1.0 | 1.3 | 14% |
| P2 GraphRAG | 47% | 5,484 | 5,206 | 0.096 | 3.6 | 2.3 | 20% |
| **P3 Agentic** | **99%** | **4,284** | **140** | **0.776** | 2.2 | 3.0 | 0% |

**The agent is both more accurate and cheaper.** It does not pay for its
reasoning with context: counting and comparison run in the database, so it puts
under 200 tokens of retrieved text in front of the model where the baselines put
about 5,000. Its extra 1.7 s per question over RAG is one more model call and a
Savanna round trip per tool call (median 274 ms), not retrieval volume.

### Where the difference comes from

| qtype | n | RAG | GraphRAG | Agentic | tokens R / G / A | agent context |
|---|---:|---:|---:|---:|---:|---:|
| lookup | 19 | 100% | 89% | **100%** | 4,732 / 5,451 / 4,071 | 118 |
| temporal | 22 | 82% | 68% | **100%** | 4,726 / 5,463 / 4,107 | 192 |
| multi_hop | 28 | 50% | 50% | **96%** | 5,137 / 5,328 / 4,132 | 209 |
| aggregation | 21 | 5% | 5% | **100%** | 5,118 / 5,631 / 3,931 | 6 |
| superlative | 10 | 0% | 0% | **100%** | 5,049 / 5,721 / 6,246 | 152 |

Aggregation and superlative are the benchmark's point. One aggregation question
spans 43 gold documents; no top-k window answers it, and both baselines sit near
zero. The agent asks the database for the count instead: six tokens of context
per aggregation question, and every one of them right.

### When does the agent earn its cost?

Against GraphRAG, per question type:

| qtype | Δ accuracy | Δ tokens | verdict |
|---|---:|---:|---|
| aggregation | +95% | -1,700 | strictly better |
| superlative | +100% | +526 | worth the cost |
| multi_hop | +46% | -1,197 | strictly better |
| temporal | +32% | -1,356 | strictly better |
| lookup | +11% | -1,380 | strictly better |

Against RAG the picture is the same except for `lookup`, where both are at 100%
and the agent is 661 tokens cheaper but no more accurate: **for a question one
document answers, the agent is overkill**, and the routing layer exists because
of it. 53 of 100 questions took the fast path.

### What the one miss is

`pub-099` is a venue-and-date question where two events share the venue and the
exact date string. The agent copied the right four-name medal cell exactly and
then appended a name from the other event it found at that venue that day, so
strict exact match scores it wrong. The organisers have said multi-hop scoring
accepts any event held at that venue in that range, and any member of a
gold-medal team, so their scorer may well count it; this repository keeps the
strict score. It is also the one question the tool-layer ceiling does not reach
from the question text alone (99%), so the agent is at the ceiling of what the
graph can express.

**Variance.** Temperature is 0, but tool-calling is not fully deterministic:
between the 96% run and the final build, three questions changed answer for
reasons unrelated to the code change (one lookup was answered, another lookup
was refused, and one ambiguous venue-and-date question went from a refusal to a
misspelt copy). Single-run numbers should be read with that in mind; the
per-type margins over the baselines are far larger than the variance.

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
change. The verbatim check fired on 11 of the 100 questions and is recorded as a
step in each of those traces, together with what the model wrote.

A last pass read the four remaining failed traces of the 96% build and found two
more defects, both general and both unit-tested. A number inside a `contains`
value matched inside other numbers ("6" inside "2016"), so a date filter returned
every event held at a venue that year and the right one fell past the rows the
model is shown; a digit run now matches only as a whole number, in both backends
(TigerGraph gained an `s_number` GSQL op), and the parity check grew to 38/38.
And the verbatim check could replace an answer that named both the medallist
and the event with the event's title; it now stands down whenever the answer
contains a medal cell of a cited event. Both are described in
[docs/ARCHITECTURE.md §11b](docs/ARCHITECTURE.md). The agentic pipeline was
re-run in full on the fixed build, on both backends; the baselines do not touch
the changed code.

### The same 100 questions on the local backend

The headline file `data/results/public.jsonl` holds two TigerGraph runs of the
final build: the agent's, and the two baselines', which were re-run on
TigerGraph on 2026-09-25 because the organisers asked for TigerGraph's own
vector search to be the one scored. Every row is stamped `tigergraph`. All
three pipelines were also run over the same 100 questions against the in-memory
backend, which reads the same graph from `data/graph/*.jsonl` and scans the
same embeddings exhaustively; those rows are kept as a check in
`data/results/public_local.jsonl`:

| Pipeline | TigerGraph (headline) | local | Answers identical |
|---|---:|---:|---:|
| RAG | 52% | 53% | 89 of 100 |
| GraphRAG | 47% | 48% | 91 of 100 |
| Agentic | 99% | 98% | 98 of 100 |

The agent's two differing answers are the two local misses: a lookup the model
refused locally it answered on TigerGraph, and on the venue-and-date question
it misspelt the medal cell locally (`Schempt` for `Schempp`) where on TigerGraph
it copied it exactly and appended a second name. The baselines put 5,000 tokens
of context in front of the model and read from an approximate vector index on
TigerGraph against an exhaustive scan locally, so more of their answers move;
the net is one point each, in neither pipeline's favour. The backend adds
latency, not error: a tool call on Savanna takes a median 274 ms against 2 to
15 ms in memory.

### Hidden set

The 50 held-out questions were run through all three pipelines on the final
build against TigerGraph (`python eval2/run_hidden.py`,
`GRAPH_BACKEND=tigergraph`), because the organisers asked for every question
answered by every pipeline with its tokens and latency.

| | RAG | GraphRAG | Agentic |
|---|---:|---:|---:|
| Answered | 42 / 50 | 42 / 50 | **50 / 50** |
| Declined | 8 | 8 | **0** |
| Tokens per question | 4,936 | 5,263 | **4,537** |
| Retrieved context | 4,600 | 4,977 | **167** |
| Median active time | 2.0 s | 4.6 s | 3.2 s |

There are no gold answers for this set, so the only comparison it supports is
cost and coverage, not accuracy. Both baselines decline 8 of the 50; the agent
declines none, and does it on the least context of the three. The agent's own
run is 2.4 steps per question with 14 on the fast path.

Latency is reported as a median of active time — time not slept on the
provider's rate limit — because the run met Groq timeouts that put one
GraphRAG question at 1,114 s. That is provider behaviour, not pipeline cost, and
a mean would report it as though it were.

The bundle with full traces is `data/results/submission.json`; its
`backend` label is derived from the rows, each of which is stamped
`tigergraph`, and every one of the 62 tool calls in those traces took 263 to
360 ms, which is a Savanna round trip and not the 2 to 15 ms of the in-memory
backend. These rows were written before the per-row timestamp existed, so the
bundle's `run_finished_at` is null rather than guessed; the run was made on
2026-09-15. Against the previous build's bundle, 48 answers are identical; the
one refusal was answered, and one ambiguous venue-and-date question (two
events, one day) now names both gold medallists rather than one.

## Status

| Layer | State |
|---|---|
| Ingestion, property graph | done, coverage measured |
| TigerGraph schema, GSQL primitives, load | done, counts verified |
| Both backends | done, 38/38 parity checks including vector search and the final predicate semantics; the agent's public answers agree across backends on 98/100 |
| Vector index | done, 19,832 chunks, local ONNX embeddings |
| Three pipelines | done |
| Agent harness, orchestrator, coverage gate, routing | done |
| Public benchmark | done, 300 runs on the final build against TigerGraph (agent 99/100); the same 300 on the local backend in `data/results/public_local.jsonl` |
| Hidden-50 bundle | done, `data/results/submission.json`: all three pipelines x 50 on TigerGraph, agent 50/50 answered, baselines 42/50 |
| Dashboard | done, static, reads `frontend/public/report.json`; live at [Hugging Face](https://huggingface.co/spaces/Yash-1903/agentic-graphrag) |
| Unit tests | done, 73, `python -m pytest` |
| Demo video | delivered with the submission, not in the repo |

## Attribution

Corpus text is derived from English Wikipedia, licensed
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), and is redistributed
here unmodified under `hackathon-resources/` as provided by the organisers.
