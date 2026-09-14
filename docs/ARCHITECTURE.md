# Agentic GraphRAG — Architecture Specification (locked)

Binding design document. Any change to a **[LOCKED]** decision requires an explicit
note in this file explaining why, so the repo never silently drifts from the spec.

---

## 0. What we are answering

TigerGraph Agentic GraphRAG Hackathon, Round 1. Three pipelines over one corpus,
benchmarked head to head, to answer: **when does an agentic multi-step investigation
beat a single GraphRAG or RAG retrieval, and when is it overkill?**

Corpus: 2,951 Wikipedia documents, ~5.47M tokens. 2,162 are Olympic event articles
with machine-parseable infoboxes; ~789 are films / people / companies that no
question touches (distractors that punish naive vector similarity).

Questions: 100 public (with answers), 50 hidden (answers held out). All Olympic.
Every answer is a single short string. Gold document ids are given for the public set.

| qtype | public | hidden | what it structurally requires |
|---|---:|---:|---|
| aggregation | 21 | 15 | enumerate a complete set, numeric filter, count |
| multi_hop | 28 | 10 | reverse lookup by venue + date, then medal edge |
| temporal | 22 | 8 | step the Games edition chain, then medal edge |
| lookup | 19 | 7 | one document |
| superlative | 10 | 10 | enumerate a complete set, argmax |

Half the hidden set is aggregation + superlative. Top-k similarity retrieval cannot
answer those at any k that fits a context window: one question spans up to 43 gold
documents. That is theheart  of the benchmark story.

---

## 1. Non-negotiable design decisions

### [LOCKED-1] Tools are general primitives, never question-type handlers
There is no `answer_aggregation_question()`. There is `graph_filter` and
`graph_aggregate`, and the orchestrator composes them. Hardcoding one handler per
qtype fits the public benchmark, fails unseen phrasings in the hidden set, and is
visible to any judge reading the repo. **The composition IS the agentic behaviour
being scored.**

### [LOCKED-2] The LLM never does arithmetic or set enumeration
Counting, comparison, max/min and completeness checks execute in GSQL. The model
plans, picks tools, and reads results back into natural language. This is both more
accurate than generative counting and far cheaper in tokens. It is the position we
defend in the Q&A.

### [LOCKED-3] All three pipelines share one tool layer
RAG, GraphRAG and Agentic GraphRAG call the same underlying retrieval functions.
They differ only in *control flow*. A benchmark where the pipelines have different
retrieval quality underneath is not a benchmark of agentic reasoning.

### [LOCKED-4] Ingestion is deterministic, with an LLM fallback that is not used here
Infoboxes are structured key-value text; titles follow `Sport at the YEAR Season
Olympics – Event`. Parsing is rule-based: zero LLM cost, ~100% field fidelity.
An LLM extraction path for unstructured corpora (the bring-your-own-data bonus) is
optional and, if built, stays **off** the scored path. It is not built today, and
the README does not claim otherwise.

### [LOCKED-5] No answer is emitted from an incomplete evidence set
For any question requiring enumeration, the agent asks the graph for the expected
cardinality, compares it against what it retrieved, and refuses to answer while the
set is incomplete. Stopping is a decision the agent justifies, not a step counter
running out.

### [LOCKED-6] Failure is reported, never papered over
No snapshot fallbacks, no auto-pass in scoring, no silent substitution of a weaker
path. A retrieval miss is `status="insufficient_evidence"` and scores as wrong.
(Carried forward from the previous round; it is why that submission was credible.)

### [LOCKED-7] Equal generation budget across pipelines
Same model, same temperature, same max output tokens for all three. Token deltas
must come from retrieval precision, not from one pipeline being allowed to say less.

Enforced mechanically: `agentic/config.py` is the only place these are read and
`agentic/llm.py::complete` the only place they are applied, so a pipeline cannot
quietly widen its own budget. Current setting is Groq `openai/gpt-oss-120b`,
temperature 0, 1024 output tokens, reasoning effort `low`. gpt-oss bills hidden
reasoning as output tokens, which makes reasoning effort part of the budget rather
than a free knob.

---

## 2. Layers

```
                 ┌──────────────────────────────────────────────┐
  corpus.jsonl ─▶│ L1  INGESTION (deterministic, zero LLM)      │
                 │  infobox parser → typed property graph       │
                 │  chunker + embedder → vector index           │
                 └───────────────┬──────────────────────────────┘
                                 ▼
                 ┌──────────────────────────────────────────────┐
                 │ L2  STORAGE — TigerGraph                     │
                 │  property graph (GSQL)  +  Vector DB         │
                 └───────────────┬──────────────────────────────┘
                                 ▼
                 ┌──────────────────────────────────────────────┐
                 │ L3  TOOL LAYER (general primitives)          │
                 │  vector_search   doc_fetch    entity_link    │
                 │  graph_neighbors graph_filter graph_aggregate│
                 │  edition_step    medal_lookup                │
                 └───┬──────────────┬───────────────┬───────────┘
                     ▼              ▼               ▼
              ┌───────────┐  ┌────────────┐  ┌──────────────────┐
              │ P1  RAG   │  │ P2 GraphRAG│  │ P3 AGENTIC       │
              │ vector→gen│  │ link→fixed │  │ orchestrator loop│
              │           │  │ traverse   │  │ over L4 harness  │
              └───────────┘  └────────────┘  └──────────────────┘
                     └──────────────┴───────────────┘
                                 ▼
                 ┌──────────────────────────────────────────────┐
                 │ L5  EVALUATION — exact match, retrieval P/R, │
                 │     token split, trace metrics               │
                 └───────────────┬──────────────────────────────┘
                                 ▼
                 ┌──────────────────────────────────────────────┐
                 │ L6  DASHBOARD — 3-way compare, trace viewer, │
                 │     "when does the agent earn its cost"      │
                 └──────────────────────────────────────────────┘
```

---

## 3. L1 — Ingestion

`agentic/ingest/parse_infobox.py`   parse infobox block → dict
`agentic/ingest/build_graph.py`     docs → vertices.jsonl + edges.jsonl
`agentic/ingest/schema.gsql`        the graph schema, one schema-change job
`agentic/ingest/load_tigergraph.py` schema + batched REST upsert, count-verified
`agentic/ingest/build_vectors.py`   chunk + embed → TigerGraph Vector DB

Normalisation rules (all deterministic, all unit-tested):
- `competitors: "23 teams"` → int 23, flag `unit="teams"` (23 such rows).
- `date` / `dates` unify to `date_raw` + parsed `date_start` / `date_end`.
- `games: "2012 Summer"` → Games vertex `2012-Summer`, year 2012, season Summer.
- title split on ` – ` → sport, event name. Title kept verbatim as `title`
  (superlative answers ARE the full title string — exact match matters).
- `prev`/`next` are years → resolve to Games vertices within the same sport+event
  to build the edition chain.

Coverage measured on the 2,162 Olympic docs: games/event/gold 100%, date 98.9%,
competitors 98.5%, nations 98.4%, venue 96.3%, next 97.6%, prev 93.4%.
Missing fields are recorded as absent, never imputed.

**Absence across backends.** TigerGraph has no NULL for a primitive attribute, so
the loader writes `-1` for an absent INT and `""` for an absent STRING, and the
TigerGraph backend maps both back to `None` on read. Without that round trip a
predicate such as `competitors > 0` would match the 32 events whose competitor
count the corpus never stated -- a wrong aggregate that nothing downstream could
detect.

---

## 4. L2 — Graph schema

```
Games(id "2012-Summer", year, season)
Sport(id, name)
Venue(id, name)
Athlete(id, name)
Nation(id NOC code, name)
Event(id = doc_id, title, event_name, competitors INT, competitors_unit,
      nations INT, date_raw, date_start, date_end, win_value, url, text)

Event  -AT_GAMES->     Games
Event  -IN_SPORT->     Sport
Event  -HELD_AT->      Venue
Event  -WON_GOLD->     Athlete      (also WON_SILVER, WON_BRONZE)
Athlete-REPRESENTS->   Nation       (edge attr: event_id, medal)
Event  -PRECEDED_BY->  Event        (edition chain from prev/next)
Games  -PRECEDED_BY->  Games        (season-scoped edition chain)
```

Why this shape:
- `competitors` / `nations` as **numeric vertex attributes** turn every aggregation
  and superlative question into one GSQL query instead of a 43-document context.
- `HELD_AT` + date attributes make the multi_hop questions (which name **no entity**,
  only a venue and a date) a reverse index lookup that embeddings cannot do.
- `PRECEDED_BY` on Games answers "the Summer Olympics immediately before 2016"
  from the corpus rather than from model memory — the README is explicit that the
  corpus, not the world, is ground truth.

---

## 5. L3 — Tool contract

Every tool returns `ToolResult(data, evidence[], tokens, latency_ms, status)`.
Evidence items carry `doc_id`, `snippet`, `source_tool` so citations are automatic.

| tool | signature | used by |
|---|---|---|
| `vector_search` | (query, k) → chunks | P1, P2, P3 |
| `doc_fetch` | (doc_ids) → documents | P1, P2, P3 |
| `entity_link` | (text) → vertices + type + confidence | P2, P3 |
| `graph_neighbors` | (vertex, edge_types, hops) → subgraph | P2, P3 |
| `graph_filter` | (vertex_type, predicates) → vertices | P3 |
| `graph_aggregate` | (vertex_type, predicates, op, field) → scalar | P3 |
| `edition_step` | (games, direction, n) → Games | P3 |
| `medal_lookup` | (event, medal) → Athlete + Nation | P2, P3 |

`graph_filter` and `graph_aggregate` are the primitives that make LOCKED-1 possible:
the agent supplies the predicate, so an unseen phrasing still composes.

---

## 6. L4 — Agent harness

`agentic/harness/state.py`     InvestigationState: question, plan, steps, evidence,
                               open gaps, budget, stop reason
`agentic/harness/evidence.py`  dedup by doc_id, provenance, coverage accounting
`agentic/harness/budget.py`    max steps, max tokens, wall clock; soft + hard caps
`agentic/harness/registry.py`  tool registry + JSON schemas handed to the model
`agentic/harness/trace.py`     ordered step log (see §8)

Stop criteria, evaluated in order, each recording an explicit reason:
1. `sufficient_evidence` — evaluator says the open gaps are closed.
2. `coverage_complete` — enumerated set size == expected cardinality (LOCKED-5).
3. `no_progress` — two consecutive steps added no new evidence.
4. `budget_exhausted` — step / token / time cap hit.
5. `unrecoverable` — every applicable tool returned empty.

Outcomes 3–5 emit `status="insufficient_evidence"` and score as wrong (LOCKED-6).

---

## 7. L4 — Orchestrator and specialists

The orchestrator does **plan → select → execute → evaluate → decide**, re-deciding
after every step from (a) the question, (b) evidence so far, (c) remaining gaps.
It is not a fixed chain and it is allowed to change strategy mid-investigation,
which is recorded in the trace as a `strategy_change` event.

Specialists (the seven the brief names), each a focused prompt + tool subset:

| agent | job |
|---|---|
| `entity_linker` | question spans → graph vertices, with confidence |
| `graph_traverser` | pick edge types and hop depth, walk them |
| `similarity_searcher` | vector query formulation, k selection |
| `doc_retriever` | pull full documents for chosen ids |
| `aggregator` | build predicate + op for `graph_aggregate` |
| `multihop_reasoner` | chain intermediate results into the next hop |
| `evidence_evaluator` | does current evidence answer it; what is missing |

**Cost-aware routing.** Before planning, the orchestrator estimates whether the
question needs an investigation at all. A single-document lookup short-circuits to
the plain retrieval path and records `route="fast_path"`. This builds the
hackathon's own research question into the product instead of only charting it.

---

## 8. Trace schema (what makes §"Agentic effectiveness" scoreable)

Per step: `step_no, agent, tool, retrieval_method, args_digest, status,
latency_ms, input_tokens, output_tokens, context_tokens, chunks_returned,
new_evidence_count, citations, gap_before, gap_after, strategy_change`.

Per run: `route, total_steps, tools_used, total_tokens (split three ways),
wall_clock_s, evidence_docs, citations, stop_reason, coverage_expected,
coverage_actual, confidence`.

---

## 9. L5 — Evaluation

`eval2/run_public.py`  100 questions × 3 pipelines → per-row results + traces
`eval2/run_hidden.py`  50 questions × agentic → submission bundle
`eval2/score.py`       metrics
`eval2/report.py`      dashboard feed

Metrics:
- **Accuracy** — normalised exact match (case, articles, punctuation, unicode
  dashes, numeric words↔digits). Reported strict and lenient; strict is headline.
- **Grounding** — did the cited docs include the gold docs. Precision/recall/F1
  against `gold_doc_ids`. This is the evidence-quality score, and it is objective
  because the gold ids are given.
- **Completeness** — for enumeration questions, retrieved set vs expected set.
- **Token efficiency** — context / input / output tokens split, per pipeline.
- **Trace metrics** — §8, agentic only.
- **Cost of agency** — Δaccuracy per 1k extra tokens vs GraphRAG, **per qtype**.
  This is the number that answers the hackathon's headline question.

An LLM judge is retained only as a secondary check on free-text answers; exact
match against a provided gold string is the primary and is not overridable.

---

## 10. L6 — Dashboard

Three views: (1) three-way comparison with token split and accuracy per qtype;
(2) live investigation trace viewer — step timeline, tool calls, evidence
accumulating, the stop decision; (3) **"when does the agent earn its cost"** —
accuracy gain vs token cost per question type, the chart that states the finding.

---

## 11. Known risks and the mitigation for each

| risk | mitigation |
|---|---|
| Overfitting to the 5 public qtypes | LOCKED-1; hidden-set phrasings compose from primitives |
| Deterministic parsing does not generalise | LLM extraction path exists for BYO data; stated plainly in README |
| Hidden question needs an attribute we did not model | vector_search + doc_fetch remain a fallback on every path |
| Agent looks "better" only because it has better tools | LOCKED-3: shared tool layer |
| Graph makes the task trivial, undermining the agent story | Honest finding, reported as such: the agent's value is planning and coverage verification, not retrieval it alone can do |

---

## 11b. Implementation notes — where the build differs from this spec, and why

Recorded here because §0 says a LOCKED decision may not drift silently. None of
these changes a locked decision; each is a correction found while building.

**Trace lives in `agentic/pipelines/base.py`, not `harness/trace.py`.** All three
pipelines emit the same `TraceStep`, and the RAG baseline must not import the
agent harness to do it. One definition, one file, no duplicate schema.

**The coverage gate is keyed by predicate signature.** §6 describes coverage as
"enumerated set size == expected cardinality". Implemented as a single global
comparison it is wrong: a count of 238 events in 2012 has nothing to say about a
later one-row lookup for the largest of them, and the gate blocked every
superlative answer. It now compares a listing against a count **of the same
predicates**, and fires only when the agent enumerated the set it is answering
from. Aggregation answers computed in-database are not gated at all -- the
database did the enumerating.

**Progress means usable data, not new documents.** `graph_aggregate` cites no
document and is the most informative call in the system. Counting only new
documents made every counting question stop as `no_progress`. A repeat of a call
already made still counts as standing still.

**Only retrieval steps can stall a run.** A rejected generation or a coverage
push-back is bookkeeping. Counting those toward `no_progress` ended runs that had
made one real attempt.

**A malformed tool call is recoverable.** The provider validates tool arguments
and rejects a bad generation with a 400 before it reaches us. That is the model's
mistake and it can correct it, so it is surfaced as a response, recorded in the
trace, and answered -- not raised.

**Ingestion: the Olympic infobox is not always the first block.** Twenty-five
tennis articles open with `[Infobox tennis tournament event]` and carry the
Olympic infobox below it. The original `startswith` test dropped all 25 into the
distractor pile with their venues, dates and medals. Event count went from 2,162
to 2,187 when fixed. Coverage percentages in §3 are unchanged; they were measured
over the events that parsed.

**TigerGraph specifics found by testing, not documentation.** `getAttr` on an
edge type that lacks the attribute discards the whole accumulated row silently,
so the neighbour query branches by edge type. The undirected form `-(:e)-`
reports every edge twice, once under each name, losing direction; `-(_>:e)-` does
not. A vector attribute cannot be added in the same schema-change job that
creates its vertex. Vector search runs only in an installed query, never an
interpreted one.

**Predicate values are coerced in one place.** The planner writes a numeric
threshold as a string about half the time (`"63"`). The TigerGraph backend
always converted it; the local backend compared a string against an integer,
got a `TypeError`, and returned False for every row -- a count of zero, with
`status="ok"`. Two aggregation questions were lost to it, and it was invisible
because the two backends disagreed only for that input shape.
`graph_schema.coerce_value` is now called by both, so a predicate means the same
thing wherever it runs.

**`contains` is token-wise.** The corpus writes some venue names with joined
words (`Xiaohaituo Bobsleigh and Luge TrackBeijing`, `Beijing Science and
TechnologyUniversity Gymnasium`), and a question that spaces them correctly
could not reach them by substring. `contains` now requires every
whitespace-separated token of the value to occur in the stored string; the
TigerGraph backend expands one predicate into one `LIKE` per token, applied in
sequence, which is the same AND. A single-token value is the old substring
test, so nothing that matched before stops matching. Verified against Savanna:
`validate_backends.py` agrees on 35/35 cases including three added for these
semantics.

**A field the type lacks is an error, not an empty result.** Filtering `Games`
by `title` returned nothing, and the planner's loosening rule then spent two
more steps on a filter that could never match. `VERTEX_FIELDS` lists what each
type carries; the tool layer rejects the call with the real field list, and the
planner corrects in one step. The same table generates the field guide in the
tool schema, so the two cannot drift.

**A count of zero is `empty`.** `graph_aggregate` returned `{"matched": 0}` as
`status="ok"`, so the planner read zero as an answer rather than as a filter
that needs loosening. It is now reported as empty, with the same recovery cue a
listing gives.

**Tool results are rendered compactly.** A vertex row rendered as full JSON is
~600 characters, most of it keys and bookkeeping (`url`, `approx_tokens`, the
`_key` companions). A 700-token window therefore held four rows, and the planner
learned to pass `max_results=5` -- which is how the right event, sixth of six at
one venue on one day, was never seen. Rows are now rendered without the noise
fields and cut at a row boundary with a count of what was hidden; fourteen fit.

**The verbatim check.** The benchmark scores an exact string and the corpus
writes team medallists as one unbroken string. When the model's answer differs
from a string in a cited document only in separators, case or accents (compared
with `graph_schema.squash`, which strips everything but letters and digits), the
stored string is the answer, and the substitution is recorded as a trace step.
It never fires on a difference in a letter or a digit, and it is not question
aware. Titles and venues are covered as well as medal cells, which also handles
the model's curly apostrophes.

**Only retrieval steps count toward `unrecoverable`.** A malformed tool call is
recorded as an error step, and it was counted alongside empty filters toward the
three-failure stop, so one bad generation plus two empty filters ended a run
that had made two real attempts. It is now bookkeeping, like the coverage gate.

**Latency is reported net of throttling.** `PipelineResult.throttled_s` records
time slept on the provider's rate limit; the scorer reports `active_s` and the
median of it. A mean over a throttled run described the free tier's queue, not
the system.

---

## 12. Open items (owner: organisers)

- Submission file format for the hidden-50 outputs — ask on Discord/WhatsApp.
- Whether Round 2 ships additional data for evolving/conflicting facts.
