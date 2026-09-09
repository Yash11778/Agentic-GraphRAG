# GraphRAG vs Basic RAG — Architecture, Inference Flow, and Evaluation

All answers below are grounded in the actual code (`pipelines/`, `scripts/`, `eval/`) and the
committed benchmark results (`eval/results/run1.csv`). File references are included so any
claim can be verified directly in the code.

---

## 1. Walk us through the end-to-end architecture of your RAG and GraphRAG pipelines

We compare three pipelines over the same corpus — **63,632 U.S. court opinions, 117.5M
Gemini-verified tokens** (verified via Gemini's official `count_tokens` API, recorded in
`data/token_count_official.json`):

- **LLM-only** (`pipelines/pipeline1_llm.py`) — no retrieval at all. This baseline exists to
  prove the questions genuinely require the corpus: it collapses to 1.8% accuracy.
- **Basic RAG** (`pipelines/pipeline2_rag.py`) — flat FAISS vector search over 500,959
  chunks. This baseline represents standard out-of-the-box vector RAG.
- **GraphRAG** (`pipelines/pipeline3_graphrag.py`) — TigerGraph citation-graph traversal
  over real `CITES` edges.

**Controlled variables** — all three pipelines share:
- the same generator (Gemini 2.5 Flash),
- the same **512-token completion cap**,
- the same tiktoken `cl100k_base` token accounting,
- and the **identical question text** — so any difference in results is purely retrieval.

**Offline build** — one pipeline feeds both retrievers from the same raw data:

```
data/raw/dataset_100m_enriched.jsonl   (63,632 opinions, eyecite-enriched metadata)
        │
        ├── preprocess.py + build_faiss.py  →  500,959-chunk FAISS index   (Basic RAG)
        ├── ingest_pilot_graph.py --full    →  LegalCase + CITES edges     (GraphRAG)
        └── ingest_courts.py               →  Court + DECIDED_BY edges     (GraphRAG)
```

The knowledge graph that results: **63,632 `LegalCase` vertices, 1,413 `Court` vertices,
9,632 `CITES` edges, 63,632 `DECIDED_BY` edges**, live on a TigerGraph Savanna instance
(`LegalGraph`).

**Online serving** — a FastAPI backend (`api/app.py`) runs all three pipelines side by side
(in parallel threads), pre-warms the FAISS index, Gemini client, and TigerGraph auth token at
startup so the first user query is fast, and a React dashboard reports per-query tokens,
latency, cost, LLM-judge verdict, and BERTScore.

---

## 2. How does the ingestion and transformation pipeline work (chunking, indexing, enrichment, summarization)?

Step by step:

1. **Source acquisition.** U.S. court opinions (state + federal) from the Pile of Law
   `courtlistener_opinions` split → `data/raw/dataset_100m_enriched.jsonl` (63,632
   opinions). The underlying opinions are public domain (government edict doctrine).
2. **Metadata enrichment** (`scripts/enrich_dataset_metadata.py`): each opinion's case name,
   court, and year parsed with **eyecite + courts-db** (Free Law Project tooling — the same
   libraries real legal-tech uses for citation parsing).
3. **Citation edges** (`scripts/download_citations.py`): from CourtListener's public
   `citation-map` bulk data, filtered to edges where **both endpoints are in-corpus** —
   9,632 real citation pairs, zero synthetic edges.
4. **Chunking** (`scripts/preprocess.py`): text cleaned (markup stripped, whitespace
   normalized), then token-window chunking with **tiktoken cl100k_base: 256-token windows,
   32-token overlap** → **500,959 chunks**. The overlap prevents a fact from being split
   across a chunk boundary and lost to retrieval.
5. **Vector indexing** (`scripts/build_faiss.py`): **all-MiniLM-L6-v2** via fastembed
   (384-dim), embeddings L2-normalized and added to a **FAISS `IndexFlatIP`** — inner
   product on normalized vectors = exact cosine similarity, no ANN approximation error.
   Built streaming with checkpointing every 50 batches so a crash resumes mid-build.
6. **Graph ingestion**: `scripts/ingest_pilot_graph.py --full` loads LegalCase vertices +
   CITES edges; `scripts/ingest_courts.py` adds Court vertices + DECIDED_BY edges. Schema
   defined in `scripts/legal_schema.gsql` + `scripts/add_court_schema.gsql`.

**Deliberate design choice — no LLM in the ingestion loop.** We do **no LLM entity
extraction and no LLM summarization**. Generic GraphRAG frameworks extract entities per
chunk with an LLM — expensive (hundreds of thousands of calls at our scale) and
hallucination-prone. For case law the citation graph **is** the real relationship
structure: it comes from structured metadata and a public citation-map file, costs zero
LLM calls to build, and every edge is a verifiable fact. That's a per-domain insight, not
a shortcut — pick the graph your domain already contains.

---

## 3. What are the main architectural differences between your RAG and GraphRAG implementations?

| Dimension | Basic RAG | GraphRAG |
|---|---|---|
| Knowledge store | Flat FAISS index, 500,959 chunks | TigerGraph: 63,632 LegalCase + 1,413 Court vertices; 9,632 CITES + 63,632 DECIDED_BY edges |
| Retrieval unit | 256-token chunk | Whole case, compressed to a relevant 420-token window |
| Retrieval signal | Embedding cosine similarity | Explicit citation relationships (GSQL multi-hop traversal) |
| Question understanding | None — the raw question is just embedded | Entity resolution: citation ids / case names → graph vertices |
| Multi-hop capability | Emergent at best — must hope chunks from every chain case land in top-k simultaneously | First-class — walks A←B←C edges directly, with hop distance tracked |
| Context size | Fixed top-k=8 chunks (~2,000+ tokens), every question | Seeds + ≤2 neighbours under an adaptive ~1,300-token budget |
| Failure mode | Retrieves plausible-looking but wrong/incomplete chunks (silent) | Explicit `no_context` status when the graph has no match (loud, counted as FAIL) |

**One-liner:** Basic RAG asks *"what text looks like the question?"*; GraphRAG asks
*"which cases is this question about, and what do they actually cite?"* Similarity is a
guess; a citation edge is a fact.

---

## 4. Walk us through the RAG inference flow from user question to the context sent to the LLM

(`pipelines/pipeline2_rag.py`, `pipeline2()`)

1. **Embed the question** with the same **all-MiniLM-L6-v2** model used at index time
   (symmetry matters — a different query encoder would degrade recall), then L2-normalize
   so inner product equals cosine similarity.
2. **Search**: FAISS `IndexFlatIP` exact search returns the **top-k = 8** chunks. Exact
   (not approximate) search means Basic RAG is shown at its best — no ANN recall loss to
   blame.
3. **Assemble context**: the 8 chunks joined with `---` separators, in similarity order,
   with their source case names tracked for the dashboard.
4. **Generate**: one grounded prompt to Gemini — "use ONLY the context below… extract every
   relevant fact, name, date" — with the 512-token completion cap.
5. **Account**: prompt tokens + completion tokens counted with tiktoken cl100k_base;
   latency and cost recorded per call.

Typical cost of this flow: ~2,050 tokens of context per question → 2,288 avg total tokens.

---

## 5. How do you retrieve, rank, filter, and select the final context in the RAG pipeline?

Deliberately standard — a methodological choice:

- **Ranking**: pure cosine similarity — no re-ranker, no MMR, no recency weighting.
- **Filtering**: none — no similarity threshold, no metadata filters.
- **Selection**: fixed top-8, always, regardless of question complexity.

This represents what flat vector RAG gives you out of the box, and its failure mode is
exactly what the benchmark measures. Cosine similarity retrieves chunks that *look like*
the question — chunks sharing legal vocabulary, party names, or topic words — but a
multi-hop question needs chunks from **two or three specific opinions simultaneously**, and
top-k similarity has no mechanism to guarantee that joint coverage. One chain case usually
dominates the ranking and the others never make the top-8.

The numbers show it precisely: Basic RAG spends **2,288 avg tokens** (the most of any
pipeline) for **9.1% accuracy overall and 0.0% on 3-hop** — maximum spend, minimum
multi-hop return. If we'd added a re-ranker we'd be benchmarking our re-ranker; keeping the
baseline canonical keeps the comparison about the graph.

---

## 6. Walk us through the GraphRAG inference flow from user question to the context sent to the LLM

(`pipelines/pipeline3_graphrag.py`, `pipeline3()`) — four stages:

1. **Entity resolution (question → seed vertices).**
   - Primary path: explicit citation ids in the question — e.g. *"People v. Batson
     (6047231)"* — extracted with regex `\((\d{5,})\)` and mapped **directly to vertex
     ids**. This mirrors how real legal queries pin down which of several same-named
     opinions is meant. (Not answer leakage: Basic RAG receives the identical question
     text.)
   - Fallback path: extract "X v. Y" case-name patterns (regex over capitalized captions)
     and keyword n-grams (stopword-filtered unigrams/bigrams/trigrams, longest first), then
     resolve via the installed `find_case_by_keyword` GSQL query.
   - Capped at **2 seeds** (`max_seeds=2`) — measured: seeding from every same-named
     caption collision dilutes the context with wrong cases and lowers accuracy.
2. **Graph traversal.** The installed `citation_multihop_retrieve` GSQL query
   (`scripts/legal_schema.gsql`) walks real CITES edges **in both directions** — outgoing
   (what this case relies on) and incoming (what relies on this case) — out to **2 hops,
   max 5 cases per hop**. Each returned case carries its `@hop_distance` and a direction
   mask; a `@visited` accumulator guarantees no vertex is returned twice.
3. **Context compression.** Each case's full opinion is reduced to a **420-token relevance
   window** (`PER_CHUNK_CAP = 420`): document head + best-scoring sentence window, with a
   10× boost for citing sentences (full detail in Q8).
4. **Context assembly + generation.** Hop-0 seed cases are guaranteed in first (they are
   the opinions the question explicitly cites), then at most **2 citation neighbours**,
   under an adaptive token budget (~1,300, scales with seed count). Same Gemini call, same
   512-token cap as the other pipelines.

**Reliability engineering:** TigerGraph auth tokens
are cached and transparently re-acquired if the Savanna instance auto-suspends and
invalidates them mid-run; transient transport failures get exactly one retry after a 2s
pause, so an infra blip is never mis-recorded as "the graph had nothing" — while a genuine
empty result is honestly reported as `status="no_context"` and scored as a **FAIL**, never
silently excused. No snapshot fallback, no curated subset — every answer is a live GSQL
query.

---

## 7. How do entity extraction, graph traversal, semantic search, and other retrieval techniques work together?

The techniques are staged, each doing what it's structurally best at:

- **Entity linking (text → graph coordinates).** The question is unstructured text; the
  graph needs vertex ids. Citation-id extraction handles the precise case; case-name regex
  + keyword lookup via `find_case_by_keyword` handle the fuzzy case. Longest/most-specific
  candidates are tried first ("X v. Y" full captions before bigrams before unigrams).
- **Graph traversal (the multi-hop reasoning).** `citation_multihop_retrieve` does what no
  embedding can: it follows *actual relationships*. "What did the case this question cites
  rely on, and what later relied on it" is a two-line frontier expansion in GSQL and
  structurally impossible for flat top-k similarity.
- **Lexical-semantic scoring (sentence selection inside each document).** Once the graph
  has picked the right opinions, we still can't send whole opinions. Sentences are scored
  by question-keyword overlap, **plus a 10× boost for any sentence containing distinctive
  party-name tokens of another retrieved case** (`_distinctive_name_tokens` +
  `link_tokens`). That boost is the clever part: the sentence in opinion B that names
  opinion A is almost always the *citing sentence* — the exact place B applies or
  distinguishes A, i.e. the multi-hop link the question asks about. Generic caption words
  ("People", "State", "United") are stopworded so the boost only fires on genuinely
  distinctive names.

Division of labour in one line: **the graph finds the right documents; scoring finds the
right sentences inside them.** Neither alone suffices — scoring without the graph can't
cross documents, the graph without scoring would blow the token budget.

---

## 8. How do you rank, filter, fuse, deduplicate, or otherwise optimize the evidence retrieved by GraphRAG?

- **Ranking**: results sorted by hop distance. Hop-0 seeds (the cases the question cites)
  anchor the context first; 1-hop citation neighbours next; 2-hop after. Rationale: the
  seed cases are certain to be relevant; certainty decreases with distance.
- **Filtering / compression** (`_select_relevant_window`): the **head + best window**
  strategy. Court opinions front-load the disposition and citation context but often bury
  the actual holding mid-to-late in the text. We measured both naive strategies failing:
  head-only truncation loses buried holdings; a pure keyword window wanders into
  keyword-dense but irrelevant sections and loses the front matter. So: **⅓ of the 420-token
  budget keeps the document head**, then a sentence window grows greedily around the
  highest-scoring sentence from the remainder, extending forward on score ties (holdings
  tend to *follow* the sentence sharing the question's keywords, not precede it). Joined
  as `head [...] window`.
- **Fusion (cross-document linking)**: the `link_tokens` mechanism — distinctive party-name
  tokens from the *other* retrieved cases give a 10× score boost to citing sentences, so
  the passages that connect the retrieved opinions to each other are preferentially kept.
  This is what makes the compressed windows still *jointly* answer a multi-hop question.
- **Deduplication**: two layers — the GSQL query's `@visited` accumulator ensures no vertex
  is retrieved twice across hops/directions, and a `seen` set at assembly time drops any
  duplicate text chunk.
- **Budget optimization**: every hop-0 seed is guaranteed inclusion first, then at most 2
  neighbours, under a budget that **scales with seed count: 1,300 + 420 × (seeds − 2)**.
  This fix came from a measured failure: a flat 1,300 budget fit two full seed windows but
  starved 3-seed (3-hop) questions — 3 × 420 = 1,260 left zero room for citation
  neighbours, which showed up directly as a lower 3-hop pass rate. Each extra seed now buys
  exactly one extra window.

---

## 9. What parts of your architecture are primarily responsible for reducing token usage?

Three mechanisms, all on the **retrieval/input side**:

1. **Structural precision** — the graph returns the 3–4 cases actually connected to the
   question by citation, instead of 8 similarity chunks retrieved as "redundancy insurance"
   against embedding uncertainty. Basic RAG over-retrieves because it's guessing; GraphRAG
   doesn't have to, because relevance is a fact recorded in the graph.
2. **Relevance-window compression** — 420 tokens of the most relevant text per case (head +
   citing passage) instead of whole opinions that run thousands of tokens.
3. **Hard context budgeting** — seeds + max 2 neighbours under ~1,300 context tokens,
   enforced at assembly time. Basic RAG has no equivalent; its context is whatever top-8
   costs.

**Fairness guards**: the completion cap is identical (512) across all pipelines, so none of
the 37.8% comes from shorter answers — it is all input-side. And a `no_context` miss is
flagged as a **failure**, never counted as a token "win".

The deeper point: the token saving isn't a compression trick bolted on at the end — it
falls directly out of how retrieval works. Knowing *which* documents matter is the same
capability that lets you send *less* of them.

---

## 10. Show us your token reduction results and how they were measured

**Results** (run1, 55 questions × 3 pipelines):

| Pipeline | Avg tokens/question | vs Basic RAG | Avg cost/question | Pass rate |
|----------|:------------------:|:------------:|:-----------------:|:---------:|
| LLM-only | 609 | −73.4% | $0.00031 | 1.8% |
| Basic RAG | 2,288 | — | $0.00046 | 9.1% |
| **GraphRAG** | **1,423** | **−37.8%** | **$0.00032** | **50.9%** |

- LLM-only is cheapest but useless — proof that low tokens alone mean nothing; the metric
  that matters is tokens *at* accuracy. GraphRAG costs barely more than no-retrieval while
  passing 28× more often.
- The reduction is stable across hop depth: 1,419 avg tokens on 2-hop, 1,433 on 3-hop —
  the adaptive budget keeps spend flat even as questions get harder, while Basic RAG grows
  (2,256 → 2,374).

**How measured**:
- Every pipeline call counts **prompt tokens + completion tokens** with the same tiktoken
  `cl100k_base` encoder (`pipelines/utils.py`, `count_tokens`, `make_result`).
- The benchmark (`eval/evaluate.py`) records per-row totals to CSV and computes
  `reduction = 1 − mean(graphrag) / mean(basic_rag)` over all 55 questions.
- Per-question raw numbers are committed in `eval/results/run1.csv` — any row can be
  audited.
- The 117.5M corpus-size figure is measured separately via Gemini's official
  `count_tokens` API (`data/token_count_official.json`), not estimated.

---

## 11. Show us your accuracy results, including the evaluation methodology

**QA set construction** (`scripts/generate_multihop_qa.py`): 55 multi-hop questions —
**40 two-hop, 15 three-hop** — each built from an **actual in-corpus citation chain**
(A cites B [cites C]) found in the real citations data. Gemini receives the chain cases'
real texts and must write a question answerable **only by combining them**, plus a
ground-truth answer. Each question cites its cases by identifier (e.g. "People v. Batson
(6047231)") the way a real legal query disambiguates same-named captions — and both
retrieval pipelines receive the identical question text.

**Results (run1 — with token columns):**

| Pipeline | Pass rate | 2-hop | 3-hop | Avg tokens | Token reduction vs Basic RAG | BERTScore (raw / rescaled) | Avg latency |
|----------|:---------:|:-----:|:-----:|:----------:|:---------------------------:|:--------------------------:|:-----------:|
| LLM-only | 1.8% | 2.5% | 0.0% | 609 | −73.4% (but 1.8% accuracy) | 0.787 / 0.362 | 4.6s |
| Basic RAG | 9.1% | 12.5% | 0.0% | 2,288 | — (baseline) | 0.807 / 0.422 | 3.7s |
| **GraphRAG** | **50.9%** | **55.0%** | **40.0%** | **1,423** | **−37.8%** | **0.841 / 0.524** | 7.0s |

Headline: **GraphRAG passes 5.6× more often than Basic RAG while using 37.8% fewer
tokens** — better and cheaper simultaneously, because both effects share one cause
(retrieving the right evidence). The 3-hop column is the structural proof: 40.0% vs 0.0%.

**Methodology**:
- **Judge** (`eval/judge.py`): **one strict Gemini judge for the entire run** — single
  rubric ("PASS only if the answer correctly answers the question AND its key facts agree
  with ground truth; partial, hedged, or off-question answers FAIL"), no fallback judge, no
  mid-run judge swap, no auto-pass, judge errors count as FAIL. All 165 rows carry
  `judge_source=gemini_strict`. Absolute pass rates are low **by design** — a strict judge
  fails hedged answers for every pipeline equally; **the relative gap under one uniform
  judge is the measurement**.
- **BERTScore**: `evaluate.load("bertscore")` with `rescale_with_baseline=True`, computed
  for all three pipelines — a continuous semantic-similarity check alongside the binary
  judge, and the two agree on the ordering.
- **Reproducibility**: the full strict-judge benchmark was run **three times**
  (`run1.csv`–`run3.csv`, all committed): GraphRAG 50.9% / 52.7% / 50.9% (mean 51.5%);
  Basic RAG 9.1% and LLM-only 1.8% identical across all three runs — inside the ±3–5pt
  noise floor an LLM judge has over 55 questions.
- **Latency honesty**: GraphRAG is slower (7.0s vs 3.7s) — entity resolution + live graph
  query + generation vs a single vector lookup. That's the trade-off we accept for 5.6×
  accuracy; caching and query warm-up already reduce it, and it's engineering, not
  architecture.

---

## 12. Representative examples — including where GraphRAG performed worse than RAG

**Showcase win (3-hop, qid 53).** *"What principle regarding legal sufficiency of evidence
originated in People v. Contes (5905793), how did People v. Contes (5998617) apply it…"* —
a three-opinion chain where two cases even **share the same caption** and differ only by
citation id. Basic RAG **FAILED** (2,048 tokens): embeddings cannot tell two "People v.
Contes" opinions apart, let alone assemble the chain. GraphRAG resolved both ids to
distinct vertices, walked the chain, and **PASSED** with 1,918 tokens. Across the run,
**24 questions were GraphRAG-PASS / RAG-FAIL** (vs 1 the other way); on 3-hop, similarity
search went **0-for-15** — top-k has no mechanism to jointly retrieve an A←B←C chain.

**Where GraphRAG lost — exactly 1 of 55 (qid 29).** *Matter Torres v. Coughlin* on
substantial evidence for weapon possession. GraphRAG retrieved the right cases and answered
the Torres holding correctly, but its answer ended by asserting the court "did not apply"
the principle from *Matter Bryant v. Coughlin* — the ground truth says Torres aligns with
Bryant. Root cause: the 420-token relevance window selected from the Torres opinion missed
the passage connecting it to Bryant, so the model denied a link it couldn't see. Basic
RAG's 8 similarity chunks happened to include that connection and passed. Honest takeaway:
the failure mode is context-window selection — aggressive compression can drop the citing
passage — which is **exactly where the token savings come from**. It's a precision/recall
trade-off we tuned toward precision, and the measured price was 1 question out of 55
against 24 wins.

**No-context misses — 5 of 55 (qids 22, 41, 42, 49, 50).** Example: qid 22 cites its cases
in NY Slip Op format ("2020 NY Slip Op 04893") rather than a numeric corpus id, so seed
resolution found no vertex. All 5 returned `status="no_context"`, were scored FAIL, and are
**inside** the 50.9% — we count our misses. (Fixing the citation-format normalizer is the
obvious next-round improvement and would raise the ceiling to ~60%.)

---

## Token reduction — deep dive

**Core idea in one sentence:** Basic RAG has to **over-retrieve because it's guessing**;
GraphRAG can **retrieve exactly enough because the graph tells it what's relevant**. The
37.8% saving is not a compression trick bolted on at the end — it falls directly out of how
retrieval works.

**Why Basic RAG spends 2,288 tokens per question:** vector search returns a similarity
ranking, not an answer to "which documents matter." It can't know whether the 3rd chunk is
essential and the 7th is noise — so the standard defense is a fixed top-8 (8 × 256 ≈ 2,000
tokens of context) and hope. That padding is redundancy insurance against retrieval
uncertainty, and every question pays it.

**The three GraphRAG mechanisms:**

1. **Structural precision** — resolve cited cases to vertices, walk real CITES edges;
   typically 2 seeds + ≤2 neighbours (3–4 documents) instead of 8 blind chunks. Relevance
   isn't a similarity guess; it's a fact recorded in the graph.
2. **Relevance-window compression** — each case reduced to 420 tokens: ⅓ head + the
   best-scoring sentence window, with a 10× boost for sentences naming another retrieved
   case (the citing passage where the multi-hop link lives).
3. **Hard context budget** — 1,300 tokens base, +420 per seed beyond two; seeds guaranteed
   first, neighbours only while budget remains. Basic RAG has no equivalent.

**Why the comparison is honest:** equal 512-token output caps (all saving is on the input
side); same tiktoken encoder everywhere; reduction = 1 − mean(GraphRAG)/mean(Basic RAG)
over all 55 questions; and a no-context miss is a FAIL, never a token "win".

**Closing point:** normally cutting context costs accuracy. Here tokens went **down 37.8%**
while the pass rate went **up 5.6×** (50.9% vs 9.1%) — both for the same reason: the graph
puts the *right* evidence in the context, so we need less of it. That's the whole thesis of
GraphRAG in one result.
