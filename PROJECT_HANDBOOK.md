# GraphRAG for Patent Intelligence — Major Project Handbook

Everything needed to pitch, build, and write up this project: the domain + SDG decision,
the "why this isn't just ChatGPT" defense, the academic writing package (thesis, report,
review paper, research paper), the business model, and the cost/feasibility plan.

> **CRITICAL INTEGRITY NOTE.** The benchmark numbers throughout your current repo
> (50.9% pass, 37.8% token reduction, 40% vs 0% on 3-hop) were measured on **legal** data.
> The architecture and the *shape* of these results will transfer to patents, but you
> **must re-run the full benchmark on patent data** before citing any number in a report or
> paper. Until then, write results as "expected/illustrative, pending re-measurement on the
> patent corpus." A reviewer will catch legal numbers on a patent system.

---

## 1. The decision: Domain + SDG

**Domain: Patent & Innovation Intelligence.**
**Primary SDG: SDG 9 — Industry, Innovation and Infrastructure.**

Why this domain (and not the others you ruled out):

- **It is a citation graph, so migration is trivial.** Patents cite *prior art* — earlier
  patents they build on — exactly the "documents that cite documents" structure your legal
  version already handles. You rename `LegalCase` → `Patent`, keep the `CITES` edge, and
  ~90% of your code is unchanged.
- **The strongest revenue story of any option.** Commercial patent-intelligence platforms
  (PatSnap, Clarivate Derwent, Questel, IPRally, Patlytics) cost **tens of thousands of
  dollars per year**. That is a large, validated, expensive market — the clearest "there is
  real money here" argument you can give a jury.
- **Clean SDG fit, no overlap with excluded areas.** Innovation and industrial capability,
  not health, not education.
- **Free, high-quality data with the graph already built in** (see Section 4).

**SDG 9 target mapping (cite these exact targets):**

- **Target 9.5** — "Enhance scientific research, upgrade the technological capabilities of
  industrial sectors... encouraging innovation." Your tool directly lowers the cost of the
  research step every innovator must do (prior-art search).
- **Target 9.b** — "Support domestic technology development, research and innovation in
  developing countries." Small inventors, startups, MSMEs, and innovators in developing
  economies are precisely the people locked out of ₹15–20 lakh/year patent tools.

**The one-sentence SDG pitch:** *"Innovation starts with knowing what already exists. Today
that knowledge is locked behind patent-search tools only large corporations can afford. We
make grounded, multi-hop patent search free and accessible — so a student, a startup, or a
small-town inventor can navigate the prior-art landscape as well as a corporate IP team."*

---

## 2. What the system does (one paragraph you can memorize)

Given a natural-language question about technology or prior art, the system resolves the
relevant patent(s) to vertices in a Neo4j citation graph, traverses real citation edges
outward and inward up to N hops to gather connected prior art, compresses each patent to a
relevance-scored text window, and generates a grounded, **source-attributed** answer with an
LLM. It never invents a patent — every cited document is a real node, and the exact citation
path that produced the answer can be shown to the user.

---

## 3. The "why this is NOT just ChatGPT" arsenal (your #1 objection)

Faculty will say "ChatGPT already does this." Here is how you dismantle that. You have
something most teams don't: **an empirical benchmark that proves it.**

**1. Hallucinated sources — the fatal flaw you fix.** General LLMs fabricate patent numbers,
titles, and claims that don't exist. In patent work that is not a nuisance, it's legal
liability — a freedom-to-operate opinion built on a hallucinated patent is worthless or
dangerous. Your system returns **only real patents that exist as graph nodes**, each
traceable. *"Ask ChatGPT for the prior art behind a claim and it may invent it. Ask us, and
every citation is a real node we can point to."*

**2. ChatGPT can't see your corpus.** An LLM only knows its training data up to a cutoff and
never reliably memorized niche or recent patents. Your system grounds a general model on a
specific, **updatable** patent corpus — add a patent today, it's answerable today. That is
the entire reason retrieval-augmented generation exists.

**3. Multi-hop reasoning — and you have the numbers.** This is your moat. Prior art builds in
chains: patent C improves B, which extended A. Answering "trace the prior-art lineage of this
method" requires *traversing* those links. In a controlled experiment — same LLM, same
questions, three retrieval strategies — flat vector search and plain LLM score **0%** on
three-hop questions; graph traversal scores **40%**. Overall, graph retrieval beats standard
RAG **5.6×** while using **37.8% fewer tokens**. *(Re-measure on patent data before quoting.)*

**4. Explainability and provenance.** You can display the exact citation subgraph that
produced each answer. A chatbot is a black box; auditability is the product in any
high-stakes domain.

**5. Reframe what the project is.** ChatGPT is not your competitor — it is the *generator you
use*. Your contribution is the **retrieval architecture** that makes a general LLM reliable
over a domain it doesn't know, benchmarked against baselines. That is an Information
Retrieval / systems research contribution, not a chatbot wrapper.

**The kill shot:** *"We're not competing with ChatGPT. We fix the two things it is worst at —
inventing sources and failing at multi-hop reasoning over a corpus it can't see — and we
measured exactly how much better we are."*

---

## 4. The dataset (free, and the graph is pre-built)

| Source | What you get | Cost |
|---|---|---|
| **PatentsView (USPTO)** | Free API + bulk downloads: patents, abstracts, claims, **citations**, assignees, inventors, CPC classifications | $0 |
| **Google Patents Public Data (BigQuery)** | Full patent corpus, citation graph, full text; BigQuery free tier (1 TB queries/month) | $0 |
| **EPO Open Patent Services (OPS)** | European patents, free tier | $0 |

**Scoping (keep it free and manageable):** don't ingest all patents. Do what you did with
court opinions — take **one technology slice** via CPC class (e.g. a subfield of computing,
electronics, or mechanical engineering), roughly **50k–100k patents with their citation
edges**. To avoid any healthcare overlap, simply pick a non-medical CPC section.

**Graph you build:** `Patent` vertices (number, title, abstract, claims, year, assignee,
CPC), `CITES` edges (prior-art citations), plus optional `ASSIGNED_TO` → `Assignee` and
`CLASSIFIED_AS` → `CPC` enrichment (the analog of your Court/DECIDED_BY enrichment).

---

## 5. Migration effort: a weekend (TigerGraph → Neo4j, legal → patents)

| Legal (current) | Patents (new) | Change |
|---|---|---|
| `LegalCase` vertex | `Patent` vertex | rename; `text` → `abstract`+`claims` |
| `CITES` edge | `CITES` (prior-art) edge | identical structure |
| `Court` + `DECIDED_BY` | `Assignee`/`CPC` + `ASSIGNED_TO`/`CLASSIFIED_AS` | analogous enrichment |
| `citation_multihop_retrieve` (GSQL) | Cypher `MATCH (p)-[:CITES*1..2]-(n)` | rewrite 2 queries |
| `find_case_by_keyword` (GSQL) | Neo4j full-text index | simpler than a LIKE scan |
| eyecite metadata parsing | PatentsView returns structured metadata | *less* work than before |

Pipelines 1 & 2 (LLM-only, vector RAG), the relevance-window logic, the eval harness, the
strict judge, and the dashboard are all graph-agnostic and **unchanged**.

**Cypher for your core query (drop-in replacement for `citation_multihop_retrieve`):**

```cypher
MATCH (seed:Patent) WHERE seed.patent_id IN $seedIds
MATCH path = (seed)-[:CITES*1..2]-(n:Patent)
RETURN n.patent_id, n.title, n.abstract, n.year,
       min(length(path)) AS hop_distance
ORDER BY hop_distance
```

---

## 6. Cost: genuinely near-zero

| Component | Free choice |
|---|---|
| Data | PatentsView / Google Patents / EPO OPS — all free |
| Graph DB | **Neo4j Community Edition** via Docker, local — no node/storage limits |
| Vector index | FAISS (local, CPU) |
| Embeddings | fastembed MiniLM (local) |
| LLM generation + judge | **Gemini free tier** (throttle the benchmark with sleeps), or **Ollama + Llama 3.1 8B / Qwen 2.5** fully offline |
| BERTScore / tokenizer | Local |
| Demo hosting | Hugging Face Spaces / Streamlit Community free tier |

Honest caveats: (a) the free Gemini tier is rate-limited, so the ~330-call benchmark runs
over an hour or two — fine for a project; the Ollama fallback lets you claim "runs with **no**
paid API," which is stronger. (b) Use **local Neo4j Community**, not AuraDB Free — full patent
text would exceed AuraDB's storage cap, but locally there is no limit.

---

## 7. Making it a "big, well-designed" major project

Your GraphRAG core is already strong. These additions make it a *system*, and each one also
reinforces the anti-ChatGPT case:

1. **Provenance / citation-graph visualization** — for every answer, render the actual
   subgraph (seed patents + traversed edges). This is the demo "wow" moment and the visual
   proof of grounding. Neo4j Bloom or a D3 graph in your React dashboard.
2. **The three-way benchmark, front and center** — most college projects have *zero*
   evaluation. Yours has a strict LLM judge + BERTScore + multi-run reproducibility. Put the
   comparison table on a slide; it puts you in the top tier immediately.
3. **Live "add a patent" demo** — ingest a new patent on stage, then answer a question about
   it, proving the "ChatGPT can't do this" claim in real time.
4. **Source & confidence panel** — show retrieved patents, hop distance, and the citation
   path next to each answer for verification.
5. **A written report framed as an IR contribution** — problem, related work, method,
   controlled evaluation, results, limitations. That structure reads as serious research.

---

## 8. Business model (tie revenue to the SDG mission)

Frame revenue as what *sustains* free access — the standard SDG-venture pitch.

| Tier | Who | Price |
|---|---|---|
| **Free** | Students, individual inventors, hobbyists | ₹0 — the SDG-access core |
| **Pro** | IP professionals, patent agents — exports, alerts, saved searches, landscape reports | ~$15–30/mo |
| **Enterprise / API** | Law firms, R&D teams, tech-transfer offices embedding it | Annual license / metered API |
| **Custom corpus** | An org points it at *their* private R&D documents or patent portfolio | Setup + hosting fee |

**Market validation (name these to prove demand is real):** PatSnap, Clarivate Derwent,
Questel, **IPRally** and **Patlytics** (both use graph/ML for patent search — proof the
graph approach is commercially serious). Use-cases that map to revenue: prior-art search,
freedom-to-operate (FTO), patentability/novelty search, invalidity search, technology
landscaping, competitive intelligence.

---

## 9. Academic writing package

### 9.1 Title options
1. *GraphRAG for Patent Intelligence: Multi-Hop Prior-Art Retrieval over Citation Networks
   using Neo4j and Large Language Models*
2. *Democratizing Patent Search: A Graph-Augmented Retrieval System for Grounded, Multi-Hop
   Prior-Art Discovery*
3. *Grounded Multi-Hop Question Answering over Patent Citation Graphs: A Comparative Study of
   LLM-only, Vector RAG, and GraphRAG*

### 9.2 Abstract (adapt this)

> Access to patent intelligence — prior-art search, freedom-to-operate analysis, and
> technology landscaping — is dominated by commercial platforms costing tens of thousands of
> dollars annually, excluding individual inventors, startups, and innovators in developing
> economies. General-purpose large language models (LLMs) are an appealing free alternative
> but fail at this task in two critical ways: they fabricate non-existent patent numbers and
> claims (hallucination), and they cannot reliably reason across chains of prior-art
> citations (multi-hop reasoning). We present a Graph-Augmented Retrieval (GraphRAG) system
> that grounds LLM answers in a real patent citation network stored in Neo4j. Given a
> natural-language question, the system resolves relevant patents to graph vertices,
> traverses real citation edges up to N hops, compresses each document to a relevance-scored
> window, and generates a grounded, source-attributed answer. On a benchmark of multi-hop
> questions built from actual in-corpus citation chains, GraphRAG substantially outperforms
> standard vector RAG and LLM-only baselines — most sharply on the hardest multi-hop
> questions, where flat similarity search fails entirely — while using materially fewer
> tokens. The system is built entirely on free and open components (open patent data, Neo4j
> Community Edition, open embedding and LLM models), making rigorous, grounded patent search
> accessible at near-zero cost, in direct support of UN SDG 9 (Industry, Innovation and
> Infrastructure). *(Insert exact figures after re-running the benchmark on the patent
> corpus.)*

### 9.3 Problem statement
1. Patent prior-art search is essential to innovation but locked behind expensive tools.
2. Free LLMs hallucinate sources and fail multi-hop reasoning, making them unsafe for it.
3. Flat vector RAG cannot assemble multi-document citation chains.
4. There is no free, grounded, auditable, multi-hop patent QA system.

### 9.4 Objectives
1. Construct a patent citation knowledge graph from open data in Neo4j.
2. Implement three retrieval pipelines: LLM-only, vector RAG, and GraphRAG.
3. Design a token-efficient, relevance-window context-compression method.
4. Build a rigorous multi-hop QA benchmark with an LLM-judge and BERTScore.
5. Demonstrate GraphRAG's advantage in accuracy and token cost, especially on multi-hop.
6. Deliver a free, reproducible, deployable system aligned with SDG 9.

### 9.5 Contributions (for the research paper)
1. A GraphRAG architecture grounding LLM answers in a **native** patent citation graph —
   avoiding the cost and hallucination of LLM-based entity extraction used by generic
   GraphRAG frameworks.
2. A relevance-window + citation-link compression method that cuts token usage while
   improving multi-hop accuracy.
3. A controlled three-way benchmark isolating retrieval strategy (same LLM, same questions),
   showing graph traversal solves multi-hop questions that vector RAG cannot.
4. A rigorous, reproducible evaluation methodology (single strict LLM judge + BERTScore,
   multi-run) for multi-hop retrieval.
5. A fully open, near-zero-cost, deployable system with a societal-impact (SDG 9) framing.

### 9.6 Novelty defense (vs Microsoft GraphRAG and others)
- **Microsoft/generic GraphRAG** builds an entity graph by running an LLM over every chunk —
  expensive and hallucination-prone. **Ours uses the domain's native citation graph** — free,
  exact, and verifiable. That domain-grounding insight is a genuine contribution.
- Your **relevance-window + link-token compression** is a specific, novel optimization.
- Your **controlled 3-way benchmark with a strict judge** is methodologically stronger than
  the typical single-system demo.
- **Application to patents for innovation access + SDG 9** is a novel application framing.

---

## 10. Document structures

### 10.1 Research paper (IEEE conference style)
1. Abstract
2. Introduction — problem, motivation, contributions
3. Related Work — RAG, GraphRAG, knowledge-graph QA, patent retrieval systems
4. System Architecture — the three pipelines, the graph
5. Dataset & Graph Construction — sources, schema, statistics
6. Methodology — entity resolution, traversal, compression, generation
7. Experimental Setup — QA generation, judge, metrics, baselines
8. Results & Discussion — the comparison, per-hop analysis, token analysis, failure cases
9. Limitations & Future Work
10. Conclusion (with SDG 9 impact statement)
11. References

### 10.2 Review / survey paper (the literature-survey deliverable)
1. Introduction — scope: retrieval-augmented and graph-augmented QA
2. Background — RAG fundamentals, knowledge graphs, LLMs
3. Taxonomy — vector RAG vs graph RAG vs hybrid; entity-graph vs native-relationship graphs
4. Domain applications — legal, scientific, financial, patent retrieval
5. Evaluation methods — LLM-as-judge, BERTScore, multi-hop benchmarks
6. Comparative analysis table — approaches, strengths, weaknesses, cost
7. Research gaps — hallucination, multi-hop, cost, auditability
8. Positioning of the proposed work
9. Conclusion & open problems

### 10.3 Thesis (full)
- Ch 1 Introduction — problem, motivation, objectives, SDG context, contributions
- Ch 2 Literature Review — (your review paper, expanded)
- Ch 3 Proposed System & Methodology — architecture, the three pipelines
- Ch 4 Implementation — data pipeline, Neo4j, retrieval, dashboard, tech stack
- Ch 5 Results & Evaluation — benchmark, metrics, per-hop analysis, failure analysis
- Ch 6 Societal Impact & Business Model — SDG 9 alignment, cost, sustainability model
- Ch 7 Conclusion & Future Scope
- References, Appendices (code listings, sample QA, full result tables)

### 10.4 Project report (shorter than the thesis)
Same spine as the thesis, condensed: Introduction, Literature Review, System Design,
Implementation, Results, Conclusion. Include the architecture diagram, the benchmark table,
and the SDG-9 + business-model section.

---

## 11. Suggested future-work list (fills the thesis, shows depth)
- Hybrid retrieval: combine citation traversal with semantic embeddings on graph nodes.
- Claim-level analysis: reason over individual patent claims, not just abstracts.
- Cross-lingual patents (EPO/WIPO) for global prior-art coverage.
- Real-time ingestion of newly granted patents.
- A user study with real inventors / patent agents measuring time saved.
- Confidence calibration and automated freedom-to-operate risk flags.

---

## 12. Talking-points cheat sheet (for the viva / review)
- **Domain:** Patent & innovation intelligence.
- **SDG:** 9 (Industry, Innovation, Infrastructure), targets 9.5 and 9.b.
- **Problem:** grounded, multi-hop patent search is expensive; free LLMs hallucinate and
  can't do multi-hop.
- **Our answer:** graph-grounded retrieval over a real patent citation network — no invented
  sources, real multi-hop traversal, 37.8% fewer tokens *(re-measure on patents)*.
- **Proof:** controlled 3-way benchmark; graph traversal solves 3-hop where vector RAG scores
  0%.
- **Cost:** free stack (Neo4j Community, open patent data, open models).
- **Business:** freemium + enterprise/API; validated market (PatSnap, IPRally cost $$$).
- **Not ChatGPT because:** grounding, provenance, updatable private corpus, measured
  multi-hop advantage.
