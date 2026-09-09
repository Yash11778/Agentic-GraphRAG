---
title: GraphRAG API
emoji: 🧠
colorFrom: green
colorTo: blue
sdk: docker
app_port: 8080
pinned: false
---

# GraphRAG vs Basic RAG vs LLM-only — TigerGraph Hackathon Round 2

Three retrieval strategies compared on **multi-hop legal questions** over a real U.S.
case-law citation graph. GraphRAG walks actual citation edges in TigerGraph; Basic RAG
does flat FAISS similarity; LLM-only has no retrieval. A live React dashboard runs all
three side by side and reports tokens, latency, cost, LLM-as-a-Judge verdict, and BERTScore.

**Corpus: 63,632 U.S. court opinions · 117.5M tokens (Gemini `count_tokens` verified) · 9,632 real citation edges**

## Architecture

![Architecture diagram](architecture_diagram.png)

## Headline result

Benchmark on **55 multi-hop questions** built from real in-corpus citation chains
(`data/qa/qa_pairs_legal_multihop.json`). Reproduce with `BASIC_RAG_FULL=1 python eval/evaluate.py`.

| Pipeline | Pass rate | 2-hop | 3-hop | Avg tokens | BERTScore (raw / rescaled) |
|----------|:---------:|:-----:|:-----:|:----------:|:--------------------------:|
| LLM-only | 1.8% | 2.5% | 0.0% | 609 | 0.787 / 0.362 |
| Basic RAG (FAISS top-k) | 9.1% | 12.5% | 0.0% | 2,288 | 0.807 / 0.422 |
| **GraphRAG (TigerGraph)** | **50.9%** | **55.0%** | **40.0%** | **1,423** | **0.841 / 0.524** |

- **GraphRAG passes 5.6× more often than Basic RAG** while using **37.8% fewer tokens**.
- **3-hop is the clearest proof:** GraphRAG 40.0% vs Basic RAG 0.0%. Flat chunk similarity
  structurally cannot assemble a three-case citation chain (A←B←C); graph traversal can.
- Every one of the 165 verdicts came from the **single strict Gemini judge**
  (`judge_source=gemini_strict` on every row of `eval/results/run1.csv`) — no fallback
  judge, no mid-run judge swap, no auto-pass. A strict judge fails partial or hedged
  answers, so absolute pass rates are low for all three pipelines by design; the
  **relative gap** is the result.

## Honesty guarantees (verifiable)

- **GraphRAG reads the live graph on every call.** `pipeline3_graphrag.py` resolves the
  cited case(s) to `LegalCase` vertex IDs and runs the installed `citation_multihop_retrieve`
  GSQL query over real `CITES` edges. There is **no snapshot file and no curated subset**. If
  the graph returns nothing, the answer is `status="no_context"` and scored as a **FAIL** — in
  the benchmark run, **5 of 55** GraphRAG answers were no-context and all 5 are counted as
  failures in the 50.9%.
- **Strict judge, no auto-pass.** `eval/judge.py` returns PASS only when the judge says PASS;
  errors count as FAIL; there is no "default to PASS", no BERTScore override.
- **Equal output budget.** All three pipelines use the same 512-token completion cap, so the
  token reduction comes from GraphRAG's precise retrieval, not from shorter answers.
- **Token count is official.** 117.5M via Gemini `count_tokens` (`data/token_count_official.json`).

## Data Source & Licensing

- **Source**: U.S. court opinions (state and federal), via the [Pile of Law](https://huggingface.co/datasets/pile-of-law/pile-of-law) `courtlistener_opinions` split, which aggregates public [CourtListener](https://www.courtlistener.com/) data.
- **Underlying content is public domain**: judicial opinions authored by judges in their official capacity are not copyrightable in the U.S. (government edict doctrine, *Georgia v. Public.Resource.Org*, 2020).
- **Pile of Law's compiled dataset artifact** is `CC-BY-NC-SA-4.0` (non-commercial, share-alike) — applying to their packaging, not the public-domain opinion text. This submission is non-commercial (competition use), consistent with that license.
- Case metadata (name, court, year) parsed with [eyecite](https://github.com/freelawproject/eyecite) + [courts-db](https://github.com/freelawproject/courts-db) (Free Law Project). Citation edges built from CourtListener's public `citation-map` bulk data, filtered to edges where both endpoints are in-corpus.

## Knowledge graph schema

```
LegalCase(id, case_id, case_name, court_name, year, text)
    │
    ├── CITES ──────────▶ LegalCase        (9,632 real citation edges; powers multi-hop)
    └── DECIDED_BY ─────▶ Court             (63,632 edges; 1,413 courts)
```

Defined in `scripts/legal_schema.gsql` (+ `scripts/add_court_schema.gsql`), live on the
TigerGraph Savanna `LegalGraph` instance: 63,632 `LegalCase` + 1,413 `Court` vertices.

## How GraphRAG retrieves (pipeline3_graphrag.py)

```
Question
  → resolve cited case(s) to LegalCase IDs
      (citation id in the question, e.g. "People v. Batson (6047231)", else name lookup)
  → citation_multihop_retrieve(seedCaseIds, hops=2): walk real CITES edges both directions
  → assemble context: seed cases first, then a few citation neighbours (tight budget)
  → Gemini generates the answer
```

The multi-hop questions cite their cases by identifier the way a real legal query does;
Basic RAG receives the **identical** question text, so the comparison is fair — GraphRAG
simply exploits the graph structure that Basic RAG cannot.

## Pipeline build

```
data/raw/dataset_100m_enriched.jsonl   63,632 opinions, enriched metadata (eyecite)
        │
        ├── scripts/preprocess.py + build_faiss.py  → 500,959-chunk FAISS index (Basic RAG)
        ├── scripts/ingest_pilot_graph.py --full     → LegalCase + CITES in TigerGraph
        └── scripts/ingest_courts.py                 → Court + DECIDED_BY enrichment
```

## Quick start

```bash
pip install -r api/requirements.txt
cp .env.example .env          # add GEMINI_API_KEY, HF_TOKEN, TG_HOST, TG_PASSWORD
# Reproduce the benchmark against the live graph + full FAISS index:
BASIC_RAG_FULL=1 python eval/evaluate.py
# Or run the dashboard API:
python api/app.py             # then open the React frontend
```

## Environment variables

| Var | Purpose |
|-----|---------|
| `GEMINI_API_KEY` | Gemini (generation + token counting) |
| `HF_TOKEN` | HuggingFace InferenceClient (LLM-as-a-Judge); needs Inference Providers scope |
| `TG_HOST`, `TG_PASSWORD`, `TG_GRAPH` | TigerGraph Savanna instance + secret + `LegalGraph` |
| `BASIC_RAG_FULL` | `1` = benchmark Basic RAG on the full 500k-chunk index |

## Judging criteria mapping

| Criterion | Where |
|-----------|-------|
| Dataset ≥100M tokens | 117.5M, Gemini-verified (`data/token_count_official.json`) |
| Multi-hop GraphRAG advantage | 3-hop 40.0% vs Basic RAG 0.0% (`eval/results/run1.csv`) |
| Token efficiency | 37.8% reduction vs Basic RAG, equal 512-token output caps |
| LLM-as-a-Judge | `eval/judge.py` — one strict Gemini judge for the whole run, no fallback |
| BERTScore | `evaluate.load("bertscore", rescale_with_baseline=True)`, all 3 pipelines |

> **Note for reviewers:** an earlier version of this project reported inflated numbers
> (91.7% / 80.9% / 0.889) from a retrieval path that silently fell back to a hand-built
> snapshot, plus an auto-pass judge. Both are gone: GraphRAG now retrieves only from the
> live graph, the judge is a single strict Gemini judge with no fallback and no auto-pass,
> output caps are equal, and every number above comes from one complete strict-judge run
> whose raw per-question verdicts are committed (`eval/results/run1.csv`). LLM-judge runs
> on 55 questions have a ±3–5% run-to-run noise floor, so we ran the full strict-judge
> benchmark **three times** (`run1.csv`, `run2.csv`, `run3.csv`, all committed): GraphRAG
> 50.9% / 52.7% / 50.9% (mean 51.5%), Basic RAG 9.1% and LLM-only 1.8% identical across
> all three runs. The headline table reports run1; the other runs only strengthen it.
