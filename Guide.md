# How to Run This Project

Three retrieval pipelines compared on multi-hop legal questions over a real U.S. case-law
citation graph in TigerGraph.

| Pipeline | Strategy | Avg tokens | Pass rate |
|----------|----------|:----------:|:---------:|
| LLM-only | Raw Gemini, no retrieval | 609 | 32.7% |
| Basic RAG | FAISS top-k → Gemini | 2,281 | 14.5% |
| GraphRAG | TigerGraph `CITES` traversal → Gemini | 1,546 | **69.1%** |

## Prerequisites

| Tool | Why |
|------|-----|
| Python 3.10+ | pipelines, eval |
| Gemini API key | generation + token counting |
| HuggingFace token | LLM-as-a-Judge (needs "Inference Providers" scope) |
| TigerGraph Savanna instance | the live citation graph (`LegalGraph`) |

## Setup

```bash
pip install -r api/requirements.txt
cp .env.example .env
# Fill in: GEMINI_API_KEY, HF_TOKEN, TG_HOST, TG_PASSWORD, TG_GRAPH=LegalGraph
```

## Build the data (one-time; skip if data/ is already populated)

```bash
# 1. Dataset + enriched metadata (already built into data/raw/dataset_100m_enriched.jsonl)
python scripts/build_dataset_v2_concentrated.py      # source opinions + eyecite metadata
python scripts/download_citations.py                 # citation edges -> data/raw/citations.csv
python scripts/count_tokens_gemini.py                # official 117.5M token count

# 2. Basic RAG index (full corpus)
python scripts/preprocess.py                         # -> data/chunks/chunks.jsonl (500,959)
python scripts/build_faiss.py                        # -> data/chunks/rag_index.faiss
python scripts/build_faiss_demo.py                   # small deploy index (committed)

# 3. TigerGraph knowledge graph
python scripts/ingest_pilot_graph.py --install-schema
python scripts/ingest_pilot_graph.py --full          # LegalCase + CITES
python scripts/ingest_courts.py                      # Court + DECIDED_BY enrichment
```

## Generate QA + run the benchmark

```bash
python scripts/generate_multihop_qa.py --target-2hop 40 --target-3hop 15
BASIC_RAG_FULL=1 python eval/evaluate.py             # all 3 pipelines, live graph
```

Results → `eval/results/eval_results.csv`; summary printed with per-hop breakdown, token
reduction, and BERTScore.

## Run the dashboard

```bash
python api/app.py        # FastAPI backend on :8080
# then start the React frontend (frontend/) pointed at the API
```

## Notes

- The TigerGraph Savanna workspace **auto-suspends when idle**. If GraphRAG returns
  "no context", resume the `LegalGraph` workspace in the TigerGraph console, then retry.
- `BASIC_RAG_FULL=1` benchmarks Basic RAG on the full 500k-chunk index; without it, the
  small committed demo index is used (same top-k answers, loads instantly, no OOM).
- Every GraphRAG answer comes from a live GSQL query — there is no offline snapshot.
