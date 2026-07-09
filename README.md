# E-commerce Domain RAG System

An end-to-end RAG system for e-commerce product catalogs, combining LLM-based catalog enrichment with hybrid search retrieval.

## Problem

Sparse, inconsistent product listings hurt both catalog quality and customer search — missing attributes mean products don't surface for relevant queries. This system solves both halves: it enriches incomplete listings using a domain knowledge base, then makes the enriched catalog searchable through hybrid (BM25 + dense) retrieval with metadata filtering.

## Architecture

Full details in [`ARCHITECTURE.md`](./ARCHITECTURE.md). Summary:

```
Raw sparse listings + Knowledge base (schemas, style guides, references)
        |
        v
Enrichment pipeline (retriever.py -> enricher.py -> pipeline.py)
   - Retry/backoff + circuit breaker for LLM call resilience
   - In-memory cache to avoid redundant LLM calls
   - Grounding rule: no hallucinated attributes, honest "unknown" when unsupported
        |
        v
enriched_listings.json
        |
        v
Ingestion & indexing (loader.py -> chunkers.py -> vector_store.py + bm25_index.py)
   - 2 chunk strategies per product: attribute-table + prose
        |
        v
Hybrid search (hybrid_search.py)
   - BM25 + dense fused via Reciprocal Rank Fusion
   - Category and price metadata filtering
```

## Tech stack

Python, Groq (Llama 3.3 70B), Gemini embeddings, ChromaDB, rank-bm25, Pydantic, FastAPI, Gradio, Locust, RAGAS, Docker.

## Project structure

```
ecommerce-rag/
├── data/
│   ├── knowledge_base/       # per-category schemas, style guides, reference listings
│   ├── raw/                  # 40 sparse source listings
│   └── processed/            # enriched_listings.json, chroma_db/
├── src/
│   ├── config.py
│   ├── logging_config.py
│   ├── api.py                 # FastAPI search endpoint
│   ├── app.py                 # Gradio UI
│   ├── enrichment/            # schemas, prompts, retriever, enricher, pipeline, resilience, cache
│   ├── ingestion/              # loader, chunkers
│   └── retrieval/              # vector_store, bm25_index, hybrid_search
├── eval/                      # eval_dataset.json, ragas_eval.py, ragas_results.csv
├── observability/              # metrics.py
├── loadtest/                   # locustfile.py
├── tests/
└── .github/workflows/ci.yml
```

## Evaluation results (RAGAS)

| Chunk strategy | Faithfulness | Answer relevancy | Context recall | Context precision |
|---|---|---|---|---|
| Attribute-table | *(fill after run)* | | | |
| Prose | | | | |
| Combined (hybrid) | | | | |

Full results: [`eval/ragas_results.csv`](./eval/ragas_results.csv)

## Latency

| Percentile | Retrieval (baseline) |
|---|---|
| p50 | 1673.6 ms |
| p95 | 2386.9 ms |
| max | 2652.0 ms |

Load test results (concurrent users, throughput, degradation curve): [`loadtest/results_stats.csv`](./loadtest/results_stats.csv)

## Limitations & what's simulated vs battle-tested

- **Simulated:** load test uses a local single-process FastAPI instance, not a production deployment behind a load balancer — real concurrency limits will differ.
- **Latency bottleneck:** each query makes a live Gemini embedding API call, which dominates response time. Documented paths forward: (1) cache embeddings for repeated/similar queries, (2) run BM25 and dense embedding calls in parallel instead of sequentially, (3) swap to a local embedding model to remove the network round-trip.
- **Cache is in-memory:** resets on process restart. Production would use Redis with the same get/set interface already documented in `cache.py`.
- **No rate limiting on the API layer** — would add per-client throttling before production use.
- **No real monitoring dashboard** — `metrics.py` logs structured JSON; a real deployment would ship this to Grafana/Datadog instead of stdout.
- **No A/B testing infrastructure** — chunk strategy comparison here is offline (RAGAS), not a live A/B test against real user behavior.

## Running it

```bash
uv sync
```

```bash
uv run python -m src.enrichment.pipeline
```

```bash
uv run python -m scripts.reindex_enriched
```

```bash
uv run pytest tests/ -v
```

```bash
uv run python eval/ragas_eval.py
```

```bash
uv run uvicorn src.api:app --port 8000
```

```bash
uv run python -m src.app
```

## License

MIT — see [`LICENSE`](./LICENSE).