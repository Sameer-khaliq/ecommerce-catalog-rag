# Architecture

## Design goals

1. Demonstrate two distinct RAG applications in one system: content-generation RAG (enrichment) and retrieval RAG (search).
2. Keep every stage independently testable — each pipeline stage has its own test file and can run without the others.
3. Fail gracefully — a single bad listing or LLM call should never crash the whole pipeline.
4. Be honest about uncertainty — never let the LLM invent facts; log every inferred field.

## Three-stage pipeline

### Stage 1 — Enrichment

| File | Responsibility |
|---|---|
| `enrichment/schemas.py` | `EnrichedListing` Pydantic model. `attributes` is an open dict (not fixed fields) because required attributes differ per category. `confidence_notes` tracks every inferred (not stated) field. |
| `enrichment/prompts.py` | 3 prompts: attribute completion, description generation, SEO tag generation. Every prompt embeds a mandatory grounding rule — no facts beyond what's in the raw listing, reference listings, or schema. |
| `enrichment/retriever.py` | Loads the category-specific knowledge base (attribute schema, brand/style guide, reference listings) and returns the right slice for a given listing's category. |
| `enrichment/enricher.py` | Runs the 3-prompt sequence per listing, parses LLM JSON output into `EnrichedListing`, defensively backfills `confidence_notes` for any attribute the LLM marked unknown but forgot to explain. |
| `enrichment/resilience.py` | Retry with exponential backoff, a circuit breaker that stops calling a repeatedly-failing API, and graceful degradation (returns a flagged partial result instead of crashing). |
| `enrichment/cache.py` | Hashes each raw listing; caches its enrichment result to avoid redundant LLM calls on reruns. In-memory for this scope — documented swap path to Redis. |
| `enrichment/pipeline.py` | Orchestrates retrieve -> enrich -> validate across all listings, writes `enriched_listings.json`. |

**Key data-quality decision:** `brand` is a required attribute in every category schema but never present in the raw sparse data — it is always an inferred field. The system is designed to say "unknown" rather than guess a plausible-sounding brand name.

### Stage 2 — Ingestion & indexing

| File | Responsibility |
|---|---|
| `ingestion/loader.py` | Reads a listings JSON file into a list of dicts. No transformation logic — used for both raw and enriched data. |
| `ingestion/chunkers.py` | Converts one product dict into two text representations: an attribute-table chunk (structured, good for filter-heavy queries) and a prose chunk (natural language, good for semantic queries). Handles raw-data quirks defensively: `raw_attributes` vs `attributes` key, missing `brand`, `null` price. |
| `retrieval/vector_store.py` | Embeds chunks via Gemini and stores them in ChromaDB with metadata (`product_id`, `category`, `brand`, `price`, `chunk_strategy`). |
| `retrieval/bm25_index.py` | Builds a keyword-based sparse index over the same chunks using `rank_bm25`. |

The pipeline is re-run after enrichment completes (`scripts/reindex_enriched.py`) so the searchable index reflects the enriched catalog, not the original sparse data. The ChromaDB collection is deleted and recreated on reindex to avoid stale sparse-data chunks persisting alongside enriched ones.

### Stage 3 — Search & retrieval

| File | Responsibility |
|---|---|
| `retrieval/hybrid_search.py` | Runs BM25 and dense search in parallel candidate sets, fuses them with Reciprocal Rank Fusion (rank-based, not score-based, since BM25 scores and cosine distances aren't on the same scale), then applies category/price metadata filters. |

## Cross-cutting

- `config.py` — centralized `pydantic-settings` configuration, loaded once and cached.
- `logging_config.py` — structured JSON logging used by every module, wired in from the start rather than retrofitted.
- `observability/metrics.py` — per-stage latency tracking (p50/p95/p99) via a context manager (`track_stage`).

## Why RRF instead of score-based fusion

BM25 returns unbounded relevance scores; dense search returns cosine distances. These are not comparable on the same scale, so combining them by raw score would bias toward whichever method produces larger numbers. RRF instead uses each result's *rank position* in its own list, which is always comparable regardless of the underlying scoring method.

## Why two chunk strategies

Product search queries vary in kind — exact-match/filter-style queries ("red, size 32") are better served by structured text, while natural-language queries ("comfortable pants for the office") are better served by prose. Both chunk types are indexed for every product; `eval/ragas_eval.py` evaluates them separately to measure which performs better per category, rather than assuming one is universally superior.

## Known trade-offs

See the "Limitations" section in `README.md` for a full list of what's simulated versus production-ready.