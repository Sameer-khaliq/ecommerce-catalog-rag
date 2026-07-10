import json
from pathlib import Path

from src.config import get_settings
from src.logging_config import get_logger, log_with_context
from src.enrichment.retriever import load_knowledge_base, get_retrieval_context
from src.enrichment.enricher import enrich_listing
from src.enrichment.cache import get_cached, set_cached
from src.ingestion.loaders import load_listings

logger = get_logger(__name__)

settings = get_settings()
def run_enrichment_pipeline(raw_path: str, output_path: str, limit: int | None = None) -> list[dict]:
    raw_listings = load_listings(raw_path)
    if limit:
        raw_listings = raw_listings[:limit]
    kb = load_knowledge_base()

    enriched_results = []
    for raw_listing in raw_listings:
        cached = get_cached(raw_listing)
        if cached:
            enriched_results.append(cached)
            continue

        context = get_retrieval_context(raw_listing, kb)
        try:
            enriched_dict = enrich_listing(raw_listing, context).model_dump()
            set_cached(raw_listing, enriched_dict)
            enriched_results.append(enriched_dict)
            import time
            log_with_context(logger, "info", "sleeping for 5 seconds",)     
            time.sleep(5)  
        except Exception as e:
            log_with_context(
                logger, "error", "listing failed validation, skipping",
                product_id=raw_listing.get("product_id"), error=str(e),
            )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(enriched_results, f, indent=2)

    log_with_context(
        logger, "info", "enrichment pipeline complete",
        total_raw=len(raw_listings), total_enriched=len(enriched_results),
    )
    return enriched_results


if __name__ == "__main__":
    run_enrichment_pipeline(
        raw_path=f"{settings.raw_data_dir}/sparse_listings.json",
        output_path=f"{settings.processed_data_dir}/enriched_listings.json",
        
    )