import json
import random

from src.ingestion.loaders import load_listings
from src.logging_config import get_logger, log_with_context

logger = get_logger(__name__)


def generate_eval_dataset(
    enriched_path: str = "data/processed_data/enriched_listings.json",
    output_path: str = "eval/eval_dataset.json",
    samples_per_category: int = 5,
    seed: int = 42,
) -> list[dict]:
    """Ground truth ab seedha real enriched listings se aati hai — hand-guessed
    paraphrase nahi, jo indexed data se mismatch kar sakti thi."""
    listings = load_listings(enriched_path)

    by_category: dict[str, list[dict]] = {}
    for listing in listings:
        by_category.setdefault(listing["category"], []).append(listing)

    random.seed(seed)
    eval_set = []
    qid = 1
    for category, items in by_category.items():
        sample = random.sample(items, min(samples_per_category, len(items)))
        for listing in sample:
            query = listing["seo_tags"][0] if listing.get("seo_tags") else listing["title"]
            eval_set.append({
                "query_id": f"q{qid:02d}",
                "query": query,
                "category": category,
                "expected_product_id": listing["product_id"],
                "reference_answer": listing["description"],
            })
            qid += 1

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(eval_set, f, indent=2)

    log_with_context(logger, "info", "eval dataset generated", total_queries=len(eval_set))
    return eval_set


if __name__ == "__main__":
    generate_eval_dataset()