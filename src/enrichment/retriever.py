import json
from pathlib import Path

from src.config import get_settings
from src.logging_config import get_logger, log_with_context

logger = get_logger(__name__)

CATEGORIES = ["beauty", "electronics", "fashion", "home"]
settings = get_settings()

def _load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_knowledge_base() -> dict:
    """Loads all 3 per-category KB file sets once, indexes them by category name
    (as declared inside the schema JSON, e.g. "Beauty") for fast lookup."""
    kb_dir = Path(settings.knowledge_base_dir)

    schemas: dict[str, dict] = {}
    brand_guides: dict[str, str] = {}
    references: dict[str, list[dict]] = {}

    for cat in CATEGORIES:
        schema = _load_json(kb_dir / f"{cat}_attribute_schema.json")
        category_name = schema["category"]
        schemas[category_name] = schema

        guide_path = kb_dir / f"{cat}_brand_guide.md"
        brand_guides[category_name] = guide_path.read_text(encoding="utf-8")

        references[category_name] = _load_json(kb_dir / f"{cat}_reference_listings.json")

    log_with_context(logger, "info", "knowledge base loaded", categories=list(schemas.keys()))
    return {"schemas": schemas, "brand_guides": brand_guides, "references": references}


def get_retrieval_context(raw_listing: dict, kb: dict, max_references: int = 3) -> dict:
    """Pulls the schema, style guide, and top reference listings for this listing's category.
    Note: 'brand_guide' here is a category-level tone/style guide (how to write about
    Beauty products in general), not a per-brand guide — the raw data has no brand field yet."""
    category = raw_listing.get("category")
    schema = kb["schemas"].get(category, {})
    references = kb["references"].get(category, [])[:max_references]
    style_guide = kb["brand_guides"].get(category)

    log_with_context(
        logger, "info", "retrieval context built",
        product_id=raw_listing.get("product_id"), category=category,
        reference_count=len(references), has_style_guide=style_guide is not None,
    )
    return {"schema": schema, "references": references, "style_guide": style_guide}