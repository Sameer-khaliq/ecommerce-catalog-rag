""" Two types of chunking will be performed
    (i) Table chunking for the attribute heavy data
    (ii) Prose chunking for normal data """



from src.logging_config import get_logger, log_with_context

logger = get_logger(__name__)


def _get_attributes(product: dict) -> dict:
    """Raw listings use 'raw_attributes' (often empty), enriched listings use 'attributes'.
    Check both so this works on raw AND enriched data without changes."""
    return product.get("attributes") or product.get("raw_attributes") or {}


def _get_price_display(product: dict) -> str:
    price = product.get("price")
    return f"${price}" if price is not None else "price not listed"


def chunk_attribute_table(product: dict) -> str:
    header = (
        f"Product: {product.get('title', 'Unknown')} | "
        f"Category: {product.get('category', 'Unknown')}"
    )
    brand = product.get("brand")
    if brand:
        header += f" | Brand: {brand}"

    attributes = _get_attributes(product)
    attr_lines = [f"{key}: {value}" for key, value in attributes.items()]
    body = "\n".join(attr_lines) if attr_lines else "No attributes listed."
    return header + "\n" + body


def chunk_prose(product: dict) -> str:
    title = product.get("title", "Unknown product")
    category = product.get("category", "product")
    brand = product.get("brand")
    description = product.get("description", "")
    attributes = _get_attributes(product)

    attrs_sentence = ", ".join(f"{k} is {v}" for k, v in attributes.items())

    sentence = f"{title} is a {category.lower()} product"
    sentence += f" by {brand}." if brand else "."
    parts = [sentence]
    if description:
        parts.append(description)
    if attrs_sentence:
        parts.append(f"Key details: {attrs_sentence}.")
    parts.append(f"Priced at {_get_price_display(product)}.")

    return " ".join(parts)


def build_chunk_metadata(product: dict, chunk_strategy: str) -> dict:
    return {
        "product_id": product.get("product_id", product.get("id", "unknown")),
        "category": product.get("category", "Unknown"),
        "brand": product.get("brand", "Unknown"),
        "price": product.get("price") if product.get("price") is not None else 0,
        "chunk_strategy": chunk_strategy,
    }


def chunk_product(product: dict) -> list[dict]:
    chunks = [
        {"text": chunk_attribute_table(product), "metadata": build_chunk_metadata(product, "attribute_table")},
        {"text": chunk_prose(product), "metadata": build_chunk_metadata(product, "prose")},
    ]
    log_with_context(
        logger, "info", "product chunked",
        product_id=product.get("product_id", product.get("id", "unknown")),
        chunk_count=len(chunks),
    )
    return chunks


def chunk_all_products(products: list[dict]) -> list[dict]:
    all_chunks = []
    for product in products:
        all_chunks.extend(chunk_product(product))
    log_with_context(logger, "info", "catalog chunked", total_chunks=len(all_chunks))
    return all_chunks