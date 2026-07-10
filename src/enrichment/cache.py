import hashlib
import json

from src.logging_config import get_logger, log_with_context

logger = get_logger(__name__)

_cache: dict[str, dict] = {}


def _hash_listing(raw_listing: dict) -> str:
    """Stable hash of listing content — cache key. In production this dict swaps
    for a Redis client with the same get/set interface (documented, not built here
    — portfolio scope)."""
    content = json.dumps(raw_listing, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()


def get_cached(raw_listing: dict) -> dict | None:
    key = _hash_listing(raw_listing)
    cached = _cache.get(key)
    if cached:
        log_with_context(logger, "info", "cache hit", product_id=raw_listing.get("product_id"))
    return cached


def set_cached(raw_listing: dict, enriched: dict) -> None:
    _cache[_hash_listing(raw_listing)] = enriched
    log_with_context(logger, "info", "cache set", product_id=raw_listing.get("product_id"))