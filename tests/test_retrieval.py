import time
import pytest

from src.ingestion.loaders import load_listings
from src.ingestion.chunking import chunk_all_products
from src.retrieval.vectorstore import get_client, get_or_create_collection
from src.retrieval.bm25_index import build_bm25_index
from src.retrieval.hybrid_search import hybrid_search

RAW_PATH = "data/raw/sparse_listings.json"


@pytest.fixture(scope="module")
def collection():
    client = get_client()
    return get_or_create_collection(client)


@pytest.fixture(scope="module")
def bm25_idx():
    listings = load_listings(RAW_PATH)
    chunks = chunk_all_products(listings)
    return build_bm25_index(chunks)


def _get_metadata_for_ids(collection, ids: list[str]) -> list[dict]:
    result = collection.get(ids=ids)
    return result["metadatas"]


class TestCategoryFiltering:
    @pytest.mark.parametrize("category", ["Electronics", "Fashion", "Beauty", "Home"])
    def test_filter_returns_only_matching_category(self, collection, bm25_idx, category):
        results = hybrid_search(
            f"good {category.lower()} product", collection, bm25_idx,
            category=category, top_k=5,
        )
        metadatas = _get_metadata_for_ids(collection, results)
        assert all(m["category"] == category for m in metadatas)

    def test_no_filter_still_returns_relevant_results(self, collection, bm25_idx):
        results = hybrid_search("wireless earbuds", collection, bm25_idx, top_k=5)
        metadatas = _get_metadata_for_ids(collection, results)
        categories = {m["category"] for m in metadatas}
        assert "Electronics" in categories


class TestPriceFiltering:
    def test_price_filter_respected(self, collection, bm25_idx):
        results = hybrid_search("affordable item", collection, bm25_idx, max_price=50, top_k=5)
        metadatas = _get_metadata_for_ids(collection, results)
        for m in metadatas:
            if m["price"]:  # skip chunks where price wasn't listed (raw sparse data)
                assert m["price"] <= 50


class TestLatency:
    def test_query_latency_is_logged_and_reasonable(self, collection, bm25_idx):
        start = time.perf_counter()
        hybrid_search("test query", collection, bm25_idx, top_k=5)
        latency_ms = (time.perf_counter() - start) * 1000
        assert latency_ms < 3000  # generous ceiling — tune once you have a real baseline