# scripts/latency_baseline.py
import time
import numpy as np

from src.retrieval.vectorstore import get_client, get_or_create_collection
from src.ingestion.loaders import load_listings
from src.ingestion.chunking import chunk_all_products
from src.retrieval.bm25_index import build_bm25_index
from src.retrieval.hybrid_search import hybrid_search

SAMPLE_QUERIES = [
    "wireless earbuds", "red running shoes", "moisturizer for dry skin",
    "steel kitchen utensils", "affordable gaming mouse", "waterproof jacket",
    "vitamin c serum", "ceramic dinner plates", "bluetooth speaker",
    "leather handbag",
]


def main():
    client = get_client()
    collection = get_or_create_collection(client)
    listings = load_listings("data/raw/sparse_listings.json")
    chunks = chunk_all_products(listings)
    bm25_idx = build_bm25_index(chunks)

    latencies = []
    for query in SAMPLE_QUERIES:
        start = time.perf_counter()
        hybrid_search(query, collection, bm25_idx, top_k=5)
        latencies.append((time.perf_counter() - start) * 1000)

    print(f"\np50: {np.percentile(latencies, 50):.1f}ms")
    print(f"p95: {np.percentile(latencies, 95):.1f}ms")
    print(f"max: {max(latencies):.1f}ms")


if __name__ == "__main__":
    main()
