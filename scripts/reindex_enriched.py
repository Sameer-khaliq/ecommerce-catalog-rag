from src.ingestion.loaders import load_listings
from src.ingestion.chunking import chunk_all_products
from src.retrieval.vectorstore import get_client, get_or_create_collection, add_chunks
from src.retrieval.bm25_index import build_bm25_index

ENRICHED_PATH = "data/processed_data/enriched_listings.json"


def main():
    listings = load_listings(ENRICHED_PATH)
    chunks = chunk_all_products(listings)
    print(f"Loaded {len(listings)} enriched listings -> {len(chunks)} chunks")

    client = get_client()
    collection = get_or_create_collection(client)


    client.delete_collection("ecommerce_catalog")
    collection = get_or_create_collection(client)

    add_chunks(collection, chunks)
    print("Dense index rebuilt on enriched data.")

    bm25_idx = build_bm25_index(chunks)
    print("Sparse (BM25) index rebuilt on enriched data.")

    # quick sanity check
    from src.retrieval.hybrid_search import hybrid_search
    results = hybrid_search("wireless earbuds", collection, bm25_idx, top_k=5)
    print("\nSanity check query results:")
    print(results)


if __name__ == "__main__":
    main()