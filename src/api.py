# src/api.py — minimal search endpoint for load testing (separate from Gradio UI)
from fastapi import FastAPI
from pydantic import BaseModel

from src.retrieval.vectorstore import get_client, get_or_create_collection
from src.retrieval.bm25_index import build_bm25_index
from src.retrieval.hybrid_search import hybrid_search
from src.ingestion.loaders import load_listings
from src.ingestion.chunking import chunk_all_products

app = FastAPI()

_client = get_client()
_collection = get_or_create_collection(_client)
_listings = load_listings("data/processed_data/enriched_listings.json")
_chunks = chunk_all_products(_listings)
_bm25_idx = build_bm25_index(_chunks)


class SearchRequest(BaseModel):
    query: str
    category: str | None = None
    max_price: float | None = None


@app.post("/search")
def search(req: SearchRequest):
    results = hybrid_search(req.query, _collection, _bm25_idx, category=req.category, max_price=req.max_price, top_k=5)
    return {"results": results}