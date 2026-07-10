import gradio as gr

from src.retrieval.vectorstore import get_client, get_or_create_collection
from src.retrieval.bm25_index import build_bm25_index
from src.retrieval.hybrid_search import hybrid_search
from src.ingestion.loaders import load_listings
from src.ingestion.chunking import chunk_all_products
from src.logging_config import get_logger, log_with_context

logger = get_logger(__name__)

_client = get_client()
_collection = get_or_create_collection(_client)
_listings = load_listings("data/processed/enriched_listings.json")
_chunks = chunk_all_products(_listings)
_bm25_idx = build_bm25_index(_chunks)

CATEGORIES = ["All", "Electronics", "Fashion", "Beauty", "Home"]


def search_products(query: str, category: str, max_price: float | None):
    if not query or not query.strip():
        return "⚠️ Please enter a search query."

    try:
        cat_filter = None if category == "All" else category
        price_filter = max_price if max_price and max_price > 0 else None

        result_ids = hybrid_search(query, _collection, _bm25_idx, category=cat_filter, max_price=price_filter, top_k=5)
        if not result_ids:
            return "No matching products found. Try a broader query or remove filters."

        metadatas = _collection.get(ids=result_ids)["metadatas"]
        documents = _collection.get(ids=result_ids)["documents"]

        output = ""
        for meta, doc in zip(metadatas, documents):
            output += f"### {meta['product_id']} — {meta['category']}\n"
            output += f"**Price:** ${meta['price']} | **Chunk style:** {meta['chunk_strategy']}\n\n{doc}\n\n---\n\n"
        return output

    except Exception as e:
        log_with_context(logger, "error", "search failed in UI", error=str(e))
        return "⚠️ Something went wrong while searching. Please try again or refine your query."


with gr.Blocks(title="E-commerce Catalog Search") as demo:
    gr.Markdown("# E-commerce Domain RAG — Product Search")
    with gr.Row():
        query_input = gr.Textbox(label="Search query", placeholder="e.g. wireless earbuds, mineral sunscreen...")
    with gr.Row():
        category_input = gr.Dropdown(choices=CATEGORIES, value="All", label="Category")
        price_input = gr.Number(label="Max price (optional)", value=None)
    search_btn = gr.Button("Search", variant="primary")
    output = gr.Markdown()

    search_btn.click(fn=search_products, inputs=[query_input, category_input, price_input], outputs=output)
    query_input.submit(fn=search_products, inputs=[query_input, category_input, price_input], outputs=output)

if __name__ == "__main__":
    demo.launch()