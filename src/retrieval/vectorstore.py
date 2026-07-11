import chromadb
from google import genai
from src.config import settings
from src.logging_config import get_logger, log_with_context

logger = get_logger(__name__)

COLLECTION_NAME = "ecommerce_catalog"


def get_client() -> chromadb.PersistentClient:
    
    return chromadb.PersistentClient(path=settings.chroma_db_path)


def get_or_create_collection(client: chromadb.PersistentClient):
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embeds a batch of texts using Gemini embeddings."""
    gemini_client = genai.Client(api_key=settings.google_api_key)

    result = gemini_client.models.embed_content(
        model=settings.gemini_embedding_model,
        contents=texts,
    )
    embeddings = [e.values for e in result.embeddings]
    log_with_context(logger, "info", "texts embedded", count=len(texts))
    return embeddings


def add_chunks(collection, chunks: list[dict], batch_size: int = 50) -> None:
    """Embeds and stores chunks in ChromaDB, in batches to avoid API limits.
    Each chunk is expected as {"text": ..., "metadata": {...}}.
    """
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c["text"] for c in batch]
        metadatas = [c["metadata"] for c in batch]
        ids = [f"{m['product_id']}_{m['chunk_strategy']}" for m in metadatas]

        embeddings = embed_texts(texts)

        collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )
        log_with_context(
            logger, "info", "batch added to chroma",
            batch_start=i, batch_size=len(batch),
        )

    log_with_context(logger, "info", "all chunks indexed", total=len(chunks))


def query_dense(collection, query_text: str, top_k: int = 10, where: dict | None = None) -> dict:
    """Runs a dense/semantic query against the collection with optional metadata filter."""
    query_embedding = embed_texts([query_text])[0]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where,
    )
    log_with_context(
        logger, "info", "dense query executed",
        query=query_text, result_count=len(results["ids"][0]),
    )
    return results