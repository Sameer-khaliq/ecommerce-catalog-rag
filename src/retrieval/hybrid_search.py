from src.retrieval.vectorstore import query_dense
from src.retrieval.bm25_index import query_bm25
from src.logging_config import get_logger, log_with_context

logger = get_logger(__name__)


def _build_where_filter(category=None, max_price=None, chunk_strategy=None) -> dict | None:
    """Wraps multiple clauses in $and — some Chroma versions reject flat multi-key dicts."""
    clauses = []
    if category:
        clauses.append({"category": category})
    if max_price is not None:
        clauses.append({"price": {"$lte": max_price}})
    if chunk_strategy:
        clauses.append({"chunk_strategy": chunk_strategy})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def reciprocal_rank_fusion(bm25_ids: list[str], dense_ids: list[str], k: int = 60) -> list[str]:
    scores: dict[str, float] = {}
    for rank, doc_id in enumerate(bm25_ids):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    for rank, doc_id in enumerate(dense_ids):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)


# src/retrieval/hybrid_search.py mein sirf function signature aur where_filter build karne wala hissa
def hybrid_search(
    query_text: str,
    collection,
    bm25_index: dict,
    category: str | None = None,
    max_price: float | None = None,
    chunk_strategy: str | None = None,   # <-- add this param
    top_k: int = 10,
) -> list[str]:
    where_filter = {}
    if category:
        where_filter["category"] = category
    if max_price is not None:
        where_filter["price"] = {"$lte": max_price}
    if chunk_strategy:
        where_filter["chunk_strategy"] = chunk_strategy

    # agar 2+ conditions hain, Chroma ko $and wrapper chahiye ho sakta hai
    if len(where_filter) > 1:
        where_filter = {"$and": [{k: v} for k, v in where_filter.items()]}

    dense_results = query_dense(collection, query_text, top_k=top_k * 2, where=where_filter or None)
    dense_ids = dense_results["ids"][0]

    bm25_ids_all = query_bm25(bm25_index, query_text, top_k=len(bm25_index["ids"]))
    bm25_ids = [d for d in bm25_ids_all if d in dense_ids] if where_filter else bm25_ids_all[: top_k * 2]

    fused_ids = reciprocal_rank_fusion(bm25_ids, dense_ids)
    log_with_context(logger, "info", "hybrid search executed", query=query_text, chunk_strategy=chunk_strategy, result_count=len(fused_ids[:top_k]))
    return fused_ids[:top_k]