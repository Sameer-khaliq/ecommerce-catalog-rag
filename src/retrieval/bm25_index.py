from rank_bm25 import BM25Okapi

from src.logging_config import get_logger, log_with_context

logger = get_logger(__name__)


def _tokenize(text: str) -> list[str]:
    """Simple lowercase whitespace tokenizer — good enough for product text."""
    return text.lower().split()


def build_bm25_index(chunks: list[dict]) -> dict:
    """Builds a BM25 index over the same chunks used for the dense index.
    Returns a dict bundling the index with the id/text/metadata lists,
    since BM25Okapi itself only knows about token lists, not your chunk objects.
    """
    ids = [f"{c['metadata']['product_id']}_{c['metadata']['chunk_strategy']}" for c in chunks]
    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    tokenized_corpus = [_tokenize(text) for text in texts]
    bm25 = BM25Okapi(tokenized_corpus)

    log_with_context(logger, "info", "bm25 index built", total_chunks=len(chunks))

    return {
        "bm25": bm25,
        "ids": ids,
        "texts": texts,
        "metadatas": metadatas,
    }


def query_bm25(index: dict, query_text: str, top_k: int = 10) -> list[str]:
    """Returns the top_k chunk ids ranked by BM25 score, highest first."""
    tokenized_query = _tokenize(query_text)
    scores = index["bm25"].get_scores(tokenized_query)

    ranked = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True,
    )[:top_k]

    result_ids = [index["ids"][i] for i in ranked]
    log_with_context(
        logger, "info", "bm25 query executed",
        query=query_text, result_count=len(result_ids),
    )
    return result_ids