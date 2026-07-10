import json
from pathlib import Path

import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# --- WRAPPER COMPATIBLE 4 METRICS ---
from ragas.metrics._faithfulness import Faithfulness
from ragas.metrics._context_recall import ContextRecall
from ragas.metrics._context_precision import ContextPrecision
# -------------------------------------
# -----------------------------------

from src.config import get_settings
from src.logging_config import get_logger, log_with_context
from src.retrieval.vectorstore import get_client, get_or_create_collection
from src.retrieval.bm25_index import build_bm25_index
from src.retrieval.hybrid_search import hybrid_search
from src.ingestion.loaders import load_listings
from src.ingestion.chunking import chunk_all_products

logger = get_logger(__name__)
settings = get_settings()

def load_eval_dataset(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_chunk_texts(collection, ids: list[str]) -> list[str]:
    result = collection.get(ids=ids)
    return result["documents"]


def build_ragas_dataset(eval_queries: list[dict], collection, bm25_idx, chunk_strategy: str | None = None) -> Dataset:
    """Runs hybrid search per query, optionally filtered to one chunk_strategy so we
    can compare attribute-table vs prose retrieval quality separately."""
    questions, contexts, answers, ground_truths = [], [], [], []

    for item in eval_queries:
        result_ids = hybrid_search(item["query"], collection, bm25_idx, top_k=5)

        if chunk_strategy:
            metas = collection.get(ids=result_ids)["metadatas"]
            result_ids = [rid for rid, m in zip(result_ids, metas) if m["chunk_strategy"] == chunk_strategy]

        retrieved_texts = _get_chunk_texts(collection, result_ids) if result_ids else [""]

        questions.append(item["query"])
        contexts.append(retrieved_texts)
        answers.append(retrieved_texts[0] if retrieved_texts else "")
        ground_truths.append(item["reference_answer"])

    return Dataset.from_dict({
        "question": questions,
        "contexts": contexts,
        "answer": answers,
        "ground_truth": ground_truths,
    })


def run_ragas_eval(eval_path: str, output_path: str) -> pd.DataFrame:
    eval_queries = load_eval_dataset(eval_path)

    client = get_client()
    collection = get_or_create_collection(client)

    listings = load_listings("data/processed_data/enriched_listings.json")
    chunks = chunk_all_products(listings)
    bm25_idx = build_bm25_index(chunks)

    all_results = []
    
    
   # 1. Instantiate the raw LangChain objects
    raw_groq = ChatGroq(
        model=settings.groq_model, 
        temperature=0,
        groq_api_key=settings.groq_api_key
    )
    raw_embeddings = GoogleGenerativeAIEmbeddings(
        model=settings.gemini_embedding_model,
        google_api_key=settings.google_api_key
    )
    
    evaluator_llm = LangchainLLMWrapper(raw_groq)
    evaluator_embeddings = LangchainEmbeddingsWrapper(raw_embeddings)
    for strategy in ["attribute_table", "prose", None]:
        label = strategy or "combined"
        dataset = build_ragas_dataset(eval_queries, collection, bm25_idx, chunk_strategy=strategy)
        
        # --- UPDATE THE EVALUATE LLM/EMBEDDING PARAMETERS ---
        scores = evaluate(
            dataset, 
            metrics=[
                Faithfulness(llm=evaluator_llm),  
                ContextRecall(llm=evaluator_llm), 
                ContextPrecision(llm=evaluator_llm)
            ],
            llm=evaluator_llm,
            embeddings=evaluator_embeddings
        )
        # ----------------------------------------------------
        
        df = scores.to_pandas()
        df["chunk_strategy"] = label
        all_results.append(df)
        log_with_context(
            logger, "info", "ragas eval completed", strategy=label,
            faithfulness=float(df["faithfulness"].mean()),
            context_recall=float(df["context_recall"].mean()),
            context_precision=float(df["context_precision"].mean()),
        )

    combined = pd.concat(all_results, ignore_index=True)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)

    print("\n=== RAGAS Summary (mean per chunk strategy) ===")
    print(combined.groupby("chunk_strategy")[["faithfulness", "context_recall", "context_precision"]].mean())

    return combined


if __name__ == "__main__":
    run_ragas_eval("eval/eval_dataset.json", "eval/ragas_results.csv")