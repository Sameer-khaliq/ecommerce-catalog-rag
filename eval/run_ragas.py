import json
from pathlib import Path

import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.run_config import RunConfig
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from ragas.metrics._faithfulness import Faithfulness
from ragas.metrics._context_recall import ContextRecall
from ragas.metrics._context_precision import ContextPrecision

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
    """Filtering ab hybrid_search ke andar hoti hai (ChromaDB where clause), post-hoc
    nahi — pehle top_k=5 mixed results maang ke baad mein filter karte the, jisse
    context bohot chhota reh jaata tha aur recall/precision artificially kharab dikhte the."""
    questions, contexts, answers, ground_truths = [], [], [], []

    for item in eval_queries:
        result_ids = hybrid_search(item["query"], collection, bm25_idx, chunk_strategy=chunk_strategy, top_k=5)
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


def run_ragas_eval(eval_path: str, output_path: str, sample_size: int | None = None) -> pd.DataFrame:
    eval_queries = load_eval_dataset(eval_path)
    if sample_size:
        eval_queries = eval_queries[:sample_size]

    client = get_client()
    collection = get_or_create_collection(client)

    listings = load_listings("data/processed_data/enriched_listings.json")
    chunks = chunk_all_products(listings)
    bm25_idx = build_bm25_index(chunks)

    raw_groq = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        groq_api_key=settings.groq_api_key,
    )
    raw_embeddings = GoogleGenerativeAIEmbeddings(
        model=settings.gemini_embedding_model,
        google_api_key=settings.google_api_key,
    )
    evaluator_llm = LangchainLLMWrapper(raw_groq)
    evaluator_embeddings = LangchainEmbeddingsWrapper(raw_embeddings)

    # Low concurrency + generous timeout — pehle wale run mein isi ki kami se
    # TimeoutError cascade hua tha (60 parallel calls Groq ko overwhelm kar rahe the)
    run_config = RunConfig(timeout=180, max_retries=3, max_wait=90, max_workers=2)

    all_results = []
    for strategy in ["attribute_table", "prose", None]:
        label = strategy or "combined"
        dataset = build_ragas_dataset(eval_queries, collection, bm25_idx, chunk_strategy=strategy)

        scores = evaluate(
            dataset,
            metrics=[
                Faithfulness(llm=evaluator_llm),
                ContextRecall(llm=evaluator_llm),
                ContextPrecision(llm=evaluator_llm),
            ],
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
            run_config=run_config,
        )

        df = scores.to_pandas()
        df["chunk_strategy"] = label
        all_results.append(df)

        available_metrics = [c for c in ["faithfulness", "context_recall", "context_precision"] if c in df.columns]
        log_with_context(
            logger, "info", "ragas eval completed", strategy=label,
            **{m: float(df[m].mean()) for m in available_metrics},
        )

    combined = pd.concat(all_results, ignore_index=True)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)

    metric_cols = [c for c in ["faithfulness", "context_recall", "context_precision"] if c in combined.columns]
    print("\n=== RAGAS Summary (mean per chunk strategy) ===")
    print(combined.groupby("chunk_strategy")[metric_cols].mean())

    missing = set(["faithfulness", "context_recall", "context_precision"]) - set(metric_cols)
    if missing:
        print(f"\n Metrics not computed: {missing}")

    return combined


if __name__ == "__main__":
    run_ragas_eval("eval/eval_dataset.json", "eval/ragas_results.csv", sample_size=6)  # pehle chhota test