from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ---- Application ----
    app_name: str = "ecommerce-catalog-rag"
    app_env: str = "development"
    log_level: str = "INFO"

    # ---- Groq (generation) ----
    groq_api_key: str
    groq_model: str = "openai/gpt-oss-120b"

    # ---- Gemini (embeddings) ----
    google_api_key: str
    gemini_embedding_model: str = "gemini-embedding-001"

    # ---- Vector store ----
    chroma_db_path: str = "./data/chroma_db"
    chroma_collection_name: str = "ecommerce_catalog"

    # ---- Retrieval ----
    retrieval_top_k: int = 5
    hybrid_search_alpha: float = 0.5

    # ---- Caching (diskcache) ----
    cache_enabled: bool = True
    cache_dir: str = "./data/cache"
    cache_ttl_seconds: int = 3600

    # ---- Resilience ----
    llm_max_retries: int = 3
    llm_retry_backoff_seconds: int = 2
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_recovery_seconds: int = 30

    # ---- Server ----
    host: str = "0.0.0.0"
    port: int = 8000
    gradio_port: int = 7860

    # ---- Observability ----
    enable_metrics_logging: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

settings = Settings()