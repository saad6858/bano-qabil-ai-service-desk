"""Project configuration loaded from environment variables."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed settings keep credentials and rate-limit controls out of code."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    google_api_key: str
    gemini_chat_model: str = "gemini-3.5-flash-lite"
    gemini_backup_chat_model: str = "gemini-3.1-flash-lite"
    gemini_embedding_model: str = "models/gemini-embedding-001"

    pinecone_api_key: str
    pinecone_environment: str | None = None
    pinecone_website_index: str = "bq-website-knowledge"
    pinecone_curriculum_index: str = "bq-curriculum-knowledge"
    pinecone_website_namespace: str = "website"
    pinecone_curriculum_namespace: str = "curriculum"

    rag_top_k: int = 5
    rag_min_score: float = 0.35

    gemini_chat_delay_seconds: float = 3.0
    gemini_embed_delay_seconds: float = 5.0
    gemini_429_cooldown_seconds: float = 60.0
    gemini_chat_rpm: int = 15
    gemini_chat_rpd: int = 500
    gemini_embedding_rpm: int = 100
    gemini_embedding_rpd: int = 1000
    gemini_max_retries: int = 2

    google_calendar_id: str | None = None
    google_service_account_file: str | None = None
    google_calendar_timezone: str = "Asia/Karachi"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
