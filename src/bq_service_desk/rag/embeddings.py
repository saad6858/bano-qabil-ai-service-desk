"""Gemini embedding helpers.

Document embedding is intentionally batch-oriented because the project guide
observed 429 errors when every chunk was embedded through separate requests.
"""

from __future__ import annotations

from collections.abc import Iterable

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from ..config import Settings
from ..rate_limit import GeminiRateLimiter


class GeminiEmbeddingService:
    """Shared Gemini Embedding 001 client for both RAG indexes."""

    def __init__(self, settings: Settings, limiter: GeminiRateLimiter) -> None:
        self.settings = settings
        self.limiter = limiter

        self.document_embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.gemini_embedding_model,
            google_api_key=settings.google_api_key,
            task_type="retrieval_document",
        )
        self.query_embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.gemini_embedding_model,
            google_api_key=settings.google_api_key,
            task_type="retrieval_query",
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed many documents in API batches, never one document at a time."""

        if not texts:
            return []

        # The Google embedding batch interface is commonly capped at 100 inputs.
        # Keeping the batch bounded prevents oversized requests while preserving
        # the guide's core rule: one API call per batch, not one per chunk.
        all_vectors: list[list[float]] = []
        for start in range(0, len(texts), 100):
            batch = texts[start : start + 100]
            self.limiter.before_embedding()
            try:
                vectors = self.document_embeddings.embed_documents(batch)
            except Exception as exc:
                if "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc):
                    self.limiter.cooldown_after_429()
                    self.limiter.before_embedding()
                    vectors = self.document_embeddings.embed_documents(batch)
                else:
                    raise
            all_vectors.extend(vectors)
        return all_vectors

    def embed_query(self, question: str) -> list[float]:
        """Embed one user query after the required 5-second pacing delay."""

        self.limiter.before_embedding()
        try:
            return self.query_embeddings.embed_query(question)
        except Exception as exc:
            if "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc):
                self.limiter.cooldown_after_429()
                self.limiter.before_embedding()
                return self.query_embeddings.embed_query(question)
            raise
