"""Small Pinecone data-plane wrapper for Gemini 3072-dimensional vectors."""

from __future__ import annotations

from dataclasses import dataclass

from pinecone import Pinecone

from ..config import Settings
from .embeddings import GeminiEmbeddingService


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    score: float
    source: str = ""


class PineconeRAG:
    """Retrieves pre-embedded records from a standard Pinecone vector index."""

    def __init__(
        self,
        settings: Settings,
        embeddings: GeminiEmbeddingService,
    ) -> None:
        self.settings = settings
        self.embeddings = embeddings
        self.client = Pinecone(api_key=settings.pinecone_api_key)

    def verify_index_dimension(self, index_name: str, expected: int = 3072) -> None:
        """Fail early if a Python Gemini index uses the wrong vector dimension."""

        description = self.client.describe_index(index_name)
        dimension = getattr(description, "dimension", None)
        if dimension is None:
            # Newer SDK shapes expose dimension under model metadata for some
            # index types. In that case, leave the server to validate queries.
            return
        if int(dimension) != expected:
            raise ValueError(
                f"Index '{index_name}' has dimension {dimension}, but "
                f"gemini-embedding-001 for this project expects {expected}."
            )

    def retrieve(
        self,
        *,
        question: str,
        index_name: str,
        namespace: str,
        top_k: int,
        min_score: float,
    ) -> list[RetrievedChunk]:
        """Embed the question once, then query Pinecone once."""

        self.verify_index_dimension(index_name)
        index = self.client.Index(index_name)
        query_vector = self.embeddings.embed_query(question)

        response = index.query(
            vector=query_vector,
            top_k=top_k,
            namespace=namespace,
            include_metadata=True,
        )

        chunks: list[RetrievedChunk] = []
        for match in response.matches:
            score = float(getattr(match, "score", 0.0) or 0.0)
            if score < min_score:
                continue

            metadata = getattr(match, "metadata", {}) or {}
            text = (
                metadata.get("text")
                or metadata.get("page_content")
                or metadata.get("content")
                or ""
            )
            if not text:
                continue

            chunks.append(
                RetrievedChunk(
                    text=str(text),
                    score=score,
                    source=str(metadata.get("source", "")),
                )
            )
        return chunks
