"""Guide-compliant local document ingestion into Pinecone."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..config import Settings
from ..rate_limit import GeminiRateLimiter
from .embeddings import GeminiEmbeddingService
from pinecone import Pinecone


def load_text_documents(source_dir: Path) -> list[Document]:
    """Load text/Markdown files; PDF support is intentionally explicit."""

    documents: list[Document] = []
    for path in sorted(source_dir.rglob("*")):
        if path.suffix.lower() in {".txt", ".md"}:
            documents.append(
                Document(
                    page_content=path.read_text(encoding="utf-8"),
                    metadata={"source": str(path)},
                )
            )
    return documents


def load_pdf_documents(source_dir: Path) -> list[Document]:
    """Extract page text from PDF files for curriculum ingestion."""

    import fitz

    documents: list[Document] = []
    for path in sorted(source_dir.rglob("*.pdf")):
        pdf = fitz.open(path)
        try:
            for page_number, page in enumerate(pdf, start=1):
                text = page.get_text("text").strip()
                if text:
                    documents.append(
                        Document(
                            page_content=text,
                            metadata={
                                "source": str(path),
                                "page": page_number,
                            },
                        )
                    )
        finally:
            pdf.close()
    return documents


def index_documents(
    *,
    source_dir: Path,
    index_name: str,
    namespace: str,
    settings: Settings,
) -> int:
    """Batch embed chunks once per batch, then manually upsert vectors."""

    raw_docs = load_text_documents(source_dir)
    raw_docs.extend(load_pdf_documents(source_dir))

    if not raw_docs:
        raise ValueError(f"No supported documents found in {source_dir}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=300,
    )
    chunks = splitter.split_documents(raw_docs)

    limiter = GeminiRateLimiter(
        chat_delay_seconds=settings.gemini_chat_delay_seconds,
        embedding_delay_seconds=settings.gemini_embed_delay_seconds,
        cooldown_seconds=settings.gemini_429_cooldown_seconds,
        chat_rpm=settings.gemini_chat_rpm,
        chat_rpd=settings.gemini_chat_rpd,
        embedding_rpm=settings.gemini_embedding_rpm,
        embedding_rpd=settings.gemini_embedding_rpd,
    )
    embedder = GeminiEmbeddingService(settings, limiter)

    pinecone = Pinecone(api_key=settings.pinecone_api_key)
    description = pinecone.describe_index(index_name)
    dimension = getattr(description, "dimension", 3072)
    if int(dimension) != 3072:
        raise ValueError(
            f"Index '{index_name}' must be 3072-dimensional for "
            "gemini-embedding-001 in this implementation."
        )

    texts = [doc.page_content for doc in chunks]
    metadatas = [doc.metadata for doc in chunks]

    # This is the guide's core fix: batch embedding first, then manual upsert.
    vectors = embedder.embed_documents(texts)
    payload = []
    for index, (vector, text, metadata) in enumerate(
        zip(vectors, texts, metadatas, strict=True)
    ):
        payload.append(
            {
                "id": f"{uuid4()}-{index}",
                "values": vector,
                "metadata": {
                    "text": text,
                    "source": str(metadata.get("source", "")),
                    "page": int(metadata["page"]) if "page" in metadata else 0,
                },
            }
        )

    index_client = pinecone.Index(index_name)
    for start in range(0, len(payload), 1000):
        index_client.upsert(
            vectors=payload[start : start + 1000],
            namespace=namespace,
        )

    return len(payload)
