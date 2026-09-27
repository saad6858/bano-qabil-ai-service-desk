from types import SimpleNamespace

from bq_service_desk.rag.pinecone import PineconeRAG


def test_indexed_match_metadata_supports_text_field() -> None:
    match = SimpleNamespace(
        id="x",
        score=0.9,
        metadata={"text": "Bano Qabil website content", "source": "faq"},
    )
    assert match.metadata["text"]
