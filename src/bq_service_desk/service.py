"""Top-level orchestration for the four-agent Service Desk."""

from __future__ import annotations

from .agents.curriculum import CurriculumAgent
from .agents.router import MainManagerAgent
from .agents.schedule import ScheduleAgent
from .agents.website import WebsiteKnowledgeAgent
from .config import get_settings
from .llm import SharedGeminiLLM
from .rag.embeddings import GeminiEmbeddingService
from .rag.pinecone import PineconeRAG
from .rate_limit import GeminiRateLimiter
from .services.calendar import CalendarService


class BanoQabilServiceDesk:
    """Build the four agents around shared, rate-limited services."""

    def __init__(self) -> None:
        settings = get_settings()

        # One limiter and one Gemini chat client are shared by every agent.
        limiter = GeminiRateLimiter(
            chat_delay_seconds=settings.gemini_chat_delay_seconds,
            embedding_delay_seconds=settings.gemini_embed_delay_seconds,
            cooldown_seconds=settings.gemini_429_cooldown_seconds,
            chat_rpm=settings.gemini_chat_rpm,
            chat_rpd=settings.gemini_chat_rpd,
            embedding_rpm=settings.gemini_embedding_rpm,
            embedding_rpd=settings.gemini_embedding_rpd,
        )

        llm = SharedGeminiLLM(
            model=settings.gemini_chat_model,
            api_key=settings.google_api_key,
            limiter=limiter,
        )

        embeddings = GeminiEmbeddingService(settings, limiter)
        pinecone_rag = PineconeRAG(settings, embeddings)
        calendar = CalendarService(settings)

        self.settings = settings
        self.manager = MainManagerAgent(llm)
        self.website = WebsiteKnowledgeAgent(llm)
        self.curriculum = CurriculumAgent(llm)
        self.schedule = ScheduleAgent(llm, calendar, settings)
        self.pinecone_rag = pinecone_rag

    def answer(self, question: str) -> str:
        route = self.manager.route(question)

        if route == "WEBSITE":
            chunks = self.pinecone_rag.retrieve(
                question=question,
                index_name=self.settings.pinecone_website_index,
                namespace=self.settings.pinecone_website_namespace,
                top_k=self.settings.rag_top_k,
                min_score=self.settings.rag_min_score,
            )
            return self.website.answer(question, chunks)

        if route == "CURRICULUM":
            chunks = self.pinecone_rag.retrieve(
                question=question,
                index_name=self.settings.pinecone_curriculum_index,
                namespace=self.settings.pinecone_curriculum_namespace,
                top_k=self.settings.rag_top_k,
                min_score=self.settings.rag_min_score,
            )
            return self.curriculum.answer(question, chunks)

        return self.schedule.answer(question)
