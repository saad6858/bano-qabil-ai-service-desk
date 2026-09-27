"""Website Knowledge Agent."""

from crewai import Agent, Crew, Process, Task

from .prompts import WEBSITE_TASK
from ..llm import SharedGeminiLLM
from ..rag.pinecone import RetrievedChunk


class WebsiteKnowledgeAgent:
    def __init__(self, llm: SharedGeminiLLM) -> None:
        self.agent = Agent(
            role="Website Knowledge Agent",
            goal="Answer public Bano Qabil questions only from retrieved website evidence.",
            backstory=(
                "You are the public-information specialist for Bano Qabil. "
                "You never invent facts that were not retrieved."
            ),
            llm=llm,
            allow_delegation=False,
            max_iter=1,
            verbose=False,
        )

    def answer(self, question: str, chunks: list[RetrievedChunk]) -> str:
        context = _format_context(chunks)
        task = Task(
            description=f"{WEBSITE_TASK}\n\nRetrieved context:\n{context}\n\nQuestion:\n{question}",
            expected_output="A grounded plain-text user answer or NOT_FOUND_IN_WEBSITE.",
            agent=self.agent,
        )
        crew = Crew(
            agents=[self.agent],
            tasks=[task],
            process=Process.sequential,
            verbose=False,
            cache=False,
        )
        result = crew.kickoff()
        return str(getattr(result, "raw", result)).strip()


def _format_context(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "NO_RELEVANT_CONTEXT"
    return "\n\n".join(
        f"[Source: {chunk.source or 'unknown'} | score={chunk.score:.3f}]\n{chunk.text}"
        for chunk in chunks
    )
