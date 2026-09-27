"""Curriculum Agent."""

from crewai import Agent, Crew, Process, Task

from .prompts import CURRICULUM_TASK
from ..llm import SharedGeminiLLM
from ..rag.pinecone import RetrievedChunk


class CurriculumAgent:
    def __init__(self, llm: SharedGeminiLLM) -> None:
        self.agent = Agent(
            role="Curriculum Agent",
            goal="Answer course-curriculum questions only from retrieved curriculum evidence.",
            backstory=(
                "You are the syllabus specialist. You never invent curriculum "
                "content and never substitute public website facts."
            ),
            llm=llm,
            allow_delegation=False,
            max_iter=1,
            verbose=False,
        )

    def answer(self, question: str, chunks: list[RetrievedChunk]) -> str:
        context = _format_context(chunks)
        task = Task(
            description=f"{CURRICULUM_TASK}\n\nRetrieved context:\n{context}\n\nQuestion:\n{question}",
            expected_output="A grounded plain-text user answer or NOT_FOUND_IN_CURRICULUM.",
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
