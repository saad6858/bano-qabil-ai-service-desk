"""Main Manager Agent.

The manager only routes. It does not answer the user's factual question.
"""

from crewai import Agent, Crew, Process, Task

from .prompts import ROUTER_TASK
from ..llm import SharedGeminiLLM


class MainManagerAgent:
    """One-step routing agent with delegation disabled."""

    def __init__(self, llm: SharedGeminiLLM) -> None:
        self.agent = Agent(
            role="Main Manager",
            goal="Route each request to exactly one correct specialist.",
            backstory=(
                "You are the central Service Desk router. You never fabricate "
                "answers and you do not perform specialist work yourself."
            ),
            llm=llm,
            allow_delegation=False,
            max_iter=1,
            verbose=False,
        )

    def route(self, question: str) -> str:
        task = Task(
            description=f"{ROUTER_TASK}\n\nUser request:\n{question}",
            expected_output="Exactly one word: WEBSITE, CURRICULUM, or SCHEDULE.",
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
        route = str(getattr(result, "raw", result)).strip().upper()
        if "SCHEDULE" in route:
            return "SCHEDULE"
        if "CURRICULUM" in route:
            return "CURRICULUM"
        return "WEBSITE"
