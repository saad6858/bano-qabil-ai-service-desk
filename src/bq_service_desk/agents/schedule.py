"""Schedule Agent backed by live Google Calendar data."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from crewai import Agent, Crew, Process, Task

from .prompts import SCHEDULE_TASK
from ..config import Settings
from ..llm import SharedGeminiLLM
from ..services.calendar import CalendarService


class ScheduleAgent:
    def __init__(
        self,
        llm: SharedGeminiLLM,
        calendar: CalendarService,
        settings: Settings,
    ) -> None:
        self.calendar = calendar
        self.settings = settings
        self.agent = Agent(
            role="Schedule Agent",
            goal="Explain live Google Calendar entries accurately.",
            backstory=(
                "You are the live schedule specialist. You use only the "
                "current calendar data provided to you."
            ),
            llm=llm,
            allow_delegation=False,
            max_iter=1,
            verbose=False,
        )

    def answer(self, question: str, date: datetime | None = None) -> str:
        tz = ZoneInfo(self.settings.google_calendar_timezone)
        target = date.astimezone(tz) if date else datetime.now(tz)
        events = self.calendar.events_for_day(target)

        if not events:
            live_context = "NO_SCHEDULE_ENTRIES"
        else:
            live_context = "\n".join(
                f"- {event['summary']} | {event['start']} | {event['location']}"
                for event in events
            )

        task = Task(
            description=f"{SCHEDULE_TASK}\n\nLive calendar data:\n{live_context}\n\nQuestion:\n{question}",
            expected_output="A concise plain-text schedule answer.",
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
