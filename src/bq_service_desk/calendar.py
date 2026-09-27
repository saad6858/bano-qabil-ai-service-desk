"""Read-only Google Calendar service for the live Schedule Agent."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from ..config import Settings


class CalendarService:
    """Queries a live calendar without storing schedule data in RAG."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.timezone = ZoneInfo(settings.google_calendar_timezone)

    def events_for_day(self, day: datetime) -> list[dict[str, str]]:
        if not self.settings.google_calendar_id:
            return []

        from googleapiclient.discovery import build
        from google.oauth2 import service_account

        if not self.settings.google_service_account_file:
            return []

        credentials = service_account.Credentials.from_service_account_file(
            self.settings.google_service_account_file,
            scopes=["https://www.googleapis.com/auth/calendar.readonly"],
        )
        service = build("calendar", "v3", credentials=credentials, cache_discovery=False)

        local_day = day.astimezone(self.timezone)
        start = local_day.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1)

        response = (
            service.events()
            .list(
                calendarId=self.settings.google_calendar_id,
                timeMin=start.astimezone(timezone.utc).isoformat(),
                timeMax=end.astimezone(timezone.utc).isoformat(),
                singleEvents=True,
                orderBy="startTime",
            )
            .execute()
        )

        events: list[dict[str, str]] = []
        for event in response.get("items", []):
            start_data = event.get("start", {})
            start_value = start_data.get("dateTime") or start_data.get("date", "")
            events.append(
                {
                    "summary": event.get("summary", "Untitled event"),
                    "start": start_value,
                    "location": event.get("location", ""),
                }
            )
        return events
