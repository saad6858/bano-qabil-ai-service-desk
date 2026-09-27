"""Gemini free-tier pacing, quota guarding, and 429 recovery.

The project guide observed:
- 3 seconds between Gemini chat calls.
- 5 seconds before every RAG embedding query.
- 60 seconds after a 429 before retrying.
- 15 RPM / 500 RPD for Gemini chat models in the guide snapshot.
- 100 RPM / 1000 RPD for Gemini Embedding 1 in the guide snapshot.

Google's live limits can change, so these are local safety guards rather than
claims that the provider will always expose exactly these values.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo


class DailyQuotaGuardError(RuntimeError):
    """Raised when the local safety guard reaches the guide's daily snapshot."""


class GeminiRateLimiter:
    """One shared limiter keeps all agents on the same Gemini API budget."""

    PACIFIC_TZ = ZoneInfo("America/Los_Angeles")

    def __init__(
        self,
        chat_delay_seconds: float = 3.0,
        embedding_delay_seconds: float = 5.0,
        cooldown_seconds: float = 60.0,
        chat_rpm: int = 15,
        chat_rpd: int = 500,
        embedding_rpm: int = 100,
        embedding_rpd: int = 1000,
    ) -> None:
        self.chat_delay_seconds = chat_delay_seconds
        self.embedding_delay_seconds = embedding_delay_seconds
        self.cooldown_seconds = cooldown_seconds
        self.chat_rpm = chat_rpm
        self.chat_rpd = chat_rpd
        self.embedding_rpm = embedding_rpm
        self.embedding_rpd = embedding_rpd
        self._lock = threading.Lock()
        self._last_chat_at = 0.0
        self._last_embedding_at = 0.0
        self._chat_requests: list[float] = []
        self._embedding_requests: list[float] = []
        self._chat_day = self._current_pacific_day()
        self._embedding_day = self._current_pacific_day()
        self._chat_daily_count = 0
        self._embedding_daily_count = 0

    def before_chat(self) -> None:
        with self._lock:
            self._refresh_daily_counters()
            self._enforce_daily_limit(
                self._chat_daily_count,
                self.chat_rpd,
                "Gemini chat",
            )
            self._wait_since(self._last_chat_at, self.chat_delay_seconds)
            self._wait_for_rpm(self._chat_requests, self.chat_rpm)
            self._record(self._chat_requests)
            self._last_chat_at = time.monotonic()
            self._chat_daily_count += 1

    def before_embedding(self) -> None:
        with self._lock:
            self._refresh_daily_counters()
            self._enforce_daily_limit(
                self._embedding_daily_count,
                self.embedding_rpd,
                "Gemini embedding",
            )
            self._wait_since(self._last_embedding_at, self.embedding_delay_seconds)
            self._wait_for_rpm(self._embedding_requests, self.embedding_rpm)
            self._record(self._embedding_requests)
            self._last_embedding_at = time.monotonic()
            self._embedding_daily_count += 1

    def cooldown_after_429(self) -> None:
        time.sleep(self.cooldown_seconds)

    def _refresh_daily_counters(self) -> None:
        day = self._current_pacific_day()
        if self._chat_day != day:
            self._chat_day = day
            self._chat_daily_count = 0
        if self._embedding_day != day:
            self._embedding_day = day
            self._embedding_daily_count = 0

    @classmethod
    def _current_pacific_day(cls):
        return datetime.now(cls.PACIFIC_TZ).date()

    @staticmethod
    def _wait_since(last_call: float, minimum_gap: float) -> None:
        elapsed = time.monotonic() - last_call
        remaining = minimum_gap - elapsed
        if remaining > 0:
            time.sleep(remaining)

    @staticmethod
    def _wait_for_rpm(timestamps: list[float], max_rpm: int) -> None:
        while True:
            now = time.monotonic()
            one_minute_ago = now - 60.0
            while timestamps and timestamps[0] <= one_minute_ago:
                timestamps.pop(0)
            if len(timestamps) < max_rpm:
                return
            time.sleep(max(0.1, 60.0 - (now - timestamps[0])))

    @staticmethod
    def _record(timestamps: list[float]) -> None:
        timestamps.append(time.monotonic())

    @staticmethod
    def _enforce_daily_limit(current: int, limit: int, name: str) -> None:
        if current >= limit:
            raise DailyQuotaGuardError(
                f"{name} local daily safety guard reached. "
                "The project guide says RPD resets at midnight Pacific Time; "
                "stop rather than hammer the API."
            )
