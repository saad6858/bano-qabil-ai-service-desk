"""Shared Gemini LLM adapter for CrewAI.

CrewAI expects a BaseLLM. Wrapping LangChain's Google integration lets the
project keep one shared Gemini client while still demonstrating CrewAI agents
and LangChain components in the same implementation.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from crewai.llms.base_llm import BaseLLM
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import PrivateAttr

from .rate_limit import GeminiRateLimiter


class SharedGeminiLLM(BaseLLM):
    """CrewAI-compatible adapter around one shared LangChain Gemini instance."""

    llm_type: str = "gemini_shared"
    provider: str = "google"

    _client: Any = PrivateAttr()
    _limiter: GeminiRateLimiter = PrivateAttr()

    def __init__(
        self,
        model: str,
        api_key: str,
        limiter: GeminiRateLimiter,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> None:
        super().__init__(
            model=model,
            provider="google",
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
        )
        self._client = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        self._limiter = limiter

    def call(
        self,
        messages: str | list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
        response_model: Any | None = None,
    ) -> str:
        """Make one controlled Gemini chat request."""

        if tools:
            raise RuntimeError(
                "Tool-calling is intentionally disabled in this Python MVP. "
                "Retrieve data before the specialist call to avoid free-tier "
                "multi-step tool loops."
            )

        self._limiter.before_chat()
        prompt_messages = self._convert_messages(messages)
        try:
            response = self._client.invoke(prompt_messages)
        except Exception as exc:
            if "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc):
                self._limiter.cooldown_after_429()
                self._limiter.before_chat()
                response = self._client.invoke(prompt_messages)
            else:
                raise

        return self._extract_text(response)

    @staticmethod
    def _convert_messages(
        messages: str | list[dict[str, Any]],
    ) -> Sequence[BaseMessage]:
        if isinstance(messages, str):
            return [HumanMessage(content=messages)]

        converted: list[BaseMessage] = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            if role == "system":
                converted.append(SystemMessage(content=content))
            elif role == "assistant":
                converted.append(AIMessage(content=content))
            else:
                converted.append(HumanMessage(content=content))
        return converted

    @staticmethod
    def _extract_text(response: Any) -> str:
        content = getattr(response, "content", response)
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and "text" in item:
                    parts.append(str(item["text"]))
            return "".join(parts).strip()
        return str(content).strip()
