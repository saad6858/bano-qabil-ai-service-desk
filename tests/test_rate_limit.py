from bq_service_desk.rate_limit import GeminiRateLimiter


def test_limiter_accepts_project_guide_values() -> None:
    limiter = GeminiRateLimiter(
        chat_delay_seconds=3,
        embedding_delay_seconds=5,
        cooldown_seconds=60,
    )
    assert limiter.chat_delay_seconds == 3
    assert limiter.embedding_delay_seconds == 5
    assert limiter.cooldown_seconds == 60
