from bq_service_desk.agents.prompts import ROUTER_TASK


def test_router_has_exact_routes() -> None:
    assert "WEBSITE" in ROUTER_TASK
    assert "CURRICULUM" in ROUTER_TASK
    assert "SCHEDULE" in ROUTER_TASK
