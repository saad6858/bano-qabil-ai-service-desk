from datetime import datetime
from zoneinfo import ZoneInfo


def test_project_timezone_is_valid() -> None:
    dt = datetime.now(ZoneInfo("Asia/Karachi"))
    assert dt.tzinfo is not None
