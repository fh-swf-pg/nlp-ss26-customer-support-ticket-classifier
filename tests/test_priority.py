from app.priority_engine import calculate_priority
from app.schemas import PriorityEnum


def test_calculate_priority_is_case_insensitive_for_categories() -> None:
    assert calculate_priority("system outage", "positive") == PriorityEnum.CRITICAL


def test_calculate_priority_escalates_network_issues_on_negative_sentiment() -> None:
    assert calculate_priority("Network & Connectivity", "ANGRY") == PriorityEnum.HIGH