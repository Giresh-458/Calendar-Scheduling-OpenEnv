from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Literal, Tuple


def utc_datetime(year: int, month: int, day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=timezone.utc)


BASE_CURRENT_TIME = utc_datetime(2026, 4, 1, 9, 0)
BASE_TARGET_DAY = BASE_CURRENT_TIME + timedelta(days=1)


def tomorrow_at(hour: int, minute: int = 0) -> datetime:
    return BASE_TARGET_DAY.replace(hour=hour, minute=minute)


@dataclass(frozen=True)
class MeetingTemplate:
    request_id: str
    title: str
    start_time: datetime
    duration_hours: float
    participants: Tuple[str, ...] = ()

    @property
    def end_time(self) -> datetime:
        return self.start_time + timedelta(hours=self.duration_hours)


@dataclass(frozen=True)
class TaskDefinition:
    task_id: str
    name: str
    difficulty: Literal["easy", "medium", "hard"]
    description: str
    current_time: datetime
    max_steps: int
    requested_meetings: Tuple[MeetingTemplate, ...]
    initial_events: Tuple[MeetingTemplate, ...] = ()
    required_preserved_event_ids: Tuple[str, ...] = ()


TASKS: Dict[str, TaskDefinition] = {
    "task_easy": TaskDefinition(
        task_id="task_easy",
        name="One Meeting",
        difficulty="easy",
        description=(
            "Schedule a single one-hour meeting for tomorrow at 10:00 without creating conflicts."
        ),
        current_time=BASE_CURRENT_TIME,
        max_steps=5,
        requested_meetings=(
            MeetingTemplate(
                request_id="req_easy_primary",
                title="Planning Sync",
                start_time=tomorrow_at(10),
                duration_hours=1.0,
                participants=("alex@example.com", "sam@example.com"),
            ),
        ),
    ),
    "task_medium": TaskDefinition(
        task_id="task_medium",
        name="Conflict Resolution",
        difficulty="medium",
        description=(
            "A requested meeting needs tomorrow's 10:00-11:00 slot, but an existing event blocks it. "
            "Resolve the blocker and place the requested meeting at the target time."
        ),
        current_time=BASE_CURRENT_TIME,
        max_steps=6,
        requested_meetings=(
            MeetingTemplate(
                request_id="req_medium_primary",
                title="Customer Review",
                start_time=tomorrow_at(10),
                duration_hours=1.0,
                participants=("alex@example.com", "jordan@example.com"),
            ),
        ),
        initial_events=(
            MeetingTemplate(
                request_id="seed_medium_blocker",
                title="Team Sync",
                start_time=tomorrow_at(10),
                duration_hours=1.0,
                participants=("alex@example.com", "sam@example.com"),
            ),
        ),
    ),
    "task_hard": TaskDefinition(
        task_id="task_hard",
        name="Multi-Meeting Coordination",
        difficulty="hard",
        description=(
            "Clear two blocking meetings, then schedule two back-to-back meetings tomorrow at "
            "10:00-11:00 and 11:00-12:00 while preserving the surrounding anchor events."
        ),
        current_time=BASE_CURRENT_TIME,
        max_steps=8,
        requested_meetings=(
            MeetingTemplate(
                request_id="req_hard_first",
                title="Design Review",
                start_time=tomorrow_at(10),
                duration_hours=1.0,
                participants=("alex@example.com", "maya@example.com"),
            ),
            MeetingTemplate(
                request_id="req_hard_second",
                title="Hiring Sync",
                start_time=tomorrow_at(11),
                duration_hours=1.0,
                participants=("alex@example.com", "lee@example.com"),
            ),
        ),
        initial_events=(
            MeetingTemplate(
                request_id="seed_hard_morning",
                title="Event A",
                start_time=tomorrow_at(9),
                duration_hours=1.0,
                participants=("alex@example.com",),
            ),
            MeetingTemplate(
                request_id="seed_hard_blocker_one",
                title="Budget Review",
                start_time=tomorrow_at(10),
                duration_hours=1.0,
                participants=("alex@example.com", "finance@example.com"),
            ),
            MeetingTemplate(
                request_id="seed_hard_blocker_two",
                title="Vendor Check-In",
                start_time=tomorrow_at(11),
                duration_hours=1.0,
                participants=("alex@example.com", "ops@example.com"),
            ),
            MeetingTemplate(
                request_id="seed_hard_afternoon",
                title="Event B",
                start_time=tomorrow_at(12),
                duration_hours=1.0,
                participants=("alex@example.com",),
            ),
        ),
        required_preserved_event_ids=("seed_hard_morning", "seed_hard_afternoon"),
    ),
}
