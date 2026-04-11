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
class CandidateSlotTemplate:
    start_time: datetime
    duration_hours: float
    label: str = ""
    preference: Literal["preferred", "acceptable", "fallback"] = "acceptable"

    @property
    def end_time(self) -> datetime:
        return self.start_time + timedelta(hours=self.duration_hours)


def candidate_slot(
    hour: int,
    minute: int = 0,
    *,
    duration_hours: float = 1.0,
    label: str = "",
    preference: Literal["preferred", "acceptable", "fallback"] = "acceptable",
) -> CandidateSlotTemplate:
    return CandidateSlotTemplate(
        start_time=tomorrow_at(hour, minute),
        duration_hours=duration_hours,
        label=label,
        preference=preference,
    )


@dataclass(frozen=True)
class MeetingTemplate:
    request_id: str
    title: str
    start_time: datetime
    duration_hours: float
    participants: Tuple[str, ...] = ()
    priority: int = 1
    alternate_slots: Tuple[CandidateSlotTemplate, ...] = ()
    relocation_candidates: Tuple[CandidateSlotTemplate, ...] = ()
    movable: bool = True
    protected: bool = False
    notes: Tuple[str, ...] = ()

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
    preferred_preserved_event_ids: Tuple[str, ...] = ()
    scheduler_notes: Tuple[str, ...] = ()
    scenario_type: str = "calendar_coordination"


TASKS: Dict[str, TaskDefinition] = {
    "task_easy": TaskDefinition(
        task_id="task_easy",
        name="One Meeting",
        difficulty="easy",
        description=(
            "Schedule a single one-hour planning meeting tomorrow at 10:00 without creating conflicts."
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
                priority=3,
                notes=("The team strongly prefers the morning slot.",),
            ),
        ),
        scheduler_notes=(
            "This is the warm-up task. Place the meeting cleanly with no collateral edits.",
        ),
        scenario_type="team_coordination",
    ),
    "task_medium": TaskDefinition(
        task_id="task_medium",
        name="Conflict Resolution",
        difficulty="medium",
        description=(
            "A requested customer review needs tomorrow's 10:00-11:00 slot, but an internal sync blocks it. "
            "Move the blocker to its fallback slot and place the requested meeting at the preferred time."
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
                priority=4,
                alternate_slots=(
                    candidate_slot(15, duration_hours=1.0, label="late afternoon fallback"),
                ),
                notes=(
                    "The customer prefers the original morning slot.",
                    "An afternoon fallback is acceptable but not ideal.",
                ),
            ),
        ),
        initial_events=(
            MeetingTemplate(
                request_id="seed_medium_blocker",
                title="Team Sync",
                start_time=tomorrow_at(10),
                duration_hours=1.0,
                participants=("alex@example.com", "sam@example.com"),
                relocation_candidates=(
                    candidate_slot(13, duration_hours=1.0, label="approved spillover", preference="fallback"),
                ),
                notes=("This blocker can be moved to 13:00 without hurting anyone.",),
            ),
        ),
        preferred_preserved_event_ids=("seed_medium_blocker",),
        scheduler_notes=(
            "Prefer rescheduling over cancellation when a blocker has an approved fallback slot.",
        ),
        scenario_type="customer_coordination",
    ),
    "task_hard": TaskDefinition(
        task_id="task_hard",
        name="Multi-Meeting Coordination",
        difficulty="hard",
        description=(
            "Clear two blocking meetings, then schedule two back-to-back meetings tomorrow at "
            "10:00-11:00 and 11:00-12:00 while preserving the protected focus and lunch anchors."
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
                priority=4,
                alternate_slots=(
                    candidate_slot(15, duration_hours=1.0, label="afternoon fallback"),
                ),
            ),
            MeetingTemplate(
                request_id="req_hard_second",
                title="Hiring Sync",
                start_time=tomorrow_at(11),
                duration_hours=1.0,
                participants=("alex@example.com", "lee@example.com"),
                priority=3,
                alternate_slots=(
                    candidate_slot(16, duration_hours=1.0, label="late fallback"),
                ),
            ),
        ),
        initial_events=(
            MeetingTemplate(
                request_id="seed_hard_focus",
                title="Focus Block",
                start_time=tomorrow_at(9),
                duration_hours=1.0,
                participants=("alex@example.com",),
                movable=False,
                protected=True,
                notes=("This focus block is protected and must remain untouched.",),
            ),
            MeetingTemplate(
                request_id="seed_hard_blocker_one",
                title="Budget Review",
                start_time=tomorrow_at(10),
                duration_hours=1.0,
                participants=("alex@example.com", "finance@example.com"),
                relocation_candidates=(
                    candidate_slot(13, duration_hours=1.0, label="approved afternoon move", preference="fallback"),
                ),
                notes=("Reschedule this blocker instead of deleting it.",),
            ),
            MeetingTemplate(
                request_id="seed_hard_blocker_two",
                title="Vendor Check-In",
                start_time=tomorrow_at(11),
                duration_hours=1.0,
                participants=("alex@example.com", "ops@example.com"),
                relocation_candidates=(
                    candidate_slot(14, duration_hours=1.0, label="approved vendor fallback", preference="fallback"),
                ),
                notes=("This check-in should also survive in the final calendar.",),
            ),
            MeetingTemplate(
                request_id="seed_hard_lunch",
                title="Lunch Buffer",
                start_time=tomorrow_at(12),
                duration_hours=1.0,
                participants=("alex@example.com",),
                movable=False,
                protected=True,
                notes=("The lunch buffer is a protected anchor event.",),
            ),
        ),
        required_preserved_event_ids=("seed_hard_focus", "seed_hard_lunch"),
        preferred_preserved_event_ids=("seed_hard_blocker_one", "seed_hard_blocker_two"),
        scheduler_notes=(
            "Protect focus and lunch anchors.",
            "Reschedule movable blockers rather than deleting them if you want full credit.",
        ),
        scenario_type="project_coordination",
    ),
    "task_exec_dense_day": TaskDefinition(
        task_id="task_exec_dense_day",
        name="Executive Dense Day",
        difficulty="hard",
        description=(
            "Coordinate an executive calendar with three requested meetings, protected anchors, "
            "and movable internal blockers. Preferred morning slots are best; late-day fallbacks earn only partial credit."
        ),
        current_time=BASE_CURRENT_TIME,
        max_steps=10,
        requested_meetings=(
            MeetingTemplate(
                request_id="req_exec_board_prep",
                title="Board Prep",
                start_time=tomorrow_at(10),
                duration_hours=1.0,
                participants=("alex@example.com", "chief_of_staff@example.com"),
                priority=5,
                alternate_slots=(
                    candidate_slot(15, duration_hours=1.0, label="late board prep"),
                ),
                notes=("This is the highest-priority request in the scenario.",),
            ),
            MeetingTemplate(
                request_id="req_exec_investor_sync",
                title="Investor Sync",
                start_time=tomorrow_at(11),
                duration_hours=1.0,
                participants=("alex@example.com", "finance@example.com"),
                priority=4,
                alternate_slots=(
                    candidate_slot(17, duration_hours=1.0, label="end-of-day investor fallback"),
                ),
                notes=("Morning placement is strongly preferred by the CFO.",),
            ),
            MeetingTemplate(
                request_id="req_exec_cos",
                title="Chief of Staff 1:1",
                start_time=tomorrow_at(14),
                duration_hours=0.5,
                participants=("alex@example.com", "chief_of_staff@example.com"),
                priority=3,
                alternate_slots=(
                    candidate_slot(17, 0, duration_hours=0.5, label="late-day 1:1"),
                ),
            ),
        ),
        initial_events=(
            MeetingTemplate(
                request_id="seed_exec_focus",
                title="Executive Focus Block",
                start_time=tomorrow_at(9),
                duration_hours=1.0,
                participants=("alex@example.com",),
                movable=False,
                protected=True,
                notes=("Protected anchor. Do not modify.",),
            ),
            MeetingTemplate(
                request_id="seed_exec_staff_sync",
                title="Staff Sync",
                start_time=tomorrow_at(10),
                duration_hours=1.0,
                participants=("alex@example.com", "staff@example.com"),
                relocation_candidates=(
                    candidate_slot(13, duration_hours=1.0, label="staff spillover", preference="fallback"),
                ),
            ),
            MeetingTemplate(
                request_id="seed_exec_finance_review",
                title="Finance Review",
                start_time=tomorrow_at(11),
                duration_hours=1.0,
                participants=("alex@example.com", "finance@example.com"),
                relocation_candidates=(
                    candidate_slot(17, duration_hours=1.0, label="late finance review", preference="fallback"),
                ),
            ),
            MeetingTemplate(
                request_id="seed_exec_lunch",
                title="Lunch Buffer",
                start_time=tomorrow_at(12),
                duration_hours=1.0,
                participants=("alex@example.com",),
                movable=False,
                protected=True,
            ),
            MeetingTemplate(
                request_id="seed_exec_office_hours",
                title="Office Hours",
                start_time=tomorrow_at(14),
                duration_hours=0.5,
                participants=("alex@example.com", "team@example.com"),
                relocation_candidates=(
                    candidate_slot(15, duration_hours=0.5, label="shifted office hours", preference="fallback"),
                ),
            ),
            MeetingTemplate(
                request_id="seed_exec_board_read",
                title="Board Read",
                start_time=tomorrow_at(16),
                duration_hours=1.0,
                participants=("alex@example.com",),
                movable=False,
                protected=True,
            ),
        ),
        required_preserved_event_ids=("seed_exec_focus", "seed_exec_lunch", "seed_exec_board_read"),
        preferred_preserved_event_ids=(
            "seed_exec_staff_sync",
            "seed_exec_finance_review",
            "seed_exec_office_hours",
        ),
        scheduler_notes=(
            "Morning placements are preferred for executive-facing meetings.",
            "Protected anchors must remain in place.",
            "Full credit requires preserving movable internal meetings by rescheduling them.",
        ),
        scenario_type="executive_assistance",
    ),
    "task_recruiting_loop": TaskDefinition(
        task_id="task_recruiting_loop",
        name="Recruiting Panel Loop",
        difficulty="hard",
        description=(
            "Protect recruiting anchors while creating a candidate panel and a debrief in the preferred morning window."
        ),
        current_time=BASE_CURRENT_TIME,
        max_steps=9,
        requested_meetings=(
            MeetingTemplate(
                request_id="req_recruit_panel",
                title="Candidate Panel",
                start_time=tomorrow_at(10),
                duration_hours=1.0,
                participants=(
                    "alex@example.com",
                    "candidate@example.com",
                    "eng_manager@example.com",
                ),
                priority=5,
                alternate_slots=(
                    candidate_slot(15, duration_hours=1.0, label="late candidate panel"),
                ),
                notes=("Candidate-facing meetings should stay in the preferred morning window.",),
            ),
            MeetingTemplate(
                request_id="req_recruit_debrief",
                title="Interview Debrief",
                start_time=tomorrow_at(11),
                duration_hours=0.5,
                participants=("alex@example.com", "eng_manager@example.com"),
                priority=3,
                alternate_slots=(
                    candidate_slot(16, duration_hours=0.5, label="late debrief"),
                ),
            ),
        ),
        initial_events=(
            MeetingTemplate(
                request_id="seed_recruiting_prep",
                title="Interview Prep",
                start_time=tomorrow_at(9, 30),
                duration_hours=0.5,
                participants=("alex@example.com",),
                movable=False,
                protected=True,
            ),
            MeetingTemplate(
                request_id="seed_recruiting_blocker_one",
                title="Team Sync",
                start_time=tomorrow_at(10),
                duration_hours=1.0,
                participants=("alex@example.com", "staff@example.com"),
                relocation_candidates=(
                    candidate_slot(13, duration_hours=1.0, label="team sync fallback", preference="fallback"),
                ),
            ),
            MeetingTemplate(
                request_id="seed_recruiting_blocker_two",
                title="Hiring Calibration",
                start_time=tomorrow_at(11),
                duration_hours=0.5,
                participants=("alex@example.com", "recruiting@example.com"),
                relocation_candidates=(
                    candidate_slot(14, duration_hours=0.5, label="calibration fallback", preference="fallback"),
                ),
            ),
            MeetingTemplate(
                request_id="seed_recruiting_lunch",
                title="Candidate Lunch Buffer",
                start_time=tomorrow_at(12),
                duration_hours=1.0,
                participants=("alex@example.com",),
                movable=False,
                protected=True,
            ),
        ),
        required_preserved_event_ids=("seed_recruiting_prep", "seed_recruiting_lunch"),
        preferred_preserved_event_ids=(
            "seed_recruiting_blocker_one",
            "seed_recruiting_blocker_two",
        ),
        scheduler_notes=(
            "Keep the candidate-facing meetings in the preferred morning window if possible.",
            "Do not disturb the prep block or lunch buffer.",
        ),
        scenario_type="recruiting_operations",
    ),
}
