from __future__ import annotations

from models import CalendarAction
from server.environment import CalendarSchedulingEnvironment
from task_definitions import TASKS


def test_easy_task_reaches_full_score_in_one_step():
    env = CalendarSchedulingEnvironment()
    reset_result = env.reset("task_easy")
    request = TASKS["task_easy"].requested_meetings[0]

    result = env.step(
        reset_result.episode_id,
        CalendarAction(
            action_type="schedule_event",
            title=request.title,
            start_time=request.start_time,
            duration_hours=request.duration_hours,
            participants=list(request.participants),
        ),
    )

    assert result.done is True
    assert result.observation.score == 1.0
    assert result.reward > 0


def test_medium_task_requires_conflict_resolution():
    env = CalendarSchedulingEnvironment()
    reset_result = env.reset("task_medium")
    request = TASKS["task_medium"].requested_meetings[0]
    blocker_id = reset_result.observation.events[0].event_id

    failed_attempt = env.step(
        reset_result.episode_id,
        CalendarAction(
            action_type="schedule_event",
            title=request.title,
            start_time=request.start_time,
            duration_hours=request.duration_hours,
            participants=list(request.participants),
        ),
    )
    assert failed_attempt.observation.score == 0.0
    assert failed_attempt.reward < 0
    assert failed_attempt.observation.last_action_error is not None
    assert failed_attempt.observation.reward_breakdown.invalid_action_penalty < 0

    env.step(
        reset_result.episode_id,
        CalendarAction(action_type="cancel_event", event_id=blocker_id),
    )
    solved = env.step(
        reset_result.episode_id,
        CalendarAction(
            action_type="schedule_event",
            title=request.title,
            start_time=request.start_time,
            duration_hours=request.duration_hours,
            participants=list(request.participants),
        ),
    )

    assert solved.observation.score == 1.0
    assert solved.done is True


def test_hard_task_grants_partial_credit_for_one_correct_meeting():
    env = CalendarSchedulingEnvironment()
    reset_result = env.reset("task_hard")
    first_request = TASKS["task_hard"].requested_meetings[0]
    blocker = next(
        event for event in reset_result.observation.events if event.start_time == first_request.start_time
    )

    env.step(
        reset_result.episode_id,
        CalendarAction(action_type="cancel_event", event_id=blocker.event_id),
    )

    partial = env.step(
        reset_result.episode_id,
        CalendarAction(
            action_type="schedule_event",
            title=first_request.title,
            start_time=first_request.start_time,
            duration_hours=first_request.duration_hours,
            participants=list(first_request.participants),
        ),
    )

    assert partial.done is False
    assert partial.observation.score == 0.5
