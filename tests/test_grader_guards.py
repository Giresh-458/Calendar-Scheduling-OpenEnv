from __future__ import annotations

from models import CalendarAction
from server.environment import CalendarSchedulingEnvironment
from task_definitions import TASKS


def test_hard_task_does_not_allow_deleting_required_seed_events_for_full_credit():
    env = CalendarSchedulingEnvironment()
    reset_result = env.reset("task_hard")
    episode_id = reset_result.episode_id

    for event in list(reset_result.observation.events):
        env.step(episode_id, CalendarAction(action_type="cancel_event", event_id=event.event_id))

    for request in TASKS["task_hard"].requested_meetings:
        env.step(
            episode_id,
            CalendarAction(
                action_type="schedule_event",
                title=request.title,
                start_time=request.start_time,
                duration_hours=request.duration_hours,
                participants=list(request.participants),
            ),
        )

    grade = env.grade_episode(episode_id)
    assert grade.score < 1.0


def test_wrong_title_does_not_receive_full_credit():
    env = CalendarSchedulingEnvironment()
    reset_result = env.reset("task_easy")
    request = TASKS["task_easy"].requested_meetings[0]

    result = env.step(
        reset_result.episode_id,
        CalendarAction(
            action_type="schedule_event",
            title="Wrong Title",
            start_time=request.start_time,
            duration_hours=request.duration_hours,
            participants=list(request.participants),
        ),
    )

    assert result.observation.score < 1.0
