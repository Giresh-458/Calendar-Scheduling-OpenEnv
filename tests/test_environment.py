from __future__ import annotations

from collections.abc import Mapping, Sequence

from models import CalendarAction, CalendarEvent
from server.environment import CalendarSchedulingEnvironment, MAX_PUBLIC_SCORE, MIN_PUBLIC_SCORE
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
    assert result.observation.score == MAX_PUBLIC_SCORE
    assert result.reward > 0


def test_medium_task_requires_rescheduling_for_full_score():
    env = CalendarSchedulingEnvironment()
    reset_result = env.reset("task_medium")
    request = TASKS["task_medium"].requested_meetings[0]
    blocker = reset_result.observation.events[0]
    relocation = blocker.relocation_candidates[0]

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
    assert failed_attempt.observation.score == MIN_PUBLIC_SCORE
    assert failed_attempt.reward < 0
    assert failed_attempt.observation.last_action_error is not None
    assert failed_attempt.observation.reward_breakdown.invalid_action_penalty < 0

    env.step(
        reset_result.episode_id,
        CalendarAction(
            action_type="reschedule_event",
            event_id=blocker.event_id,
            new_start_time=relocation.start_time,
            duration_hours=relocation.duration_hours,
        ),
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

    assert solved.observation.score == MAX_PUBLIC_SCORE
    assert solved.done is True


def test_medium_task_cancelling_relocatable_blocker_loses_full_credit():
    env = CalendarSchedulingEnvironment()
    reset_result = env.reset("task_medium")
    request = TASKS["task_medium"].requested_meetings[0]
    blocker_id = reset_result.observation.events[0].event_id

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

    assert solved.observation.score < MAX_PUBLIC_SCORE
    assert solved.done is False


def test_hard_task_grants_partial_credit_for_one_correct_meeting():
    env = CalendarSchedulingEnvironment()
    reset_result = env.reset("task_hard")
    first_request = TASKS["task_hard"].requested_meetings[0]
    blocker = next(
        event for event in reset_result.observation.events if event.start_time == first_request.start_time
    )
    relocation = blocker.relocation_candidates[0]

    env.step(
        reset_result.episode_id,
        CalendarAction(
            action_type="reschedule_event",
            event_id=blocker.event_id,
            new_start_time=relocation.start_time,
            duration_hours=relocation.duration_hours,
        ),
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


def test_task_catalog_exposes_richer_metadata():
    env = CalendarSchedulingEnvironment()
    tasks = {task.task_id: task for task in env.list_tasks()}

    assert "task_exec_dense_day" in tasks
    assert "task_recruiting_loop" in tasks
    assert tasks["task_medium"].supports_reschedule is True
    assert tasks["task_exec_dense_day"].request_count == 3


def test_state_exposes_history_and_action_metadata():
    env = CalendarSchedulingEnvironment()
    reset_result = env.reset("task_medium")
    blocker = reset_result.observation.events[0]
    relocation = blocker.relocation_candidates[0]

    env.step(
        reset_result.episode_id,
        CalendarAction(
            action_type="reschedule_event",
            event_id=blocker.event_id,
            new_start_time=relocation.start_time,
            duration_hours=relocation.duration_hours,
        ),
    )
    state = env.state(reset_result.episode_id)

    assert blocker.event_id in state.movable_event_ids
    assert state.scheduler_notes
    assert any("Rescheduled" in entry for entry in state.history)


def test_grader_scores_stay_strictly_inside_open_interval():
    env = CalendarSchedulingEnvironment()
    request = TASKS["task_easy"].requested_meetings[0]

    empty_grade = env.grade_explicit("task_easy", [])
    solved_grade = env.grade_explicit(
        "task_easy",
        [
            CalendarEvent(
                event_id=1,
                title=request.title,
                start_time=request.start_time,
                end_time=request.end_time,
                participants=list(request.participants),
            )
        ],
    )

    assert empty_grade.score == MIN_PUBLIC_SCORE
    assert solved_grade.score == MAX_PUBLIC_SCORE
    assert 0.0 < empty_grade.score < 1.0
    assert 0.0 < solved_grade.score < 1.0
    assert all(0.0 < score < 1.0 for score in solved_grade.details["request_scores"].values())


def _iter_score_named_numeric_values(payload, path="$"):
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            child_path = f"{path}.{key}"
            if "score" in str(key).casefold():
                if isinstance(value, (int, float)):
                    yield child_path, float(value)
                elif isinstance(value, Mapping):
                    for nested_key, nested_value in value.items():
                        if isinstance(nested_value, (int, float)):
                            yield f"{child_path}.{nested_key}", float(nested_value)
            yield from _iter_score_named_numeric_values(value, child_path)
    elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        for index, item in enumerate(payload):
            yield from _iter_score_named_numeric_values(item, f"{path}[{index}]")


def test_all_score_named_json_fields_stay_inside_open_interval():
    env = CalendarSchedulingEnvironment()

    for task_id in TASKS:
        reset_result = env.reset(task_id)
        episode_id = reset_result.episode_id

        payloads = [reset_result.model_dump(mode="json")]

        for request in TASKS[task_id].requested_meetings:
            blockers = [
                event
                for event in env.state(episode_id).events
                if event.start_time == request.start_time and event.end_time == request.end_time
            ]
            for blocker in blockers:
                payloads.append(
                    env.step(
                        episode_id,
                        CalendarAction(action_type="cancel_event", event_id=blocker.event_id),
                    ).model_dump(mode="json")
                )

            payloads.append(
                env.step(
                    episode_id,
                    CalendarAction(
                        action_type="schedule_event",
                        title=request.title,
                        start_time=request.start_time,
                        duration_hours=request.duration_hours,
                        participants=list(request.participants),
                    ),
                ).model_dump(mode="json")
            )

        payloads.append(env.state(episode_id).model_dump(mode="json"))
        payloads.append(env.grade_episode(episode_id).model_dump(mode="json"))

        for payload in payloads:
            for score_path, score_value in _iter_score_named_numeric_values(payload):
                assert 0.0 < score_value < 1.0, f"{task_id} produced out-of-range score field {score_path}={score_value}"
