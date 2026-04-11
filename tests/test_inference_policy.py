from __future__ import annotations

from inference import plan_next_action
from server.environment import CalendarSchedulingEnvironment, MAX_PUBLIC_SCORE
from task_definitions import TASKS


def solve_with_policy(task_id: str) -> tuple[float, int]:
    env = CalendarSchedulingEnvironment()
    result = env.reset(task_id)
    episode_id = result.episode_id
    steps_taken = 0

    for step_number in range(1, TASKS[task_id].max_steps + 1):
        if result.done:
            break
        action = plan_next_action(result.observation)
        result = env.step(episode_id, action)
        steps_taken = step_number

    grade = env.grade_episode(episode_id)
    return grade.score, steps_taken


def test_policy_baseline_solves_all_tasks():
    for task_id in TASKS:
        score, _ = solve_with_policy(task_id)
        assert score == MAX_PUBLIC_SCORE


def test_complex_exec_day_requires_more_policy_steps_than_medium():
    _, medium_steps = solve_with_policy("task_medium")
    _, exec_steps = solve_with_policy("task_exec_dense_day")

    assert medium_steps == 2
    assert exec_steps == 6


def test_hard_task_requires_more_policy_steps_than_medium():
    _, medium_steps = solve_with_policy("task_medium")
    _, hard_steps = solve_with_policy("task_hard")

    assert medium_steps == 2
    assert hard_steps == 4
