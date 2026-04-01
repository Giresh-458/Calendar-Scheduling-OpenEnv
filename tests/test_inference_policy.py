from __future__ import annotations

from inference import plan_next_action
from server.environment import CalendarSchedulingEnvironment
from task_definitions import TASKS


def solve_with_policy(task_id: str) -> float:
    env = CalendarSchedulingEnvironment()
    result = env.reset(task_id)
    episode_id = result.episode_id

    for _ in range(TASKS[task_id].max_steps):
        if result.done:
            break
        action = plan_next_action(result.observation)
        result = env.step(episode_id, action)

    grade = env.grade_episode(episode_id)
    return grade.score


def test_policy_baseline_solves_all_tasks():
    for task_id in TASKS:
        assert solve_with_policy(task_id) == 1.0
