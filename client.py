from __future__ import annotations

from typing import List, Optional

import requests

from models import (
    CalendarAction,
    CalendarState,
    GraderResponse,
    ResetRequest,
    StepRequest,
    StepResult,
    TaskSummary,
)
from server.environment import CalendarSchedulingEnvironment


class CalendarSchedulingEnvClient:
    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 10.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def health(self) -> dict:
        response = self.session.get(f"{self.base_url}/health", timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def tasks(self) -> List[TaskSummary]:
        response = self.session.get(f"{self.base_url}/tasks", timeout=self.timeout)
        response.raise_for_status()
        return [TaskSummary.model_validate(item) for item in response.json()]

    def reset(self, task_id: Optional[str] = None) -> StepResult:
        payload = ResetRequest(task_id=task_id).model_dump(exclude_none=True)
        response = self.session.post(f"{self.base_url}/reset", json=payload, timeout=self.timeout)
        response.raise_for_status()
        return StepResult.model_validate(response.json())

    def step(self, episode_id: str, action: CalendarAction) -> StepResult:
        payload = StepRequest(episode_id=episode_id, action=action).model_dump(mode="json")
        response = self.session.post(f"{self.base_url}/step", json=payload, timeout=self.timeout)
        response.raise_for_status()
        return StepResult.model_validate(response.json())

    def state(self, episode_id: str) -> CalendarState:
        response = self.session.get(
            f"{self.base_url}/state",
            params={"episode_id": episode_id},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return CalendarState.model_validate(response.json())

    def grade(self, episode_id: str) -> GraderResponse:
        response = self.session.post(
            f"{self.base_url}/grader",
            json={"episode_id": episode_id},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return GraderResponse.model_validate(response.json())

    def close(self) -> None:
        self.session.close()


class EmbeddedCalendarSchedulingEnvClient:
    """In-process client used by the baseline script for reproducible local runs."""

    def __init__(self) -> None:
        self.env = CalendarSchedulingEnvironment()

    def health(self) -> dict:
        return {"status": "healthy", "service": "calendar-scheduling-env", "mode": "embedded"}

    def tasks(self) -> List[TaskSummary]:
        return self.env.list_tasks()

    def reset(self, task_id: Optional[str] = None) -> StepResult:
        return self.env.reset(task_id=task_id)

    def step(self, episode_id: str, action: CalendarAction) -> StepResult:
        return self.env.step(episode_id, action)

    def state(self, episode_id: str) -> CalendarState:
        return self.env.state(episode_id)

    def grade(self, episode_id: str) -> GraderResponse:
        return self.env.grade_episode(episode_id)

    def close(self) -> None:
        return None
