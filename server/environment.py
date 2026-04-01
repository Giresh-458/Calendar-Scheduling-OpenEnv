from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from itertools import combinations
from typing import Dict, Iterable, List, Optional, Tuple
from uuid import uuid4

from models import (
    CalendarAction,
    CalendarEvent,
    CalendarObservation,
    CalendarState,
    GraderResponse,
    MeetingRequest,
    StepResult,
    TaskSummary,
)
from task_definitions import MeetingTemplate, TASKS, TaskDefinition


def ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def overlaps(
    first_start: datetime,
    first_end: datetime,
    second_start: datetime,
    second_end: datetime,
) -> bool:
    return first_start < second_end and second_start < first_end


@dataclass
class Episode:
    episode_id: str
    task: TaskDefinition
    events: List[CalendarEvent]
    created_at: datetime
    step_count: int = 0
    max_steps: int = 0
    cumulative_reward: float = 0.0
    last_score: float = 0.0
    done: bool = False
    next_event_id: int = 1
    last_feedback: str = "Episode initialized."
    last_reward: float = 0.0


class CalendarSchedulingEnvironment:
    def __init__(self) -> None:
        self._episodes: Dict[str, Episode] = {}
        self._latest_episode_id: Optional[str] = None

    def list_tasks(self) -> List[TaskSummary]:
        return [
            TaskSummary(
                task_id=task.task_id,
                name=task.name,
                difficulty=task.difficulty,
                description=task.description,
                max_steps=task.max_steps,
            )
            for task in TASKS.values()
        ]

    def reset(self, task_id: Optional[str] = None) -> StepResult:
        resolved_task = TASKS[task_id or "task_easy"]
        events = [
            self._template_to_event(index + 1, template)
            for index, template in enumerate(resolved_task.initial_events)
        ]
        episode_id = str(uuid4())
        initial_score, _ = self._grade_events(resolved_task, events)
        episode = Episode(
            episode_id=episode_id,
            task=resolved_task,
            events=events,
            created_at=datetime.now(timezone.utc),
            max_steps=resolved_task.max_steps,
            last_score=initial_score,
            next_event_id=len(events) + 1,
            last_feedback=f"Task '{resolved_task.name}' loaded.",
        )
        self._episodes[episode_id] = episode
        self._latest_episode_id = episode_id
        observation = self._build_observation(episode)
        return StepResult(
            episode_id=episode_id,
            observation=observation,
            reward=0.0,
            done=False,
            info={"score": initial_score},
        )

    def step(self, episode_id: str, action: CalendarAction) -> StepResult:
        episode = self._episodes[episode_id]
        if episode.done:
            observation = self._build_observation(episode)
            return StepResult(
                episode_id=episode_id,
                observation=observation,
                reward=0.0,
                done=True,
                info={"score": episode.last_score, "message": "Episode already complete."},
            )

        reward_delta = -0.01
        extra_penalty = 0.0
        feedback = ""

        if action.action_type == "noop":
            feedback = "No action taken."
        elif action.action_type == "schedule_event":
            feedback, extra_penalty = self._handle_schedule(episode, action)
        elif action.action_type == "cancel_event":
            feedback, extra_penalty = self._handle_cancel(episode, action)

        episode.step_count += 1
        score, details = self._grade_events(episode.task, episode.events)
        reward = (score - episode.last_score) + reward_delta + extra_penalty

        if score >= 1.0:
            reward += 0.2
            episode.done = True
        elif episode.step_count >= episode.max_steps:
            episode.done = True

        reward = round(max(-1.0, min(1.0, reward)), 4)
        episode.cumulative_reward += reward
        episode.last_score = score
        episode.last_reward = reward
        episode.last_feedback = feedback

        observation = self._build_observation(episode)
        return StepResult(
            episode_id=episode_id,
            observation=observation,
            reward=reward,
            done=episode.done,
            info=details,
        )

    def state(self, episode_id: Optional[str] = None) -> CalendarState:
        resolved_episode_id = episode_id or self._latest_episode_id
        if resolved_episode_id is None:
            raise KeyError("No episodes have been created yet.")
        episode = self._episodes[resolved_episode_id]
        return CalendarState(
            episode_id=episode.episode_id,
            task_id=episode.task.task_id,
            step_count=episode.step_count,
            max_steps=episode.max_steps,
            cumulative_reward=round(episode.cumulative_reward, 4),
            last_score=episode.last_score,
            done=episode.done,
            created_at=episode.created_at,
            current_time=self._current_time_for_step(episode),
            events=deepcopy(episode.events),
        )

    def grade_episode(self, episode_id: str) -> GraderResponse:
        episode = self._episodes[episode_id]
        score, details = self._grade_events(episode.task, episode.events)
        return GraderResponse(
            task_id=episode.task.task_id,
            score=score,
            passed=score >= 1.0,
            details=details,
        )

    def grade_explicit(self, task_id: str, events: List[CalendarEvent]) -> GraderResponse:
        task = TASKS[task_id]
        score, details = self._grade_events(task, events)
        return GraderResponse(task_id=task_id, score=score, passed=score >= 1.0, details=details)

    def _build_observation(self, episode: Episode) -> CalendarObservation:
        return CalendarObservation(
            task_id=episode.task.task_id,
            task_name=episode.task.name,
            task_description=episode.task.description,
            requested_meetings=[self._template_to_request(item) for item in episode.task.requested_meetings],
            current_time=self._current_time_for_step(episode),
            events=deepcopy(episode.events),
            step=episode.step_count,
            max_steps=episode.max_steps,
            done=episode.done,
            feedback=episode.last_feedback,
            score=episode.last_score,
            last_reward=episode.last_reward,
        )

    def _current_time_for_step(self, episode: Episode) -> datetime:
        return episode.task.current_time + timedelta(minutes=15 * episode.step_count)

    def _handle_schedule(self, episode: Episode, action: CalendarAction) -> Tuple[str, float]:
        start_time = ensure_utc(action.start_time)
        end_time = start_time + timedelta(hours=float(action.duration_hours))
        conflicts = self._find_conflicts(episode.events, start_time, end_time)
        if conflicts:
            return (
                "Schedule rejected because the requested time overlaps an existing event.",
                -0.5,
            )

        new_event = CalendarEvent(
            event_id=episode.next_event_id,
            title=action.title or "Untitled Event",
            start_time=start_time,
            end_time=end_time,
            participants=list(action.participants),
        )
        episode.events.append(new_event)
        episode.next_event_id += 1
        return (f"Scheduled '{new_event.title}' at {new_event.start_time.isoformat()}.", 0.0)

    def _handle_cancel(self, episode: Episode, action: CalendarAction) -> Tuple[str, float]:
        for index, event in enumerate(episode.events):
            if event.event_id == action.event_id:
                removed = episode.events.pop(index)
                return (f"Cancelled '{removed.title}'.", 0.0)
        return ("Cancel rejected because the event_id does not exist.", -0.5)

    def _find_conflicts(
        self,
        events: Iterable[CalendarEvent],
        start_time: datetime,
        end_time: datetime,
    ) -> List[CalendarEvent]:
        return [
            event
            for event in events
            if overlaps(start_time, end_time, event.start_time, event.end_time)
        ]

    def _has_any_overlap(self, events: Iterable[CalendarEvent]) -> bool:
        event_list = list(events)
        return any(
            overlaps(left.start_time, left.end_time, right.start_time, right.end_time)
            for left, right in combinations(event_list, 2)
        )

    def _grade_events(
        self,
        task: TaskDefinition,
        events: List[CalendarEvent],
    ) -> Tuple[float, Dict[str, object]]:
        request_scores = {}
        request_exact_matches = {}

        for request in task.requested_meetings:
            best_score = 0.0
            exact_match = False
            for event in events:
                similarity = self._meeting_similarity(request, event)
                best_score = max(best_score, similarity)
                if similarity >= 1.0:
                    exact_match = True
            request_scores[request.request_id] = best_score
            request_exact_matches[request.request_id] = exact_match

        score = round(sum(request_scores.values()) / len(task.requested_meetings), 4)
        conflicts_present = self._has_any_overlap(events)

        if task.task_id == "task_medium":
            target = task.requested_meetings[0]
            target_blockers = [
                event
                for event in events
                if event.start_time == target.start_time
                and event.end_time == target.end_time
                and self._meeting_similarity(target, event) < 1.0
            ]
            if request_exact_matches[target.request_id] and not target_blockers:
                score = 1.0
            elif request_exact_matches[target.request_id]:
                score = 0.5

        if conflicts_present:
            score = max(0.0, round(score - 0.25, 4))

        details = {
            "task_name": task.name,
            "request_scores": request_scores,
            "conflicts_present": conflicts_present,
            "event_count": len(events),
        }
        return score, details

    def _meeting_similarity(self, request: MeetingTemplate, event: CalendarEvent) -> float:
        same_day = request.start_time.date() == event.start_time.date()
        same_start = request.start_time == event.start_time
        same_end = request.end_time == event.end_time
        same_duration = request.end_time - request.start_time == event.end_time - event.start_time
        title_match = request.title.strip().casefold() == event.title.strip().casefold()
        participants_match = set(request.participants).issubset(set(event.participants))

        if same_start and same_end and participants_match:
            return 1.0
        if same_start and same_end and title_match:
            return 0.85
        if same_day and same_duration and (title_match or participants_match):
            return 0.5
        if same_day and (title_match or participants_match):
            return 0.25
        return 0.0

    def _template_to_event(self, event_id: int, template: MeetingTemplate) -> CalendarEvent:
        return CalendarEvent(
            event_id=event_id,
            title=template.title,
            start_time=template.start_time,
            end_time=template.end_time,
            participants=list(template.participants),
        )

    def _template_to_request(self, template: MeetingTemplate) -> MeetingRequest:
        return MeetingRequest(
            request_id=template.request_id,
            title=template.title,
            start_time=template.start_time,
            end_time=template.end_time,
            participants=list(template.participants),
        )
