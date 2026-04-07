from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from itertools import combinations
from typing import Dict, Iterable, List, Optional, Tuple
from uuid import uuid4

from models import (
    CalendarAction,
    CalendarEvent,
    CalendarObservation,
    CalendarReward,
    CalendarState,
    GraderResponse,
    MeetingRequest,
    StepResult,
    TaskSummary,
)
from task_definitions import MeetingTemplate, TASKS, TaskDefinition


MIN_PUBLIC_SCORE = 0.001
MAX_PUBLIC_SCORE = 0.999
SOLVED_RAW_SCORE = 1.0


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


def zero_reward_breakdown() -> CalendarReward:
    return CalendarReward(
        total=0.0,
        score_delta=0.0,
        step_penalty=0.0,
        invalid_action_penalty=0.0,
        destructive_action_penalty=0.0,
        completion_bonus=0.0,
    )


def normalize_public_score(value: float) -> float:
    bounded = round(max(0.0, min(SOLVED_RAW_SCORE, value)), 4)
    if bounded <= MIN_PUBLIC_SCORE:
        return MIN_PUBLIC_SCORE
    if bounded >= MAX_PUBLIC_SCORE:
        return MAX_PUBLIC_SCORE
    return bounded


def is_solved_raw_score(value: float) -> bool:
    return value >= SOLVED_RAW_SCORE


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
    last_action_error: Optional[str] = None
    last_reward_breakdown: CalendarReward = field(default_factory=zero_reward_breakdown)


class CalendarSchedulingEnvironment:
    def __init__(self) -> None:
        self._episodes: Dict[str, Episode] = {}
        self._latest_episode_id: Optional[str] = None

    def close(self) -> None:
        return None

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
        initial_score, _, _ = self._grade_events(resolved_task, events)
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
            info={
                "score": initial_score,
                "reward_breakdown": episode.last_reward_breakdown.model_dump(mode="json"),
            },
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
                info={
                    "score": episode.last_score,
                    "message": "Episode already complete.",
                    "reward_breakdown": episode.last_reward_breakdown.model_dump(mode="json"),
                },
            )

        reward_delta = -0.01
        invalid_action_penalty = 0.0
        destructive_action_penalty = 0.0
        completion_bonus = 0.0
        feedback = ""
        action_error = None

        if action.action_type == "noop":
            feedback = "No action taken."
        elif action.action_type == "schedule_event":
            feedback, invalid_action_penalty = self._handle_schedule(episode, action)
        elif action.action_type == "cancel_event":
            feedback, invalid_action_penalty = self._handle_cancel(episode, action)
            if invalid_action_penalty == 0.0:
                destructive_action_penalty = -0.05

        if invalid_action_penalty < 0.0:
            action_error = feedback

        episode.step_count += 1
        previous_score = episode.last_score
        score, solved, details = self._grade_events(episode.task, episode.events)
        score_delta = round(score - previous_score, 4)
        reward = score_delta + reward_delta + invalid_action_penalty + destructive_action_penalty

        if solved:
            completion_bonus = 0.2
            reward += completion_bonus
            episode.done = True
        elif episode.step_count >= episode.max_steps:
            episode.done = True

        reward = round(reward, 4)
        reward_breakdown = CalendarReward(
            total=reward,
            score_delta=score_delta,
            step_penalty=reward_delta,
            invalid_action_penalty=invalid_action_penalty,
            destructive_action_penalty=destructive_action_penalty,
            completion_bonus=completion_bonus,
        )
        episode.cumulative_reward += reward
        episode.last_score = score
        episode.last_reward = reward
        episode.last_action_error = action_error
        episode.last_reward_breakdown = reward_breakdown
        if action_error is None:
            if score_delta > 0:
                feedback = f"{feedback} Score improved to {score:.3f}."
            else:
                feedback = f"{feedback} Current score is {score:.3f}."
        episode.last_feedback = feedback

        observation = self._build_observation(episode)
        details["reward_breakdown"] = reward_breakdown.model_dump(mode="json")
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
            last_reward=episode.last_reward,
            last_action_error=episode.last_action_error,
            last_reward_breakdown=episode.last_reward_breakdown,
            done=episode.done,
            created_at=episode.created_at,
            current_time=self._current_time_for_step(episode),
            events=deepcopy(episode.events),
        )

    def grade_episode(self, episode_id: str) -> GraderResponse:
        episode = self._episodes[episode_id]
        score, solved, details = self._grade_events(episode.task, episode.events)
        return GraderResponse(
            task_id=episode.task.task_id,
            score=score,
            passed=solved,
            details=details,
        )

    def grade_explicit(self, task_id: str, events: List[CalendarEvent]) -> GraderResponse:
        task = TASKS[task_id]
        score, solved, details = self._grade_events(task, events)
        return GraderResponse(task_id=task_id, score=score, passed=solved, details=details)

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
            last_action_error=episode.last_action_error,
            score=episode.last_score,
            last_reward=episode.last_reward,
            reward_breakdown=episode.last_reward_breakdown,
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
    ) -> Tuple[float, bool, Dict[str, object]]:
        raw_request_scores = {}
        request_scores = {}
        request_exact_matches = {}

        for request in task.requested_meetings:
            best_raw_score = 0.0
            exact_match = False
            for event in events:
                similarity = self._meeting_similarity(request, event)
                best_raw_score = max(best_raw_score, similarity)
                if similarity >= 1.0:
                    exact_match = True
            raw_request_scores[request.request_id] = best_raw_score
            request_scores[request.request_id] = normalize_public_score(best_raw_score)
            request_exact_matches[request.request_id] = exact_match

        raw_score = round(sum(raw_request_scores.values()) / len(task.requested_meetings), 4)
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
                raw_score = SOLVED_RAW_SCORE
            elif request_exact_matches[target.request_id]:
                raw_score = 0.5

        preserved_initial_events = {
            template.request_id: any(
                self._template_matches_event(template, event) for event in events
            )
            for template in task.initial_events
            if template.request_id in task.required_preserved_event_ids
        }

        if (
            task.task_id == "task_hard"
            and is_solved_raw_score(raw_score)
            and not all(preserved_initial_events.values())
        ):
            raw_score = 0.75

        if conflicts_present:
            raw_score = max(0.0, round(raw_score - 0.25, 4))

        solved = is_solved_raw_score(raw_score)
        score = normalize_public_score(raw_score)

        details = {
            "task_name": task.name,
            "request_scores": request_scores,
            "conflicts_present": conflicts_present,
            "event_count": len(events),
            "preserved_initial_events": preserved_initial_events,
        }
        return score, solved, details

    def _meeting_similarity(self, request: MeetingTemplate, event: CalendarEvent) -> float:
        same_day = request.start_time.date() == event.start_time.date()
        same_start = request.start_time == event.start_time
        same_end = request.end_time == event.end_time
        same_duration = request.end_time - request.start_time == event.end_time - event.start_time
        title_match = request.title.strip().casefold() == event.title.strip().casefold()
        participants_match = set(request.participants).issubset(set(event.participants))

        if same_start and same_end and title_match and participants_match:
            return 1.0
        if same_start and same_end and title_match:
            return 0.85
        if same_day and same_duration and (title_match or participants_match):
            return 0.5
        if same_day and (title_match or participants_match):
            return 0.25
        return 0.0

    def _template_matches_event(self, template: MeetingTemplate, event: CalendarEvent) -> bool:
        return (
            template.title.strip().casefold() == event.title.strip().casefold()
            and template.start_time == event.start_time
            and template.end_time == event.end_time
            and set(template.participants).issubset(set(event.participants))
        )

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
