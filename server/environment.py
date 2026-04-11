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
    CandidateSlot,
    GraderResponse,
    MeetingRequest,
    StepResult,
    TaskSummary,
)
from task_definitions import CandidateSlotTemplate, MeetingTemplate, TASKS, TaskDefinition


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
        progress_delta=0.0,
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
    history: List[str] = field(default_factory=list)


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
                scenario_type=task.scenario_type,
                request_count=len(task.requested_meetings),
                supports_reschedule=any(template.relocation_candidates for template in task.initial_events),
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
            history=[f"Loaded task '{resolved_task.name}'."],
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
        elif action.action_type == "reschedule_event":
            feedback, invalid_action_penalty = self._handle_reschedule(episode, action)

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
            progress_delta=score_delta,
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
            episode.history.append(feedback)

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
            protected_event_ids=self._protected_event_ids(episode.events),
            movable_event_ids=self._movable_event_ids(episode.events),
            scheduler_notes=list(episode.task.scheduler_notes),
            history=list(episode.history),
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
            protected_event_ids=self._protected_event_ids(episode.events),
            movable_event_ids=self._movable_event_ids(episode.events),
            scheduler_notes=list(episode.task.scheduler_notes),
            recent_history=list(episode.history[-5:]),
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
            movable=True,
            protected=False,
        )
        episode.events.append(new_event)
        episode.next_event_id += 1
        return (f"Scheduled '{new_event.title}' at {new_event.start_time.isoformat()}.", 0.0)

    def _handle_cancel(self, episode: Episode, action: CalendarAction) -> Tuple[str, float]:
        target = self._find_event(episode.events, action.event_id)
        if target is None:
            return ("Cancel rejected because the event_id does not exist.", -0.5)
        if target.protected:
            return ("Cancel rejected because the event is protected.", -0.75)

        episode.events = [event for event in episode.events if event.event_id != target.event_id]
        return (f"Cancelled '{target.title}'.", 0.0)

    def _handle_reschedule(self, episode: Episode, action: CalendarAction) -> Tuple[str, float]:
        target = self._find_event(episode.events, action.event_id)
        if target is None:
            return ("Reschedule rejected because the event_id does not exist.", -0.5)
        if target.protected:
            return ("Reschedule rejected because the event is protected.", -0.75)
        if not target.movable:
            return ("Reschedule rejected because the event is locked.", -0.5)

        new_start_time = ensure_utc(action.new_start_time)
        duration_hours = action.duration_hours or self._event_duration_hours(target)
        new_end_time = new_start_time + timedelta(hours=float(duration_hours))

        if target.relocation_candidates and not any(
            self._event_matches_slot(target, slot.start_time, slot.duration_hours, new_start_time, new_end_time)
            for slot in target.relocation_candidates
        ):
            return (
                "Reschedule rejected because the new slot is not one of the approved relocation candidates.",
                -0.4,
            )

        conflicts = [
            event
            for event in episode.events
            if event.event_id != target.event_id
            and overlaps(new_start_time, new_end_time, event.start_time, event.end_time)
        ]
        if conflicts:
            return (
                "Reschedule rejected because the new time overlaps another event.",
                -0.5,
            )

        target.start_time = new_start_time
        target.end_time = new_end_time
        return (
            f"Rescheduled '{target.title}' to {target.start_time.isoformat()}.",
            0.0,
        )

    def _find_event(self, events: Iterable[CalendarEvent], event_id: Optional[int]) -> Optional[CalendarEvent]:
        for event in events:
            if event.event_id == event_id:
                return event
        return None

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
        raw_request_scores: Dict[str, float] = {}
        request_scores: Dict[str, float] = {}
        request_slot_quality: Dict[str, str] = {}

        for request in task.requested_meetings:
            best_raw_score = 0.0
            best_quality = "missing"
            for event in events:
                similarity, quality = self._meeting_similarity(request, event)
                if similarity > best_raw_score:
                    best_raw_score = similarity
                    best_quality = quality
            raw_request_scores[request.request_id] = best_raw_score
            request_scores[request.request_id] = normalize_public_score(best_raw_score)
            request_slot_quality[request.request_id] = best_quality

        raw_score = round(sum(raw_request_scores.values()) / len(task.requested_meetings), 4)
        conflicts_present = self._has_any_overlap(events)

        required_preserved_events = {
            template.request_id: any(self._template_matches_event(template, event) for event in events)
            for template in task.initial_events
            if template.request_id in task.required_preserved_event_ids
        }
        preferred_preserved_events = {
            template.request_id: any(self._template_preserved_or_relocated(template, event) for event in events)
            for template in task.initial_events
            if template.request_id in task.preferred_preserved_event_ids
        }

        missing_required = sum(not is_preserved for is_preserved in required_preserved_events.values())
        missing_preferred = sum(not is_preserved for is_preserved in preferred_preserved_events.values())

        if missing_required:
            raw_score = max(0.0, round(raw_score - (0.25 * missing_required), 4))

        if missing_preferred:
            raw_score = max(0.0, round(raw_score - (0.08 * missing_preferred), 4))

        if conflicts_present:
            raw_score = max(0.0, round(raw_score - 0.25, 4))

        all_requests_preferred = all(quality == "preferred" for quality in request_slot_quality.values())
        solved = (
            all_requests_preferred
            and missing_required == 0
            and missing_preferred == 0
            and not conflicts_present
        )
        if solved:
            raw_score = SOLVED_RAW_SCORE

        score = normalize_public_score(raw_score)

        details = {
            "task_name": task.name,
            "request_scores": request_scores,
            "request_slot_quality": request_slot_quality,
            "conflicts_present": conflicts_present,
            "event_count": len(events),
            "preserved_initial_events": required_preserved_events,
            "preferred_preserved_events": preferred_preserved_events,
            "missing_required_preservations": missing_required,
            "missing_preferred_preservations": missing_preferred,
        }
        return score, solved, details

    def _meeting_similarity(self, request: MeetingTemplate, event: CalendarEvent) -> Tuple[float, str]:
        title_match = request.title.strip().casefold() == event.title.strip().casefold()
        participants_match = set(request.participants).issubset(set(event.participants))
        duration_match = (request.end_time - request.start_time) == (event.end_time - event.start_time)
        same_day = request.start_time.date() == event.start_time.date()

        if title_match and participants_match and self._event_matches_slot(
            event,
            request.start_time,
            request.duration_hours,
        ):
            return 1.0, "preferred"

        if title_match and participants_match:
            for slot in request.alternate_slots:
                if self._event_matches_slot(event, slot.start_time, slot.duration_hours):
                    return 0.92, "alternate"

        if title_match and self._event_matches_slot(event, request.start_time, request.duration_hours):
            return 0.85, "preferred_title_only"

        if same_day and duration_match and (title_match or participants_match):
            return 0.5, "partial"

        if same_day and (title_match or participants_match):
            return 0.25, "weak"

        return 0.0, "missing"

    def _template_matches_event(self, template: MeetingTemplate, event: CalendarEvent) -> bool:
        return (
            template.title.strip().casefold() == event.title.strip().casefold()
            and self._participants_match(template.participants, event.participants)
            and self._event_matches_slot(event, template.start_time, template.duration_hours)
        )

    def _template_preserved_or_relocated(self, template: MeetingTemplate, event: CalendarEvent) -> bool:
        if template.title.strip().casefold() != event.title.strip().casefold():
            return False
        if not self._participants_match(template.participants, event.participants):
            return False

        if self._event_matches_slot(event, template.start_time, template.duration_hours):
            return True

        return any(
            self._event_matches_slot(event, slot.start_time, slot.duration_hours)
            for slot in template.relocation_candidates
        )

    def _participants_match(self, expected: Iterable[str], actual: Iterable[str]) -> bool:
        return set(expected).issubset(set(actual))

    def _event_matches_slot(
        self,
        event: CalendarEvent,
        slot_start: datetime,
        slot_duration_hours: float,
        actual_start: Optional[datetime] = None,
        actual_end: Optional[datetime] = None,
    ) -> bool:
        resolved_start = actual_start or event.start_time
        resolved_end = actual_end or event.end_time
        expected_end = ensure_utc(slot_start) + timedelta(hours=slot_duration_hours)
        return resolved_start == ensure_utc(slot_start) and resolved_end == expected_end

    def _event_duration_hours(self, event: CalendarEvent) -> float:
        return (event.end_time - event.start_time).total_seconds() / 3600.0

    def _protected_event_ids(self, events: Iterable[CalendarEvent]) -> List[int]:
        return [event.event_id for event in events if event.protected]

    def _movable_event_ids(self, events: Iterable[CalendarEvent]) -> List[int]:
        return [event.event_id for event in events if event.movable and not event.protected]

    def _template_slot_to_model(self, slot: CandidateSlotTemplate) -> CandidateSlot:
        return CandidateSlot(
            start_time=slot.start_time,
            duration_hours=slot.duration_hours,
            label=slot.label,
            preference=slot.preference,
        )

    def _template_to_event(self, event_id: int, template: MeetingTemplate) -> CalendarEvent:
        return CalendarEvent(
            event_id=event_id,
            title=template.title,
            start_time=template.start_time,
            end_time=template.end_time,
            participants=list(template.participants),
            movable=template.movable,
            protected=template.protected,
            relocation_candidates=[self._template_slot_to_model(slot) for slot in template.relocation_candidates],
            notes=list(template.notes),
        )

    def _template_to_request(self, template: MeetingTemplate) -> MeetingRequest:
        return MeetingRequest(
            request_id=template.request_id,
            title=template.title,
            start_time=template.start_time,
            end_time=template.end_time,
            participants=list(template.participants),
            priority=template.priority,
            alternate_slots=[self._template_slot_to_model(slot) for slot in template.alternate_slots],
            notes=list(template.notes),
        )
