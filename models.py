from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

try:
    from openenv.core.env_server.types import Action, Observation, Reward, State
except ImportError:
    class Action(BaseModel):
        model_config = {"extra": "forbid"}

    class Observation(BaseModel):
        model_config = {"extra": "forbid"}

    class Reward(BaseModel):
        model_config = {"extra": "forbid"}

    class State(BaseModel):
        model_config = {"extra": "forbid"}


class CalendarEvent(BaseModel):
    event_id: int = Field(..., description="Stable identifier for an event in the current episode.")
    title: str = Field(..., description="Human-readable meeting title.")
    start_time: datetime = Field(..., description="Inclusive event start time.")
    end_time: datetime = Field(..., description="Exclusive event end time.")
    participants: List[str] = Field(default_factory=list, description="Meeting participants.")


class MeetingRequest(BaseModel):
    request_id: str = Field(..., description="Stable identifier for a requested meeting.")
    title: str = Field(..., description="Requested meeting title.")
    start_time: datetime = Field(..., description="Target start time.")
    end_time: datetime = Field(..., description="Target end time.")
    participants: List[str] = Field(default_factory=list, description="Requested participants.")


class CalendarAction(Action):
    action_type: Literal["schedule_event", "cancel_event", "noop"] = Field(
        ..., description="The action variant to execute."
    )
    title: Optional[str] = Field(None, description="Meeting title for schedule_event.")
    start_time: Optional[datetime] = Field(None, description="Start time for schedule_event.")
    duration_hours: Optional[float] = Field(None, description="Duration in hours for schedule_event.")
    participants: List[str] = Field(default_factory=list, description="Meeting participants.")
    event_id: Optional[int] = Field(None, description="Event identifier for cancel_event.")

    @model_validator(mode="after")
    def validate_for_action_type(self) -> "CalendarAction":
        if self.action_type == "schedule_event":
            missing = [
                field_name
                for field_name, value in {
                    "title": self.title,
                    "start_time": self.start_time,
                    "duration_hours": self.duration_hours,
                }.items()
                if value is None
            ]
            if missing:
                raise ValueError(f"schedule_event requires: {', '.join(missing)}")
            if self.duration_hours is not None and self.duration_hours <= 0:
                raise ValueError("duration_hours must be greater than 0")
        if self.action_type == "cancel_event" and self.event_id is None:
            raise ValueError("cancel_event requires event_id")
        return self


class CalendarReward(Reward):
    total: float = Field(..., description="Final scalar reward returned by the transition.")
    score_delta: float = Field(..., description="Improvement in the normalized grader score.")
    step_penalty: float = Field(..., description="Small cost applied to every action.")
    invalid_action_penalty: float = Field(
        ...,
        description="Penalty applied when the action is invalid or creates a conflict.",
    )
    destructive_action_penalty: float = Field(
        ...,
        description="Penalty for canceling an event instead of preserving the calendar.",
    )
    completion_bonus: float = Field(
        ...,
        description="Bonus applied when the task reaches a perfect score.",
    )


class CalendarObservation(Observation):
    task_id: str = Field(..., description="Current task identifier.")
    task_name: str = Field(..., description="Short task name.")
    task_description: str = Field(..., description="Human-readable task objective.")
    requested_meetings: List[MeetingRequest] = Field(
        default_factory=list,
        description="Meetings that must exist in the final schedule.",
    )
    current_time: datetime = Field(..., description="Current time reference for the agent.")
    events: List[CalendarEvent] = Field(default_factory=list, description="Current calendar state.")
    step: int = Field(..., description="Current step count.")
    max_steps: int = Field(..., description="Maximum allowed steps for the episode.")
    done: bool = Field(..., description="Whether the episode has terminated.")
    feedback: str = Field(..., description="Short feedback message from the previous action.")
    last_action_error: Optional[str] = Field(
        None,
        description="Raw error text from the last invalid or rejected action, if any.",
    )
    score: float = Field(
        ...,
        gt=0.0,
        lt=1.0,
        description="Current deterministic graded score, normalized to the open interval (0, 1).",
    )
    last_reward: float = Field(..., description="Reward returned by the previous transition.")
    reward_breakdown: CalendarReward = Field(
        ...,
        description="Structured components that produced the previous scalar reward.",
    )
    available_actions: List[str] = Field(
        default_factory=lambda: ["schedule_event", "cancel_event", "noop"],
        description="Action types supported by the environment.",
    )


class CalendarState(State):
    episode_id: str = Field(..., description="Episode identifier.")
    task_id: str = Field(..., description="Active task identifier.")
    step_count: int = Field(..., description="Steps taken so far.")
    max_steps: int = Field(..., description="Maximum steps for the episode.")
    cumulative_reward: float = Field(..., description="Total accumulated reward.")
    last_score: float = Field(
        ...,
        gt=0.0,
        lt=1.0,
        description="Most recent deterministic score, normalized to the open interval (0, 1).",
    )
    last_reward: float = Field(..., description="Most recent scalar reward.")
    last_action_error: Optional[str] = Field(
        None,
        description="Raw error text from the last invalid or rejected action, if any.",
    )
    last_reward_breakdown: CalendarReward = Field(
        ...,
        description="Structured components that produced the previous scalar reward.",
    )
    done: bool = Field(..., description="Whether the episode has ended.")
    created_at: datetime = Field(..., description="Episode creation time.")
    current_time: datetime = Field(..., description="Current environment time.")
    events: List[CalendarEvent] = Field(default_factory=list, description="Current calendar state.")


class TaskSummary(BaseModel):
    task_id: str
    name: str
    difficulty: Literal["easy", "medium", "hard"]
    description: str
    max_steps: int


class ResetRequest(BaseModel):
    task_id: Optional[str] = Field(None, description="Task identifier to reset into.")


class StepRequest(BaseModel):
    episode_id: str = Field(..., description="Episode to advance.")
    action: CalendarAction = Field(..., description="Typed action payload.")


class StepResult(BaseModel):
    episode_id: str
    observation: CalendarObservation
    reward: float
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


class GraderRequest(BaseModel):
    episode_id: Optional[str] = None
    task_id: Optional[str] = None
    events: List[CalendarEvent] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_payload(self) -> "GraderRequest":
        if self.episode_id is None and self.task_id is None:
            raise ValueError("Provide either episode_id or task_id")
        if self.episode_id is None and not self.events:
            raise ValueError("When grading explicit events, provide a non-empty events list")
        return self


class GraderResponse(BaseModel):
    task_id: str
    score: float = Field(..., gt=0.0, lt=1.0)
    passed: bool
    details: Dict[str, Any] = Field(default_factory=dict)
