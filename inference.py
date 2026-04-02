from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from openai import OpenAI

from client import CalendarSchedulingEnvClient, EmbeddedCalendarSchedulingEnvClient
from models import CalendarAction, CalendarEvent, CalendarObservation, MeetingRequest
from task_definitions import TASKS


DEFAULT_API_BASE_URL = "https://router.huggingface.co/v1"
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-72B-Instruct"
DEFAULT_BENCHMARK_NAME = "calendar_scheduling_env"
TASK_ORDER = tuple(TASKS.keys())

SYSTEM_PROMPT = """You are controlling a calendar scheduling environment.
Return exactly one JSON object and nothing else.

Valid action schemas:
{"action_type":"schedule_event","title":"...","start_time":"ISO-8601","duration_hours":1.0,"participants":["..."]}
{"action_type":"cancel_event","event_id":1}
{"action_type":"noop"}

If the observation includes a suggested_safe_action, prefer returning it exactly when it is valid.
"""


def build_user_prompt(
    observation: CalendarObservation,
    suggested_safe_action: CalendarAction,
) -> str:
    payload = observation.model_dump(mode="json")
    payload["suggested_safe_action"] = suggested_safe_action.model_dump(
        mode="json",
        exclude_none=True,
        exclude_defaults=True,
    )
    return json.dumps(payload, indent=2)


def parse_action(raw_text: str) -> Optional[CalendarAction]:
    if not raw_text:
        return None

    try:
        payload = json.loads(raw_text)
        return CalendarAction.model_validate(payload)
    except Exception:
        pass

    first_brace = raw_text.find("{")
    last_brace = raw_text.rfind("}")
    if first_brace == -1 or last_brace == -1 or last_brace <= first_brace:
        return None

    try:
        payload = json.loads(raw_text[first_brace : last_brace + 1])
        return CalendarAction.model_validate(payload)
    except Exception:
        return None


def event_matches_request(event: CalendarEvent, request: MeetingRequest) -> bool:
    return (
        event.start_time == request.start_time
        and event.end_time == request.end_time
        and set(request.participants).issubset(set(event.participants))
    )


def has_overlap(
    left_start,
    left_end,
    right_start,
    right_end,
) -> bool:
    return left_start < right_end and right_start < left_end


def exact_duration_hours(request: MeetingRequest) -> float:
    duration_seconds = (request.end_time - request.start_time).total_seconds()
    return duration_seconds / 3600.0


def plan_next_action(observation: CalendarObservation) -> CalendarAction:
    for request in observation.requested_meetings:
        if any(event_matches_request(event, request) for event in observation.events):
            continue

        for event in observation.events:
            if event_matches_request(event, request):
                continue
            if has_overlap(
                request.start_time,
                request.end_time,
                event.start_time,
                event.end_time,
            ):
                return CalendarAction(action_type="cancel_event", event_id=event.event_id)

        return CalendarAction(
            action_type="schedule_event",
            title=request.title,
            start_time=request.start_time,
            duration_hours=exact_duration_hours(request),
            participants=list(request.participants),
        )

    return CalendarAction(action_type="noop")


def same_action(left: CalendarAction, right: CalendarAction) -> bool:
    return left.model_dump(
        mode="json",
        exclude_none=True,
        exclude_defaults=True,
    ) == right.model_dump(
        mode="json",
        exclude_none=True,
        exclude_defaults=True,
    )


def benchmark_name() -> str:
    return os.getenv("BENCHMARK_NAME", DEFAULT_BENCHMARK_NAME)


def success_score_threshold() -> float:
    return float(os.getenv("SUCCESS_SCORE_THRESHOLD", "1.0"))


def max_agent_steps() -> int:
    return int(os.getenv("MAX_AGENT_STEPS", "8"))


def build_env_client() -> Any:
    env_base_url = os.getenv("ENV_BASE_URL")
    if env_base_url:
        return CalendarSchedulingEnvClient(base_url=env_base_url)
    return EmbeddedCalendarSchedulingEnvClient()


def build_model_client() -> tuple[Optional[OpenAI], Optional[str]]:
    api_base_url = os.getenv("API_BASE_URL", DEFAULT_API_BASE_URL)
    model_name = os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME)
    api_key = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")

    if not api_key:
        return None, None

    return OpenAI(base_url=api_base_url, api_key=api_key), model_name


def choose_action(
    observation: CalendarObservation,
    model_client: Optional[OpenAI],
    model_name: Optional[str],
) -> CalendarAction:
    policy_action = plan_next_action(observation)
    if model_client is None or not model_name:
        return policy_action

    try:
        response = model_client.chat.completions.create(
            model=model_name,
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": build_user_prompt(observation, policy_action),
                },
            ],
        )
        content = response.choices[0].message.content or ""
    except Exception:
        return policy_action

    model_action = parse_action(content)
    if model_action is None:
        return policy_action

    if same_action(model_action, policy_action):
        return model_action

    if policy_action.action_type != "noop":
        return policy_action

    return model_action


def requested_task_ids() -> List[str]:
    raw_task_ids = os.getenv("TASK_IDS")
    if not raw_task_ids:
        return list(TASK_ORDER)

    task_ids = [item.strip() for item in raw_task_ids.split(",") if item.strip()]
    unknown = [task_id for task_id in task_ids if task_id not in TASKS]
    if unknown:
        raise ValueError(f"Unknown TASK_IDS entries: {', '.join(unknown)}")
    return task_ids


def normalize_log_value(value: Optional[str]) -> str:
    if not value:
        return "null"
    return " ".join(value.split())


def action_to_log_string(action: CalendarAction) -> str:
    return json.dumps(
        action.model_dump(
            mode="json",
            exclude_none=True,
            exclude_defaults=True,
        ),
        separators=(",", ":"),
    )


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int,
    action: CalendarAction,
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    print(
        "[STEP] "
        f"step={step} "
        f"action={action_to_log_string(action)} "
        f"reward={reward:.2f} "
        f"done={str(done).lower()} "
        f"error={normalize_log_value(error)}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}",
        flush=True,
    )


def run_task(
    env_client: Any,
    model_client: Optional[OpenAI],
    model_name: Optional[str],
    task_id: str,
) -> Dict[str, Any]:
    rewards: List[float] = []
    steps_taken = 0
    success = False
    score = 0.0
    exception: Optional[BaseException] = None
    episode_id: Optional[str] = None

    log_start(
        task=task_id,
        env=benchmark_name(),
        model=model_name or "deterministic-policy",
    )

    try:
        result = env_client.reset(task_id=task_id)
        episode_id = result.episode_id

        for step_index in range(1, max_agent_steps() + 1):
            if result.done:
                break

            action = choose_action(result.observation, model_client, model_name)
            result = env_client.step(episode_id, action)

            rewards.append(result.reward)
            steps_taken = step_index
            log_step(
                step=step_index,
                action=action,
                reward=result.reward,
                done=result.done,
                error=result.observation.last_action_error,
            )

            if result.done:
                break

        if episode_id is not None:
            grade = env_client.grade(episode_id)
            score = grade.score
            success = grade.score >= success_score_threshold()
    except BaseException as exc:
        exception = exc
    finally:
        log_end(success=success, steps=steps_taken, rewards=rewards)

    return {
        "task_id": task_id,
        "score": score,
        "success": success,
        "steps_taken": steps_taken,
        "exception": exception,
    }


def main() -> None:
    env_client = build_env_client()
    model_client, model_name = build_model_client()
    first_exception: Optional[BaseException] = None

    try:
        for task_id in requested_task_ids():
            result = run_task(
                env_client=env_client,
                model_client=model_client,
                model_name=model_name,
                task_id=task_id,
            )
            if first_exception is None and result["exception"] is not None:
                first_exception = result["exception"]
    finally:
        env_client.close()

    if first_exception is not None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
