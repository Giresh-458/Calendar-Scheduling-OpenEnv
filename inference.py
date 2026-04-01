from __future__ import annotations

import json
import os
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional

from openai import OpenAI

from client import CalendarSchedulingEnvClient, EmbeddedCalendarSchedulingEnvClient
from models import CalendarAction, CalendarEvent, CalendarObservation, MeetingRequest
from task_definitions import TASKS


DEFAULT_API_BASE_URL = "https://router.huggingface.co/v1"
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
    return left.model_dump(mode="json", exclude_none=True) == right.model_dump(
        mode="json",
        exclude_none=True,
    )


def build_env_client() -> tuple[Any, str]:
    env_base_url = os.getenv("ENV_BASE_URL")
    if env_base_url:
        return CalendarSchedulingEnvClient(base_url=env_base_url), f"http:{env_base_url}"
    return EmbeddedCalendarSchedulingEnvClient(), "embedded"


def build_model_client() -> tuple[Optional[OpenAI], Optional[str], bool]:
    api_base_url = os.getenv("API_BASE_URL", DEFAULT_API_BASE_URL)
    model_name = os.getenv("MODEL_NAME")
    api_key = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
    llm_enabled = bool(model_name and api_key)

    if not llm_enabled:
        return None, model_name, False

    return OpenAI(base_url=api_base_url, api_key=api_key), model_name, True


def choose_action(
    observation: CalendarObservation,
    model_client: Optional[OpenAI],
    model_name: Optional[str],
) -> tuple[CalendarAction, Dict[str, Any]]:
    policy_action = plan_next_action(observation)
    trace: Dict[str, Any] = {
        "policy_action": policy_action.model_dump(mode="json", exclude_none=True),
        "selection": "policy",
    }

    if model_client is None or not model_name:
        return policy_action, trace

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
    except Exception as exc:
        trace["selection"] = "policy_after_model_error"
        trace["model_error"] = str(exc)
        return policy_action, trace

    trace["model_response"] = content
    model_action = parse_action(content)
    if model_action is None:
        trace["selection"] = "policy_after_parse_failure"
        return policy_action, trace

    trace["model_action"] = model_action.model_dump(mode="json", exclude_none=True)
    if same_action(model_action, policy_action):
        trace["selection"] = "model"
        return model_action, trace

    if policy_action.action_type != "noop":
        trace["selection"] = "policy_guardrail"
        return policy_action, trace

    trace["selection"] = "model"
    return model_action, trace


def run_task(
    env_client: Any,
    model_client: Optional[OpenAI],
    model_name: Optional[str],
    task_id: str,
    max_agent_steps: int,
) -> Dict[str, Any]:
    result = env_client.reset(task_id=task_id)
    episode_id = result.episode_id
    steps: List[Dict[str, Any]] = []

    for _ in range(max_agent_steps):
        if result.done:
            break

        action, trace = choose_action(result.observation, model_client, model_name)
        result = env_client.step(episode_id, action)
        steps.append(
            {
                "step": result.observation.step,
                "executed_action": action.model_dump(mode="json", exclude_none=True),
                "selection": trace["selection"],
                "reward": result.reward,
                "score": result.observation.score,
                "done": result.done,
            }
        )

    grade = env_client.grade(episode_id)
    return {
        "task_id": task_id,
        "score": grade.score,
        "passed": grade.passed,
        "details": grade.details,
        "steps_taken": len(steps),
        "trajectory": steps,
    }


def requested_task_ids() -> List[str]:
    raw_task_ids = os.getenv("TASK_IDS")
    if not raw_task_ids:
        return list(TASK_ORDER)

    task_ids = [item.strip() for item in raw_task_ids.split(",") if item.strip()]
    unknown = [task_id for task_id in task_ids if task_id not in TASKS]
    if unknown:
        raise ValueError(f"Unknown TASK_IDS entries: {', '.join(unknown)}")
    return task_ids


def main() -> None:
    max_agent_steps = int(os.getenv("MAX_AGENT_STEPS", "8"))
    env_client, env_mode = build_env_client()
    model_client, model_name, llm_enabled = build_model_client()
    task_ids = requested_task_ids()

    try:
        task_results = [
            run_task(
                env_client=env_client,
                model_client=model_client,
                model_name=model_name,
                task_id=task_id,
                max_agent_steps=max_agent_steps,
            )
            for task_id in task_ids
        ]
        average_score = round(mean(item["score"] for item in task_results), 4)
        summary = {
            "environment_mode": env_mode,
            "llm_enabled": llm_enabled,
            "model_name": model_name,
            "task_count": len(task_results),
            "scores": {item["task_id"]: item["score"] for item in task_results},
            "average_score": average_score,
            "results": task_results,
        }
        print(json.dumps(summary, indent=2))
    finally:
        env_client.close()


if __name__ == "__main__":
    main()
