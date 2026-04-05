---
title: Calendar Scheduling OpenEnv
emoji: 📅
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
short_description: OpenEnv calendar scheduling environment.
tags:
  - openenv
---

# Calendar Scheduling OpenEnv Environment

This repository provides a deterministic OpenEnv-compatible calendar scheduling environment. It models a real workflow that operations assistants, executive assistants, recruiters, and project managers handle every day: placing meetings into constrained calendars while preserving existing commitments.

The environment exposes a Gym-style interaction loop over HTTP:

- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /tasks`
- `POST /grader`
- `GET /health`

It includes three deterministic tasks:

- `task_easy`: schedule one meeting in an empty calendar
- `task_medium`: resolve a blocking meeting, then schedule the requested meeting
- `task_hard`: clear two blockers, then coordinate two back-to-back meetings while preserving anchor events

## Project Layout

```text
.
|-- Dockerfile
|-- README.md
|-- client.py
|-- inference.py
|-- models.py
|-- openenv.yaml
|-- pyproject.toml
|-- requirements.txt
|-- scripts/
|   `-- validate-submission.sh
|-- task_definitions.py
|-- uv.lock
|-- tests/
|   |-- test_environment.py
|   |-- test_grader_guards.py
|   |-- test_inference_logging.py
|   `-- test_inference_policy.py
`-- server/
    |-- __init__.py
    |-- app.py
    `-- environment.py
```

## Quick Start

### Local Python

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### OpenEnv Validation

```bash
pip install openenv-core uv
uv lock
openenv validate
```

Optional pre-submission validator:

```bash
bash scripts/validate-submission.sh https://your-space-name.hf.space .
```

### Docker

```bash
docker build -t calendar-scheduling-env:latest .
docker run --rm -p 8000:8000 calendar-scheduling-env:latest
curl http://localhost:8000/health
```

Expected health response:

```json
{"status":"healthy","service":"calendar-scheduling-env"}
```

## Hugging Face Space Deployment

1. Create a new Hugging Face Space and choose the `Docker` SDK.
2. Push this repository to the Space repository root.
3. Keep the root `README.md` and `Dockerfile` exactly at the repository root.
4. In Space Settings, add secrets only if you want to run `inference.py` inside the Space:
   `HF_TOKEN`, `MODEL_NAME`, and optionally `API_BASE_URL`.
5. Wait for the build logs to finish, then verify:
   `GET /health` and `POST /reset`.

If you only want to host the environment server itself, no secret is required for the Space runtime.

## Environment Model

### Observation

Each step returns a structured observation with:

- current task metadata
- requested meetings
- the current calendar state
- step count and max steps
- last reward, current score, and feedback
- `last_action_error` for rejected actions
- `reward_breakdown` with typed reward components

Observation fields:

- `task_id`, `task_name`, `task_description`
- `requested_meetings`
- `current_time`
- `events`
- `step`, `max_steps`, `done`
- `feedback`, `last_action_error`
- `score`, `last_reward`, `reward_breakdown`
- `available_actions`

### Action

Supported actions:

- `schedule_event`
- `cancel_event`
- `noop`

Action fields:

- `action_type`
- `title`, `start_time`, `duration_hours`, `participants` for `schedule_event`
- `event_id` for `cancel_event`

Example action payload:

```json
{
  "episode_id": "your-episode-id",
  "action": {
    "action_type": "schedule_event",
    "title": "Design Review",
    "start_time": "2026-04-02T10:00:00Z",
    "duration_hours": 1.0,
    "participants": ["alex@example.com", "maya@example.com"]
  }
}
```

## Tasks

### `task_easy`

- Current time: `2026-04-01T09:00:00Z`
- Goal: schedule a one-hour meeting tomorrow at `10:00`
- Initial calendar: empty

### `task_medium`

- Goal: schedule a one-hour meeting tomorrow at `10:00`
- Initial calendar: `"Team Sync"` already occupies `10:00-11:00`
- Expected behavior: remove or move the blocker, then schedule the requested meeting

### `task_hard`

- Goal: place two back-to-back meetings at `10:00-11:00` and `11:00-12:00`
- Initial calendar: anchor meetings at `09:00-10:00` and `12:00-13:00`
- Blocking meetings already occupy `10:00-11:00` and `11:00-12:00`
- Expected behavior: remove both blockers, schedule both requested meetings, and keep the anchor meetings intact

## Reward Shaping

The environment uses dense rewards:

- small step penalty on every action
- positive reward when the current graded score improves
- additional penalties for invalid actions and scheduling conflicts
- a destructive-action penalty when the agent cancels an event
- completion bonus when a task reaches full score

The deterministic grader always computes a final normalized score between `0.0` and `1.0`. Each transition also exposes a typed `CalendarReward` breakdown so agents can learn from progress, mistakes, and destructive edits separately.

## Endpoints

### `GET /tasks`

Returns the task catalog.

### `POST /reset`

Starts a new episode.

Request:

```json
{
  "task_id": "task_easy"
}
```

### `POST /step`

Applies one typed action to an existing episode.

### `GET /state?episode_id=<id>`

Returns the current internal episode state.

### `POST /grader`

Grades either:

- a live episode by `episode_id`, or
- an explicit `{task_id, events}` payload

## Baseline Inference Script

`inference.py` uses the OpenAI-compatible client and expects:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`
- optional `LOCAL_IMAGE_NAME` only for Docker-image-backed environments using `from_docker_image()`

Defaults are provided only for `API_BASE_URL` and `MODEL_NAME`. `HF_TOKEN` must be supplied in the submission environment when you want model-backed inference.

It evaluates all three tasks in order and prints only the required structured stdout lines:

```text
[START] task=<task_id> env=<benchmark> model=<model_name>
[STEP] step=<n> action=<json_action> reward=<0.00> done=<true|false> error=<msg|null>
[END] success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...,rn>
```

The script uses the OpenAI client for all LLM calls when `HF_TOKEN` is present, but still applies a deterministic safety policy so baseline scores remain reproducible.

This repository does not require a Docker image name inside `inference.py`. If you see `IMAGE_NAME` or `LOCAL_IMAGE_NAME` in the sample materials, treat them as sample-only variables for image-backed environments rather than a requirement for this HTTP/embedded environment.

For local reproducibility, the script defaults to an embedded in-process environment when `ENV_BASE_URL` is not set. If `ENV_BASE_URL` is provided, it will target the running HTTP server or deployed HF Space instead.

Optional environment variables:

- `ENV_BASE_URL` points to a running environment endpoint when you want remote execution
- `TASK_IDS` can restrict evaluation to a comma-separated subset such as `task_easy,task_medium`
- `MAX_AGENT_STEPS` defaults to `8`
- `BENCHMARK_NAME` overrides the benchmark label printed in the `[START]` line
- `SUCCESS_SCORE_THRESHOLD` defaults to `1.0`

Example local run:

```bash
python inference.py
```

### Reference Baseline Scores

With the embedded deterministic baseline policy, the environment reaches:

- `task_easy`: `1.0`
- `task_medium`: `1.0`
- `task_hard`: `1.0`
- average: `1.0`

## Tests

The test suite covers:

- solving the easy task in one move
- conflict handling for the medium task
- partial credit on the hard task
- full-score deterministic baseline planning across all three tasks
- a longer policy trajectory for `task_hard` than for `task_medium`
- strict `[START]` / `[STEP]` / `[END]` logging for the submission script

Run:

```bash
pytest
```
