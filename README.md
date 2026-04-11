---
title: Calendar Scheduling OpenEnv
emoji: 📅
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
short_description: Rich OpenEnv calendar coordination benchmark with rescheduling and protected events.
tags:
  - openenv
  - scheduling
  - benchmark
---

# Calendar Scheduling OpenEnv Environment

This repository provides a deterministic OpenEnv-compatible calendar coordination benchmark. Instead of only testing whether an agent can place one meeting, it evaluates whether the agent can preserve protected anchors, reschedule movable blockers, respect preferred time slots, and avoid destructive edits when solving a realistic day-planning problem.

The server exposes a Gym-style interaction loop over HTTP:

- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /tasks`
- `POST /grader`
- `GET /health`

## Why This Version Is Stronger

Compared with a toy scheduling demo, this benchmark now includes:

- protected anchor events that must remain intact
- movable internal meetings with approved relocation candidates
- preferred slots plus acceptable fallback slots for requested meetings
- denser grading that rewards good calendar stewardship, not just end-state matching
- five deterministic scenarios across team coordination, executive assistance, customer work, recruiting, and project management
- a deterministic baseline policy that solves every included task to the maximum public score

## Task Catalog

The environment ships with five deterministic tasks:

- `task_easy`: schedule one clean meeting into an empty calendar
- `task_medium`: move a blocker to its approved fallback slot, then place the customer review
- `task_hard`: preserve protected anchors while coordinating two back-to-back meetings
- `task_exec_dense_day`: coordinate three executive requests around focus, lunch, and board-read anchors
- `task_recruiting_loop`: protect recruiting anchors while scheduling a candidate panel and debrief

`GET /tasks` returns richer metadata for each task, including:

- `scenario_type`
- `request_count`
- `supports_reschedule`

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

## Environment Model

### Observation

Each step returns a structured observation with:

- current task metadata and requested meetings
- current calendar state with `movable`, `protected`, and `relocation_candidates`
- protected and movable event IDs for quick policy use
- scheduler notes describing the scenario constraints
- recent action history
- current step, score, reward, and feedback

Key observation fields:

- `task_id`, `task_name`, `task_description`
- `requested_meetings`
- `current_time`
- `events`
- `protected_event_ids`, `movable_event_ids`
- `scheduler_notes`, `recent_history`
- `step`, `max_steps`, `done`
- `feedback`, `last_action_error`
- `score`, `last_reward`, `reward_breakdown`
- `available_actions`

### Actions

Supported actions:

- `schedule_event`
- `cancel_event`
- `reschedule_event`
- `noop`

`reschedule_event` is the key addition in this version. It lets an agent preserve internal meetings by moving them to approved fallback slots instead of deleting them.

Example `schedule_event` payload:

```json
{
  "episode_id": "your-episode-id",
  "action": {
    "action_type": "schedule_event",
    "title": "Board Prep",
    "start_time": "2026-04-02T10:00:00Z",
    "duration_hours": 1.0,
    "participants": ["alex@example.com", "chief_of_staff@example.com"]
  }
}
```

Example `reschedule_event` payload:

```json
{
  "episode_id": "your-episode-id",
  "action": {
    "action_type": "reschedule_event",
    "event_id": 2,
    "new_start_time": "2026-04-02T13:00:00Z",
    "duration_hours": 1.0
  }
}
```

## Grading and Rewards

The grader combines end-state correctness with schedule quality:

- full credit requires requested meetings in their preferred slots
- acceptable fallback slots earn strong partial credit
- protected anchors must remain intact
- movable blockers that have approved fallback slots should be preserved by rescheduling
- overlapping events reduce the final score

The environment also exposes dense reward shaping on every step:

- small step penalty for efficiency
- progress reward when the deterministic grader improves
- invalid action penalties for rejected operations
- destructive action penalty for cancellations
- completion bonus on a perfect solve

Scores are always normalized into the open interval `(0, 1)`:

- unsolved floor: `0.001`
- solved ceiling: `0.999`

## Endpoints

### `GET /tasks`

Returns the task catalog and scenario metadata.

### `POST /reset`

Starts a new episode.

Request:

```json
{
  "task_id": "task_exec_dense_day"
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

### `GET /metadata`

Returns environment metadata plus the repository README contents.

### `GET /schema`

Returns the action, observation, state, and task-summary JSON schemas.

## Baseline Inference Script

`inference.py` includes a deterministic safety-first policy that:

- keeps protected anchors intact
- reschedules movable blockers into approved fallback slots when possible
- cancels only when a clean relocation is unavailable
- prefers the highest-priority request and preferred slot first

The script prints only the required structured stdout lines:

```text
[START] task=<task_id> env=<benchmark> model=<model_name>
[STEP] step=<n> action=<json_action> reward=<0.00> done=<true|false> error=<msg|null>
[END] success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...,rn>
```

For local reproducibility, the script defaults to an embedded in-process environment when `ENV_BASE_URL` is not set. If `ENV_BASE_URL` is provided, it targets the running HTTP server or deployed HF Space instead.

Optional environment variables:

- `ENV_BASE_URL`
- `TASK_IDS`
- `MAX_AGENT_STEPS`
- `BENCHMARK_NAME`
- `SUCCESS_SCORE_THRESHOLD`
- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`

### Reference Baseline Scores

With the embedded deterministic policy, all included tasks reach `0.999`.

## Hugging Face Space Deployment

1. Create a new Hugging Face Space using the `Docker` SDK.
2. Push this repository to the Space repository root.
3. Keep `README.md`, `Dockerfile`, and `openenv.yaml` at the repo root.
4. Wait for the build to finish, then verify:
   `GET /health`, `GET /tasks`, and `POST /reset`.

## Tests

The test suite now covers:

- full-score solves for the easy and richer multi-step tasks
- guardrails that prevent full credit after destructive blocker deletion
- score-range checks across all API payloads
- structured inference logging
- deterministic baseline policy success across the full task catalog

Run:

```bash
pytest
```
