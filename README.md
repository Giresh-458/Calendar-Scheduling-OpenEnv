---
title: Calendar Scheduling OpenEnv
emoji: 📅
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
short_description: Rich calendar coordination benchmark for OpenEnv agents.
tags:
  - openenv
  - scheduling
  - benchmark
---

# Calendar Scheduling OpenEnv Environment

This repository provides a deterministic OpenEnv-compatible calendar coordination benchmark. Instead of only testing whether an agent can place one meeting, it evaluates whether the agent can preserve protected anchors, reschedule movable blockers, respect preferred time slots, and avoid destructive edits when solving a realistic day-planning problem.

The server exposes a Gym-style interaction loop over HTTP.

## Features

Compared to a basic scheduling demo, this benchmark includes:
- Protected anchor events that must remain intact.
- Movable internal meetings with approved relocation candidates.
- Preferred slots plus acceptable fallback slots for requested meetings.
- Dense grading that rewards good calendar stewardship, not just end-state matching.
- Five deterministic scenarios (team coordination, executive assistance, customer work, recruiting, project management).
- A deterministic baseline policy that solves every included task to the maximum score.

## Task Catalog

The environment ships with five deterministic tasks:
- `task_easy`: schedule one clean meeting into an empty calendar
- `task_medium`: move a blocker to its approved fallback slot, then place the customer review
- `task_hard`: preserve protected anchors while coordinating two back-to-back meetings
- `task_exec_dense_day`: coordinate three executive requests around focus, lunch, and board-read anchors
- `task_recruiting_loop`: protect recruiting anchors while scheduling a candidate panel and debrief

`GET /tasks` returns richer metadata for each task, including `scenario_type`, `request_count`, and `supports_reschedule`.

## Quick Start

### Local Python

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
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
- Current task metadata and requested meetings
- Current calendar state with `movable`, `protected`, and `relocation_candidates`
- Protected and movable event IDs for quick policy use
- Scheduler notes describing the scenario constraints
- Recent action history
- Current step, score, reward, and feedback

### Actions
Supported actions:
- `schedule_event`
- `cancel_event`
- `reschedule_event` (Lets an agent preserve internal meetings by moving them to approved fallback slots instead of deleting them)
- `noop`

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
- Full credit requires requested meetings in their preferred slots.
- Acceptable fallback slots earn strong partial credit.
- Protected anchors must remain intact.
- Movable blockers that have approved fallback slots should be preserved by rescheduling.
- Overlapping events reduce the final score.

Scores are normalized into the open interval `(0, 1)` (floor: `0.001`, ceiling: `0.999`). The environment also exposes dense reward shaping on every step (step penalties, progress rewards, destructive action penalties, and completion bonuses).

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/tasks` | `GET` | Returns the task catalog and scenario metadata. |
| `/reset` | `POST` | Starts a new episode. Pass `{"task_id": "task_id"}` in payload. |
| `/step` | `POST` | Applies one typed action to an existing episode. |
| `/state` | `GET` | Returns the current internal episode state (`?episode_id=<id>`). |
| `/grader` | `POST` | Grades a live episode by `episode_id`, or an explicit `{task_id, events}` payload. |
| `/metadata`| `GET` | Returns environment metadata plus the repository README contents. |
| `/schema` | `GET` | Returns the action, observation, state, and task-summary JSON schemas. |

## Baseline Inference Script

`inference.py` includes a deterministic safety-first policy that:
- Keeps protected anchors intact.
- Reschedules movable blockers into approved fallback slots when possible.
- Cancels only when a clean relocation is unavailable.
- Prefers the highest-priority request and preferred slot first.

For local reproducibility, the script defaults to an embedded in-process environment when `ENV_BASE_URL` is not set. If `ENV_BASE_URL` is provided, it targets the running HTTP server or deployed HF Space instead.

With the embedded deterministic policy, all included tasks reach `0.999`.

## Deployment (Hugging Face Spaces)

1. Create a new Hugging Face Space using the Docker SDK.
2. Push this repository to the Space repository root.
3. Keep `README.md`, `Dockerfile`, and `openenv.yaml` at the repo root.
4. Wait for the build to finish, then verify `/health`, `/tasks`, and `/reset`.

## Tests

The test suite covers full-score solves, guardrails, score-range checks, and deterministic baseline success across the catalog. 

```bash
pytest
```
