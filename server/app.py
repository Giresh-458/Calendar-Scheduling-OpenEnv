from __future__ import annotations

import os
from pathlib import Path

import uvicorn
from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse

from models import (
    CalendarAction,
    CalendarObservation,
    CalendarState,
    GraderRequest,
    ResetRequest,
    StepRequest,
    TaskSummary,
)
from server.environment import CalendarSchedulingEnvironment


env = CalendarSchedulingEnvironment()
app = FastAPI(title="Calendar Scheduling Environment", version="0.2.0")
README_PATH = Path(__file__).resolve().parents[1] / "README.md"


@app.get("/")
def index() -> dict:
    return {
        "name": "calendar-scheduling-env",
        "status": "ok",
        "openenv": True,
        "benchmark": "rich_calendar_coordination",
        "task_count": len(env.list_tasks()),
        "capabilities": [
            "schedule_event",
            "cancel_event",
            "reschedule_event",
            "dense_reward_shaping",
            "protected_events",
            "fallback_slots",
        ],
        "endpoints": ["/health", "/tasks", "/reset", "/step", "/state", "/grader"],
    }


@app.get("/metadata")
def metadata() -> dict:
    readme_content = README_PATH.read_text(encoding="utf-8") if README_PATH.exists() else None
    return {
        "name": "calendar-scheduling-env",
        "description": "Deterministic calendar coordination benchmark with protected anchors, fallback slots, and rescheduling.",
        "version": app.version,
        "readme_content": readme_content,
    }


@app.get("/schema")
def schema() -> dict:
    return {
        "action": CalendarAction.model_json_schema(),
        "observation": CalendarObservation.model_json_schema(),
        "state": CalendarState.model_json_schema(),
        "task_summary": TaskSummary.model_json_schema(),
    }


@app.post("/mcp")
def mcp(payload: dict | None = Body(default=None)):
    request_id = payload.get("id") if isinstance(payload, dict) else None
    return JSONResponse(
        status_code=200,
        content={
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": -32601,
                "message": "MCP is not implemented for this environment.",
            },
        },
    )


@app.get("/health")
def health() -> dict:
    return {"status": "healthy", "service": "calendar-scheduling-env"}


@app.get("/tasks")
def tasks():
    return [task.model_dump(mode="json") for task in env.list_tasks()]


@app.post("/reset")
def reset(payload: ResetRequest | None = Body(default=None)):
    try:
        result = env.reset(task_id=payload.task_id if payload else None)
        return result.model_dump(mode="json")
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/step")
def step(payload: StepRequest):
    try:
        result = env.step(payload.episode_id, payload.action)
        return result.model_dump(mode="json")
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown episode_id: {payload.episode_id}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/state")
def state(episode_id: str | None = Query(default=None)):
    try:
        return env.state(episode_id).model_dump(mode="json")
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/grader")
def grader(payload: GraderRequest):
    try:
        if payload.episode_id is not None:
            result = env.grade_episode(payload.episode_id)
        else:
            result = env.grade_explicit(payload.task_id, payload.events)
        return result.model_dump(mode="json")
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def main() -> None:
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("server.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
