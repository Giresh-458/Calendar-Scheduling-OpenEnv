from __future__ import annotations

import os

import uvicorn
from fastapi import Body, FastAPI, HTTPException, Query

from models import GraderRequest, ResetRequest, StepRequest
from server.environment import CalendarSchedulingEnvironment


env = CalendarSchedulingEnvironment()
app = FastAPI(title="Calendar Scheduling Environment", version="0.1.0")


@app.get("/")
def index() -> dict:
    return {
        "name": "calendar-scheduling-env",
        "status": "ok",
        "openenv": True,
        "endpoints": ["/health", "/tasks", "/reset", "/step", "/state", "/grader"],
    }


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
