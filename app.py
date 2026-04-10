"""
app.py — FastAPI server exposing the OpenEnv HTTP API
=====================================================
Endpoints:
  POST /reset          → CodeDebugObservation
  POST /step           → {observation, reward, done, info}
  GET  /state          → CodeDebugState
  GET  /tasks          → list[str]
  GET  /health         → {"status": "ok"}
"""

import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from env import CodeDebugAction, CodeDebugEnv, TASK_MAP

app = FastAPI(
    title="Code Debug Environment",
    description="OpenEnv: Code Debugging / Repair",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# One env instance per task (simple; good enough for hackathon scope)
_envs: dict[str, CodeDebugEnv] = {}


def _get_env(task_name: str) -> CodeDebugEnv:
    if task_name not in TASK_MAP:
        raise HTTPException(status_code=400, detail=f"Unknown task: {task_name}")
    if task_name not in _envs:
        _envs[task_name] = CodeDebugEnv(task_name)
    return _envs[task_name]


# ── Request bodies ──────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_name: str = "easy_syntax_fix"


class StepRequest(BaseModel):
    task_name: str = "easy_syntax_fix"
    fixed_code: str


# ── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/tasks")
def list_tasks():
    return {"tasks": list(TASK_MAP.keys())}


@app.post("/reset")
def reset(req: ResetRequest):
    env = _get_env(req.task_name)
    obs = env.reset()
    return obs.model_dump()


@app.post("/step")
def step(req: StepRequest):
    env = _get_env(req.task_name)
    if env._state is None:
        env.reset()
    action = CodeDebugAction(fixed_code=req.fixed_code)
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": reward,
        "done": done,
        "info": info,
    }


@app.get("/state")
def get_state(task_name: str = "easy_syntax_fix"):
    env = _get_env(task_name)
    if env._state is None:
        raise HTTPException(status_code=400, detail="Episode not started. Call /reset first.")
    return env.state().model_dump()


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 7860))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
