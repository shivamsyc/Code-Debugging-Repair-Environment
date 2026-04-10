"""
inference.py — OpenEnv Code Debug Environment
==============================================
Mandatory variables (set via environment or .env):
  API_BASE_URL     LLM API base URL
  MODEL_NAME       Model identifier
  HF_TOKEN         Hugging Face / API key

Stdout format (strictly followed):
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import os
import sys
import json
import textwrap
import requests
from openai import OpenAI

# ── Config ──────────────────────────────────────────────────────────────────
API_BASE_URL  = os.getenv("API_BASE_URL",  "https://router.huggingface.co/v1")
MODEL_NAME    = os.getenv("MODEL_NAME",    "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN      = os.getenv("HF_TOKEN",      "")
ENV_BASE_URL  = os.getenv("ENV_BASE_URL",  "http://localhost:7860")
BENCHMARK     = "code_debug_env"
MAX_STEPS     = 5
TEMPERATURE   = 0.2

TASKS = ["easy_syntax_fix", "medium_logic_fix", "hard_runtime_fix"]

client = OpenAI(
    api_key=HF_TOKEN or "dummy",
    base_url=API_BASE_URL,
)

# ── Helpers ──────────────────────────────────────────────────────────────────

def env_reset(task_name: str) -> dict:
    r = requests.post(f"{ENV_BASE_URL}/reset", json={"task_name": task_name}, timeout=30)
    r.raise_for_status()
    return r.json()

def env_step(task_name: str, fixed_code: str) -> dict:
    r = requests.post(
        f"{ENV_BASE_URL}/step",
        json={"task_name": task_name, "fixed_code": fixed_code},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()

def build_prompt(obs: dict) -> str:
    return textwrap.dedent(f"""\
        You are an expert Python debugger.

        ## Task
        {obs['description']}

        ## Buggy Code
        ```python
        {obs['buggy_code']}
        ```

        ## Error / Wrong-Output Log
        ```
        {obs['error_log']}
        ```

        {"## Hint\\n" + obs['hint'] if obs.get('hint') else ""}

        ## Instructions
        Return ONLY the complete corrected Python code with no markdown fences,
        no explanations, and no extra text. The code must be runnable as-is.
    """).strip()

def call_llm(prompt: str) -> str:
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=TEMPERATURE,
        max_tokens=1024,
    )
    text = resp.choices[0].message.content or ""
    # Strip accidental markdown fences
    lines = text.strip().splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines)

def safe_action_str(code: str) -> str:
    """Single-line representation for the [STEP] log."""
    return repr(code[:120])

# ── Main loop ────────────────────────────────────────────────────────────────

def run_task(task_name: str) -> float:
    obs = env_reset(task_name)
    print(f"[START] task={task_name} env={BENCHMARK} model={MODEL_NAME}", flush=True)

    rewards = []
    score = 0.0
    done = False
    steps = 0
    success = False

    try:
        for step_num in range(1, MAX_STEPS + 1):
            prompt = build_prompt(obs)
            fixed_code = call_llm(prompt)
            action_str = safe_action_str(fixed_code)

            result = env_step(task_name, fixed_code)
            reward = result["reward"]
            done   = result["done"]
            error  = result["observation"].get("last_action_error") or "null"

            rewards.append(reward)
            steps = step_num
            score = reward

            print(
                f"[STEP] step={step_num} action={action_str} "
                f"reward={reward:.2f} done={str(done).lower()} error={error}",
                flush=True,
            )

            obs = result["observation"]

            if done:
                success = reward >= 1.0
                break

    except Exception as exc:
        error_msg = str(exc).replace("\n", " ")
        print(
            f"[STEP] step={steps+1} action=null reward=0.00 done=true error={error_msg}",
            flush=True,
        )

    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True,
    )
    return score


def main():
    task_arg = os.getenv("TASK_NAME", "")
    task_list = [task_arg] if task_arg and task_arg in TASKS else TASKS

    all_scores = {}
    for task in task_list:
        s = run_task(task)
        all_scores[task] = s

    # Summary
    avg = sum(all_scores.values()) / len(all_scores) if all_scores else 0.0
    print(f"\n# Final Scores: {json.dumps(all_scores, indent=2)}", flush=True)
    print(f"# Average Score: {avg:.4f}", flush=True)


if __name__ == "__main__":
    main()
