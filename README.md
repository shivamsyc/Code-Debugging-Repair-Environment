# 🐛 Code Debugging / Repair — OpenEnv Environment

> **OpenEnv Hackathon Round 1 Submission**

## Overview

This environment presents an AI agent with **buggy Python code + an error / wrong-output log** and asks it to produce the **correct, runnable version**.

The environment tests an agent's ability to:
- Parse and understand error messages
- Identify syntax, logic, and runtime bugs
- Produce valid, executable Python

---

## Tasks

| Task | Difficulty | Description |
|------|-----------|-------------|
| `easy_syntax_fix`   | Easy   | Fix a `SyntaxError` in a recursive factorial function |
| `medium_logic_fix`  | Medium | Fix logic errors in a "two largest numbers" function |
| `hard_runtime_fix`  | Hard   | Fix multiple bugs in a merge-sorted implementation |

---

## Reward Function

| Score | Condition |
|-------|-----------|
| 0.00  | Submitted code has a syntax error |
| 0.25  | Syntax is valid |
| 0.50  | Code runs without raising an exception |
| 0.75  | Code produces non-empty stdout |
| 1.00  | Stdout **exactly** matches expected output |

Episode ends at `reward == 1.0` or after **5 steps**.

---

## Action / Observation Spaces

**Action**
```json
{ "fixed_code": "<complete corrected Python source>" }
```

**Observation**
```json
{
  "task_id":           "easy_syntax_fix",
  "task_name":         "easy_syntax_fix",
  "buggy_code":        "def factorial(n)\n    ...",
  "error_log":         "SyntaxError: expected ':'",
  "description":       "Fix the syntax error ...",
  "hint":              "Python function definitions must end with a colon.",
  "last_action_error": null,
  "step_count":        1,
  "done":              false,
  "reward":            0.25
}
```

---

## API Endpoints

| Method | Path      | Body / Query            | Description |
|--------|-----------|-------------------------|-------------|
| `GET`  | `/health` | —                       | Health check |
| `GET`  | `/tasks`  | —                       | List available tasks |
| `POST` | `/reset`  | `{"task_name": "..."}` | Start new episode |
| `POST` | `/step`   | `{"task_name": "...", "fixed_code": "..."}` | Submit fix |
| `GET`  | `/state`  | `?task_name=...`        | Current state |

---

## Setup

### Local (Docker)

```bash
docker build -t code-debug-env .
docker run -p 7860:7860 code-debug-env
```

### Local (Python)

```bash
pip install -r requirements.txt
python app.py
```

---

## Running Inference

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="hf_..."
export ENV_BASE_URL="http://localhost:7860"

python inference.py
```

Optional — run a single task:
```bash
TASK_NAME=hard_runtime_fix python inference.py
```

---

## Project Structure

```
code_debug_env/
├── app.py           # FastAPI HTTP server (OpenEnv API)
├── env.py           # Environment core logic + graders
├── inference.py     # Mandatory inference script
├── openenv.yaml     # OpenEnv spec
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Evaluation Criteria Mapping

| Criterion | How we address it |
|-----------|------------------|
| Real-world utility (30%) | Debugging is a universal dev task; graders execute real code |
| Task & grader quality (25%) | 3 tasks easy→hard; graders deterministic and reproducible |
| Environment design (20%) | Clean state per episode; partial reward signals at 4 levels |
| Code quality (15%) | Typed Pydantic models; FastAPI; Docker; spec-compliant |
| Creativity (10%) | Code-execution grader with AST + subprocess; 4-level partial reward |
