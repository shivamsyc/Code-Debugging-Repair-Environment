"""
Code Debugging / Repair Environment
=====================================
Input:  buggy code + error logs
Output: fixed code
Goal:   produce correct + runnable code
Focus:  reasoning + correctness
"""

import ast
import subprocess
import sys
import tempfile
import textwrap
import uuid
from typing import Any, Optional

from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Pydantic models (typed state / action / observation)
# ---------------------------------------------------------------------------

class CodeDebugState(BaseModel):
    task_id: str
    task_name: str
    buggy_code: str
    error_log: str
    description: str
    hint: Optional[str] = None
    fixed_code: Optional[str] = None
    last_action_error: Optional[str] = None
    step_count: int = 0
    done: bool = False
    reward: float = 0.0
    episode_id: str = ""


class CodeDebugAction(BaseModel):
    fixed_code: str


class CodeDebugObservation(BaseModel):
    task_id: str
    task_name: str
    buggy_code: str
    error_log: str
    description: str
    hint: Optional[str] = None
    last_action_error: Optional[str] = None
    step_count: int
    done: bool
    reward: float


# ---------------------------------------------------------------------------
# Task bank  (easy → medium → hard)
# ---------------------------------------------------------------------------

TASKS = [
    # ── EASY ──────────────────────────────────────────────────────────────
    {
        "id": "easy_syntax_fix",
        "name": "easy_syntax_fix",
        "difficulty": "easy",
        "description": (
            "Fix the syntax error in this Python function that is supposed "
            "to compute the factorial of n recursively."
        ),
        "buggy_code": textwrap.dedent("""\
            def factorial(n)
                if n == 0:
                    return 1
                return n * factorial(n - 1)

            print(factorial(5))
        """),
        "error_log": (
            "  File \"solution.py\", line 1\n"
            "    def factorial(n)\n"
            "                   ^\n"
            "SyntaxError: expected ':'"
        ),
        "hint": "Python function definitions must end with a colon.",
        "reference": textwrap.dedent("""\
            def factorial(n):
                if n == 0:
                    return 1
                return n * factorial(n - 1)

            print(factorial(5))
        """),
        "expected_output": "120",
    },
    # ── MEDIUM ────────────────────────────────────────────────────────────
    {
        "id": "medium_logic_fix",
        "name": "medium_logic_fix",
        "difficulty": "medium",
        "description": (
            "Fix the logic errors in this function that is supposed to "
            "return the two largest numbers from a list (sorted descending)."
        ),
        "buggy_code": textwrap.dedent("""\
            def two_largest(nums):
                nums.sort()
                return nums[:2]

            print(two_largest([3, 1, 4, 1, 5, 9, 2, 6]))
        """),
        "error_log": (
            "No runtime error — but output is wrong.\n"
            "Expected: [9, 6]\n"
            "Got:      [1, 1]"
        ),
        "hint": (
            "sort() defaults to ascending order. "
            "Also think about which slice you need."
        ),
        "reference": textwrap.dedent("""\
            def two_largest(nums):
                nums.sort(reverse=True)
                return nums[:2]

            print(two_largest([3, 1, 4, 1, 5, 9, 2, 6]))
        """),
        "expected_output": "[9, 6]",
    },
    # ── HARD ──────────────────────────────────────────────────────────────
    {
        "id": "hard_runtime_fix",
        "name": "hard_runtime_fix",
        "difficulty": "hard",
        "description": (
            "Fix all bugs in this function that is supposed to merge two "
            "sorted lists into one sorted list (merge-sort merge step). "
            "There are multiple bugs: an off-by-one error, an incorrect "
            "comparison operator, and a missing remainder append."
        ),
        "buggy_code": textwrap.dedent("""\
            def merge_sorted(a, b):
                result = []
                i, j = 0, 0
                while i < len(a) and j < len(b):
                    if a[i] > b[j]:        # bug 1
                        result.append(a[i])
                        i += 1
                    else:
                        result.append(b[j])
                        j += 1
                # bug 2: missing remainder
                return result

            print(merge_sorted([1, 3, 5], [2, 4, 6]))
        """),
        "error_log": (
            "No exception — but output is wrong.\n"
            "Expected: [1, 2, 3, 4, 5, 6]\n"
            "Got:      [1, 2, 3]"
        ),
        "hint": (
            "Check the comparison direction. "
            "After the while-loop, one list may still have elements left."
        ),
        "reference": textwrap.dedent("""\
            def merge_sorted(a, b):
                result = []
                i, j = 0, 0
                while i < len(a) and j < len(b):
                    if a[i] <= b[j]:
                        result.append(a[i])
                        i += 1
                    else:
                        result.append(b[j])
                        j += 1
                result.extend(a[i:])
                result.extend(b[j:])
                return result

            print(merge_sorted([1, 3, 5], [2, 4, 6]))
        """),
        "expected_output": "[1, 2, 3, 4, 5, 6]",
    },
]

TASK_MAP = {t["name"]: t for t in TASKS}


# ---------------------------------------------------------------------------
# Grader helpers
# ---------------------------------------------------------------------------

def _run_code(code: str, timeout: int = 10) -> tuple[bool, str, str]:
    """Execute *code* in a subprocess. Returns (success, stdout, stderr)."""
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(code)
        fname = f.name
    try:
        result = subprocess.run(
            [sys.executable, fname],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode == 0, result.stdout.strip(), result.stderr.strip()
    except subprocess.TimeoutExpired:
        return False, "", "TimeoutExpired"
    except Exception as e:
        return False, "", str(e)
    finally:
        import os
        try:
            os.unlink(fname)
        except OSError:
            pass


def _syntax_ok(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def grade(task: dict, fixed_code: str) -> float:
    """
    Returns a score in [0.0, 1.0].
    0.0  — no improvement at all
    0.25 — code is syntactically valid
    0.5  — code runs without error
    0.75 — output is non-empty and looks reasonable
    1.0  — output exactly matches expected
    """
    score = 0.0
    if not _syntax_ok(fixed_code):
        return score
    score = 0.25

    ok, stdout, stderr = _run_code(fixed_code)
    if not ok:
        return score
    score = 0.5

    expected = task["expected_output"].strip()
    if stdout:
        score = 0.75
    if stdout == expected:
        score = 1.0
    return score


# ---------------------------------------------------------------------------
# Environment class (OpenEnv spec)
# ---------------------------------------------------------------------------

class CodeDebugEnv:
    """OpenEnv-compatible Code Debugging / Repair environment."""

    metadata = {
        "name": "code_debug_env",
        "description": (
            "An RL environment where an agent receives buggy Python code "
            "plus an error log and must return the corrected, runnable code."
        ),
        "tasks": [t["name"] for t in TASKS],
        "action_space": "CodeDebugAction(fixed_code: str)",
        "observation_space": "CodeDebugObservation",
        "reward_range": [0.0, 1.0],
    }

    def __init__(self, task_name: str = "easy_syntax_fix"):
        if task_name not in TASK_MAP:
            raise ValueError(
                f"Unknown task '{task_name}'. "
                f"Available: {list(TASK_MAP.keys())}"
            )
        self.task_name = task_name
        self._task = TASK_MAP[task_name]
        self._state: Optional[CodeDebugState] = None

    # ── OpenEnv API ─────────────────────────────────────────────────────

    def reset(self) -> CodeDebugObservation:
        self._state = CodeDebugState(
            task_id=self._task["id"],
            task_name=self._task["name"],
            buggy_code=self._task["buggy_code"],
            error_log=self._task["error_log"],
            description=self._task["description"],
            hint=self._task.get("hint"),
            episode_id=str(uuid.uuid4()),
        )
        return self._obs()

    def step(self, action: CodeDebugAction) -> tuple[CodeDebugObservation, float, bool, dict]:
        if self._state is None:
            raise RuntimeError("Call reset() before step().")
        if self._state.done:
            raise RuntimeError("Episode is done. Call reset() to start a new one.")

        self._state.step_count += 1
        self._state.last_action_error = None

        try:
            reward = grade(self._task, action.fixed_code)
        except Exception as exc:
            self._state.last_action_error = str(exc)
            reward = 0.0

        self._state.fixed_code = action.fixed_code
        self._state.reward = reward
        self._state.done = reward >= 1.0 or self._state.step_count >= 5

        info = {
            "step": self._state.step_count,
            "task": self._task["name"],
            "difficulty": self._task["difficulty"],
        }
        return self._obs(), reward, self._state.done, info

    def state(self) -> CodeDebugState:
        if self._state is None:
            raise RuntimeError("Call reset() first.")
        return self._state

    def close(self):
        self._state = None

    # ── internal ─────────────────────────────────────────────────────────

    def _obs(self) -> CodeDebugObservation:
        s = self._state
        return CodeDebugObservation(
            task_id=s.task_id,
            task_name=s.task_name,
            buggy_code=s.buggy_code,
            error_log=s.error_log,
            description=s.description,
            hint=s.hint,
            last_action_error=s.last_action_error,
            step_count=s.step_count,
            done=s.done,
            reward=s.reward,
        )
