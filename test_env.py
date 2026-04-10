"""
tests/test_env.py — Local sanity tests (no external calls needed)
Run: python tests/test_env.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from env import CodeDebugEnv, CodeDebugAction, TASK_MAP, grade

CORRECT_FIXES = {
    "easy_syntax_fix": (
        "def factorial(n):\n"
        "    if n == 0:\n"
        "        return 1\n"
        "    return n * factorial(n - 1)\n\n"
        "print(factorial(5))\n"
    ),
    "medium_logic_fix": (
        "def two_largest(nums):\n"
        "    nums.sort(reverse=True)\n"
        "    return nums[:2]\n\n"
        "print(two_largest([3, 1, 4, 1, 5, 9, 2, 6]))\n"
    ),
    "hard_runtime_fix": (
        "def merge_sorted(a, b):\n"
        "    result = []\n"
        "    i, j = 0, 0\n"
        "    while i < len(a) and j < len(b):\n"
        "        if a[i] <= b[j]:\n"
        "            result.append(a[i])\n"
        "            i += 1\n"
        "        else:\n"
        "            result.append(b[j])\n"
        "            j += 1\n"
        "    result.extend(a[i:])\n"
        "    result.extend(b[j:])\n"
        "    return result\n\n"
        "print(merge_sorted([1, 3, 5], [2, 4, 6]))\n"
    ),
}

BROKEN_CODE = "def broken(:\n    pass\n"

passed = 0
failed = 0

def check(desc, condition):
    global passed, failed
    if condition:
        print(f"  ✅  {desc}")
        passed += 1
    else:
        print(f"  ❌  {desc}")
        failed += 1

print("=" * 60)
print("Code Debug Env — Test Suite")
print("=" * 60)

# ── 1. Task enumeration ──────────────────────────────────────────────────────
print("\n[1] Task enumeration")
check("3 tasks defined", len(TASK_MAP) == 3)
for name in ["easy_syntax_fix", "medium_logic_fix", "hard_runtime_fix"]:
    check(f"  task '{name}' present", name in TASK_MAP)

# ── 2. reset() ───────────────────────────────────────────────────────────────
print("\n[2] reset()")
for task_name in TASK_MAP:
    env = CodeDebugEnv(task_name)
    obs = env.reset()
    check(f"  {task_name}: obs.task_name correct", obs.task_name == task_name)
    check(f"  {task_name}: obs.done == False",     not obs.done)
    check(f"  {task_name}: obs.reward == 0.0",     obs.reward == 0.0)
    check(f"  {task_name}: buggy_code non-empty",  bool(obs.buggy_code))
    check(f"  {task_name}: error_log non-empty",   bool(obs.error_log))

# ── 3. Grader: broken code → score < 0.25 ────────────────────────────────────
print("\n[3] Grader — broken code")
for task_name, task in TASK_MAP.items():
    s = grade(task, BROKEN_CODE)
    check(f"  {task_name}: broken code scores 0.0", s == 0.0)

# ── 4. Grader: correct fix → score == 1.0 ────────────────────────────────────
print("\n[4] Grader — correct fix")
for task_name, fix in CORRECT_FIXES.items():
    task = TASK_MAP[task_name]
    s = grade(task, fix)
    check(f"  {task_name}: correct fix scores 1.0", s == 1.0)

# ── 5. step() with correct fix ───────────────────────────────────────────────
print("\n[5] step() — correct fix → done")
for task_name, fix in CORRECT_FIXES.items():
    env = CodeDebugEnv(task_name)
    env.reset()
    obs, reward, done, info = env.step(CodeDebugAction(fixed_code=fix))
    check(f"  {task_name}: reward == 1.0", reward == 1.0)
    check(f"  {task_name}: done == True",  done)

# ── 6. step() max-steps guard ────────────────────────────────────────────────
print("\n[6] step() — max steps (5) ends episode")
env = CodeDebugEnv("easy_syntax_fix")
env.reset()
done_any = False
for _ in range(6):
    try:
        obs, reward, done, _ = env.step(CodeDebugAction(fixed_code="# noop\n"))
        if done:
            done_any = True
            break
    except RuntimeError:
        done_any = True
        break
check("  episode terminates within 5 steps", done_any)

# ── 7. reward range ──────────────────────────────────────────────────────────
print("\n[7] Reward always in [0.0, 1.0]")
for task_name, fix in CORRECT_FIXES.items():
    task = TASK_MAP[task_name]
    for code in [BROKEN_CODE, "print('hello')\n", fix]:
        s = grade(task, code)
        check(f"  {task_name}: grade in [0,1]", 0.0 <= s <= 1.0)

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
total = passed + failed
print(f"Results: {passed}/{total} passed", "✅" if failed == 0 else "❌")
print("=" * 60)
sys.exit(0 if failed == 0 else 1)
