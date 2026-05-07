#!/usr/bin/env python3
"""Executable grader: run snippets the agent produced and check their output.

Strongest possible signal for a code-suggesting skill: the agent's reply
isn't just substring-matched, it's *executed*. If the suggested commands
or Python snippets actually run and produce the expected output, the
scenario passes.

How it works
------------

For each scenario in the skill's assertions file, this grader looks for
an optional ``executable`` block::

    {
      "S3": {
        "must_include": [...],
        "executable": {
          "code_extractor": "first_python_block" | "first_bash_block" | "all_python_blocks",
          "preamble": "import os\\nimport sys\\n",
          "harness": "{code}",
          "expected_stdout_substr": "Steane",
          "expected_exit": 0,
          "timeout_s": 30,
          "interpreter": "python3" | "bash",
          "skip_if_missing": ["cudaq", "cudaq_qec"]
        }
      }
    }

If the block is absent, the scenario is recorded with ``status: skipped``
and does not contribute to pass-rate. If the block is present, the
extracted code is run in a subprocess (no shell injection unless
``interpreter == "bash"``) and the result is compared.

Sandbox notes
-------------

This is *not* a hardened sandbox. It assumes the agent under test is
trusted (it is your own model). It only protects against accidental
infinite loops via ``timeout_s``. Run inside a container if you need
isolation.

Output
------

``grading.executable.json`` next to ``responses.json`` (same shape as
``programmatic.json``).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

EVALS_ROOT = Path(__file__).resolve().parents[1]
ASSERTIONS_DIR = EVALS_ROOT / "assertions"

SKILL_DIRS = {
    "solvers": "cuda-qx-solvers",
    "qec": "cuda-qx-qec",
    "build": "cuda-qx-build",
}


def _resolve_assertions(skill: str, override: Path | None) -> Path:
    if override is not None:
        return override
    if skill not in SKILL_DIRS:
        raise SystemExit(f"Unknown skill alias '{skill}'. Known: {sorted(SKILL_DIRS)}")
    p = ASSERTIONS_DIR / f"{SKILL_DIRS[skill]}.json"
    if not p.exists():
        raise SystemExit(f"Assertions not found at {p}")
    return p


_FENCE_PATTERNS = {
    "first_python_block": re.compile(r"```(?:python|py)\s*\n(.*?)\n```", re.DOTALL),
    "first_bash_block":   re.compile(r"```(?:bash|sh|shell)\s*\n(.*?)\n```", re.DOTALL),
    "all_python_blocks":  re.compile(r"```(?:python|py)\s*\n(.*?)\n```", re.DOTALL),
}


def _extract_code(response: str, mode: str) -> str | None:
    pat = _FENCE_PATTERNS.get(mode)
    if pat is None:
        return None
    matches = pat.findall(response or "")
    if not matches:
        return None
    if mode == "all_python_blocks":
        return "\n\n".join(matches)
    return matches[0]


def _module_present(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError):
        return False


def _run_scenario(scenario_id: str, spec: dict, response: str) -> dict:
    rules = spec.get("executable")
    if not rules:
        return {"id": scenario_id, "status": "skipped",
                "reason": "no executable block in assertions"}

    # Skip if a required module is missing (saves time + noise).
    for needed in rules.get("skip_if_missing", []):
        if not _module_present(needed):
            return {"id": scenario_id, "status": "skipped",
                    "reason": f"required module '{needed}' not installed"}

    extractor = rules.get("code_extractor", "first_python_block")
    code = _extract_code(response, extractor)
    if code is None:
        return {"id": scenario_id, "status": "no_code",
                "reason": f"no `{extractor}` block found in response"}

    interp = rules.get("interpreter", "python3" if "python" in extractor else "bash")
    preamble = rules.get("preamble", "")
    harness = rules.get("harness", "{code}")
    program = harness.format(code=preamble + code) if "{code}" in harness else preamble + code
    timeout_s = float(rules.get("timeout_s", 30))

    cmd = [interp, "-c", program] if interp != "bash" else ["bash", "-c", program]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        rc, out, err, timed_out = r.returncode, r.stdout, r.stderr, False
    except subprocess.TimeoutExpired:
        rc, out, err, timed_out = 124, "", f"timeout after {timeout_s}s", True

    expected_exit = int(rules.get("expected_exit", 0))
    needle = rules.get("expected_stdout_substr", "")
    fail_substr = rules.get("must_not_in_stderr", "")

    exit_ok = rc == expected_exit
    stdout_ok = (needle in out) if needle else True
    stderr_ok = True if not fail_substr else (fail_substr not in err)

    passed = exit_ok and stdout_ok and stderr_ok and not timed_out

    return {
        "id": scenario_id,
        "status": "passed" if passed else "failed",
        "interpreter": interp,
        "exit_code": rc,
        "expected_exit": expected_exit,
        "exit_ok": exit_ok,
        "stdout_ok": stdout_ok,
        "stderr_ok": stderr_ok,
        "timed_out": timed_out,
        "stdout_tail": out[-500:],
        "stderr_tail": err[-500:],
    }


def grade(skill: str, responses: dict, assertions_path: Path) -> dict:
    bench = json.loads(assertions_path.read_text())
    scenarios = []
    for sid, spec in bench["scenarios"].items():
        scenarios.append(_run_scenario(sid, spec, responses.get(sid, "")))

    counted = [s for s in scenarios if s["status"] in {"passed", "failed", "no_code"}]
    passed = sum(1 for s in scenarios if s["status"] == "passed")
    failed = sum(1 for s in scenarios if s["status"] == "failed")
    no_code = sum(1 for s in scenarios if s["status"] == "no_code")
    skipped = sum(1 for s in scenarios if s["status"] == "skipped")

    return {
        "grader": "executable",
        "skill": skill,
        "assertions_path": str(assertions_path),
        "scenario_count": len(scenarios),
        "passed": passed,
        "failed": failed,
        "no_code": no_code,
        "skipped": skipped,
        "scenario_pass_rate": (passed / len(counted)) if counted else 0.0,
        "scenarios": scenarios,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--skill", required=True,
                   help=f"Skill alias. Known: {sorted(SKILL_DIRS)}")
    p.add_argument("--responses", type=Path, required=True)
    p.add_argument("--assertions", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    if not args.responses.exists():
        raise SystemExit(f"Responses file not found: {args.responses}")
    responses = json.loads(args.responses.read_text())

    assertions_path = _resolve_assertions(args.skill, args.assertions)
    result = grade(args.skill, responses, assertions_path)

    out = args.out or args.responses.parent / "grading.executable.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))

    if not args.quiet:
        print(f"Skill: {result['skill']}  ({assertions_path})")
        print(f"  Scenarios:   {result['scenario_count']}")
        print(f"  Passed:      {result['passed']}")
        print(f"  Failed:      {result['failed']}")
        print(f"  No code:     {result['no_code']}")
        print(f"  Skipped:     {result['skipped']}")
        print(f"  Pass rate:   {result['scenario_pass_rate']:.0%} (over passed+failed+no_code)")
        for s in result["scenarios"]:
            if s["status"] == "failed":
                print(f"  [{s['id']}] FAILED  exit={s['exit_code']}/{s['expected_exit']}")
                print(f"    stderr_tail: {s['stderr_tail'][:200]}")
        print(f"  Wrote: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
