#!/usr/bin/env python3
"""Programmatic grader: substring + activation checks for CUDA-Q Libraries skills.

A coarse but useful first-pass grader. Reads agent responses, applies the
``must_include`` / ``must_not_include`` substring rules from the skill's
assertions file, and checks activation behavior. Cheap to run and fully
deterministic, so it can be re-run without re-spending tokens on the agent.

Usage::

    programmatic.py --skill solvers --responses responses.json
    programmatic.py --skill qec     --responses responses.json
    programmatic.py --skill build   --responses responses.json
    programmatic.py --skill <name>  --responses responses.json \\
                    --assertions /path/to/cudaq-name.json

The responses file is JSON: ``{"S1": "agent reply", "A1": "...", ...}``. Keys
are ``S1`` ... ``S12`` for scenarios and ``A1`` ... ``A10`` for activation
prompts. Substring matching is case-insensitive.

Output: a ``grading.programmatic.json`` file conforming to the shared grader
schema (see references/schemas.md once it lands). When ``--out`` is omitted
the file is written next to the responses file.

Why standalone: this grader is one of three (the others are ``executable.py``
and ``judge.py``). Each grader runs as its own process so we can re-grade
without re-running the agent and combine multiple strategies (substring +
executable + judge) for inter-grader agreement analysis.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Map of skill cli alias -> full skill name. Used to locate the per-skill
# assertions file and as the ``skill`` field in the grading output. Add a new
# entry here when introducing a new skill.
SKILL_DIRS = {
    "solvers": "cudaq-solvers-algorithms",
    "qec": "cudaq-qec-decode",
    "build": "cudaq-build",
    "qec-realtime": "cudaq-qec-realtime",
    "chemistry": "cudaq-solvers-chemistry",
    "benchmarking": "cudaq-benchmarking",
    "contributing": "cudaq-contributing",
    "profiling": "cudaq-profiling-perf",
    "qec-ai": "cudaq-qec-ai-decoders",
    "qec-extending": "cudaq-qec-extending",
    "quickstart": "cudaq-quickstart",
    "skills-authoring": "cudaq-skills-authoring",
    "solvers-extending": "cudaq-solvers-extending",
}

# Repo-relative location of the assertions tree. Resolved relative to this
# file so the script works regardless of cwd.
EVALS_ROOT = Path(__file__).resolve().parents[1]
ASSERTIONS_DIR = EVALS_ROOT / "assertions"


@dataclass
class ScenarioScore:
    id: str
    coverage: int = 0
    coverage_max: int = 0
    purity: int = 0
    purity_max: int = 0
    missing: list[str] = field(default_factory=list)
    forbidden: list[str] = field(default_factory=list)


def _resolve_assertions_path(skill: str, override: Path | None) -> Path:
    if override is not None:
        return override
    if skill not in SKILL_DIRS:
        raise SystemExit(
            f"Unknown skill alias '{skill}'. Known: {sorted(SKILL_DIRS)}. "
            f"Pass --assertions <path> to score an unmapped skill.")
    candidate = ASSERTIONS_DIR / f"{SKILL_DIRS[skill]}.json"
    if not candidate.exists():
        raise SystemExit(f"Could not find assertions at {candidate}. "
                         f"Pass --assertions <path> explicitly.")
    return candidate


def _load_assertions(path: Path) -> dict:
    data = json.loads(path.read_text())
    required = {"scenarios", "activation", "activation_marker"}
    missing = required - data.keys()
    if missing:
        raise SystemExit(
            f"Assertions at {path} missing required keys: {sorted(missing)}")
    return data


def score_scenario(scenario_id: str, spec: dict,
                   response: str) -> ScenarioScore:
    text = response.lower()
    must = [s.lower() for s in spec.get("must_include", [])]
    must_not = [s.lower() for s in spec.get("must_not_include", [])]
    s = ScenarioScore(id=scenario_id)
    # Use the true list length, not max(..., 1). The previous floor-of-1
    # made empty-must_include scenarios unpassable (coverage stayed 0 while
    # coverage_max was 1) and gave empty-must_not_include scenarios a free
    # +1 (purity = 1 - 0 against purity_max = 1).
    s.coverage_max = len(must)
    s.purity_max = len(must_not)

    s.coverage = sum(1 for m in must if m in text)
    s.missing = [
        m for m in spec.get("must_include", []) if m.lower() not in text
    ]

    bad = sum(1 for m in must_not if m in text)
    s.purity = s.purity_max - bad
    s.forbidden = [
        m for m in spec.get("must_not_include", []) if m.lower() in text
    ]
    return s


def score_activation(spec: dict, response: str, marker: str) -> bool:
    activated = marker.lower() in response.lower()
    return activated == bool(spec["should_activate"])


def grade(skill: str, responses: dict[str, str], assertions_path: Path) -> dict:
    """Run the grader and return a grading.json-shaped dict.

    Pure function; does not write anywhere. ``main()`` handles I/O.
    """
    bench = _load_assertions(assertions_path)
    marker = bench["activation_marker"]

    scenarios = [
        score_scenario(sid, spec, responses.get(sid, ""))
        for sid, spec in bench["scenarios"].items()
    ]

    activations = []
    for aid, spec in bench["activation"].items():
        ok = score_activation(spec, responses.get(aid, ""), marker)
        activations.append({
            "id": aid,
            "should_activate": bool(spec["should_activate"]),
            "passed": ok,
        })

    coverage = sum(s.coverage for s in scenarios)
    coverage_max = sum(s.coverage_max for s in scenarios)
    purity = sum(s.purity for s in scenarios)
    purity_max = sum(s.purity_max for s in scenarios)
    activation_correct = sum(1 for a in activations if a["passed"])
    pass_rate = (
        sum(1 for s in scenarios
            if s.coverage == s.coverage_max and s.purity == s.purity_max) /
        max(len(scenarios), 1))

    return {
        "schema_version": "1",
        "grader": "programmatic",
        "skill": skill,
        "assertions_path": str(assertions_path),
        "coverage": coverage,
        "coverage_max": coverage_max,
        "purity": purity,
        "purity_max": purity_max,
        "activation_correct": activation_correct,
        "activation_total": len(activations),
        "scenario_pass_rate": pass_rate,
        "total": coverage + purity + activation_correct,
        "total_max": coverage_max + purity_max + len(activations),
        "scenarios": [{
            "id":
                s.id,
            "coverage":
                s.coverage,
            "coverage_max":
                s.coverage_max,
            "purity":
                s.purity,
            "purity_max":
                s.purity_max,
            "missing":
                s.missing,
            "forbidden":
                s.forbidden,
            "passed": (s.coverage == s.coverage_max and
                       s.purity == s.purity_max),
        } for s in scenarios],
        "activations": activations,
    }


def _print_summary(result: dict) -> None:
    print(f"Skill: {result['skill']}  ({result['assertions_path']})")
    print(f"  Scenario pass rate: {result['scenario_pass_rate']:.0%}")
    print(
        f"  Coverage:           {result['coverage']}/{result['coverage_max']}")
    print(f"  Purity:             {result['purity']}/{result['purity_max']}")
    print(
        f"  Activation:         {result['activation_correct']}/{result['activation_total']}"
    )
    print(f"  Total:              {result['total']}/{result['total_max']}")
    for s in result["scenarios"]:
        if s["missing"] or s["forbidden"]:
            print()
            print(f"  [{s['id']}] coverage {s['coverage']}/{s['coverage_max']} "
                  f"purity {s['purity']}/{s['purity_max']}")
            if s["missing"]:
                print(f"    missing: {s['missing']}")
            if s["forbidden"]:
                print(f"    forbidden hit: {s['forbidden']}")


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--skill",
                   required=True,
                   help=f"Skill alias. Known: {sorted(SKILL_DIRS)}.")
    p.add_argument("--responses",
                   type=Path,
                   required=True,
                   help="JSON file mapping prompt id -> agent response.")
    p.add_argument("--assertions",
                   type=Path,
                   default=None,
                   help="Path to assertions json. Auto-resolved by --skill.")
    p.add_argument("--out",
                   type=Path,
                   default=None,
                   help="Output path for grading.programmatic.json. "
                   "Defaults to <responses dir>/grading.programmatic.json.")
    p.add_argument("--quiet",
                   action="store_true",
                   help="Skip the human-readable summary.")
    args = p.parse_args()

    assertions_path = _resolve_assertions_path(args.skill, args.assertions)
    if not args.responses.exists():
        raise SystemExit(f"Responses file not found: {args.responses}")
    responses: dict[str, str] = json.loads(args.responses.read_text())

    result = grade(args.skill, responses, assertions_path)

    out = args.out or args.responses.parent / "grading.programmatic.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))

    if not args.quiet:
        _print_summary(result)
        print(f"  Wrote: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
