#!/usr/bin/env python3
"""Eval runner for CUDA-QX skills.

Orchestrator between an evaluator (a human or a sub-agent driver) and the
grading layer. The runner does NOT call any LLM and does NOT do any grading
itself; it just (1) hands prompts to the evaluator and (2) collects the
resulting responses into a workspace tree that the graders consume.

Subcommands
-----------

prompts
    Print prompts for a skill (and optionally a configuration) as JSONL on
    stdout. Each line is ``{"id": ..., "prompt": ..., "kind":
    "scenario"|"activation"}``.

    Use this to drive a sub-agent: pipe each line into your runner of choice,
    capture the response, and collect them into a ``responses.json`` map.

aggregate
    Combine the per-config grading.*.json files in an iteration directory
    into a single ``benchmark.json``. Computes deltas between ``with_skill``
    and ``without_skill`` configs when both are present. For inter-grader
    agreement analysis, see ``../aggregate.py``.

Layout convention
-----------------

::

  <workspace>/
    iteration-N/
      with_skill/
        responses.json                   <- evaluator wrote this
        grading.programmatic.json        <- grader wrote this
        grading.executable.json          <- grader wrote this (optional)
        grading.judge.json               <- grader wrote this (optional)
        timing.json                      <- evaluator wrote this
      without_skill/
        responses.json
        grading.programmatic.json
        ...
      benchmark.json                     <- aggregate wrote this

Workspaces live under ``.claude/evals/workspaces/`` (gitignored). They sit
outside the skills tree so the agent under test never sees them.

Note: scoring used to live inside this runner. It now lives in the graders
under ``.claude/evals/graders/`` (one binary per strategy). To grade a
``responses.json``, invoke a grader directly, e.g.::

    python .claude/evals/graders/programmatic.py --skill qec \
        --responses workspace/iteration-1/with_skill/responses.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Repo-relative roots, resolved from this file. The runner lives at
# .claude/evals/runners/runner.py.
EVALS_ROOT = Path(__file__).resolve().parents[1]
PROMPTS_DIR = EVALS_ROOT / "prompts"
GRADERS_DIR = EVALS_ROOT / "graders"

# Map of cli alias -> full skill name. Mirrors graders/programmatic.py;
# kept duplicated rather than imported so each tool can run standalone.
SKILL_DIRS = {
    "solvers": "cuda-qx-solvers",
    "qec": "cuda-qx-qec",
    "build": "cuda-qx-build",
}


def _prompts_path(skill: str) -> Path:
    if skill not in SKILL_DIRS:
        raise SystemExit(
            f"Unknown skill alias '{skill}'. Known: {sorted(SKILL_DIRS)}"
        )
    candidate = PROMPTS_DIR / f"{SKILL_DIRS[skill]}.evals.json"
    if not candidate.exists():
        raise SystemExit(
            f"Could not find prompts at {candidate}. "
            f"Did you move/rename the prompts file?"
        )
    return candidate


def cmd_prompts(args: argparse.Namespace) -> int:
    evals = json.loads(_prompts_path(args.skill).read_text())

    items: list[dict] = []
    if args.kind in ("scenario", "all"):
        for entry in evals.get("evals", []):
            items.append({
                "id": entry["id"],
                "name": entry.get("name"),
                "kind": "scenario",
                "prompt": entry["prompt"],
            })
    if args.kind in ("activation", "all"):
        for entry in evals.get("activation", []):
            items.append({
                "id": entry["id"],
                "kind": "activation",
                "prompt": entry["prompt"],
            })

    if args.format == "jsonl":
        for item in items:
            sys.stdout.write(json.dumps(item) + "\n")
    else:
        sys.stdout.write(json.dumps(items, indent=2) + "\n")
    return 0


def cmd_aggregate(args: argparse.Namespace) -> int:
    """Aggregate grading.*.json files across configurations in an iteration.

    For each configuration directory (with_skill, without_skill, ...),
    collect every grading.*.json, keyed by grader name. When both with_skill
    and without_skill are present and a grader has results for both, compute
    a delta. The resulting benchmark.json is the input to the HTML viewer
    and to longer-term iteration-vs-iteration comparison.
    """
    iteration = args.iteration_dir
    if not iteration.is_dir():
        raise SystemExit(f"Not a directory: {iteration}")

    out: dict = {"iteration": iteration.name, "configurations": {}}

    for cfg_dir in sorted(iteration.iterdir()):
        if not cfg_dir.is_dir():
            continue
        cfg_record: dict = {}
        for grading in sorted(cfg_dir.glob("grading.*.json")):
            grader_name = grading.stem.split(".", 1)[1]  # grading.programmatic -> programmatic
            cfg_record[grader_name] = json.loads(grading.read_text())
        if cfg_record:
            out["configurations"][cfg_dir.name] = cfg_record

    cfgs = out["configurations"]
    if "with_skill" in cfgs and "without_skill" in cfgs:
        deltas: dict = {}
        for grader in cfgs["with_skill"]:
            if grader not in cfgs["without_skill"]:
                continue
            w = cfgs["with_skill"][grader]
            b = cfgs["without_skill"][grader]
            deltas[grader] = {
                "scenario_pass_rate":
                    w.get("scenario_pass_rate", 0) - b.get("scenario_pass_rate", 0),
                "coverage": w.get("coverage", 0) - b.get("coverage", 0),
                "purity": w.get("purity", 0) - b.get("purity", 0),
                "activation_correct":
                    w.get("activation_correct", 0) - b.get("activation_correct", 0),
                "total": w.get("total", 0) - b.get("total", 0),
            }
        out["delta"] = deltas

    out_path = iteration / "benchmark.json"
    out_path.write_text(json.dumps(out, indent=2))

    print(f"Aggregated {len(cfgs)} configurations into {out_path}")
    for grader, d in (out.get("delta") or {}).items():
        print(f"  [{grader}] Δ pass_rate: {d['scenario_pass_rate']:+.0%}  "
              f"Δ coverage: {d['coverage']:+d}  Δ total: {d['total']:+d}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("prompts", help="Dump prompts for a skill.")
    pp.add_argument("--skill", required=True,
                    help=f"One of: {sorted(SKILL_DIRS)}")
    pp.add_argument("--kind", choices=["scenario", "activation", "all"],
                    default="all")
    pp.add_argument("--format", choices=["jsonl", "json"], default="jsonl")
    pp.set_defaults(func=cmd_prompts)

    pa = sub.add_parser(
        "aggregate",
        help="Combine grading.*.json files across an iteration directory."
    )
    pa.add_argument("iteration_dir", type=Path,
                    help="Path to iteration-N/ directory.")
    pa.set_defaults(func=cmd_aggregate)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
