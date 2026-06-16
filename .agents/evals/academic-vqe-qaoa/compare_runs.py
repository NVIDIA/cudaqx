#!/usr/bin/env python3
"""Initialize and compare academic VQE/QAOA eval runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from evaluate_metrics import DEFAULT_ASSERTIONS, load_assertions, normalize_run, summarize

ROOT = Path(__file__).resolve().parent
DEFAULT_PROMPTS = ROOT / "prompts.json"
DEFAULT_RUNS = ROOT / "runs"

SKILL = ".agents/skills/cudaq-academic-vqe-qaoa/SKILL.md"
REFERENCE_BY_PROMPT = {
    "INSTALL01": ".agents/skills/cudaq-academic-vqe-qaoa/references/install.md",
    "VQE01": ".agents/skills/cudaq-academic-vqe-qaoa/references/vqe.md",
    "QAOA01": ".agents/skills/cudaq-academic-vqe-qaoa/references/qaoa.md",
    "QAOA02": ".agents/skills/cudaq-academic-vqe-qaoa/references/qaoa.md",
}


def load_prompts(path: Path) -> list[dict[str, Any]]:
    prompts = json.loads(path.read_text())
    if not isinstance(prompts, list):
        raise SystemExit(f"Prompts must be a JSON array: {path}")
    return prompts


def response_template(prompt: dict[str, Any],
                      with_skill: bool) -> dict[str, Any]:
    prompt_id = prompt["id"]
    context_files = []
    if with_skill:
        context_files = [SKILL, REFERENCE_BY_PROMPT.get(prompt_id, SKILL)]

    return {
        "id": prompt_id,
        "name": prompt.get("name", ""),
        "prompt": prompt["prompt"],
        "response": "",
        "context_files": context_files,
        "duration_ms": None,
    }


def cmd_init(args: argparse.Namespace) -> int:
    prompts = load_prompts(args.prompts)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    created: list[Path] = []
    for config in ("with_skill", "without_skill"):
        path = args.out_dir / f"{args.agent}_{config}.json"
        if path.exists() and not args.force:
            print(f"exists, skipping: {path}")
            continue
        payload = {
            "agent":
                args.agent,
            "model":
                args.model,
            "config":
                config,
            "notes": ("Fill response text and runtime/context metrics after "
                      "running each prompt. Leave unavailable fields null."),
            "responses": [
                response_template(p, config == "with_skill") for p in prompts
            ],
        }
        path.write_text(json.dumps(payload, indent=2))
        created.append(path)

    for path in created:
        print(f"created: {path}")
    if not created:
        print("No files created. Pass --force to overwrite templates.")
    return 0


def numeric_delta(with_value: Any, without_value: Any) -> int | float | None:
    if with_value is None or without_value is None:
        return None
    return with_value - without_value


def pair_key(summary: dict[str, Any]) -> tuple[str, str]:
    return (summary.get("agent", "unknown"), summary.get("model", ""))


def summarize_pairs(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for summary in summaries:
        grouped.setdefault(pair_key(summary), {})[summary["config"]] = summary

    pairs: list[dict[str, Any]] = []
    for (agent, model), configs in sorted(grouped.items()):
        with_skill = configs.get("with_skill")
        without_skill = configs.get("without_skill")
        if not with_skill or not without_skill:
            pairs.append({
                "agent": agent,
                "model": model,
                "status": "missing_pair",
                "available_configs": sorted(configs),
            })
            continue

        pairs.append({
            "agent": agent,
            "model": model,
            "status": "paired",
            "with_skill_path": with_skill["path"],
            "without_skill_path": without_skill["path"],
            "delta": {
                "pass_rate":
                    numeric_delta(with_skill["pass_rate"],
                                  without_skill["pass_rate"]),
                "coverage_rate":
                    numeric_delta(with_skill["coverage_rate"],
                                  without_skill["coverage_rate"]),
                "forbidden_hits":
                    numeric_delta(with_skill["forbidden_hits"],
                                  without_skill["forbidden_hits"]),
                "context_files":
                    numeric_delta(with_skill["context_files"],
                                  without_skill["context_files"]),
                "duration_ms":
                    numeric_delta(with_skill["duration_ms"],
                                  without_skill["duration_ms"]),
            },
        })
    return pairs


def candidate_run_files(run_dir: Path) -> list[Path]:
    return sorted(path for path in run_dir.glob("*.json")
                  if not path.name.endswith("-summary.json") and
                  path.name != "comparison-summary.json")


def cmd_compare(args: argparse.Namespace) -> int:
    assertions = load_assertions(args.assertions)
    paths = args.responses or candidate_run_files(args.run_dir)
    if not paths:
        raise SystemExit(f"No response JSON files found in {args.run_dir}")

    summaries = [summarize(normalize_run(path), assertions) for path in paths]
    result = {
        "assertions": str(args.assertions),
        "runs": summaries,
        "pairs": summarize_pairs(summaries),
    }

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))

    if args.json:
        print(json.dumps(result, indent=2))
        return 0

    for summary in summaries:
        print(f"[{summary['agent']}:{summary['config']}] "
              f"pass_rate={summary['pass_rate']:.0%} "
              f"coverage={summary['coverage']}/{summary['coverage_max']} "
              f"forbidden={summary['forbidden_hits']} "
              f"context_files={summary['context_files']}")

    print()
    print("Pair deltas (with_skill - without_skill):")
    for pair in result["pairs"]:
        label = f"{pair['agent']}/{pair['model'] or 'model-unset'}"
        if pair["status"] != "paired":
            print(
                f"  {label}: missing pair; available={pair['available_configs']}"
            )
            continue
        delta = pair["delta"]
        print(
            f"  {label}: pass_rate={delta['pass_rate']:+.0%} "
            f"coverage={delta['coverage_rate']:+.0%} "
            f"forbidden={delta['forbidden_hits']:+} "
            f"context_files={delta['context_files']:+} "
            f"duration_ms={delta['duration_ms'] if delta['duration_ms'] is not None else 'n/a'}"
        )

    if args.out:
        print(f"Wrote {args.out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    init = sub.add_parser("init", help="Create editable run JSON templates.")
    init.add_argument("--agent",
                      required=True,
                      help="codex, claude, cursor, etc.")
    init.add_argument("--model", default="", help="Model label if known.")
    init.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    init.add_argument("--out-dir", type=Path, default=DEFAULT_RUNS)
    init.add_argument("--force",
                      action="store_true",
                      help="Overwrite templates.")
    init.set_defaults(func=cmd_init)

    compare = sub.add_parser("compare",
                             help="Score runs and compute pair deltas.")
    compare.add_argument(
        "responses",
        nargs="*",
        type=Path,
        help="Specific run JSON files. Defaults to run-dir/*.json.")
    compare.add_argument("--run-dir", type=Path, default=DEFAULT_RUNS)
    compare.add_argument("--assertions", type=Path, default=DEFAULT_ASSERTIONS)
    compare.add_argument("--out",
                         type=Path,
                         default=DEFAULT_RUNS / "comparison-summary.json")
    compare.add_argument("--json", action="store_true")
    compare.set_defaults(func=cmd_compare)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
