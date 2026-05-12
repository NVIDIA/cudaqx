#!/usr/bin/env python3
"""Aggregate per-iteration grading: pass-rates, deltas, inter-grader Cohen's κ.

Run AFTER you have grading.*.json files in each configuration directory
of an iteration::

    python .claude/evals/aggregate.py .claude/evals/workspaces/<iter>/

Two outputs:

* ``benchmark.json`` (or whatever you set with --out) — combined
  pass-rates per grader per configuration, plus the with-vs-without
  delta. Same shape as ``runner.py aggregate`` produces, but with
  judge/executable graders included.
* Inter-grader agreement on the with_skill configuration: pairwise
  Cohen's κ between programmatic / executable / judge. Disagreement is
  exactly where the rubric or the skill needs sharpening.

Cohen's κ
---------

For each pair of graders, label every scenario "pass" or "fail"
according to that grader's own rule, then compute:

  κ = (po - pe) / (1 - pe)

  po = observed fraction of agreement
  pe = expected fraction of agreement by chance, given each rater's
       marginal pass rate

κ ∈ [-1, 1]. >0.6 is usually called "substantial agreement"; <0.4
"poor". When two graders disagree systematically, fix the rubric or
add scenarios where they could agree.
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Any


def _grader_pass(grader: str, scenario: dict) -> bool | None:
    """Per-grader rule for what counts as a 'pass' on a single scenario.

    Returns None when the grader didn't score the scenario (skipped /
    not_configured / parse_error etc.).
    """
    if grader == "programmatic":
        if "passed" not in scenario:
            return None
        return bool(scenario["passed"])
    if grader == "executable":
        st = scenario.get("status")
        if st in (None, "skipped", "not_configured"):
            return None
        return st == "passed"
    if grader == "judge":
        st = scenario.get("status")
        if st != "scored":
            return None
        return scenario.get("total", 0) >= 6  # 6/8 rubric points
    return None


def _cohen_kappa(a: list[bool | None], b: list[bool | None]) -> dict[str, Any]:
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    n = len(pairs)
    if n == 0:
        return {"kappa": None, "n": 0, "agreement": None}
    agree = sum(1 for x, y in pairs if x == y)
    po = agree / n
    pa = sum(1 for x, _ in pairs if x) / n
    pb = sum(1 for _, y in pairs if y) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    kappa = (po - pe) / (1 - pe) if pe < 1.0 else 1.0
    return {
        "kappa": round(kappa, 3),
        "n": n,
        "agreement": round(po, 3),
        "rater_a_pass_rate": round(pa, 3),
        "rater_b_pass_rate": round(pb, 3),
    }


def _load_iteration(iter_dir: Path) -> dict[str, dict[str, dict]]:
    """Return {config_name: {grader_name: grading_dict}}."""
    out: dict[str, dict[str, dict]] = {}
    for cfg in sorted(iter_dir.iterdir()):
        if not cfg.is_dir():
            continue
        cfg_record: dict[str, dict] = {}
        for path in sorted(cfg.glob("grading.*.json")):
            grader = path.stem.split(".", 1)[1]
            cfg_record[grader] = json.loads(path.read_text())
        if cfg_record:
            out[cfg.name] = cfg_record
    return out


def _benchmark(configs: dict[str, dict[str, dict]]) -> dict[str, Any]:
    bench: dict[str, Any] = {"configurations": {}}

    for cfg_name, graders in configs.items():
        cfg_out: dict[str, Any] = {}
        for grader, payload in graders.items():
            scenarios = payload.get("scenarios", [])
            verdicts = [_grader_pass(grader, s) for s in scenarios]
            considered = [v for v in verdicts if v is not None]
            cfg_out[grader] = {
                "scenarios_total": len(scenarios),
                "scenarios_considered": len(considered),
                "pass_rate":
                    (sum(considered) / len(considered) if considered else 0.0),
                "raw": payload,
            }
        bench["configurations"][cfg_name] = cfg_out

    if "with_skill" in configs and "without_skill" in configs:
        deltas: dict[str, Any] = {}
        for grader in configs["with_skill"]:
            if grader not in configs["without_skill"]:
                continue
            w = bench["configurations"]["with_skill"][grader]["pass_rate"]
            b = bench["configurations"]["without_skill"][grader]["pass_rate"]
            deltas[grader] = round(w - b, 3)
        bench["delta_pass_rate"] = deltas

    return bench


def _agreement(configs: dict[str, dict[str, dict]]) -> dict[str, Any]:
    target = configs.get("with_skill") or next(iter(configs.values()), {})
    if not target:
        return {}

    grader_verdicts: dict[str, list[bool | None]] = {}
    for grader, payload in target.items():
        verdicts = [
            _grader_pass(grader, s) for s in payload.get("scenarios", [])
        ]
        grader_verdicts[grader] = verdicts

    pairs: dict[str, dict[str, Any]] = {}
    for a, b in combinations(grader_verdicts, 2):
        pairs[f"{a} ↔ {b}"] = _cohen_kappa(grader_verdicts[a],
                                           grader_verdicts[b])
    return {
        "configuration":
            "with_skill" if "with_skill" in configs else next(iter(configs)),
        "pairwise_kappa":
            pairs,
    }


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("iteration_dir",
                   type=Path,
                   help="Path to the iteration directory.")
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output benchmark.json path. Default: <iteration>/benchmark.json")
    args = p.parse_args()

    if not args.iteration_dir.is_dir():
        raise SystemExit(f"Not a directory: {args.iteration_dir}")

    configs = _load_iteration(args.iteration_dir)
    if not configs:
        raise SystemExit("No grading.*.json files found in any configuration.")

    bench = _benchmark(configs)
    bench["inter_grader_agreement"] = _agreement(configs)
    bench["iteration"] = args.iteration_dir.name

    out = args.out or args.iteration_dir / "benchmark.json"
    out.write_text(json.dumps(bench, indent=2))

    print(f"Wrote {out}")
    print()
    print("Pass rates:")
    for cfg, graders in bench["configurations"].items():
        print(f"  [{cfg}]")
        for g, info in graders.items():
            print(f"    {g:<14} {info['pass_rate']:>5.0%}  "
                  f"({info['scenarios_considered']}/{info['scenarios_total']})")
    if "delta_pass_rate" in bench:
        print()
        print("Δ pass-rate (with_skill - without_skill):")
        for g, d in bench["delta_pass_rate"].items():
            print(f"  {g:<14} {d:+.0%}")
    if bench["inter_grader_agreement"].get("pairwise_kappa"):
        print()
        print(
            f"Inter-grader Cohen's κ ({bench['inter_grader_agreement']['configuration']}):"
        )
        for pair, k in bench["inter_grader_agreement"]["pairwise_kappa"].items(
        ):
            kv = k.get("kappa")
            ks = "n/a" if kv is None else f"{kv:+.2f}"
            print(f"  {pair:<32}  κ = {ks}  (n={k['n']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
