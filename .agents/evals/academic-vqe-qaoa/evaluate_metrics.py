#!/usr/bin/env python3
"""Evaluate academic VQE/QAOA skill responses.

This is intentionally small and deterministic. It scores answer text against
substring assertions and rolls up context/runtime metrics when a run file
contains them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
DEFAULT_ASSERTIONS = ROOT / "assertions.json"


def load_assertions(path: Path) -> dict[str, dict[str, list[str]]]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise SystemExit(f"Assertions must be a JSON object: {path}")
    return data


def normalize_run(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    config = payload.get("config") if isinstance(payload, dict) else None
    config = config or path.stem
    agent = payload.get("agent") if isinstance(payload, dict) else None
    model = payload.get("model") if isinstance(payload, dict) else None

    records: dict[str, dict[str, Any]] = {}
    if isinstance(payload, dict) and isinstance(payload.get("responses"), list):
        for item in payload["responses"]:
            if not isinstance(item, dict) or "id" not in item:
                continue
            records[str(item["id"])] = dict(item)
    elif isinstance(payload, dict) and isinstance(payload.get("responses"),
                                                  dict):
        for key, value in payload["responses"].items():
            if isinstance(value, dict):
                rec = dict(value)
                rec.setdefault("id", key)
                records[key] = rec
            else:
                records[key] = {"id": key, "response": str(value)}
    elif isinstance(payload, dict):
        for key, value in payload.items():
            if key in {"agent", "model", "config", "metrics", "notes"}:
                continue
            records[key] = {"id": key, "response": str(value)}
    else:
        raise SystemExit(f"Unsupported response JSON shape: {path}")

    return {
        "path": str(path),
        "agent": agent or "unknown",
        "model": model or "",
        "config": config,
        "records": records,
    }


def contains(text: str, needle: str) -> bool:
    return needle.lower() in text.lower()


def score_record(prompt_id: str, record: dict[str, Any],
                 spec: dict[str, list[str]]) -> dict[str, Any]:
    text = str(record.get("response", ""))
    must = spec.get("must_include", [])
    must_not = spec.get("must_not_include", [])
    missing = [item for item in must if not contains(text, item)]
    forbidden = [item for item in must_not if contains(text, item)]

    context_files = record.get("context_files", [])
    if not isinstance(context_files, list):
        context_files = []

    raw_tokens = record.get("tokens")
    if isinstance(raw_tokens, dict):
        numeric = [
            v for v in raw_tokens.values() if isinstance(v, (int, float))
        ]
        token_total = int(sum(numeric)) if numeric else None
    elif isinstance(raw_tokens, (int, float)):
        token_total = int(raw_tokens)
    else:
        token_total = None

    return {
        "id": prompt_id,
        "passed": not missing and not forbidden,
        "coverage": len(must) - len(missing),
        "coverage_max": len(must),
        "missing": missing,
        "forbidden": forbidden,
        "context_files": len(context_files),
        "duration_ms": record.get("duration_ms"),
        "tokens": token_total,
    }


def summarize(run: dict[str, Any], assertions: dict[str,
                                                    dict]) -> dict[str, Any]:
    scores = []
    for prompt_id, spec in assertions.items():
        record = run["records"].get(prompt_id, {
            "id": prompt_id,
            "response": ""
        })
        scores.append(score_record(prompt_id, record, spec))

    coverage = sum(s["coverage"] for s in scores)
    coverage_max = sum(s["coverage_max"] for s in scores)
    forbidden_hits = sum(len(s["forbidden"]) for s in scores)

    def sum_known(field: str) -> int | None:
        values = [s[field] for s in scores if s.get(field) is not None]
        return sum(int(v) for v in values) if values else None

    return {
        "agent": run.get("agent", "unknown"),
        "model": run.get("model", ""),
        "config": run["config"],
        "path": run["path"],
        "prompt_count": len(scores),
        "passed": sum(1 for s in scores if s["passed"]),
        "pass_rate": (sum(1 for s in scores if s["passed"]) /
                      len(scores) if scores else 0.0),
        "coverage": coverage,
        "coverage_max": coverage_max,
        "coverage_rate": coverage / coverage_max if coverage_max else 0.0,
        "forbidden_hits": forbidden_hits,
        "context_files": sum(s["context_files"] for s in scores),
        "duration_ms": sum_known("duration_ms"),
        "tokens_total": sum_known("tokens"),
        "scores": scores,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("responses",
                        nargs="+",
                        type=Path,
                        help="One or more response JSON files to score.")
    parser.add_argument("--assertions",
                        type=Path,
                        default=DEFAULT_ASSERTIONS,
                        help="Assertions JSON path.")
    parser.add_argument("--out", type=Path, default=None, help="Write JSON.")
    parser.add_argument("--json",
                        action="store_true",
                        help="Print full JSON instead of text summary.")
    args = parser.parse_args()

    assertions = load_assertions(args.assertions)
    summaries = [
        summarize(normalize_run(path), assertions) for path in args.responses
    ]
    result = {"assertions": str(args.assertions), "runs": summaries}

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))

    if args.json:
        print(json.dumps(result, indent=2))
        return 0

    for summary in summaries:
        tok = summary.get("tokens_total")
        tok_str = f"tokens={tok} " if tok is not None else ""
        print(f"[{summary['agent']}:{summary['config']}] "
              f"pass_rate={summary['pass_rate']:.0%} "
              f"coverage={summary['coverage']}/{summary['coverage_max']} "
              f"forbidden={summary['forbidden_hits']} "
              f"context_files={summary['context_files']} "
              f"{tok_str}".rstrip())
        for score in summary["scores"]:
            if not score["passed"]:
                print(f"  - {score['id']}: missing={score['missing']} "
                      f"forbidden={score['forbidden']}")

    if args.out:
        print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
