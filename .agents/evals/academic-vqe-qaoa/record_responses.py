#!/usr/bin/env python3
"""Capture multiline eval responses into a run JSON file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

END_MARKER = "<<<END>>>"


def load_run(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict) or not isinstance(payload.get("responses"),
                                                       list):
        raise SystemExit(f"Expected a run JSON with a responses list: {path}")
    return payload


def read_multiline(prompt_id: str) -> str:
    print(
        f"Paste response for {prompt_id}. Finish with a line containing {END_MARKER}."
    )
    lines: list[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line == END_MARKER:
            break
        lines.append(line)
    return "\n".join(lines).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Run JSON file to update.")
    parser.add_argument(
        "--only-empty",
        action="store_true",
        help="Skip responses that already contain text.",
    )
    args = parser.parse_args()

    payload = load_run(args.run)
    for item in payload["responses"]:
        prompt_id = item.get("id", "unknown")
        if args.only_empty and str(item.get("response", "")).strip():
            continue

        print()
        print(f"== {prompt_id}: {item.get('name', '')} ==")
        print(item.get("prompt", ""))
        print()
        response = read_multiline(prompt_id)
        if response:
            item["response"] = response

    args.run.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Updated {args.run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
