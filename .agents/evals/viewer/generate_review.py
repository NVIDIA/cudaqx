#!/usr/bin/env python3
"""Render an iteration's responses + grades into a single HTML page.

Standalone: no jinja, no jupyter. Just f-strings and a tiny CSS block.
Open the result in a browser to skim: each scenario shows the prompt,
the with_skill response, the without_skill response (if present), and
the verdict from every grader that was run.

Usage::

    python .agents/evals/viewer/generate_review.py \\
        .agents/evals/workspaces/2026-05-07-iter-1 \\
        --out .agents/evals/workspaces/2026-05-07-iter-1/report.html
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Any

EVALS_ROOT = Path(__file__).resolve().parents[1]
PROMPTS_DIR = EVALS_ROOT / "prompts"

# Keep in sync with aggregate.py::GRADING_SCHEMA_VERSION. Duplicated rather
# than imported so the viewer stays standalone.
GRADING_SCHEMA_VERSION = "1"


def _check_schema(path: Path, payload: dict) -> None:
    """Warn (don't fail) when a grading file was produced under a different
    schema. The viewer is best-effort: it'll still render what it can."""
    found = payload.get("schema_version")
    if found is None:
        sys.stderr.write(
            f"warning: {path} has no schema_version field "
            f"(expected {GRADING_SCHEMA_VERSION!r}). Some columns may be blank.\n"
        )
    elif found != GRADING_SCHEMA_VERSION:
        sys.stderr.write(f"warning: {path} schema_version={found!r}, "
                         f"expected {GRADING_SCHEMA_VERSION!r}.\n")


# Skill discovery: each iteration directory carries a `skill.txt` or, failing
# that, we try to pull it from the first grading file.


def _find_skill(iter_dir: Path, configs: dict[str, dict[str, dict]]) -> str:
    sentinel = iter_dir / "skill.txt"
    if sentinel.exists():
        return sentinel.read_text().strip()
    for cfg in configs.values():
        for grading in cfg.values():
            sk = grading.get("skill")
            if sk:
                return sk
    raise SystemExit(
        "Could not infer skill. Add a skill.txt file containing the skill "
        "alias (e.g. 'qec') to the iteration directory.")


def _load_iteration(iter_dir: Path) -> dict[str, dict]:
    """Return {config_name: {responses, gradings}}."""
    out: dict[str, dict] = {}
    for cfg in sorted(iter_dir.iterdir()):
        if not cfg.is_dir():
            continue
        responses_path = cfg / "responses.json"
        timing_path = cfg / "timing.json"
        responses = json.loads(
            responses_path.read_text()) if responses_path.exists() else {}
        timing = json.loads(
            timing_path.read_text()) if timing_path.exists() else {}
        gradings: dict[str, dict] = {}
        for path in sorted(cfg.glob("grading.*.json")):
            grader = path.stem.split(".", 1)[1]
            payload = json.loads(path.read_text())
            _check_schema(path, payload)
            gradings[grader] = payload
        if responses or gradings:
            out[cfg.name] = {
                "responses": responses,
                "gradings": gradings,
                "timing": timing
            }
    return out


def _grader_verdict_html(grader: str, scenario: dict) -> str:
    """Compact one-cell summary of how this grader scored a scenario."""
    if grader == "programmatic":
        cov = f"{scenario.get('coverage', 0)}/{scenario.get('coverage_max', 0)}"
        pur = f"{scenario.get('purity', 0)}/{scenario.get('purity_max', 0)}"
        passed = scenario.get("passed", False)
        cls = "ok" if passed else "fail"
        return f'<span class="pill {cls}">cov {cov} · pur {pur}</span>'
    if grader == "executable":
        st = scenario.get("status", "?")
        cls = {
            "passed": "ok",
            "failed": "fail",
            "no_code": "warn"
        }.get(st, "skip")
        extra = f" exit={scenario.get('exit_code')}" if "exit_code" in scenario else ""
        return f'<span class="pill {cls}">{st}{extra}</span>'
    if grader == "judge":
        st = scenario.get("status")
        if st != "scored":
            return f'<span class="pill skip">{st or "?"}</span>'
        total = scenario.get("total", 0)
        cls = "ok" if total >= 6 else ("warn" if total >= 4 else "fail")
        return (
            f'<span class="pill {cls}">{total}/8 '
            f'(c{scenario["correctness"]} s{scenario["specificity"]} '
            f'cv{scenario["coverage"]} h{scenario["hallucinations"]})</span>')
    return f'<span class="pill skip">{grader}</span>'


CSS = """
body { font-family: -apple-system, system-ui, Segoe UI, sans-serif; margin: 0 auto; max-width: 1100px; padding: 24px; color: #1f2328; }
h1 { font-size: 22px; }
h2 { font-size: 18px; margin-top: 32px; border-top: 1px solid #d0d7de; padding-top: 18px; }
h3 { font-size: 15px; margin-top: 14px; }
table.summary { border-collapse: collapse; margin: 12px 0; }
table.summary th, table.summary td { padding: 4px 12px; border: 1px solid #d0d7de; text-align: left; }
.pill { display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 12px; margin-right: 6px; }
.pill.ok   { background: #dafbe1; color: #116329; }
.pill.fail { background: #ffebe9; color: #82071e; }
.pill.warn { background: #fff8c5; color: #7d4e00; }
.pill.skip { background: #eaeef2; color: #57606a; }
.scenario { border: 1px solid #d0d7de; border-radius: 6px; padding: 12px 16px; margin: 14px 0; }
.scenario .prompt { font-style: italic; color: #57606a; }
.scenario pre { background: #f6f8fa; padding: 12px; border-radius: 6px; overflow-x: auto; font-size: 13px; }
.config-block { margin-top: 8px; }
.config-name { font-weight: 600; color: #0969da; }
"""


def _render(iter_name: str, skill: str, configs: dict[str, dict],
            prompts: list[dict], assertions: dict) -> str:
    parts: list[str] = [
        f"<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>CUDA-QX eval: {html.escape(iter_name)} ({skill})</title>",
        f"<style>{CSS}</style></head><body>",
        f"<h1>{html.escape(skill)} — iteration {html.escape(iter_name)}</h1>",
    ]

    # Summary table.
    parts.append("<h2>Summary</h2>")
    parts.append("<table class='summary'><tr><th>configuration</th>")
    grader_names: list[str] = []
    for cfg in configs.values():
        for g in cfg["gradings"]:
            if g not in grader_names:
                grader_names.append(g)
    for g in grader_names:
        parts.append(f"<th>{html.escape(g)} pass rate</th>")
    parts.append("<th>tokens</th><th>duration</th></tr>")

    def _pass_rate(grader: str, payload: dict) -> str:
        if grader == "programmatic":
            r = payload.get("scenario_pass_rate", 0)
            return f"{r:.0%}"
        if grader == "executable":
            r = payload.get("scenario_pass_rate", 0)
            return f"{r:.0%}"
        if grader == "judge":
            r = payload.get("average_normalized_score", 0)
            return f"{r:.0%} avg"
        return "?"

    for cfg_name, cfg in configs.items():
        parts.append(f"<tr><td>{html.escape(cfg_name)}</td>")
        for g in grader_names:
            payload = cfg["gradings"].get(g, {})
            parts.append(f"<td>{_pass_rate(g, payload)}</td>")
        timing = cfg.get("timing", {})
        parts.append(f"<td>{timing.get('total_tokens', '?')}</td>")
        parts.append(f"<td>{timing.get('duration_ms', '?')} ms</td></tr>")
    parts.append("</table>")

    # Per-scenario.
    parts.append("<h2>Scenarios</h2>")
    scenario_specs = assertions["scenarios"]
    for entry in prompts:
        sid = entry["id"]
        if sid not in scenario_specs:
            continue
        spec = scenario_specs[sid]
        parts.append(f"<div class='scenario'>")
        parts.append(f"<h3>{html.escape(sid)} — "
                     f"{html.escape(entry.get('name') or '')}</h3>")
        parts.append(
            f"<div class='prompt'>{html.escape(entry['prompt'])}</div>")
        for cfg_name, cfg in configs.items():
            response = cfg["responses"].get(sid, "")
            parts.append("<div class='config-block'>")
            parts.append(
                f"<span class='config-name'>{html.escape(cfg_name)}</span>")
            for g in grader_names:
                payload = cfg["gradings"].get(g, {})
                # Find this scenario in the grading payload's scenarios list.
                hit = next(
                    (s for s in payload.get("scenarios", [])
                     if s.get("id") == sid),
                    None,
                )
                if hit:
                    parts.append(
                        f"<br>{html.escape(g)}: {_grader_verdict_html(g, hit)}")
            parts.append(
                f"<pre>{html.escape(response or '(no response)')}</pre>")
            parts.append("</div>")
        # Answer key (collapsed-style, plain).
        must = spec.get("must_include", [])
        must_not = spec.get("must_not_include", [])
        parts.append("<details><summary>Answer key</summary>")
        parts.append("<p><strong>must_include:</strong> " +
                     html.escape(json.dumps(must)) + "</p>")
        parts.append("<p><strong>must_not_include:</strong> " +
                     html.escape(json.dumps(must_not)) + "</p>")
        parts.append("</details>")
        parts.append("</div>")

    parts.append("</body></html>")
    return "".join(parts)


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("iteration_dir", type=Path)
    p.add_argument("--out",
                   type=Path,
                   default=None,
                   help="Output HTML path. Default: <iteration>/report.html")
    args = p.parse_args()

    if not args.iteration_dir.is_dir():
        raise SystemExit(f"Not a directory: {args.iteration_dir}")

    configs = _load_iteration(args.iteration_dir)
    if not configs:
        raise SystemExit("No responses or grading files found.")

    skill = _find_skill(args.iteration_dir, {
        n: c["gradings"] for n, c in configs.items()
    })
    full = {
        "solvers": "cuda-qx-solvers",
        "qec": "cuda-qx-qec",
        "build": "cuda-qx-build"
    }.get(skill, skill)
    prompts = json.loads(
        (PROMPTS_DIR / f"{full}.evals.json").read_text())["evals"]
    assertions = json.loads(
        (EVALS_ROOT / "assertions" / f"{full}.json").read_text())

    html_text = _render(args.iteration_dir.name, full, configs, prompts,
                        assertions)
    out = args.out or args.iteration_dir / "report.html"
    out.write_text(html_text)
    print(f"Wrote {out}")
    print(f"Open with: python3 -m http.server --directory {out.parent} 8001")
    return 0


if __name__ == "__main__":
    sys.exit(main())
