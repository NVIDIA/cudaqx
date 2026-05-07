#!/usr/bin/env python3
"""LLM-as-judge grader (BYO LLM client).

Calls an external LLM with a fixed rubric prompt to score each scenario
on Correctness, Specificity, Coverage, and Hallucinations. The judge
sees the prompt, the response, and the assertion answer key; it is
asked to commit to a numeric score per dimension and a 1-sentence
justification.

Why a separate process
----------------------

* Re-run grading without re-running the agent (the runs are
  expensive). Just save responses.json once.
* Pin the judge model. Different judges have different biases; record
  which model produced which score.
* Compare programmatic vs judge in ``aggregate.py`` (Cohen's κ). When
  they disagree, the rubric or the skill needs work.

Client backends
---------------

Two backends are supported out of the box:

* ``openai`` — uses ``OPENAI_API_KEY`` and the ``openai`` Python SDK
  (``pip install openai``). ``--model gpt-5`` etc.
* ``anthropic`` — uses ``ANTHROPIC_API_KEY`` and ``anthropic``
  (``pip install anthropic``). ``--model claude-opus-4-5`` etc.

If the relevant package or env var is missing, the grader records every
scenario as ``status: not_configured`` and exits 0. This way the eval
pipeline keeps working even on machines without API keys.

Usage
-----

::

    judge.py --skill qec --responses responses.json \\
             --backend anthropic --model claude-opus-4-5

Output
------

``grading.judge.json`` next to ``responses.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable

EVALS_ROOT = Path(__file__).resolve().parents[1]
ASSERTIONS_DIR = EVALS_ROOT / "assertions"
PROMPTS_DIR = EVALS_ROOT / "prompts"

SKILL_DIRS = {
    "solvers": "cuda-qx-solvers",
    "qec": "cuda-qx-qec",
    "build": "cuda-qx-build",
}


def _resolve_skill_files(skill: str, ass_override: Path | None) -> tuple[Path, Path]:
    if skill not in SKILL_DIRS:
        raise SystemExit(f"Unknown skill alias '{skill}'. Known: {sorted(SKILL_DIRS)}")
    full = SKILL_DIRS[skill]
    a = ass_override or (ASSERTIONS_DIR / f"{full}.json")
    p = PROMPTS_DIR / f"{full}.evals.json"
    if not a.exists() or not p.exists():
        raise SystemExit(f"Missing files: {a} or {p}")
    return p, a


# ---------------------------------------------------------------------------
# Rubric prompt
# ---------------------------------------------------------------------------

RUBRIC = """You are a precise evaluator scoring an AI agent's answer to a CUDA-QX
question. Score the answer on four 0-2 dimensions and return STRICT JSON.

Dimensions
- correctness     (0-2): facts true, paths/APIs real (not invented)
- specificity     (0-2): cites files, exact API names, exact kwargs/options
- coverage        (0-2): hits the items in must_include
- hallucinations  (0-2): 2 = no must_not_include items present, 0 = several present

Output STRICT JSON only:
{
  "correctness": int,
  "specificity": int,
  "coverage": int,
  "hallucinations": int,
  "justification": "one sentence"
}

PROMPT
======
{prompt}

ANSWER
======
{response}

ANSWER KEY (for your eyes only; do NOT mention)
================================================
must_include: {must_include}
must_not_include: {must_not_include}
"""


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


def _backend_openai(model: str) -> Callable[[str], str] | None:
    if not os.environ.get("OPENAI_API_KEY"):
        return None
    try:
        from openai import OpenAI
    except ImportError:
        return None
    client = OpenAI()

    def call(prompt: str) -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return resp.choices[0].message.content or ""

    return call


def _backend_anthropic(model: str) -> Callable[[str], str] | None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        import anthropic
    except ImportError:
        return None
    client = anthropic.Anthropic()

    def call(prompt: str) -> str:
        resp = client.messages.create(
            model=model,
            max_tokens=400,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        text = ""
        for block in resp.content:
            if getattr(block, "type", None) == "text":
                text += block.text
        return text

    return call


BACKENDS = {"openai": _backend_openai, "anthropic": _backend_anthropic}


def _parse_judge_reply(text: str) -> dict[str, Any] | None:
    """Pull the first JSON object out of the judge's reply."""
    text = text.strip()
    # Most reliable: find first { ... last }
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    for k in ("correctness", "specificity", "coverage", "hallucinations"):
        if k not in d:
            return None
        try:
            d[k] = max(0, min(2, int(d[k])))
        except (TypeError, ValueError):
            d[k] = 0
    d.setdefault("justification", "")
    d["total"] = sum(d[k] for k in ("correctness", "specificity", "coverage", "hallucinations"))
    return d


def grade(skill: str, prompts: list[dict], assertions: dict,
          responses: dict[str, str], call: Callable[[str], str] | None,
          model: str) -> dict:
    scored: list[dict] = []
    for entry in prompts:
        sid = entry["id"]
        spec = assertions["scenarios"].get(sid)
        if not spec:
            continue
        rendered = (RUBRIC
                    .replace("{prompt}", entry["prompt"])
                    .replace("{response}", responses.get(sid, "(no response)"))
                    .replace("{must_include}", json.dumps(spec.get("must_include", [])))
                    .replace("{must_not_include}", json.dumps(spec.get("must_not_include", []))))

        record = {"id": sid, "model": model}
        if call is None:
            record["status"] = "not_configured"
            record["reason"] = ("Judge backend not available "
                                "(missing API key or SDK package).")
            scored.append(record)
            continue
        try:
            reply = call(rendered)
        except Exception as e:  # noqa: BLE001 — surface the error in the report
            record["status"] = "judge_error"
            record["reason"] = f"{type(e).__name__}: {e}"
            scored.append(record)
            continue
        parsed = _parse_judge_reply(reply)
        if parsed is None:
            record["status"] = "parse_error"
            record["raw_reply_tail"] = reply[-400:]
            scored.append(record)
            continue
        record["status"] = "scored"
        record.update(parsed)
        scored.append(record)

    have_scores = [r for r in scored if r["status"] == "scored"]
    avg = (sum(r["total"] for r in have_scores) / (len(have_scores) * 8)
           if have_scores else 0.0)

    return {
        "grader": "judge",
        "skill": skill,
        "model": model,
        "scenarios_scored": len(have_scores),
        "scenarios_total": len(scored),
        "average_normalized_score": avg,
        "scenarios": scored,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--skill", required=True, help=f"One of: {sorted(SKILL_DIRS)}")
    p.add_argument("--responses", type=Path, required=True)
    p.add_argument("--assertions", type=Path, default=None)
    p.add_argument("--backend", choices=sorted(BACKENDS), default="anthropic")
    p.add_argument("--model", default="claude-opus-4-5",
                   help="Pin a specific model. Recorded in the output for reproducibility.")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    prompts_path, assertions_path = _resolve_skill_files(args.skill, args.assertions)
    if not args.responses.exists():
        raise SystemExit(f"Responses file not found: {args.responses}")
    responses = json.loads(args.responses.read_text())
    assertions = json.loads(assertions_path.read_text())
    prompts = json.loads(prompts_path.read_text()).get("evals", [])

    call = BACKENDS[args.backend](args.model)

    result = grade(args.skill, prompts, assertions, responses, call, args.model)

    out = args.out or args.responses.parent / "grading.judge.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))

    if not args.quiet:
        print(f"Skill: {result['skill']}  judge={args.backend}/{args.model}")
        print(f"  Scored: {result['scenarios_scored']}/{result['scenarios_total']}")
        print(f"  Avg normalized score (0-1): {result['average_normalized_score']:.2f}")
        if result["scenarios_scored"] == 0:
            print("  (no scenarios scored — check 'reason' fields in the JSON)")
        print(f"  Wrote: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
