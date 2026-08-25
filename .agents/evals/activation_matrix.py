#!/usr/bin/env python3
"""Cross-skill activation matrix for CUDA-Q Libraries skills.

Loads an *owner-tagged* prompt corpus and predicts, for each prompt, which
skill(s) Claude's keyword-based activation step would load. The output is a
matrix with one row per prompt and one column per skill; cell ``True`` means
the skill is predicted to activate.

The activation predictor here is intentionally a *cheap heuristic* — token
overlap between the prompt and the skill's *positive* description (with the
"Do NOT use" sentences stripped out). It will not match a real LLM
activator perfectly, but it gives us a deterministic CI-friendly regression
test that catches the contamination class we just fixed (sibling skills
sharing literal trigger tokens like ``trt_decoder`` or ``AIDecoderService``).

For deeper audits use the sub-agent flow described in
``.agents/evals/README.md`` — that flow also produces an activation matrix
but with an LLM doing the scoring.

The corpus lives in ``activation_corpus.json`` next to this script. Each
entry has::

    {
      "id": "AM01",
      "prompt": "the user-facing question",
      "owner": "cudaq-qec-realtime",          # canonical sole owner, or null
      "allowed_co_activators": ["cudaq-..."]   # other skills that may also fire
    }

If ``owner`` is ``null`` the prompt is interpreted as "no skill should fire"
(used to test that off-domain questions don't leak into any skill).

Exit codes
----------
* ``0`` — no leaks (a leak = a non-allowed skill is predicted to activate
          alongside the owner).
* ``2`` — at least one prompt has a leak. This is the only failure mode CI
          should block on.
* ``3`` — only when ``--strict`` is passed: also fail on misses (owner not
          predicted to activate). Misses are usually heuristic noise — the
          token-overlap matcher can't infer ``"8ms"`` → ``"high latency"``
          the way a real LLM activator can — so they should be reviewed by a
          human, not block the build.

Usage::

    python .agents/evals/activation_matrix.py             # text matrix
    python .agents/evals/activation_matrix.py --json      # JSON output
    python .agents/evals/activation_matrix.py --threshold 2
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

# Reuse the description tokeniser from the lint script: same regexes, same
# stoplist, same "what counts as a trigger token" rule. Keep them in sync.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lint_descriptions import (  # noqa: E402
    CODE_TOKEN, DO_NOT_USE, QUOTED, STOPLIST, _camel_tokens, _read_frontmatter,
)

EVALS_ROOT = Path(__file__).resolve().parent
SKILLS_DIR = EVALS_ROOT.parent / "skills"
CORPUS_PATH = EVALS_ROOT / "activation_corpus.json"


@dataclass
class Skill:
    name: str
    triggers: set[str] = field(default_factory=set)


def _strip_negatives(description: str) -> str:
    """Drop everything from the first ``Do NOT use`` to the end of the
    description.

    Skill descriptions in this repo follow a consistent shape: a positive
    section listing trigger tokens and use cases, then a ``Do NOT use this
    skill for: ...; ...; ...`` paragraph that hands edge cases to other
    skills. The negative paragraph contains many literal trigger tokens
    (``test_examples.sh``, ``trt_decoder``, ...) which are *meant* to be
    suppressors but, as a literal-substring matcher, the heuristic can't
    tell them apart from positive triggers. Treating the entire tail of
    the description as off-limits is the simplest reliable fix.
    """
    m = DO_NOT_USE.search(description)
    if m:
        return description[:m.start()]
    return description


def _trigger_tokens(positive_desc: str) -> set[str]:
    tokens: set[str] = set()
    for m in QUOTED.finditer(positive_desc):
        phrase = (m.group(1) or m.group(2) or "").strip().lower()
        if 2 <= len(phrase) <= 80:
            tokens.add(phrase)
    for m in CODE_TOKEN.finditer(positive_desc):
        tok = m.group(0)
        if tok.lower() in STOPLIST:
            continue
        tokens.add(tok.lower())
    for tok in _camel_tokens(positive_desc):
        tokens.add(tok.lower())
    return tokens


def _load_skills() -> list[Skill]:
    out: list[Skill] = []
    for child in sorted(SKILLS_DIR.iterdir()):
        if not child.is_dir() or child.name.startswith("_"):
            continue
        skill_md = child / "SKILL.md"
        if not skill_md.exists():
            continue
        fm = _read_frontmatter(skill_md)
        if not fm or "description" not in fm:
            continue
        positive = _strip_negatives(fm["description"])
        out.append(Skill(name=fm["name"], triggers=_trigger_tokens(positive)))
    return out


def _matches(prompt: str, triggers: set[str]) -> int:
    """Count how many trigger tokens appear (case-insensitively) in the prompt."""
    pl = prompt.lower()
    hits = 0
    for tok in triggers:
        if tok in pl:
            hits += 1
    return hits


@dataclass
class Row:
    id: str
    prompt: str
    owner: str | None
    allowed: set[str]
    hits: dict[str, int]  # skill -> trigger-hit count
    activated: list[str]  # skills predicted to activate (>= threshold)
    leaks: list[str]  # activated skills not in (owner + allowed)
    miss: bool  # True if owner expected but did not activate


def _score(
    skills: list[Skill],
    corpus: list[dict],
    threshold: int,
) -> list[Row]:
    rows: list[Row] = []
    for entry in corpus:
        prompt = entry["prompt"]
        owner = entry.get("owner")
        allowed = set(entry.get("allowed_co_activators", []))
        hits = {s.name: _matches(prompt, s.triggers) for s in skills}
        activated = sorted(
            [name for name, n in hits.items() if n >= threshold],
            key=lambda n: -hits[n],
        )
        leaks = [n for n in activated if n != owner and n not in allowed]
        miss = owner is not None and owner not in activated
        rows.append(
            Row(
                id=entry["id"],
                prompt=prompt,
                owner=owner,
                allowed=allowed,
                hits=hits,
                activated=activated,
                leaks=leaks,
                miss=miss,
            ))
    return rows


def _render_text(skills: list[Skill], rows: list[Row]) -> str:
    out: list[str] = []
    n_leaks = sum(1 for r in rows if r.leaks)
    n_miss = sum(1 for r in rows if r.miss)
    out.append(f"Activation matrix: {len(rows)} prompts x {len(skills)} skills")
    out.append(f"  Leaks (off-target activation): {n_leaks}")
    out.append(f"  Misses (owner did not activate): {n_miss}")
    out.append("")

    for r in rows:
        owner_label = r.owner or "—"
        status: list[str] = []
        if r.miss:
            status.append("MISS")
        if r.leaks:
            status.append(f"LEAK({len(r.leaks)})")
        if not status:
            status.append("ok")
        out.append(f"{r.id}  [{','.join(status):10s}]  owner={owner_label}")
        out.append(
            f"   prompt: {r.prompt[:100]}{'...' if len(r.prompt)>100 else ''}")
        if r.activated:
            top = ", ".join(f"{n}({r.hits[n]})" for n in r.activated[:5])
            out.append(f"   fired : {top}")
        if r.leaks:
            out.append(f"   leak  : {', '.join(r.leaks)}")
        out.append("")
    return "\n".join(out)


def _render_json(skills: list[Skill], rows: list[Row]) -> str:
    return json.dumps(
        {
            "skills": [s.name for s in skills],
            "rows": [{
                "id": r.id,
                "prompt": r.prompt,
                "owner": r.owner,
                "allowed": sorted(r.allowed),
                "activated": r.activated,
                "leaks": r.leaks,
                "miss": r.miss,
                "hits": r.hits,
            } for r in rows],
            "summary": {
                "total_prompts": len(rows),
                "leaks": sum(1 for r in rows if r.leaks),
                "misses": sum(1 for r in rows if r.miss),
            },
        },
        indent=2,
    )


def main(argv: Iterable[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--corpus",
        type=Path,
        default=CORPUS_PATH,
        help="Path to activation_corpus.json (default: alongside this script).",
    )
    p.add_argument(
        "--threshold",
        type=int,
        default=2,
        help=("Minimum number of trigger-token hits a skill needs to count as "
              "activated. Default: 2 (one hit is too weak; two hits is the "
              "empirical sweet spot for this corpus)."),
    )
    p.add_argument("--json",
                   action="store_true",
                   help="Emit JSON instead of text.")
    p.add_argument(
        "--strict",
        action="store_true",
        help=
        "Also exit non-zero when the heuristic fails to predict the owner activating (misses).",
    )
    args = p.parse_args(list(argv) if argv is not None else None)

    if not args.corpus.exists():
        print(f"error: corpus not found at {args.corpus}", file=sys.stderr)
        return 1
    corpus = json.loads(args.corpus.read_text())
    if not isinstance(corpus, list):
        print(f"error: corpus must be a JSON array of prompt entries",
              file=sys.stderr)
        return 1

    skills = _load_skills()
    rows = _score(skills, corpus, threshold=args.threshold)

    if args.json:
        print(_render_json(skills, rows))
    else:
        print(_render_text(skills, rows))

    leaks = sum(1 for r in rows if r.leaks)
    misses = sum(1 for r in rows if r.miss)
    if leaks:
        return 2
    if misses and args.strict:
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
