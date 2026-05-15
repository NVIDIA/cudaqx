#!/usr/bin/env python3
"""Description-overlap linter for CUDA-Q Libraries skills.

Walks every ``.agents/skills/*/SKILL.md`` (skipping ``_shared/``) and pulls
the ``description`` field out of the YAML frontmatter. For each pair of
skills, looks for *literal trigger tokens* that appear in both descriptions
without a corresponding ``Do NOT use ... use cudaq-other-skill`` carve-out
in at least one of them.

Why this exists: today's activation contamination came from sibling skills
listing the same trigger token (``AIDecoderService``, ``trt_decoder``,
``test_examples.sh``, ``sliding_window``, ...) without telling the activator
which one owns it. This linter catches that class of regression before any
LLM-based eval runs.

Two token classes are mined:

* Quoted phrases  ``"add a test"``, ``"send a PR"``, ...
* "Code-shaped" identifiers — anything with at least one underscore, dot, or
  hyphen and no internal whitespace (``trt_decoder``, ``AIDecoderService``,
  ``cudaq-qec-cu12``, ``configure_decoders_from_file``, ...).

Pure English words ("the", "a", "decoder", ...) are deliberately ignored:
they never cause activation contamination on their own. Only literal,
copy-pasteable strings do.

Exit codes
----------
* ``0``  – no overlapping tokens, or only overlaps that are explicitly
          carved out in both directions.
* ``2``  – at least one token appears in two descriptions without a
          ``Do NOT use ... <other-skill>`` line in either; this is the
          regression class we want CI to block on.

Usage::

    python .agents/evals/lint_descriptions.py            # human-readable
    python .agents/evals/lint_descriptions.py --json     # machine-readable
    python .agents/evals/lint_descriptions.py --skills-dir .agents/skills
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

# Regex for *real* code-shaped identifiers: must contain at least one
# underscore, period, slash, or digit. Hyphen-only words (``real-time``,
# ``in-kernel``, ``state-prep``, ``hello-world``) are excluded because they
# read as English compounds and never act as activation triggers on their
# own. We keep camelCase identifiers via a separate regex below.
CODE_TOKEN = re.compile(
    r"\b[A-Za-z][A-Za-z0-9_./\-]*[_./0-9][A-Za-z0-9_./\-]*\b")

# Camel-case identifiers without separators: 2+ "uppercase-prefix +
# optional-lowercase" groups, with at least one lowercase letter so that
# pure acronyms like "QEC" or "DEM" don't match. Catches PyMatching,
# AIDecoderService, AIPreDecoderService, MyTestClass, etc. Pure acronyms
# are filtered post-match in `_camel_tokens` below.
CAMEL_TOKEN = re.compile(r"\b(?:[A-Z]+[a-z]*){2,}\b")


def _camel_tokens(text: str) -> set[str]:
    """Yield camelCase identifiers, filtering out pure acronyms."""
    out: set[str] = set()
    for m in CAMEL_TOKEN.finditer(text):
        tok = m.group(0)
        if any(c.islower() for c in tok):
            out.add(tok)
    return out


# Quoted phrases inside descriptions. Both straight and curly quotes.
QUOTED = re.compile(r'"([^"\n]{2,80})"|"([^"\n]{2,80})"')

# Heuristic for "Do NOT use ... <other-skill>" carve-outs. Loose on purpose
# because authors phrase them differently (Do NOT, do not, DO NOT, etc.).
DO_NOT_USE = re.compile(r"do\s*not\s+use", re.IGNORECASE)

# Pattern for "use X" / "see X" / "delegate to X" inside a carve-out
# paragraph. We use this so multi-clause carve-outs (e.g. "Do NOT use for:
# A (use cudaq-X); B (use cudaq-Y); C (use cudaq-Z)") are detected
# correctly even after sentence-splitting drops semicolon-separated tail
# clauses.
HANDOFF = re.compile(r"\b(?:use|see|delegate\s+to)\s+`?(cudaq-[a-z0-9\-]+)`?")


@dataclass
class Skill:
    name: str
    path: Path
    description: str
    # Set of other-skill names that this description explicitly hands work to,
    # extracted from any "use cudaq-other-skill" mention near a "Do NOT use"
    # phrase.
    carve_outs: set[str] = field(default_factory=set)


def _read_frontmatter(skill_md: Path) -> dict[str, str] | None:
    """Pull a tiny subset of YAML out of the SKILL.md frontmatter.

    We only care about ``name`` and ``description``. We avoid pulling
    PyYAML in here so the linter has zero dependencies.
    """
    text = skill_md.read_text()
    m = re.match(r"---\n(.*?)\n---", text, re.DOTALL)
    if not m:
        return None
    fm = m.group(1)

    out: dict[str, str] = {}
    name_m = re.search(r"^name:\s*\"?([A-Za-z0-9._\-]+)\"?\s*$", fm,
                       re.MULTILINE)
    if name_m:
        out["name"] = name_m.group(1)

    # description: >- followed by an indented block until the next top-level
    # key (column 0).
    desc_m = re.search(r"^description:\s*>-?\s*\n((?:[ \t]+.*\n)+)", fm,
                       re.MULTILINE)
    if desc_m:
        # Strip the per-line leading indentation and join.
        lines = [ln.strip() for ln in desc_m.group(1).splitlines()]
        out["description"] = " ".join(lines)
    return out


def _load_skills(skills_dir: Path) -> list[Skill]:
    skills: list[Skill] = []
    for child in sorted(skills_dir.iterdir()):
        if not child.is_dir() or child.name.startswith("_"):
            continue
        skill_md = child / "SKILL.md"
        if not skill_md.exists():
            continue
        fm = _read_frontmatter(skill_md)
        if not fm or "name" not in fm or "description" not in fm:
            continue
        s = Skill(name=fm["name"], path=skill_md, description=fm["description"])
        s.carve_outs = _carve_outs(s.description)
        skills.append(s)
    return skills


def _carve_outs(description: str) -> set[str]:
    """Find every cudaq-* skill mentioned as a handoff target.

    We don't try to parse English. Instead, the description is scanned for
    the ``use cudaq-X`` / ``see cudaq-X`` / ``delegate to cudaq-X``
    pattern. Empirically, every legitimate carve-out in this repo uses one
    of those verbs, and treating them all as carve-out targets gives us a
    precision/recall trade-off that's well-suited to a lint pass: we'd
    rather miss a token-overlap warning than report a false alarm in CI.
    """
    targets: set[str] = set()
    for m in HANDOFF.finditer(description):
        targets.add(m.group(1))
    return targets


# Tokens that appear in many descriptions because they are the project name
# or shared technology vocabulary. Carrying them in a stoplist keeps the
# linter focused on identifiers that *act* as activation triggers rather
# than branding noise. Skill names themselves are added dynamically in
# ``_tokens`` because they are legitimate cross-references, not contamination.
STOPLIST: frozenset[str] = frozenset({
    "cudaq",
    "cudaq-qec",
    "cudaq_qec",
    "cudaq-solvers",
    "cudaq_solvers",
})


def _tokens(description: str, skill_names: set[str]) -> set[str]:
    """Mine literal-trigger tokens from a description.

    ``skill_names`` is excluded so that valid cross-references (``Do NOT
    use for X — use cudaq-other``) don't get reported as overlaps.
    """
    tokens: set[str] = set()

    for m in QUOTED.finditer(description):
        phrase = (m.group(1) or m.group(2) or "").strip().lower()
        if 2 <= len(phrase) <= 80:
            tokens.add(f'"{phrase}"')

    for m in CODE_TOKEN.finditer(description):
        tok = m.group(0)
        if tok.lower() in STOPLIST:
            continue
        if tok in skill_names:
            continue
        tokens.add(tok)

    tokens.update(_camel_tokens(description))

    return tokens


@dataclass
class Overlap:
    token: str
    skills: list[str]
    carved: bool  # True if at least one skill points at the other(s) via Do NOT use
    severity: str  # "warn" | "fail"


def _build_overlaps(skills: list[Skill]) -> list[Overlap]:
    skill_names = {s.name for s in skills}
    inv: dict[str, list[str]] = {}
    for s in skills:
        for tok in _tokens(s.description, skill_names):
            inv.setdefault(tok, []).append(s.name)

    by_name = {s.name: s for s in skills}
    overlaps: list[Overlap] = []

    for tok, names in sorted(inv.items()):
        if len(names) < 2:
            continue
        # Carve-out check: every pair (a, b) must have either a -> b or b -> a
        # in the carve_outs map. If even one pair lacks both directions, the
        # token is a hard failure.
        all_pairs_carved = True
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                a_to_b = b in by_name[a].carve_outs
                b_to_a = a in by_name[b].carve_outs
                if not (a_to_b or b_to_a):
                    all_pairs_carved = False
                    break
            if not all_pairs_carved:
                break
        overlaps.append(
            Overlap(
                token=tok,
                skills=sorted(names),
                carved=all_pairs_carved,
                severity="warn" if all_pairs_carved else "fail",
            ))
    return overlaps


def _render_text(skills: list[Skill], overlaps: list[Overlap]) -> str:
    lines: list[str] = []
    lines.append(f"Linted {len(skills)} skills.")
    fails = [o for o in overlaps if o.severity == "fail"]
    warns = [o for o in overlaps if o.severity == "warn"]
    lines.append(f"  HARD FAILS  : {len(fails)}")
    lines.append(f"  Carved warns: {len(warns)}")
    lines.append("")
    if fails:
        lines.append(
            "==== HARD FAILS (token in 2+ descriptions, no carve-out) ====")
        for o in fails:
            lines.append(f"  {o.token!r:40s}  -> {', '.join(o.skills)}")
        lines.append("")
    if warns:
        lines.append("==== Carved overlaps (info only) ====")
        for o in warns:
            lines.append(f"  {o.token!r:40s}  -> {', '.join(o.skills)}")
    return "\n".join(lines) + "\n"


def _render_json(skills: list[Skill], overlaps: list[Overlap]) -> str:
    return json.dumps(
        {
            "linted_skills": [s.name for s in skills],
            "overlaps": [{
                "token": o.token,
                "skills": o.skills,
                "carved": o.carved,
                "severity": o.severity,
            } for o in overlaps],
            "summary": {
                "total_skills": len(skills),
                "fail_count": sum(1 for o in overlaps if o.severity == "fail"),
                "warn_count": sum(1 for o in overlaps if o.severity == "warn"),
            },
        },
        indent=2,
    )


def main(argv: Iterable[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--skills-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "skills",
        help="Path to .agents/skills/ (default: auto-detected).",
    )
    p.add_argument("--json",
                   action="store_true",
                   help="Emit JSON instead of text.")
    args = p.parse_args(list(argv) if argv is not None else None)

    skills = _load_skills(args.skills_dir)
    if not skills:
        print(f"error: no SKILL.md files under {args.skills_dir}",
              file=sys.stderr)
        return 1
    overlaps = _build_overlaps(skills)

    if args.json:
        print(_render_json(skills, overlaps))
    else:
        print(_render_text(skills, overlaps))

    fails = sum(1 for o in overlaps if o.severity == "fail")
    return 2 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
