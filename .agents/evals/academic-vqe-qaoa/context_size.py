#!/usr/bin/env python3
"""Measure the *context size* each path pulls in, with a real tokenizer.

The dominant, controllable difference between answering a workshop prompt
"with the skill" vs "without the skill" is which files get pulled into the
model's context:

* With skill   -> SKILL.md (routing) + ONE curated reference + the prompt.
* Without skill -> the raw repo source files an agent must read to
                   reconstruct the same answer + the prompt.

This script tokenizes those exact files with ``tiktoken`` (``cl100k_base`` --
the BPE GPT/Claude approximate), so the numbers are real token counts of real
bytes, not API-meter guesses. It does NOT include the system prompt, tool
scaffolding, or model reasoning -- only file context, which is the term the
skill actually changes.

Usage::

    pip install tiktoken
    python context_size.py            # table for all prompts
    python context_size.py --json     # machine-readable

No API key or agent runtime required.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import tiktoken

EVAL_DIR = Path(__file__).resolve().parent
REPO = EVAL_DIR.parents[2]  # .agents/evals/academic-vqe-qaoa -> repo root
SKILL_DIR = REPO / ".agents/skills/cudaq-academic-vqe-qaoa"
SKILL_MD = SKILL_DIR / "SKILL.md"
PROMPTS = EVAL_DIR / "prompts.json"

ENC = tiktoken.get_encoding("cl100k_base")

# prompt id -> (route label, curated reference the skill loads)
ROUTE = {
    "P1": ("qaoa", "references/qaoa.md"),
    "P2": ("qaoa", "references/qaoa.md"),
    "P3": ("qaoa", "references/qaoa.md"),
    "P4": ("adapt", "references/adapt.md"),
    "P5": ("gqe", "references/gqe.md"),
    "P6": ("install", "references/install.md"),
    "P7": ("qaoa", "references/qaoa.md"),   # MaxCut via recognition table
    "P8": ("vqe", "references/vqe.md"),
}

# route -> repo source files a no-skill agent reads to reconstruct the answer.
# Seeded from SKILL.md's own "Source Of Truth" list and the files the no-skill
# subagent runs actually opened (e.g. molecular_docking_qaoa.py + test_qaoa.py
# for the MaxCut/QAOA prompts).
NO_SKILL_SOURCES = {
    "qaoa": [
        "docs/sphinx/examples/solvers/python/molecular_docking_qaoa.py",
        "libs/solvers/python/tests/test_qaoa.py",
    ],
    "vqe": [
        "docs/sphinx/examples/solvers/python/uccsd_vqe.py",
        "libs/solvers/python/tests/test_vqe.py",
    ],
    "adapt": [
        "docs/sphinx/examples/solvers/python/uccsd_vqe.py",
        "libs/solvers/python/tests/test_vqe.py",
    ],
    "gqe": [
        "docs/sphinx/examples/solvers/python/gqe_h2.py",
        "libs/solvers/python/tests/test_gqe.py",
    ],
    "install": [
        "docs/sphinx/quickstart/installation.rst",
    ],
}


def count_tokens(path: Path) -> int:
    return len(ENC.encode(path.read_text(encoding="utf-8", errors="ignore")))


def count_text(text: str) -> int:
    return len(ENC.encode(text))


def load_prompts() -> dict[str, str]:
    return {p["id"]: p["prompt"] for p in json.loads(PROMPTS.read_text())}


def measure(prompt_id: str, prompt_text: str) -> dict:
    route, ref = ROUTE[prompt_id]
    skill_tok = count_tokens(SKILL_MD)
    ref_path = SKILL_DIR / ref
    ref_tok = count_tokens(ref_path) if ref_path.exists() else 0
    prompt_tok = count_text(prompt_text)
    with_skill = skill_tok + ref_tok + prompt_tok

    no_skill_files = []
    no_skill_tok = prompt_tok
    for rel in NO_SKILL_SOURCES.get(route, []):
        p = REPO / rel
        if p.exists():
            t = count_tokens(p)
            no_skill_files.append({"file": rel, "tokens": t})
            no_skill_tok += t

    return {
        "id": prompt_id,
        "route": route,
        "reference": ref,
        "with_skill_tokens": with_skill,
        "with_skill_breakdown": {
            "SKILL.md": skill_tok, ref: ref_tok, "prompt": prompt_tok,
        },
        "no_skill_tokens": no_skill_tok,
        "no_skill_files": no_skill_files,
        "ratio": round(no_skill_tok / with_skill, 2) if with_skill else None,
    }


def response_tokens(files: list[Path]) -> int:
    """Tokenize one or more response text files; print per-file + mean."""
    counts = []
    for f in files:
        if not f.exists():
            print(f"{f}: (missing)")
            continue
        t = count_tokens(f)
        counts.append(t)
        print(f"{t:>7}  {f.name}")
    if counts:
        print("-" * 30)
        print(f"{round(sum(counts) / len(counts)):>7}  mean ({len(counts)} files)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true",
                        help="Emit machine-readable JSON.")
    parser.add_argument("--response-tokens", nargs="+", type=Path, default=None,
                        help="Tokenize response text file(s) and report counts.")
    args = parser.parse_args()

    if args.response_tokens:
        return response_tokens(args.response_tokens)

    prompts = load_prompts()
    rows = [measure(pid, prompts[pid]) for pid in ROUTE if pid in prompts]

    if args.json:
        print(json.dumps(rows, indent=2))
        return 0

    print(f"Tokenizer: cl100k_base   SKILL.md = {count_tokens(SKILL_MD)} tokens\n")
    print(f"{'id':<4}{'route':<9}{'with skill':>11}{'no skill':>10}{'ratio':>8}"
          f"   reference")
    print("-" * 70)
    ws_tot = ns_tot = 0
    for r in rows:
        ws_tot += r["with_skill_tokens"]
        ns_tot += r["no_skill_tokens"]
        print(f"{r['id']:<4}{r['route']:<9}{r['with_skill_tokens']:>11}"
              f"{r['no_skill_tokens']:>10}{r['ratio']:>7}x   {r['reference']}")
    print("-" * 70)
    overall = round(ns_tot / ws_tot, 2) if ws_tot else 0
    print(f"{'ALL':<4}{'':<9}{ws_tot:>11}{ns_tot:>10}{overall:>7}x")
    print("\nwith skill = SKILL.md + 1 reference + prompt")
    print("no skill   = repo source files for the topic + prompt")
    print("(file context only; excludes system prompt / tool scaffolding)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
