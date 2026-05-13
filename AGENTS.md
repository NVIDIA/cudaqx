# Agent Instructions

This repository ships skills for AI coding agents that work with CUDA-QX.
The skills cover building/packaging the libraries, quantum error correction
(QEC) workflows, and solver (VQE / ADAPT-VQE / QAOA / GQE) workflows.

This file is the entry point for any agent. Agent-specific loaders find
the skills in their conventional location (see "Where skills live" below);
the actual skill content is identical across locations.

## Where skills live

| Agent       | Discovery path                       | Notes                                    |
|-------------|--------------------------------------|------------------------------------------|
| Codex       | `.agents/skills/<name>/SKILL.md`     | Canonical — edit here. Open AGENTS.md convention. |
| Claude Code | `.claude/skills/<name>/SKILL.md`     | Generated mirror                         |
| Cursor      | `.cursor/skills/<name>/SKILL.md`     | Generated mirror                         |

The canonical content lives at **`.agents/skills/`** — the open
AGENTS.md convention that Codex (and other tools) discover natively. The
`.claude/skills/` and `.cursor/skills/` trees are **generated locally**
and gitignored, so only the canonical copy is tracked. Run the sync
once after cloning to populate them:

```bash
bash scripts/sync_agents_skills.sh
```

After that, edit `.agents/skills/` and re-run the sync whenever you
change a skill — `.claude/skills/` and `.cursor/skills/` are
regenerated each time.

To add a fourth agent (Copilot, Gemini, Windsurf, etc.), append one
entry to `TARGETS` in `scripts/sync_agents_skills.sh`. No skill
content needs to be duplicated by hand.

## Available skills

| Skill              | What it covers                                                              |
|--------------------|------------------------------------------------------------------------------|
| `cuda-qx-build`    | cmake / ninja / wheels / docs / dev container / release validation          |
| `cuda-qx-qec`      | decoders, codes (Steane / surface / repetition), DEMs, real-time decoding   |
| `cuda-qx-solvers`  | VQE, ADAPT-VQE, QAOA, GQE, molecular Hamiltonians, PySCF chemistry          |

Each `SKILL.md` carries its own activation triggers in the YAML
frontmatter, a workflow index, conventions to follow, and a self-check
protocol. Start there.

## Shared resources

| Need                                            | Path                                           |
|-------------------------------------------------|-------------------------------------------------|
| Preflight / import-smoke / workflow picker      | `.agents/skills/_shared/scripts/`               |
| Repo map (where things live in the source tree) | `.agents/skills/_shared/repo_map.md`            |
| Evaluation harness (prompts, assertions, graders)| `.agents/evals/`                               |
| Top-level build / format / validation scripts   | `scripts/`                                      |

## Editing the skills

1. Edit files under `.agents/skills/<name>/`.
2. Run `bash scripts/sync_agents_skills.sh` to regenerate the
   local `.claude/skills/` and `.cursor/skills/` mirrors so your
   active agent picks up the change immediately.
3. Commit only the changes under `.agents/skills/` — the mirrors
   are gitignored.
