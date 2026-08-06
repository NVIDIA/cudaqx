# Add a new target agent (Copilot, Gemini, Windsurf, …)

The sync script's design lets you onboard a new agent in one edit.

## Step 1: pick the discovery path

Each agent tool looks for skills under a specific path. Examples:

| Agent | Discovery path |
|-------|----------------|
| Codex | `.agents/skills/<name>/SKILL.md` (canonical, AGENTS.md convention) |
| Claude Code | `.claude/skills/<name>/SKILL.md` |
| Cursor | `.cursor/skills/<name>/SKILL.md` |
| Copilot | (check docs) |
| Gemini CLI | (check docs) |
| Windsurf | (check docs) |

Check the agent's documentation for the correct path. If multiple
paths are accepted, pick the one closest to the AGENTS.md
convention.

## Step 2: extend `scripts/sync_agents_skills.sh`

```diff
 TARGETS=(
   ".claude/skills:Claude Code"
   ".cursor/skills:Cursor"
+  ".copilot/skills:GitHub Copilot"
 )
```

Each entry is `<relative-path>:<agent-display-name>`. The display
name appears in the `.GENERATED` sentinel and in stderr messages.

## Step 3: gitignore the new mirror

```diff
 # .gitignore
 .claude/skills/
 .cursor/skills/
+.copilot/skills/
```

Mirrors are generated; never tracked.

## Step 4: regenerate

```bash
bash scripts/sync_agents_skills.sh
```

Verify the new mirror appeared:

```bash
ls .copilot/skills/
```

Each subdirectory should be a full copy of `.agents/skills/<name>/`
plus a `.GENERATED` sentinel.

## Step 5: update AGENTS.md

Add a row to the "Where skills live" table in `AGENTS.md`:

```markdown
| Copilot     | `.copilot/skills/<name>/SKILL.md`    | Generated mirror                         |
```

## Step 6: smoke test

Open the agent (Copilot, etc.) in a checkout of the repo and verify
it discovers the skills. If discovery works, you're done. If not:

- Path mismatch (some agents need an exact filename like
  `skills.md` instead of `SKILL.md`; check the agent's docs).
- Frontmatter format incompatibility (some agents require specific
  YAML fields; map them in a future iteration of the sync script if
  needed).

## Step 7: CI

CI runs `sync_agents_skills.sh --check`. As long as the script
itself doesn't fail (`bash -n scripts/sync_agents_skills.sh`), the
new target works in CI without further changes.

## Path-rewriting (when *might* it be needed?)

The current sync is a pure copy. Some agent tools might require
agent-specific paths in skill content (e.g.,
`.copilot/skills/_shared/...` instead of `.agents/skills/_shared/...`).

If that becomes a requirement:

1. Extend `_sync_one` in `sync_agents_skills.sh` to do a sed pass
   per-target.
2. Update the `description:` of every skill that mentions
   `.agents/skills/` to reflect this.
3. Document the rewrite rules in `cudaq-skills-authoring/SKILL.md`.

This is intentionally deferred — it adds maintenance burden and isn't
needed today.

## Common pitfalls

| Symptom | Cause |
|---------|-------|
| Mirror created but not picked up by the agent | wrong path; re-read the agent's docs |
| Mirror not created at all | `TARGETS` entry typo (missing colon, etc.) |
| Mirror tracked in git accidentally | gitignore line missing |
| `_check_one` reports stale immediately after sync | filesystem case-sensitivity or stray ignored files; check `IGNORE_NAMES` |

## Self-check

```
[ ] TARGETS entry added.
[ ] .gitignore updated.
[ ] sync_agents_skills.sh runs without error.
[ ] New mirror directory contains all skills + .GENERATED file.
[ ] AGENTS.md updated.
[ ] Smoke-tested in the actual agent tool.
```

## Where next

- Add or edit content: `new-skill.md`, `edit-skill.md`.
- Add evals to keep regressions out: `evals.md`.
