# Edit an existing skill

Safe patterns and pitfalls for modifying skill content.

## Where to edit

Always `.agents/skills/`. Never the mirrors.

```
EDIT  .agents/skills/<name>/SKILL.md            # canonical, tracked
EDIT  .agents/skills/<name>/references/*.md     # canonical, tracked
NEVER .claude/skills/<name>/...                 # mirror, gitignored
NEVER .cursor/skills/<name>/...                 # mirror, gitignored
```

Editing a mirror has no effect on tracked content and will be
overwritten the next time `sync_agents_skills.sh` runs.

## After every edit

```bash
bash scripts/sync_agents_skills.sh
```

If you forget, CI's `--check` invocation will fail.

## Pre-commit hook (recommended)

```bash
cat >> .git/hooks/pre-commit <<'EOF'
bash scripts/sync_agents_skills.sh --check
EOF
```

This blocks commits where the mirror is stale.

## Safe edit patterns

### Add a new reference file

```bash
$EDITOR .agents/skills/<name>/references/new-topic.md
# Link from SKILL.md's "Workflow Index" and "Additional resources"
$EDITOR .agents/skills/<name>/SKILL.md
bash scripts/sync_agents_skills.sh
```

### Rename a skill

```bash
git mv .agents/skills/old-name .agents/skills/new-name
# Update frontmatter `name:` field inside SKILL.md
$EDITOR .agents/skills/new-name/SKILL.md
# Search for cross-references
grep -rl "old-name" .agents/skills/
# Update each
# Update AGENTS.md
bash scripts/sync_agents_skills.sh
```

The mirror script removes the old directory automatically because
it does a full `rm -rf` + `cp -r` each run.

### Bump version (policy)

Edit `version:` in the frontmatter. SemVer with skill-specific meaning:

| Change | Bump | Examples |
|---|---|---|
| **Major** (X.0.0) | Breaking | Skill renamed, workflow removed, frontmatter `name:` changed, `Inputs:` contract changed in a non-backwards-compatible way (caller has to update) |
| **Minor** (0.X.0) | Additive | New reference file, new workflow added, new conventions, new I/O option (callers can ignore) |
| **Patch** (0.0.X) | Editorial | Typo fixes, wording cleanup, path corrections, no behavioral change |

Rules:

1. **Bump on every PR that touches a SKILL.md or its references** —
   even patch bumps. The version is the human signal that something
   changed; agents may cache by version.
2. **Start new skills at `0.1.0`**, not `1.0.0`. Promote to `1.0.0`
   only when the I/O contract is stable enough that downstream tools
   should pin to it.
3. **Don't normalize versions across skills.** Each skill's version
   reflects its own edit history. If skill A is at `0.3.0` and skill B
   is at `0.1.0`, that's fine — they evolved independently.
4. **No CI gate today.** The version field is convention only; CI does
   not block on a missing bump. Reviewers should still flag PRs that
   modify a skill without bumping.

Cross-checks before merging a skill edit:

- `version:` bumped according to the table above.
- `sync_agents_skills.sh --check` passes.
- Frontmatter still validates as YAML.

## Refactoring

If you find yourself adding a fourth reference file, ask: is this
really a reference inside the parent skill, or warrants its own skill?

Split when:

- Audience differs (e.g. "students" vs "researchers").
- The new content is independent of the parent's other workflows.
- The split would let users skip half the content they don't need.

Don't split when:

- Content is fundamentally part of the parent workflow.
- The split would force readers to bounce between two skills to
  understand one thing.

## Cross-references

When you add a new skill in the same domain, audit existing skills for places
that *should* link to it. Two patterns:

- "Delegate to" — explicit redirect at the start or end of a
  workflow.
- "Additional resources" — bullet list at the bottom of SKILL.md.

Stale cross-references are the most common rot. Re-grep when
renaming:

```bash
grep -rl "cuda-qx-OLDNAME" .agents/ AGENTS.md README.md
```

## Pitfalls

| Symptom | Cause |
|---------|-------|
| You edited the mirror, sync wiped it | always edit `.agents/skills/` |
| CI says mirrors stale, you swear you ran sync | uncommitted edit in `.agents/skills/`; `git status` to check |
| Skill name changed but routing model still picks old name | model cache; nothing to do but wait or change description |
| Frontmatter formatting broke (no skill loads) | use a YAML validator: `python -c "import yaml,sys; yaml.safe_load(open(sys.argv[1]))" SKILL.md` |
| Forgot to add the new reference to the Workflow Index | reader can't find it; agents won't autonomously discover it |

## Self-check

```
[ ] Edited the canonical (.agents/skills/) location only.
[ ] Ran sync_agents_skills.sh.
[ ] sync_agents_skills.sh --check passes.
[ ] All paths still resolve (no stale references).
[ ] Cross-references updated.
[ ] If you renamed: AGENTS.md updated.
```

## Where next

- Add a brand-new skill: `new-skill.md`.
- Add an eval to test the edited skill: `evals.md`.
- Onboard another agent: `add-target-agent.md`.
