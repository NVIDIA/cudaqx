# Add a new skill

The end-to-end procedure for adding a new top-level skill to
`.agents/skills/`.

## When to add a skill (vs a reference under an existing one)

A new top-level skill is justified when:

- A new persona / audience appears (you find yourself writing
  "audience: ... and also ..." instead of "audience: ...").
- An existing skill's `references/` folder grows past ~4 files.
- The workflows are independent (a user landing on workflow A
  shouldn't be required to read workflow B's context).

If none of these apply, add a reference file to an existing skill
instead.

## Directory layout

```
.agents/skills/cudaq-MYSKILL/
├── SKILL.md           # required: frontmatter + body
└── references/
    ├── topic-a.md     # 2-4 typical
    ├── topic-b.md
    └── topic-c.md
```

No other files inside the skill directory. (No image assets, no
scripts inside the skill — those go in `_shared/scripts/` if shared
or are kept out of the canonical tree.)

## Frontmatter template

Copy from an existing skill and adapt. Required keys (see
parent SKILL.md). Pay attention to:

- `name`: kebab-case, matches the directory name exactly.
- `description`: ~5 sentences. Include the trigger vocabulary the
  user will type. End with a "do NOT use this for X" line.
- `tags`: short list, lowercase.
- `metadata.audience`: list — `[student, researcher, developer,
  contributor, ...]`. Used by humans, not by routing.

## SKILL.md body structure

Mimic existing skills in order:

1. **Title and one-paragraph intro** — what this skill is, what it
   is *not*. Delegation to related skills explicitly.
2. **Audience** — who this is for. One paragraph.
3. **First three actions** — preflight, smoke, intent picker. Keep
   the same three across skills so users build muscle memory.
4. **Key Paths** — a table mapping concept → file path. Verify every
   path with grep.
5. **Source of Truth** — (optional) where to look up API names
   instead of guessing.
6. **Workflow Index** — table mapping user intent → reference file.
7. **Conventions** — numbered list of recurring silent bugs.
8. **Quick Start** — the smallest end-to-end procedure.
9. **Self-Check Protocol** — a checklist before declaring done.
10. **When stuck** — escape hatches.
11. **Additional resources** — links to references + related skills.

## References

Each reference is one focused page (~200-500 lines). Internal
structure:

```
# Topic title

One-paragraph intro.

## Authoritative template / source of truth

`path/to/file`.

## (Workflow body — procedures, tables, code snippets, pitfalls)

## Self-check

```
[ ] checklist of items
```

## Where next

- related skill links (peer skills, not internal references)
```

## After creating

```bash
bash scripts/sync_agents_skills.sh
# Verify the tracked mirrors got it
ls .claude/skills/cudaq-MYSKILL/
ls .cursor/skills/cudaq-MYSKILL/
```

Then update `AGENTS.md` "Available skills" table.

```bash
git add .agents/skills/cudaq-MYSKILL .claude/skills/cudaq-MYSKILL .cursor/skills/cudaq-MYSKILL AGENTS.md
git commit -s -m "skill: add cudaq-MYSKILL"
```

## Test the skill in a real agent session

Open a Claude Code (or Cursor) session and:

1. Type one of the trigger phrases from your `description`.
2. Confirm the skill loads (look for it in the available-skills
   list).
3. Walk the first workflow.
4. If anything is unclear, that's a doc gap — edit the skill and
   re-sync.

## Common pitfalls

| Symptom | Cause |
|---------|-------|
| Skill doesn't load | misspelled `name` in frontmatter; mismatch with dir name |
| Skill loads but description is empty | YAML formatting issue; check the `>-` block style |
| Mirror missing the new skill | forgot to run `sync_agents_skills.sh` |
| Mirror has it but CI fails `--check` | committed a stale mirror or forgot to stage regenerated mirror files |
| Reference paths are wrong | mass copy-paste from a peer skill without updating |

## Self-check

```
[ ] Frontmatter validates (no YAML errors).
[ ] All paths in SKILL.md and references/ exist (grep to check).
[ ] At least 2 references.
[ ] AGENTS.md updated.
[ ] sync_agents_skills.sh --check passes.
[ ] Regenerated `.claude/skills/` and `.cursor/skills/` mirrors staged.
[ ] Loaded the skill in a real session and walked one workflow.
```

## Where next

- Edit conventions for existing skills: `edit-skill.md`.
- Add an eval: `evals.md`.
- Onboard another agent target: `add-target-agent.md`.
