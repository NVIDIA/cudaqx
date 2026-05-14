---
name: "cuda-qx-contributing"
title: "Contributing to CUDA-QX"
description: >-
  Contributor-facing skill for CUDA-QX: DCO sign-off, PR workflow,
  formatting (clang-format, yapf), file-tree layout decisions (where
  new code goes), what to test, what to document, and how to file
  issues. Use whenever the user mentions "send a PR", "contribute",
  "sign off", "DCO", "where do I put this code", "clang-format",
  "yapf format", "open an issue", or otherwise wants to push changes
  upstream.
version: "0.1.3"
author: "CUDA-QX"
license: "Apache License 2.0"
compatibility: "Any platform (git, clang-format, yapf)"
tags: [cuda-qx, contributing, pr-workflow, dco, formatting, clang-format, yapf]
tools: [Read, Glob, Grep, Bash]
metadata:
  repo: [qec, solvers]
  author: "CUDA-QX"
  domain: "contributing"
  audience: [contributor, developer]
  languages: [python, c++, markdown]
---

# Contributing to CUDA-QX

End-to-end skill for sending a change upstream: from "I have a fix"
to "PR merged". Pulls from `Contributing.md`, the layout in
`AGENTS.md`, and what the CI actually checks.

If the user is *building* their change locally, delegate to
`cuda-qx-build`. If they are *testing* it, `cuda-qx-testing-ci`. If
they are writing a *new plugin*, `cuda-qx-qec-extending` (QEC) or
`cuda-qx-solvers-extending` (solvers). This skill
covers the PR workflow itself.

## Inputs

Caller provides:

- A clean code change (committed locally, or about to be).
- Branch name + scope (fix / feat / refactor / docs / ci / test).
- A `git config user.name` and `user.email` that can be public.
- (Optional) related issue number.

## Outputs

This skill produces:

- A signed-off commit series (DCO `-s`).
- Clean `clang-format` + `yapf` + skill-sync `--check` locally.
- A branch pushed to a remote.
- A `gh pr create` with structured Summary + Test plan.

Does NOT produce: actual code fixes (→ domain skills); build artifacts
(→ `cuda-qx-build`); test files (→ `cuda-qx-testing-ci`).

## Audience

External contributors and NVIDIA-internal developers. Familiarity
with git and GitHub PR workflow is assumed.

## First three actions

```bash
git status
git remote -v
bash scripts/sync_agents_skills.sh --check       # CI gate; pass before pushing
```

`git status` confirms you are on a feature branch (not main).
`scripts/sync_agents_skills.sh --check` is one of the CI gates;
running it locally catches the most common preventable CI failure.

## Key Paths

| Area | Path |
|------|------|
| Contribution guide | `Contributing.md` |
| Issue templates / discussions | `https://github.com/NVIDIA/cuda-qx/issues` |
| Format C++ | `scripts/run_clang_format.sh` |
| Format Python | `scripts/run_yapf_format.sh` |
| CI scripts | `scripts/ci/` |
| CMake top-level | `CMakeLists.txt` |
| Build skill | `.agents/skills/cuda-qx-build/SKILL.md` |
| Skill sync | `scripts/sync_agents_skills.sh` |

## Workflow Index

| If the user wants to | Read |
|----------------------|------|
| Sign off commits and submit a PR | `references/pr-workflow.md` |
| Pick where new code goes | `references/file-layout.md` |
| Format C++ and Python correctly | `references/formatting.md` |
| File a useful bug report | `references/issue-reporting.md` |

## Conventions

These prevent the recurring "PR opened, then needs three rounds of
formatting fixes" situations.

1. **DCO sign-off is mandatory.** Every commit must end with
   `Signed-off-by: Your Name <your@email.com>`. Add automatically
   with `git commit -s`. A CLA bot will tag PRs missing this.

2. **Branch from `main`, PR to `main`.** No long-lived feature
   branches.

3. **One logical change per PR.** Big PRs are reviewed badly. If you
   added a new decoder *and* refactored the build, that's two PRs.

4. **Format before pushing.** `bash scripts/run_clang_format.sh` and
   `bash scripts/run_yapf_format.sh`. CI fails on formatting
   diffs.

5. **Sync skills if you edited any.** `bash scripts/sync_agents_skills.sh`
   regenerates `.claude/skills/` and `.cursor/skills/`. CI runs
   `--check` to enforce.

6. **Tests for new features, regression tests for fixes.** A PR
   without tests is rarely merged.

7. **Update docs for user-visible changes.** New decoders / pools /
   codes need a docs page in
   `docs/sphinx/components/<lib>/` or
   `docs/sphinx/examples_rst/<lib>/`.

8. **Don't paper over license boundaries.** `libs/qec/LICENSE` is
   NVIDIA-proprietary; the rest is Apache-2.0. New QEC code must be
   compatible with the QEC license; new solvers code with Apache.

## Quick start: fix-a-bug PR

```bash
git checkout -b fix/my-decoder-bug
# ... edit ...
bash scripts/run_clang_format.sh
bash scripts/run_yapf_format.sh
git add libs/qec/lib/decoders/my_decoder.cpp libs/qec/python/tests/test_my_decoder.py
git commit -s -m "fix: decoder returns wrong shape for empty syndrome"
git push -u origin fix/my-decoder-bug
gh pr create --base main --fill
```

The `-s` flag adds the DCO sign-off. `--fill` reuses your commit
message as the PR body.

## Self-Check Protocol

```
[ ] On a feature branch (not main).
[ ] DCO sign-off in every commit.
[ ] clang-format and yapf both clean.
[ ] If you touched .agents/skills/, ran sync_agents_skills.sh.
[ ] CI green (or at least, ran locally what CI would run).
[ ] Tests added for new feature; regression test for bug fix.
[ ] Docs updated if a user-visible thing changed.
[ ] Commit messages reference an issue if one exists.
```

## When stuck

1. Read `Contributing.md` for the canonical contribution flow.
2. Read the matching `references/<topic>.md`.
3. Look at recent merged PRs on
   <https://github.com/NVIDIA/cuda-qx/pulls?q=is%3Apr+is%3Amerged> for
   patterns: how big is "small enough"?
4. If you broke CI, read `cuda-qx-testing-ci` SKILL.md to find the
   right local test.
5. For build issues, delegate to `cuda-qx-build`.

## Additional resources

- `references/pr-workflow.md` — git flow, sign-off, gh CLI.
- `references/file-layout.md` — where new code goes.
- `references/formatting.md` — clang-format and yapf details.
- `references/issue-reporting.md` — anatomy of a useful bug report.
- Build / test: `cuda-qx-build`, `cuda-qx-testing-ci`.
- Plugin authoring: `cuda-qx-qec-extending`, `cuda-qx-solvers-extending`.
- Skill authoring (you're editing AGENTS-side files): `cuda-qx-skills-authoring`.
