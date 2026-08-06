# PR workflow

The end-to-end from "I have a change" to "PR merged". Specific to
CUDA-Q Libraries (DCO sign-off, skill sync, license split).

## DCO sign-off

Every commit must end with:

```
Signed-off-by: Your Name <your@email.com>
```

Add with the `-s` flag:

```bash
git commit -s -m "feat: add X"
```

Configure git so `Your Name` and email match the public profile you
want associated with the PR. If you already committed without
sign-off:

```bash
git commit --amend --no-edit -s
```

For a series of commits without sign-off, use interactive rebase:

```bash
git rebase --signoff main
```

A CLA bot will block the merge until DCO is satisfied on every
commit.

## Branch + push + open PR

```bash
git checkout main && git pull
git checkout -b <type>/<short-description>
# ... edit, commit ...
bash scripts/run_clang_format.sh
bash scripts/run_yapf_format.sh
bash scripts/sync_agents_skills.sh        # only if you touched .agents/skills/
git add <files>
git commit -s -m "..."
git push -u origin <branch>
gh pr create --base main \
  --title "..." \
  --body "$(cat <<'EOF'
## Summary
- ...

## Test plan
- [ ] ...
EOF
)"
```

Branch naming conventions used in the repo:

| Prefix | Use |
|--------|-----|
| `fix/` | bug fix |
| `feat/` | new feature |
| `refactor/` | no behavior change |
| `docs/` | docs only |
| `ci/` | CI / scripts |
| `test/` | tests only |

## PR body conventions

A useful PR body has three sections:

1. **Summary** — what changed and why. Not what files changed
   (that's in the diff).
2. **Test plan** — what was tested and how. CI sees what ran, but
   reviewers want to know what the *user* covered.
3. **Notes for reviewers** — non-obvious choices, alternatives
   considered.

Skip "Generated with Claude Code" footers unless the policy in
`Contributing.md` says otherwise.

## CI gates (run locally before pushing)

| Local command | CI gate it satisfies |
|---------------|-----------------------|
| `bash scripts/run_clang_format.sh --check` | C++ formatting |
| `bash scripts/run_yapf_format.sh --check` | Python formatting |
| `bash scripts/sync_agents_skills.sh --check` | skill mirrors not stale |
| `bash scripts/test_cudaqx_build.sh` | top-level build + ctest |
| `bash scripts/test_libs_builds.sh` | per-library standalone builds |
| `python -m pytest libs/qec/python/tests libs/solvers/python/tests` | Python tests |

For a typical small PR, the formatting + skill-sync checks are
enough. For larger PRs (touching C++ or build), run the per-library
build script too.

## When CI fails

| Failure | First place to look |
|---------|---------------------|
| Format check | run `*_format.sh` (no `--check`) locally |
| Skill mirrors stale | run `sync_agents_skills.sh` (no `--check`) |
| Build fails on `manylinux` only | toolchain version mismatch; see `cudaq-build/references/wheels.md` |
| Unit test fails on GPU CI but passes locally | likely CUDA version or driver — check the CI job's environment |
| `import cudaq_qec` fails | stale install; rebuild with `scripts/clean.sh` |

## Code review

- Address every comment, even with "intentional, leaving as-is".
- Force-pushes are fine for small fixups; for the final review pass,
  use `git commit --amend` or `git rebase -i` to clean up history.
- Squash before merge unless individual commits are independently
  meaningful (rare in practice).

## After merge

If the PR was substantive (new decoder, new code, new pool):

1. Update the user-facing docs index if you added an example.
2. Tag the change in the release notes (maintainers will help with
   this if there's a release in progress).
3. If you authored a skill: re-run `sync_agents_skills.sh` and
   verify the mirror.

## Self-check

```
[ ] git commit -s on every commit
[ ] Formatting passes locally
[ ] Skill sync passes locally (if .agents/skills/ touched)
[ ] Tests added or updated
[ ] Docs updated (if user-visible)
[ ] PR body has Summary + Test plan
[ ] CI green
```

## Where next

- Pick file layout for new code: `file-layout.md`.
- Format details: `formatting.md`.
- Bug report (issue first): `issue-reporting.md`.
