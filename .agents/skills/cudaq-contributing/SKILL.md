---
name: "cudaq-contributing"
title: "Contributing to CUDA-Q Libraries (PRs, formatting, tests, CI)"
description: >-
  Contributor-facing skill for CUDA-Q Libraries: DCO sign-off, PR workflow,
  formatting (clang-format, yapf), file-tree layout decisions (where
  new code goes), how to file issues, AND the test / CI mechanics
  that go with a PR — pytest patterns, GoogleTest layout, ctest, the
  build/test scripts (scripts/ci/, scripts/validation/,
  test_examples.sh, test_libs_builds.sh, test_cudaqx_build.sh), wheel
  and container validation, and how to mirror CI locally. Use whenever
  the user mentions "send a PR", "contribute", "sign off", "DCO",
  "where do I put this code", "clang-format", "yapf format",
  "open an issue", "add a test", "run tests", "ctest", "pytest",
  "test failing in CI", "wheel validation", "container validation",
  "test_examples.sh", "test_libs_builds.sh", "test_cudaqx_build.sh",
  or "what does CI check". Do NOT use for: building wheels or images
  themselves (use cudaq-build); writing the production code under
  test (use the matching cudaq-qec-* / cudaq-solvers-* skill).
version: "0.2.0"
author: "CUDA-Q Libraries"
license: "Apache License 2.0"
compatibility: "Any platform (git, clang-format, yapf, pytest, ctest)"
tags: [cudaq, contributing, pr-workflow, dco, formatting, clang-format, yapf, testing, pytest, googletest, ctest, ci, validation]
tools: [Read, Glob, Grep, Bash]
metadata:
  repo: [qec, solvers]
  author: "CUDA-Q Libraries"
  domain: "contributing"
  audience: [contributor, developer]
  languages: [python, c++, markdown, bash]
---

# Contributing to CUDA-Q Libraries

End-to-end skill for sending a change upstream: from "I have a fix"
through writing the matching tests to "PR merged and CI green". Pulls
from `Contributing.md`, the layout in `AGENTS.md`, and what the CI
actually checks.

If the user is *building* their change locally, delegate to
`cudaq-build`. If they are writing a *new plugin*,
`cudaq-qec-extending` (QEC) or `cudaq-solvers-extending` (solvers).
Test-writing and CI debugging now live in this skill — they used to be
their own `cudaq-testing-ci` skill, but in practice contributors
always need both halves of "PR mechanics + test mechanics" and
splitting them just doubled activation cost.

## Inputs

Caller provides:

- A clean code change (committed locally, or about to be).
- Branch name + scope (fix / feat / refactor / docs / ci / test).
- A `git config user.name` and `user.email` that can be public.
- (Optional) related issue number.
- For tests: the test scope (unit pytest/gtest, integration, CI subset)
  and an installed build with `CUDAQX_INCLUDE_TESTS=ON` for C++ tests.

## Outputs

This skill produces:

- A signed-off commit series (DCO `-s`).
- Clean `clang-format` + `yapf` + skill-sync `--check` locally.
- New test files at the right paths (`libs/<lib>/python/tests/` for
  Python; `libs/<lib>/unittests/` for C++).
- Passing local invocation of `ctest -R <pattern>` and/or `pytest -v`.
- A subset of CI gates run locally (formatting, sync, build, tests).
- A diagnosis when "passes locally, fails in CI" — reproduced in the
  matching container.
- A branch pushed to a remote.
- A `gh pr create` with structured Summary + Test plan.

Does NOT produce: actual code fixes (→ domain skills); build artifacts
(→ `cudaq-build`); validation of published containers/wheels beyond
running `scripts/validation/` (→ release engineering for true release
gates).

## Audience

External contributors and NVIDIA-internal developers. Familiarity
with git, GitHub PR workflow, `pytest`, `gtest`, and `ctest` is
helpful.

## First three actions

```bash
git status                                       # confirm feature branch
git remote -v                                    # confirm remote
bash scripts/sync_agents_skills.sh --check       # CI gate; pass before pushing
```

`git status` confirms you are on a feature branch (not main).
`scripts/sync_agents_skills.sh --check` is one of the CI gates;
running it locally catches the most common preventable CI failure.

If the user is specifically debugging a test failure (vs preparing a
PR), substitute the third command for:

```bash
ctest --test-dir build --show-only=human 2>&1 | head -30
```

Empty output means `CUDAQX_INCLUDE_TESTS=ON` was not set during the
build; see `cudaq-build`.

## Key Paths

| Area | Path |
|------|------|
| Contribution guide | `Contributing.md` |
| Issue templates / discussions | `https://github.com/NVIDIA/cuda-qx/issues` |
| Format C++ | `scripts/run_clang_format.sh` |
| Format Python | `scripts/run_yapf_format.sh` |
| CI scripts | `scripts/ci/` |
| CMake top-level | `CMakeLists.txt` |
| Build skill | `.agents/skills/cudaq-build/SKILL.md` |
| Skill sync | `scripts/sync_agents_skills.sh` |
| Python tests (QEC) | `libs/qec/python/tests/` |
| Python tests (solvers) | `libs/solvers/python/tests/` |
| C++ tests (QEC) | `libs/qec/unittests/` |
| C++ tests (solvers) | `libs/solvers/unittests/` |
| Real-time app examples (test-shaped) | `libs/qec/unittests/realtime/app_examples/` |
| Build-and-test scripts | `scripts/test_cudaqx_build.sh`, `test_libs_builds.sh`, `test_wheels.sh` |
| Container validation | `scripts/validation/container/` |
| Wheel validation | `scripts/validation/wheel/` |
| CMake test config | top-level `CMakeLists.txt`, `CUDAQX_INCLUDE_TESTS` |

## Workflow Index

| If the user wants to | Read |
|----------------------|------|
| Sign off commits and submit a PR | `references/pr-workflow.md` |
| Pick where new code goes | `references/file-layout.md` |
| Format C++ and Python correctly | `references/formatting.md` |
| File a useful bug report | `references/issue-reporting.md` |
| Add a Python test | `references/tests-pytest.md` |
| Add a C++ / GoogleTest test | `references/tests-googletest.md` |
| Run only the subset CI runs (locally) | `references/tests-run-locally.md` |
| Understand what container / wheel validation does | `references/tests-validation.md` |

## Conventions

These prevent the recurring "PR opened, then needs three rounds of
formatting fixes" *and* the "passes locally, fails in CI" situations.

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

9. **`CUDAQX_INCLUDE_TESTS=ON` for C++ tests.** Without it, the
   build skips test executables. Default is `OFF` in some
   configurations.

10. **Python tests assume the package is on `PYTHONPATH`.** After
    `ninja install` set
    `PYTHONPATH=$CUDAQ_INSTALL_PREFIX:$CUDAQX_INSTALL_PREFIX`.

11. **One assertion per test, mostly.** A test that asserts five
    different things and fails on the second tells you less than
    five separate tests.

12. **Test at p=0 first.** For QEC tests, a "decoder returns zero
    corrections at p=0" assertion catches the most bugs.

13. **For solver tests, seed the optimizer.** `np.random.seed(0)`
    before VQE / ADAPT / GQE. Stochastic methods produce noisy
    "did it work?" otherwise.

14. **Mark slow tests.** `@pytest.mark.slow` for >30s tests; CI runs
    them on a different schedule.

15. **Skip GPU-only and arch-specific tests.** Use
    `@pytest.mark.skipif(not torch.cuda.is_available(), ...)` for
    GPU; check `platform.machine() in ("arm64", "aarch64")` for
    `stim`-dependent x86_64-only paths.

## Quick start: fix-a-bug PR (with test)

```bash
git checkout -b fix/my-decoder-bug
# ... edit code ...
# ... add a regression test (libs/<lib>/python/tests/ or unittests/) ...
bash scripts/run_clang_format.sh
bash scripts/run_yapf_format.sh
python -m pytest -v libs/qec/python/tests/test_my_decoder.py
git add libs/qec/lib/decoders/my_decoder.cpp libs/qec/python/tests/test_my_decoder.py
git commit -s -m "fix: decoder returns wrong shape for empty syndrome"
git push -u origin fix/my-decoder-bug
gh pr create --base main --fill
```

The `-s` flag adds the DCO sign-off. `--fill` reuses your commit
message as the PR body.

## Quick start: add a Python test

```python
# libs/qec/python/tests/test_my_decoder.py
import numpy as np
import pytest
import cudaq_qec as qec

def test_my_decoder_zero_syndrome():
    H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    dec = qec.get_decoder("my-decoder", H)
    r = dec.decode(np.zeros(2, dtype=np.uint8))
    assert (r.result == 0).all()
    assert r.converged
```

```bash
python -m pytest -v libs/qec/python/tests/test_my_decoder.py
```

## Quick start: add a C++ / GoogleTest test

```cpp
// libs/qec/unittests/test_my_decoder.cpp
#include <gtest/gtest.h>
#include "cudaq/qec/decoders/my_decoder.h"

TEST(MyDecoder, ZeroSyndrome) {
    cudaq::qec::heterogeneous_map opts;
    auto H = ...;
    cudaq::qec::my_decoder dec(H, opts);
    auto r = dec.decode(std::vector<float_t>(H.size(), 0.0));
    for (auto v : r.result) EXPECT_FLOAT_EQ(v, 0.0);
}
```

Add to `libs/qec/unittests/CMakeLists.txt` and rebuild. Run:

```bash
ctest --test-dir build -R MyDecoder -V
```

## Local CI mirror

```bash
bash scripts/test_cudaqx_build.sh
bash scripts/test_libs_builds.sh
python -m pytest -v libs/qec/python/tests libs/solvers/python/tests
bash scripts/build_wheels.sh && bash scripts/test_wheels.sh
```

For a normal PR the first command and the pytest run cover ~95% of
what CI catches. The wheel build / test pair is heavy and runs in a
container.

## Self-Check Protocol

```
[ ] On a feature branch (not main).
[ ] DCO sign-off in every commit.
[ ] clang-format and yapf both clean.
[ ] If you touched .agents/skills/, ran sync_agents_skills.sh.
[ ] CI green (or at least, ran locally what CI would run).
[ ] Tests added for new feature; regression test for bug fix.
[ ] New tests live at the right path (libs/<lib>/{python/tests,unittests}).
[ ] C++ tests registered in unittests/CMakeLists.txt.
[ ] Each test has a clear name and tests one thing.
[ ] `ctest` and `pytest` both pass locally.
[ ] Slow tests marked; GPU/arch-specific tests skipped appropriately.
[ ] Docs updated if a user-visible thing changed.
[ ] Commit messages reference an issue if one exists.
```

## When stuck

1. Read `Contributing.md` for the canonical contribution flow.
2. Read the matching `references/<topic>.md`.
3. Look at the nearest existing test in the same directory and copy
   its shape.
4. Look at recent merged PRs on
   <https://github.com/NVIDIA/cuda-qx/pulls?q=is%3Apr+is%3Amerged>
   for patterns: how big is "small enough"?
5. **"Passes locally, fails in CI"**: check that CI's CUDA version /
   driver / container matches your local. The CI scripts hard-code
   the container; pull and run the same image locally.
6. For container validation specifically: `references/tests-validation.md`.
7. For build issues, delegate to `cudaq-build`.

## Additional resources

- `references/pr-workflow.md` — git flow, sign-off, gh CLI.
- `references/file-layout.md` — where new code goes.
- `references/formatting.md` — clang-format and yapf details.
- `references/issue-reporting.md` — anatomy of a useful bug report.
- `references/tests-pytest.md` — pytest patterns, fixtures, marks.
- `references/tests-googletest.md` — GoogleTest, CMake test registration.
- `references/tests-run-locally.md` — how to run the same subset CI does.
- `references/tests-validation.md` — container / wheel validation scripts.
- Build itself: `cudaq-build`.
- Plugin authoring: `cudaq-qec-extending`, `cudaq-solvers-extending`.
- Skill authoring (if you're editing `.agents/skills/` files): `cudaq-skills-authoring`.
