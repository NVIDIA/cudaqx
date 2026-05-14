---
name: "cuda-qx-testing-ci"
title: "CUDA-QX Testing and CI"
description: >-
  Test layout, pytest patterns, GoogleTest layout, ctest, the build/test
  scripts (scripts/ci/, scripts/validation/), and how to mirror CI
  locally. Use whenever the user mentions "add a test", "run tests",
  "ctest", "pytest", "test failing in CI", "wheel validation",
  "container validation", "test_examples.sh", or "what does CI check".
version: "0.1.1"
author: "CUDA-QX"
license: "Apache License 2.0"
compatibility: "Python 3.11+, C++ 20, Linux x86_64/aarch64"
tags: [cuda-qx, testing, pytest, googletest, ctest, ci, validation, scripts]
tools: [Read, Glob, Grep, Bash]
metadata:
  repo: [qec, solvers]
  author: "CUDA-QX"
  domain: "testing"
  audience: [developer, contributor]
  languages: [python, c++, bash]
---

# CUDA-QX Testing and CI

The test infrastructure: layout, how to add a test, how to run the
right subset locally, and what CI is actually checking. If the user
is *writing* code, this skill comes second to the code skill itself
(`cuda-qx-qec-extending`, `cuda-qx-solvers-extending`, `cuda-qx-qec-decode`, `cuda-qx-solvers-algorithms`).

## Inputs

Caller provides:

- A code change or new feature that needs tests.
- Test scope: unit (pytest/gtest) / integration / CI subset.
- An installed build with `CUDAQX_INCLUDE_TESTS=ON` for C++ tests.
- Hardware available (GPU vs CPU-only) for marking purposes.

## Outputs

This skill produces:

- New test files at the right paths (`libs/<lib>/python/tests/` for
  Python; `libs/<lib>/unittests/` for C++).
- Updated `CMakeLists.txt` for new C++ tests.
- Passing local invocation of `ctest -R <pattern>` and/or `pytest -v`.
- A subset of CI gates run locally (formatting, sync, build).
- A diagnosis when "passes locally, fails in CI" — reproduced in the
  matching container.

Does NOT produce: the fix itself (→ domain skills); validation of
published containers/wheels (→ `validation.md` is the boundary, beyond
that delegate to release engineering).

## Audience

Contributors and maintainers. Familiarity with `pytest`,
`gtest`, and `ctest` is helpful.

## First three actions

```bash
bash   .agents/skills/_shared/scripts/preflight.sh    --json > /tmp/preflight.json
python .agents/skills/_shared/scripts/import_smoke.py --json > /tmp/import_smoke.json
ctest --test-dir build --show-only=human 2>&1 | head -30
```

The third command lists the C++ tests the current build offers.
Empty output means `CUDAQX_INCLUDE_TESTS=ON` was not set; see
`cuda-qx-build`.

## Key Paths

| Area | Path |
|------|------|
| Python tests (QEC) | `libs/qec/python/tests/` |
| Python tests (solvers) | `libs/solvers/python/tests/` |
| C++ tests (QEC) | `libs/qec/unittests/` |
| C++ tests (solvers) | `libs/solvers/unittests/` |
| Real-time app examples (test-shaped) | `libs/qec/unittests/realtime/app_examples/` |
| Build-and-test scripts | `scripts/test_cudaqx_build.sh`, `test_libs_builds.sh`, `test_wheels.sh` |
| CI scripts (run in containers) | `scripts/ci/` |
| Container validation | `scripts/validation/container/` |
| Wheel validation | `scripts/validation/wheel/` |
| CMake test config | top-level `CMakeLists.txt`, `CUDAQX_INCLUDE_TESTS` |

## Workflow Index

| If the user wants to | Read |
|----------------------|------|
| Add a Python test | `references/python-tests.md` |
| Add a C++ / GoogleTest test | `references/cpp-tests.md` |
| Run only the subset CI runs (locally) | `references/run-locally.md` |
| Understand what container / wheel validation does | `references/validation.md` |

## Conventions

These prevent the recurring "passes locally, fails in CI" pain.

1. **`CUDAQX_INCLUDE_TESTS=ON` for C++ tests.** Without it, the
   build skips test executables. Default is `OFF` in some
   configurations.

2. **Python tests assume the package is on `PYTHONPATH`.** After
   `ninja install` set `PYTHONPATH=$CUDAQ_INSTALL_PREFIX:$CUDAQX_INSTALL_PREFIX`.

3. **One assertion per test, mostly.** A test that asserts five
   different things and fails on the second tells you less than
   five separate tests.

4. **Test at p=0 first.** For QEC tests, a "decoder returns zero
   corrections at p=0" assertion catches the most bugs.

5. **For solver tests, seed the optimizer.** `np.random.seed(0)`
   before VQE / ADAPT / GQE. Stochastic methods produce noisy
   "did it work?" otherwise.

6. **Mark slow tests.** `@pytest.mark.slow` for >30s tests; CI runs
   them on a different schedule.

7. **Skip GPU-only tests on CPU-only CI.** Use
   `@pytest.mark.skipif(not torch.cuda.is_available(), reason=...)`.

8. **Skip ARM64-only tests appropriately.** Some examples / tests
   skip when `platform.machine() in ("arm64", "aarch64")` because
   `stim` is x86_64-only on manylinux.

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

@pytest.mark.skipif(not has_gpu(), reason="GPU required")
def test_my_decoder_gpu_path():
    ...
```

Run:

```bash
python -m pytest -v libs/qec/python/tests/test_my_decoder.py
```

## Quick start: add a C++ test

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
# Top-level build + tests
bash scripts/test_cudaqx_build.sh

# Per-library standalone builds
bash scripts/test_libs_builds.sh

# Python tests directly
python -m pytest -v libs/qec/python/tests libs/solvers/python/tests

# Wheel build + test (heavy; runs in container)
bash scripts/build_wheels.sh && bash scripts/test_wheels.sh
```

For a normal PR the first command and the pytest run cover ~95% of
what CI catches.

## Self-Check Protocol

```
[ ] New tests added under libs/<lib>/{python/tests,unittests}/.
[ ] C++ tests registered in unittests/CMakeLists.txt.
[ ] Each test has a clear name and tests one thing.
[ ] `ctest` and `pytest` both pass locally.
[ ] Slow tests marked; GPU/arch-specific tests skipped.
[ ] If you touched build scripts, ran test_cudaqx_build.sh locally.
```

## When stuck

1. Read the matching `references/*.md`.
2. Look at the nearest existing test in the same directory and copy
   its shape.
3. If a CI job fails but local pass: check that CI's CUDA version /
   driver / container matches your local. The CI scripts hard-code
   the container; pull and run the same image locally.
4. For container validation specifically: `references/validation.md`.

## Additional resources

- `references/python-tests.md` — pytest patterns, fixtures, marks.
- `references/cpp-tests.md` — GoogleTest, CMake test registration.
- `references/run-locally.md` — how to run the same subset CI does.
- `references/validation.md` — container / wheel validation scripts.
- Build itself: `cuda-qx-build`.
- Submitting the PR: `cuda-qx-contributing`.
