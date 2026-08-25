---
name: "cudaq-build"
title: "CUDA-Q Libraries Build, Wheels, Docs, and Container"
description: >-
  Configure, compile, package, and document CUDA-Q Libraries from this checkout.
  Use whenever the user mentions cmake, ninja install, build failures,
  wheel builds, manylinux, auditwheel, scikit-build-core, sphinx docs,
  doxygen, breathe, build_docs.sh, build_wheels.sh, dev container,
  cudaqx-dev image, or cudaq release container. Use this skill for
  "ninja install fails", "missing libgfortran during build",
  "ImportError after pip install", "rebuild only the QEC library",
  "regenerate the docs", or "build a wheel for cudaq-qec-cu12 /
  cudaq-solvers-cu13". Do NOT use this skill for: runtime API questions
  (decoders, operator pools, real-time decoding) — those go to
  cudaq-qec-decode / cudaq-solvers-algorithms; running tests or CI
  validation (test_examples.sh, ctest, wheel validation) — use
  cudaq-contributing; first-time / hello-world questions — use
  cudaq-quickstart.
version: "0.1.3"
author: "CUDA-Q Libraries"
license: "Apache License 2.0"
compatibility: "Python 3.11+, C++ 20, Linux x86_64/aarch64"
tags: [cudaq, build, cmake, ninja, wheels, manylinux, docs, sphinx, doxygen, container, dev-container]
tools: [Read, Glob, Grep, Bash]
metadata:
  repo: [qec, solvers]
  author: "CUDA-Q Libraries"
  domain: "build-and-packaging"
  languages: [bash, cmake, python, dockerfile]
---

# CUDA-Q Libraries Build Skill

Operate the CUDA-Q Libraries build pipeline on this repo: in-tree dev builds, wheels,
docs, container validation. The skill is task-driven: identify the user's
*build* intent, follow the matching workflow, then walk the **Self-Check
Protocol** before reporting done.

The runtime API skills (`cudaq-qec-decode`, `cudaq-solvers-algorithms`) deliberately do not
overlap with this skill. If the user is asking about an algorithm or decoder
behavior, delegate to them there.

## Inputs

Caller provides:

- A source checkout of cudaq (or a target dir for clone).
- Build intent: `build` (in-tree dev) / `build-wheels` / `build-docs` /
  `debug-import`.
- CUDA ABI choice if relevant: `cu12` or `cu13`.
- CMake / Ninja / Python toolchain installed (or a request to install).

## Outputs

This skill produces:

- Compiled artifacts at `$CUDAQX_INSTALL_PREFIX` (default `$HOME/.cudaq`).
- Passing `ctest` and `pytest` runs (when `CUDAQX_INCLUDE_TESTS=ON`).
- Built wheel files under `dist/` for wheel intents.
- Rendered docs at `build/docs/sphinx/` for docs intents.
- A doctor-script environment dump on failure.

Does NOT produce: runtime API answers (→ domain skills); fixes to
algorithm bugs (→ domain skills).

## How to read this skill

1. **First three actions** below: run `_shared/scripts/preflight.sh`,
   `import_smoke.py`, `pick_workflow.py`. They tell you which reference to
   read next and surface known blockers (stale install, mixed CUDA suffix,
   missing `gfortran`) before you spend tokens guessing.
2. Read **Conventions**. They prevent the most common silent breakage.
3. Find your task in the **Workflow Index** and open the matching file
   under `references/`.
4. Walk the **Self-Check** in `references/install-triage.md` before declaring done.

If you only have time for one file, read `references/build.md`. That plus
the conventions covers the bulk of "I just want to build and run my
changes" requests.

## Audience

AI coding agents and developers building CUDA-Q Libraries from a clone of this repo.
End-user pip installs (`pip install cudaq-qec`, `pip install cudaq-solvers`)
do not need this skill — they need the runtime skills.

The artifact is `SKILL.md` inside `.agents/skills/cudaq-build/`
(mirrored to `.claude/skills/` and `.cursor/skills/`). Companion files
live in `references/` (see Workflow Index below).

If the user is new to CUDA-Q Libraries and hasn't decided between pip / Docker /
source, delegate to **`cudaq-quickstart`** first; come back here once
they're committed to a source build. For PR workflow and DCO sign-off,
delegate to **`cudaq-contributing`** (which now also covers
testing, CI subsets, and container/wheel validation).

## First three actions (always, before anything else)

```bash
bash   .agents/skills/_shared/scripts/preflight.sh    --json > /tmp/preflight.json
python .agents/skills/_shared/scripts/import_smoke.py --json > /tmp/import_smoke.json
python .agents/skills/_shared/scripts/pick_workflow.py \
    --intent <pick from list below> \
    --preflight /tmp/preflight.json \
    --imports   /tmp/import_smoke.json
```

`pick_workflow.py` returns the next reference file to read, the commands
to run, and a `verify` step. Build-skill intents: `build`, `build-wheels`,
`build-docs`, `debug-import`. Run the suggested commands and only deviate
when the reference's prose tells you to.

## Key Paths

A glance map of build-relevant paths. Full repo map at
`.agents/skills/_shared/repo_map.md`.

| Area                                  | Path                                                 |
|---------------------------------------|------------------------------------------------------|
| Top-level CMake                       | `CMakeLists.txt`                                     |
| Per-library CMake                     | `libs/{qec,solvers}/CMakeLists.txt`                  |
| CMake modules                         | `cmake/Modules/`                                     |
| Dev / wheel Dockerfiles               | `docker/build_env/`                                  |
| Release Dockerfiles                   | `docker/release/`                                    |
| Build scripts                         | `scripts/`                                           |
| CI scripts (run inside containers)    | `scripts/ci/`                                        |
| Validation (container & wheel)        | `scripts/validation/`                                |
| Wheel pyproject (CUDA 12 / 13)        | `libs/{qec,solvers}/pyproject.toml.cu12`, `.cu13`    |
| Metapackage configs                   | `libs/{qec,solvers}/python/metapackages/`            |
| Build output (default)                | `build/`                                             |
| Docs build output                     | `build/docs/sphinx/`                                 |
| Install prefix (default)              | `$HOME/.cudaq/`                                     |
| Docs install prefix (default)         | `$HOME/.cudaq/docs/`                                |

## Source of Truth

| Need to know                                   | Authoritative file                          |
|------------------------------------------------|---------------------------------------------|
| Top-level CMake options                        | `CMakeLists.txt`                            |
| Human dev-build instructions                   | `Building.md`                               |
| Doc build pipeline (Doxygen + Sphinx + Breathe)| `scripts/build_docs.sh`, `docs/CMakeLists.txt`, `docs/Doxyfile.in`, `docs/sphinx/conf.py.in` |
| Wheel build orchestration                      | `scripts/build_wheels.sh` → `scripts/ci/build_{cudaq,qec,solvers}_wheel.sh` |
| Wheel test orchestration                       | `scripts/test_wheels.sh` → `scripts/ci/test_wheels.sh` |
| Dev container                                  | `docker/build_env/cudaq.dev.Dockerfile`    |
| Wheel builder container                        | `docker/build_env/cudaq.wheel.Dockerfile`  |
| Release container                              | `docker/release/Dockerfile`, `Dockerfile.wheel` |

## Workflow Index

| If the user wants to                                         | Read                          | `pick_workflow.py` intent |
|--------------------------------------------------------------|-------------------------------|---------------------------|
| Build, install, run examples, reset between branches         | `references/build.md`         | `build`                   |
| Build wheels (manylinux), test wheels, validate releases     | `references/wheels.md`        | `build-wheels`            |
| Build the docs and serve them offline (Sphinx + Breathe)     | `references/docs.md`          | `build-docs`              |
| Diagnose ImportError, version mismatch, stale install        | `references/install-triage.md`        | `debug-import`            |

`install-triage.md` carries the cross-cutting reference tables (CUDA 12 vs 13,
the troubleshooting matrix, the Self-Check Protocol). Read it any time
the build "succeeds" but the install behaves strangely.

## Conventions

These prevent the most common silent breakage. If a build "works on my
machine but not on yours", one of these is almost always the cause.

1. **Pick a CUDA ABI and stick with it.** CUDA-Q Libraries ships separate wheels for
   CUDA 12 (`cudaq-qec-cu12`, `cudaq-solvers-cu12`) and CUDA 13
   (`-cu13`). They are **not interchangeable**. The metapackages
   `cudaq-qec` / `cudaq-solvers` resolve to the matching `-cu1x` package
   based on what `cuda-quantum-cu1x` is already installed. Mixing
   `cuda-quantum-cu12` with `cudaq-qec-cu13` is a silent ABI mismatch.
   When in doubt, run `pip list | grep -E 'cuda-quantum|cudaq-(qec|solvers)'`.

2. **`$HOME/.cudaq` accumulates state across branches.** Header changes,
   moved targets, or switching libraries on/off (`CUDAQX_ENABLE_LIBS`) leave
   stale artifacts. If `import cudaq_qec` finds a stale `_pycudaqx_*.so`
   from a previous build, you will get import errors that look like ABI
   bugs. Fix: `rm -rf $HOME/.cudaq build` and rebuild. The
   `scripts/clean.sh` helper (when present) does this.

3. **`gfortran` and `libblas-dev` are required system packages.** Built-in
   solver optimizers (`cobyla`, `lbfgs`) link against `libgfortran`. Sphinx
   docs need them too because they import the package to render the API.
   Symptom: optimizer crashes at runtime, or wheel-test failures with
   `libgfortran.so.5: cannot open shared object`.

4. **`scikit-build-core`, CMake ≥ 3.28, Ninja ≥ 1.10.** The 3.28 floor is
   from the `EXCLUDE_FROM_ALL` argument to `FetchContent_Declare`. Older
   CMakes silently ignore the flag and fetch everything.

5. **CUDA-Q must already be installed before building CUDA-Q Libraries.** Default
   prefix `$HOME/.cudaq`. Pass it as `-DCUDAQ_DIR=$CUDAQ_INSTALL_PREFIX/lib/cmake/cudaq`.
   In the published dev container CUDA-Q lives at `/usr/local/cudaq` instead.

6. **C++20 toolchain. In manylinux that means `gcc-toolset-11`.** Wheel
   build scripts run `source /opt/rh/gcc-toolset-11/enable` before invoking
   `cmake`. Locally on Ubuntu 22.04+, system `gcc 11+` works.

7. **Docs build needs `CUDAQX_DOCS_GEN_IMPORT_CUDAQ=ON` and access to the
   compiled extension modules.** `scripts/build_docs.sh` sets this for you.
   If you skip the script and call `sphinx-build` directly, the API tables
   render empty.

8. **The license is split.** `libs/qec/LICENSE` is
   `LicenseRef-NVIDIA-Proprietary`; the rest of the repo (and
   `libs/solvers/`) is Apache 2.0. The release Dockerfile prepends a
   notice referencing the QEC license. Do not paper over this when
   summarizing licensing to users.

## Standard environment

```bash
export CUDAQ_INSTALL_PREFIX=$HOME/.cudaq          # CUDA-Q install root
export CUDAQX_INSTALL_PREFIX=$HOME/.cudaq        # CUDA-Q Libraries install root
export CUDAQ_DIR=$CUDAQ_INSTALL_PREFIX/lib/cmake/cudaq
export PATH=$CUDAQ_INSTALL_PREFIX/bin:$CUDAQX_INSTALL_PREFIX/bin:$PATH
export PYTHONPATH=$CUDAQ_INSTALL_PREFIX:$CUDAQX_INSTALL_PREFIX
```

In the dev container substitute `CUDAQ_INSTALL_PREFIX=/usr/local/cudaq`.

## Quick Start: in-tree build

```bash
cd /workspaces/cudaq
mkdir -p build && cd build
cmake -G Ninja -S .. \
  -DCUDAQ_DIR=$CUDAQ_INSTALL_PREFIX/lib/cmake/cudaq \
  -DCMAKE_INSTALL_PREFIX=$CUDAQX_INSTALL_PREFIX \
  -DCUDAQX_ENABLE_LIBS=all \
  -DCUDAQX_INCLUDE_TESTS=ON \
  -DCUDAQX_BINDINGS_PYTHON=ON \
  -DCMAKE_BUILD_TYPE=Release
ninja install
ctest                                    # C++ unit tests
cd .. && python3 -m pytest -v libs/qec/python/tests \
  --ignore libs/qec/python/tests/test_tensor_network_decoder.py
python3 -m pytest -v libs/solvers/python/tests \
  --ignore libs/solvers/python/tests/test_gqe.py
```

`CUDAQX_ENABLE_LIBS=qec` or `=solvers` builds only one library.

## When stuck

1. Re-run the **First three actions**. The state of the world may have
   changed since you started.
2. Read the matching reference file end-to-end (it's short).
3. Run `scripts/doctor.sh` for a human-readable environment snapshot
   suitable for a bug report. (`preflight.sh --json` is the structured
   superset used by the agent workflow.)
4. Try a clean build: `bash scripts/clean.sh && <rebuild>`.
5. Reproduce inside the published dev container — it pins the exact
   toolchain and removes "works locally" ambiguity.
6. If reporting an issue, capture `cmake --version`, `ninja --version`,
   `python3 --version`, the `pip list` output for cuda-quantum / cudaq-
   packages, and the first failing command line.

## Additional resources

- Workflow references: `references/build.md`, `references/wheels.md`,
  `references/docs.md`, `references/install-triage.md`
- Shared diagnostic scripts: `.agents/skills/_shared/scripts/`
- Shared repo map: `.agents/skills/_shared/repo_map.md`

### Related skills

- **`cudaq-quickstart`** — onboarding, pip vs Docker vs source.
- **`cudaq-qec-decode`** — runtime QEC API.
- **`cudaq-solvers-algorithms`** — runtime solvers API.
- **`cudaq-contributing`** — DCO sign-off, PR workflow, formatting,
  pytest/ctest, CI subsets, validation scripts.
- **`cudaq-skills-authoring`** — editing skill content, sync script,
  evals.
