# Running CI subsets locally

What CI actually runs, and how to mirror each piece from a developer
machine. Use this when "passes locally, fails in CI" is the bug.

## CI scripts inventory

| Script | Purpose |
|--------|---------|
| `scripts/test_cudaqx_build.sh` | Top-level `-DCUDAQX_ENABLE_LIBS=all` build + ctest. Closest to "the whole thing". |
| `scripts/test_libs_builds.sh` | Per-library standalone builds. Catches link-time issues. |
| `scripts/build_wheels.sh` | Manylinux wheel build inside `cudaqx_wheel_builder` container. |
| `scripts/test_wheels.sh` | Pip-install built wheels in a clean Ubuntu container, run pytest. |
| `scripts/ci/test_examples.sh` | Run the example programs (`qec`, `solvers`, `all`). |
| `scripts/validation/container/validate_container.sh` | Validate published cudaq container. |
| `scripts/validation/wheel/validate_wheels.sh` | Validate installed wheel set. |

## Run them in the right order

For a PR that touches code:

```bash
# 1. Format check (fast)
bash scripts/run_clang_format.sh --check
bash scripts/run_yapf_format.sh --check

# 2. Skill sync check (fast)
bash scripts/sync_agents_skills.sh --check

# 3. Top-level build + tests
bash scripts/test_cudaqx_build.sh

# 4. Examples (medium)
bash scripts/ci/test_examples.sh all
```

For a PR that touches build / packaging:

```bash
# 5. Per-library standalone builds
bash scripts/test_libs_builds.sh

# 6. Wheels (heavy; runs in container)
bash scripts/build_wheels.sh
bash scripts/test_wheels.sh
```

For a PR that touches the published container:

```bash
# 7. Container validation (heavy)
bash scripts/validation/container/validate_container.sh
```

## Subsetting

The full suite is slow. While developing:

```bash
# Just your decoder's tests
ctest --test-dir build -R my_decoder
python -m pytest -v libs/qec/python/tests/test_my_decoder.py

# Just the QEC library's tests
ctest --test-dir build -R "^qec_"
python -m pytest -v libs/qec/python/tests

# Skip slow tests
python -m pytest -v libs/qec/python/tests -m "not slow"
```

## Running inside the dev container

CI runs inside the `cudaqx_wheel_builder` or `cudaqx-dev` containers.
To exactly mirror CI's environment locally:

```bash
docker pull ghcr.io/nvidia/cudaqx-dev:latest-amd64-cu12.6
docker run --gpus all -it \
  -v $(pwd):/workspaces/cudaq \
  ghcr.io/nvidia/cudaqx-dev:latest-amd64-cu12.6
# Inside container:
cd /workspaces/cudaq
bash scripts/test_cudaqx_build.sh
```

This eliminates "different CUDA / different compiler / different
glibc" failures.

## When CI fails

| CI job | Local command that reproduces |
|--------|-------------------------------|
| `format-check` | `bash scripts/run_*_format.sh --check` |
| `skills-sync-check` | `bash scripts/sync_agents_skills.sh --check` |
| `build` | `bash scripts/test_cudaqx_build.sh` |
| `lib-builds` | `bash scripts/test_libs_builds.sh` |
| `wheel-build` | `bash scripts/build_wheels.sh` (in container) |
| `wheel-test` | `bash scripts/test_wheels.sh` |
| `examples` | `bash scripts/ci/test_examples.sh all` |
| `container-validate` | `bash scripts/validation/container/validate_container.sh` |

If the local command passes and CI fails:

1. Pull the exact container the CI job used.
2. Run the same command inside it.
3. If still passes locally: check CI logs for environment differences
   (env vars, secrets, mounted paths).
4. If reproduces in container: now it's reproducible — debug as
   normal.

## Common pitfalls

| Symptom | Cause |
|---------|-------|
| `test_cudaqx_build.sh` fails with CMake error | `CUDAQ_DIR` not set or wrong path |
| `test_wheels.sh` reports symbol-not-found at runtime | wheel built on wrong `manylinux`; check `build_wheels.sh` |
| `validate_container.sh` fails on a CI-only path | container missing a file or env; needs maintainer attention |
| Example runs locally but fails in `test_examples.sh` | example writes outside its CWD or depends on `$HOME/.cudaq` state |

## Self-check

```
[ ] Identified the failing CI job by name.
[ ] Ran the matching local command and reproduced.
[ ] If can't reproduce locally, pulled the CI container and ran inside.
[ ] After fix, re-ran the local command and confirmed pass.
```

## Where next

- Add a new test: `python-tests.md` or `cpp-tests.md`.
- Container / wheel validation specifics: `validation.md`.
- Build issues: `cudaq-build`.
