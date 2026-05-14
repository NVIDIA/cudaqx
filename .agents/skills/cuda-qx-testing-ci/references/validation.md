# Container and wheel validation

Final-mile validation: does the *published artifact* work? Different
from unit tests — these scripts run against installed packages or a
running container, not against the source tree.

## Container validation

Script: `scripts/validation/container/validate_container.sh`.

Runs inside (or against) the released `ghcr.io/nvidia/cudaqx`
container. Verifies:

- CUDA-Q imports and runs.
- CUDA-QX imports.
- A canonical example runs end-to-end.
- Plugin discovery works.
- Expected files exist at expected paths.

Supporting Python script:
`scripts/validation/container/check_cudaq_import.py`.

Usage:

```bash
docker run --gpus all -it ghcr.io/nvidia/cudaqx \
  bash /workspaces/cudaqx/scripts/validation/container/validate_container.sh
```

(The script path is relative to the source mount; adjust if your
mount is elsewhere.)

If validation fails, the container build is broken — file a P0 issue
and do not publish.

## Wheel validation

Script: `scripts/validation/wheel/validate_wheels.sh`.

Runs in a clean Ubuntu (or similar) container. Verifies:

- `pip install cudaq-qec[ , extras]` succeeds.
- `pip install cudaq-solvers[ , extras]` succeeds.
- No missing system libs in a fresh environment.
- Canonical examples run.

Supporting script:
`scripts/validation/wheel/install_packages.sh` composes the install
from a wheelhouse (a local directory of wheel files).

Usage:

```bash
# After building wheels with scripts/build_wheels.sh
bash scripts/validation/wheel/validate_wheels.sh
```

## What's *not* validated by these

- GPU correctness across all GPU generations. (CI uses one or two
  reference GPUs.)
- Real-time decoding against actual quantum hardware. (Manual
  validation only.)
- Performance regressions. (Tracked separately.)
- Documentation rendering. (Sphinx build is its own CI job; see
  `cuda-qx-build/references/docs.md`.)

## When to run validation

| Trigger | Run |
|---------|-----|
| Before tagging a release | both container + wheel validation |
| After changing Dockerfile | container validation |
| After changing `pyproject.toml.cu1*` | wheel validation |
| After changing system-dep documentation | wheel validation (catches missing apt install lines) |
| Routine PR | not required (CI runs unit tests + builds; full validation is on a slower schedule) |

## Common pitfalls

| Symptom | Cause |
|---------|-------|
| Container validation passes but pip install fails | wheel and container drifted; validate both |
| Wheel validation fails with "libgfortran" | missing in the validation container's apt setup; update the script |
| `check_cudaq_import.py` reports module-not-found | CUDA-Q wheel not in the container or wrong CUDA suffix |
| Validation timeouts on large examples | examples should be cheap by design; if not, mark them in the validation suite |

## Self-check

```
[ ] If changing wheels: ran `validate_wheels.sh` and it passed.
[ ] If changing container: ran `validate_container.sh` and it passed.
[ ] If changing both: ran both.
[ ] Wheel built for the matching cu12/cu13 ABI and tested in that container.
[ ] No skipped tests inside validation runs.
```

## Where next

- Wheel build pipeline: `cuda-qx-build/references/wheels.md`.
- Container build pipeline: `cuda-qx-build/references/build.md` (and
  `docker/` paths).
- Per-PR CI subsets: `run-locally.md`.
