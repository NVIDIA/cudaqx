# Wheel build, test, validation

How to produce manylinux wheels for CUDA-QX, test them in a clean
container, validate published artifacts, and run the formatters CI cares
about.

## Wheel build

Manylinux wheels, built inside a builder container so the resulting
artifacts work on minimal Linux distros.

**Driver script (host)**

```bash
scripts/build_wheels.sh
```

Pulls `ghcr.io/nvidia/cuda-quantum-devdeps:manylinux-...-gcc11-main`,
creates a long-lived `cudaqx_wheel_builder` container (so subsequent runs
reuse the dep cache), copies the repo in, and runs the per-library wheel
scripts. Output wheels land in `./wheels/`.

**Per-library scripts (container)**

- `scripts/ci/build_cudaq_wheel.sh` — builds the upstream CUDA-Q wheel
  and the matching CUDA-Q install used by the QEC/solvers builds. Run
  once per builder container.
- `scripts/ci/build_qec_wheel.sh` — `python -m build --wheel` from
  `libs/qec/`, then `auditwheel repair` excluding all `libcudaq*` so the
  CUDA-Q wheel provides them at install time.
- `scripts/ci/build_solvers_wheel.sh` — same shape for solvers; also
  excludes `libgfortran.so.5`, `libquadmath.so.0`, `libmvec.so.1`.

**CUDA version selection**: the scripts assume CUDA 12.6 by default. Set
`CUDA_VERSION=13.0` (and use the corresponding `cu13` builder image) to
build CUDA 13 wheels.

**Toolchain**: `source /opt/rh/gcc-toolset-11/enable` before `cmake` to
get GCC 11 with C++20.

**Metapackages**: `scripts/ci/build_metapackages.sh <version>` produces
the `cudaq-qec` / `cudaq-solvers` sdists that delegate to the cu1x
wheels at install time.

**Self-check**: `ls wheels/` shows
`cudaq_qec_cu1x-*-manylinux*.whl` and
`cudaq_solvers_cu1x-*-manylinux*.whl`. `auditwheel show
wheels/cudaq_qec_*.whl` reports a `manylinux_2_27` (or later) tag.

## Wheel test

Validates a freshly built wheel set in a clean container.

**Driver (host)**: `scripts/test_wheels.sh`. Pulls `ubuntu:22.04`, copies
the repo and the `wheels/` dir in, runs `scripts/ci/test_wheels.sh`
inside.

**`test_wheels.sh` arguments**:

```bash
test_wheels.sh <python_version> <platform> <cuda_version> <cudaq_version> <cudaqx_version>
```

E.g. `3.12 amd64 12.6 0.14.0 0.4.0`. CI normally fills these in. The
script:

1. Installs `pytest`, `openfermion`, `openfermionpyscf`, `onnxscript`.
2. Installs PyTorch (CUDA-matched if `nvidia-smi` works).
3. Installs `tensorrt-cu1x` (skipped on arm64 cu12).
4. `pip install --extra-index-url https://pypi.nvidia.com/ "cudaq-qec[all]==<v>"`.
5. Runs `pytest libs/qec/python/tests/`. Skips
   `test_trt_decoder.py` if no GPU.
6. Verifies `pip list` shows `cudaq-qec-cuXX` (matches CUDA major).
7. Same for solvers, then with `[gqe]`.
8. Runs example Python scripts. The actual source files live at
   `docs/sphinx/examples/qec/python/*.py` and
   `docs/sphinx/examples/solvers/python/*.py`. (Note: `scripts/ci/test_examples.sh`
   iterates over a bare `examples/` prefix — it expects a build-time
   staging step to copy the docs examples there. If you see "no such
   file or directory" running it directly from a fresh checkout, the
   staging hasn't run yet; invoke it from the wheel-test container or
   first copy `docs/sphinx/examples/` to `examples/`.)

**Self-check**: every pytest run completes; package suffixes match the
expected `cudaq-qec-cu<major>` / `cudaq-solvers-cu<major>`.

## Validation (published images & wheels)

End-to-end validation of the artifacts NVIDIA publishes.

**Container** (`scripts/validation/container/validate_container.sh`):

- Pulls the release image (e.g.
  `ghcr.io/nvidia/private/cuda-quantum:cu12-0.14.0-cudaqx-rc1`).
- Installs PyTorch, ONNX, lightning, mpi4py, transformers, quimb,
  opt_einsum, cuquantum-python, stim, beliefmatching.
- Runs `check_cudaq_import.py`.
- Loops `PY_TARGETS` (`nvidia`, `nvidia --option fp64`, `qpp-cpu`) and
  runs each example.
- C++ examples: compiles with `nvq++ --enable-mlir -lcudaq-<lib>` per
  target.

**Wheel** (`scripts/validation/wheel/validate_wheels.sh` and
`install_packages.sh`): pip-installs from a wheelhouse and runs pytest.

**Self-check**: container script returns 0; expected examples run on
every target. Wheel script: `pytest` reports zero failures.

## Formatting (pre-PR)

| Target            | Script                              |
|-------------------|-------------------------------------|
| C++ (clang-format)| `scripts/run_clang_format.sh`       |
| Python (yapf)     | `scripts/run_yapf_format.sh`        |

Run both before submitting a PR. CI rejects non-conforming formatting.

## Wheel pyproject locations

| Library | CUDA 12 | CUDA 13 |
|---------|---------|---------|
| qec     | `libs/qec/pyproject.toml.cu12`     | `libs/qec/pyproject.toml.cu13`     |
| solvers | `libs/solvers/pyproject.toml.cu12` | `libs/solvers/pyproject.toml.cu13` |

The wheel scripts copy the chosen `pyproject.toml.cuXX` to
`pyproject.toml` before invoking `python -m build`.
