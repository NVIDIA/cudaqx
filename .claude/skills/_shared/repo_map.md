# CUDA-QX Repo Map

This file is shared across the `cuda-qx-qec`, `cuda-qx-solvers`, and
`cuda-qx-build` skills. It lives in `.claude/skills/_shared/` (the
underscore signals "not a skill"): it has no frontmatter, no triggers,
and is read on demand by the skills that point to it.

When in doubt about where a file lives, read this first instead of guessing
or globbing the whole tree.

## Top-level layout

| Path                               | Purpose                                                           |
|------------------------------------|-------------------------------------------------------------------|
| `CMakeLists.txt`                   | Top-level build (`CUDAQX_ENABLE_LIBS={all,qec,solvers}`)          |
| `Building.md`                      | Human dev-build instructions                                      |
| `Contributing.md`                  | Contribution guide                                                |
| `cmake/`                           | Custom CMake modules (`include(CUDA-QX)`)                         |
| `libs/core/`                       | Shared C++ helpers used by both libraries                         |
| `libs/qec/`                        | QEC library (sources, headers, Python bindings, tests)            |
| `libs/solvers/`                    | Solvers library (sources, headers, Python bindings, tests)        |
| `libs/qec/python/metapackages/`    | `cudaq-qec` metapackage (delegates to cu12/cu13 wheels)           |
| `libs/solvers/python/metapackages/`| `cudaq-solvers` metapackage                                       |
| `examples/qec/`                    | Standalone Python and C++ example programs (QEC)                  |
| `examples/solvers/`                | Standalone Python and C++ example programs (solvers)              |
| `docs/`                            | Doxygen + Sphinx + Breathe docs source                            |
| `docs/sphinx/examples/`            | Curated, doc-rendered examples                                    |
| `scripts/`                         | Build, test, format, validation scripts                           |
| `docker/build_env/`                | Dev / wheel-build Dockerfiles                                     |
| `docker/release/`                  | Release Dockerfiles (NVIDIA-published containers)                 |
| `build/`                           | Default in-tree build dir (CMake/Ninja output)                    |

## Per-library layout (`libs/<lib>/`)

| Path                              | Purpose                                                            |
|-----------------------------------|--------------------------------------------------------------------|
| `CMakeLists.txt`                  | Library-level CMake (also used by `test_libs_builds.sh`)           |
| `include/cudaq/<lib>/`            | C++ public headers                                                 |
| `lib/`                            | C++ implementation                                                 |
| `python/cudaq_<lib>/`             | Python package (importable as `cudaq_<lib>`)                       |
| `python/bindings/`                | pybind11 sources for the native module                             |
| `python/tests/`                   | Python pytest suite                                                |
| `unittests/`                      | C++ GoogleTest suite                                               |
| `pyproject.toml.cu12`             | Wheel build config for CUDA 12 ABI                                 |
| `pyproject.toml.cu13`             | Wheel build config for CUDA 13 ABI                                 |
| `README.md`                       | Library overview                                                   |
| `LICENSE`                         | Apache 2.0 (solvers) / `LicenseRef-NVIDIA-Proprietary` (qec)       |

## Solvers-specific paths

| Area                 | Path                                                                  |
|----------------------|-----------------------------------------------------------------------|
| Python public API    | `libs/solvers/python/cudaq_solvers/__init__.py`                       |
| Python bindings      | `libs/solvers/python/bindings/solvers/py_solvers.cpp`, `py_optim.cpp` |
| GQE algorithm        | `libs/solvers/python/cudaq_solvers/gqe_algorithm/gqe.py`              |
| PySCF driver (C++)   | `libs/solvers/lib/operators/molecule/drivers/pyscf_driver.cpp`        |
| PySCF tool (Python)  | `libs/solvers/tools/molecule/cudaq-pyscf.py`                          |
| Examples (Python)    | `docs/sphinx/examples/solvers/python/`                                |
| Examples (C++)       | `docs/sphinx/examples/solvers/cpp/`                                   |
| API docs             | `docs/sphinx/api/solvers/`                                            |
| Component docs       | `docs/sphinx/components/solvers/`                                     |

## QEC-specific paths

| Area                       | Path                                                                            |
|----------------------------|---------------------------------------------------------------------------------|
| Python public API          | `libs/qec/python/cudaq_qec/__init__.py`                                         |
| Python bindings            | `libs/qec/python/bindings/`                                                     |
| Decoder plugins (C++)      | `libs/qec/lib/decoders/`                                                        |
| Decoder plugins (Python)   | `libs/qec/python/cudaq_qec/plugins/decoders/`                                   |
| Real-time bindings         | `libs/qec/python/bindings/py_decoding.cpp`                                      |
| Real-time C++ app examples | `libs/qec/unittests/realtime/app_examples/`                                     |
| DEM (C++)                  | `libs/qec/include/cudaq/qec/detector_error_model.h`                             |
| DEM sampling (Python)      | `libs/qec/python/cudaq_qec/dem_sampling.py`                                     |
| Built-in code headers      | `libs/qec/include/cudaq/qec/codes/`                                             |
| Examples (Python)          | `docs/sphinx/examples/qec/python/`                                              |
| Examples (C++)             | `docs/sphinx/examples/qec/cpp/`                                                 |
| Examples (RST/longform)    | `docs/sphinx/examples_rst/qec/`                                                 |
| API docs                   | `docs/sphinx/api/qec/`                                                          |
| Component docs             | `docs/sphinx/components/qec/`                                                   |

## Build / docs / scripts

| Path                                         | Use                                                          |
|----------------------------------------------|--------------------------------------------------------------|
| `scripts/build_docs.sh`                      | Build Doxygen + Sphinx docs into `build/docs/sphinx`         |
| `scripts/build_wheels.sh`                    | Drive manylinux wheel build inside `cudaqx_wheel_builder`    |
| `scripts/test_wheels.sh`                     | Test wheels in a clean Ubuntu container                      |
| `scripts/test_cudaqx_build.sh`               | Top-level `cmake -DCUDAQX_ENABLE_LIBS=all` build + tests     |
| `scripts/test_libs_builds.sh`                | Per-library standalone builds                                |
| `scripts/run_clang_format.sh`                | Format C++ via clang-format                                  |
| `scripts/run_yapf_format.sh`                 | Format Python via yapf                                       |
| `scripts/build_engine_from_onnx.py`          | Build a TensorRT engine for the predecoder workflow          |
| `scripts/patch_wheel_metadata.sh`            | Wheel metadata fixups (release plumbing)                     |
| `scripts/prune_cudaqx-dev_by_sha.sh`         | Container image pruning helper                               |
| `scripts/ci/build_cudaq_wheel.sh`            | Build the upstream CUDA-Q wheel (run inside builder image)   |
| `scripts/ci/build_qec_wheel.sh`              | Build the QEC wheel (auditwheel-repaired)                    |
| `scripts/ci/build_solvers_wheel.sh`          | Build the solvers wheel (auditwheel-repaired)                |
| `scripts/ci/build_metapackages.sh`           | Build the `cudaq-qec` / `cudaq-solvers` sdist metapackages   |
| `scripts/ci/test_examples.sh`                | Run example suites (`qec`, `solvers`, `all`)                 |
| `scripts/ci/test_wheels.sh`                  | Pip-install built wheels and run pytest                      |
| `scripts/validation/container/validate_container.sh` | Validate published container images                  |
| `scripts/validation/container/check_cudaq_import.py` | CUDA-Q sanity import                                 |
| `scripts/validation/wheel/validate_wheels.sh`        | Wheel-install validation                             |
| `scripts/validation/wheel/install_packages.sh`       | Compose a wheel install from a wheelhouse            |

## Common environment variables

| Variable                          | Purpose                                                              |
|-----------------------------------|----------------------------------------------------------------------|
| `CUDAQ_INSTALL_PREFIX`            | CUDA-Q install root (default `$HOME/.cudaq`)                         |
| `CUDAQX_INSTALL_PREFIX`           | CUDA-QX install root (default `$HOME/.cudaqx`)                       |
| `CUDAQ_DIR`                       | CUDA-Q's CMake config dir (`$CUDAQ_INSTALL_PREFIX/lib/cmake/cudaq/`) |
| `CUDAQX_ENABLE_LIBS`              | CMake list: `qec`, `solvers`, or `all`                               |
| `CUDAQX_INCLUDE_TESTS`            | CMake bool: build tests                                              |
| `CUDAQX_BINDINGS_PYTHON`          | CMake bool: build Python bindings                                    |
| `OMP_NUM_THREADS`                 | Set to `1` for reproducible PySCF Hamiltonian coefficients           |
| `CUDAQ_DEFAULT_SIMULATOR`         | `stim` for QEC kernel workflows                                      |
| `CUDAQ_QEC_DEBUG_DECODER`         | `1` to log real-time decoder uploads (Quantinuum)                    |
| `CUDAQ_QUANTINUUM_CREDENTIALS`    | Path to Quantinuum credentials                                       |
| `PYTHONPATH`                      | Should include `$CUDAQ_INSTALL_PREFIX:$CUDAQX_INSTALL_PREFIX`        |

## Container images

| Image                                            | Use                                              |
|--------------------------------------------------|--------------------------------------------------|
| `ghcr.io/nvidia/cudaqx-dev:latest-amd64-cu12.6`  | Dev container, AMD64, CUDA 12.x                  |
| `ghcr.io/nvidia/cudaqx-dev:latest-amd64-cu13.0`  | Dev container, AMD64, CUDA 13.x                  |
| `ghcr.io/nvidia/cudaqx-dev:latest-arm64-cu12.6`  | Dev container, ARM64, CUDA 12.x                  |
| `ghcr.io/nvidia/cudaqx-dev:latest-arm64-cu13.0`  | Dev container, ARM64, CUDA 13.x                  |
| `ghcr.io/nvidia/cuda-quantum-devdeps:manylinux-amd64-cu12.6-gcc11-main` | Wheel build base (used by `build_wheels.sh`) |
| `ghcr.io/nvidia/cudaqx:latest`                   | Release container (built from `docker/release/Dockerfile`) |
