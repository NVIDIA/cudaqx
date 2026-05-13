# Build & install (dev loop)

How to configure, build, install, run examples, and reset between branches.
For wheels see `wheels.md`; for docs see `docs.md`; for "something is
broken" see `triage.md`.

## Top-level CMake options

| Option                       | Default      | Notes                                                    |
|------------------------------|--------------|----------------------------------------------------------|
| `CUDAQX_ENABLE_LIBS`         | `all`        | Semicolon-separated; valid: `qec`, `solvers`, `all`      |
| `CUDAQX_INCLUDE_TESTS`       | `ON`         | C++ + Python test targets (`run_tests`, `run_python_tests`) |
| `CUDAQX_INCLUDE_DOCS`        | `ON`         | `ninja docs` target wired into top-level                 |
| `CUDAQX_BINDINGS_PYTHON`     | `ON`         | Build pybind11 modules                                   |
| `CMAKE_BUILD_TYPE`           | `Release`    | `Debug`/`RelWithDebInfo` for stack traces                |
| `CMAKE_INSTALL_PREFIX`       | `$HOME/.cudaqx` | Where install/headers/python packages land            |
| `CUDAQ_DIR`                  | (none)       | Required: `<cudaq-prefix>/lib/cmake/cudaq`               |

Per-library options (when invoking `libs/qec` or `libs/solvers` standalone):

- `CUDAQX_QEC_INCLUDE_TESTS` / `CUDAQX_SOLVERS_INCLUDE_TESTS`
- `CUDAQX_QEC_BINDINGS_PYTHON` / `CUDAQX_SOLVERS_BINDINGS_PYTHON`
- `CUDAQX_QEC_INSTALL_PYTHON` / `CUDAQX_SOLVERS_INSTALL_PYTHON`

## In-tree dev build (the default loop)

Builds and installs against an existing CUDA-Q.

```bash
export CUDAQ_INSTALL_PREFIX=${CUDAQ_INSTALL_PREFIX:-$HOME/.cudaq}
export CUDAQX_INSTALL_PREFIX=${CUDAQX_INSTALL_PREFIX:-$HOME/.cudaqx}

mkdir -p build && cd build
cmake -G Ninja -S .. \
  -DCUDAQ_DIR=$CUDAQ_INSTALL_PREFIX/lib/cmake/cudaq \
  -DCMAKE_INSTALL_PREFIX=$CUDAQX_INSTALL_PREFIX \
  -DCUDAQX_ENABLE_LIBS=all \
  -DCUDAQX_INCLUDE_TESTS=ON \
  -DCUDAQX_BINDINGS_PYTHON=ON \
  -DCMAKE_BUILD_TYPE=Release
ninja install

ctest                                  # C++ tests
cd .. && python3 -m pytest -v libs/qec/python/tests \
  --ignore libs/qec/python/tests/test_tensor_network_decoder.py
python3 -m pytest -v libs/solvers/python/tests \
  --ignore libs/solvers/python/tests/test_gqe.py
```

**Self-check**: `python3 -c "import cudaq_qec, cudaq_solvers; print(cudaq_qec, cudaq_solvers)"`
prints package paths under `$CUDAQX_INSTALL_PREFIX`. `ctest` and the two
pytest commands return zero failures.

**Helper**: `scripts/test_cudaqx_build.sh` does the configure + build +
tests in one shot (top-level `cmake -DCUDAQX_ENABLE_LIBS=all`). Pass `-i`
to also install.

## Per-library build (fast iteration)

When you only care about one library, build it standalone — much faster
than the top-level CMake re-configure.

```bash
# QEC only
cmake -S libs/qec -B build_qec \
  -DCUDAQ_DIR=$CUDAQ_INSTALL_PREFIX/lib/cmake/cudaq \
  -DCMAKE_INSTALL_PREFIX=$CUDAQX_INSTALL_PREFIX \
  -DCUDAQX_INCLUDE_TESTS=ON \
  -DCUDAQX_BINDINGS_PYTHON=ON
cmake --build build_qec -j
cmake --build build_qec --target run_tests
cmake --build build_qec --target run_python_tests
```

Substitute `libs/solvers` and `build_solvers` for solvers.

**Helper**: `scripts/test_libs_builds.sh` iterates over every library
under `libs/` (excluding `core`) and does this for each. Override with
`-l libs/qec` or `-l libs/solvers`.

**Self-check**: per-lib `run_tests` target finishes green; `pip list |
grep cudaq_<lib>` (or in-tree `import` from the install prefix) returns
the expected package.

## Run example suites end-to-end

Compile and run every example as a smoke test.

**Driver**: `scripts/ci/test_examples.sh {qec|solvers|all}`. Walks
`examples/<lib>/python/*.py` running them with `python3`, and
`examples/<lib>/cpp/*.cpp` compiling with `nvq++ --enable-mlir
-lcudaq-<lib>` (default `--target=stim` for QEC). If a C++ file contains
a leading `// nvq++ ...` line, those options are used instead.

**Pre-reqs**:

- `/cudaq-install` on `PATH` and `PYTHONPATH` (matches container layout).
- `$HOME/.cudaqx/include` and `$HOME/.cudaqx/lib` populated by `ninja install`.

**Self-check**: every example exits 0. Failed tests are accumulated and
printed at the end with non-zero exit code.

## Clean / reset

When stuck, reset to a known state.

```bash
# Drop in-tree build artifacts.
rm -rf build build_qec build_solvers _skbuild

# Drop installed CUDA-QX.
rm -rf $HOME/.cudaqx

# Drop pip-installed packages (use the matching cu suffix).
pip uninstall -y cudaq-qec-cu12 cudaq-solvers-cu12 \
                 cudaq-qec-cu13 cudaq-solvers-cu13 \
                 cudaq-qec      cudaq-solvers
```

Then rebuild. `scripts/clean.sh` does the first two steps automatically;
`scripts/clean.sh --all` does all three plus the wheel-builder container.

`docker stop cudaqx_wheel_builder && docker rm cudaqx_wheel_builder`
also resets the wheel-builder container if `build_wheels.sh` is misbehaving.
