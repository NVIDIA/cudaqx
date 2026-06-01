# Triage: ImportError, version mismatch, ABI confusion

The "something is broken at install time" entry point. When the user
reports `ImportError`, segfault on first call, or "version mismatch",
walk this file before guessing.

## CUDA 12 vs 13 (read this first)

CUDA-Q Libraries ships two ABIs in parallel. They are **not interchangeable**.

| Aspect                    | CUDA 12 (`cu12`)                                           | CUDA 13 (`cu13`)                          |
|---------------------------|------------------------------------------------------------|-------------------------------------------|
| Wheels                    | `cudaq-qec-cu12`, `cudaq-solvers-cu12`                    | `cudaq-qec-cu13`, `cudaq-solvers-cu13`    |
| Pyproject                 | `libs/{qec,solvers}/pyproject.toml.cu12`                  | `libs/{qec,solvers}/pyproject.toml.cu13`  |
| Companion CUDA-Q wheel    | `cuda-quantum-cu12`                                        | `cuda-quantum-cu13`                       |
| Companion cuQuantum       | `cuquantum-python-cu12==26.03.x`                          | `cuquantum-python-cu13==26.03.x`          |
| Dev container             | `cudaqx-dev:latest-{amd64,arm64}-cu12.6`                  | `cudaqx-dev:latest-{amd64,arm64}-cu13.0`  |
| Manylinux builder image   | `ghcr.io/nvidia/cuda-quantum-devdeps:manylinux-amd64-cu12.6-gcc11-main` | `...-cu13.0-gcc11-main` |

The metapackages `cudaq-qec` / `cudaq-solvers` are sdists that, at
install time, pick the right `-cu1x` based on the installed CUDA-Q. A
user who installs `cuda-quantum-cu12` and then `cudaq-qec` ends up with
`cudaq-qec-cu12`. Mixing versions across the boundary
(`cuda-quantum-cu12` + `cudaq-qec-cu13`) is a silent ABI mismatch that
surfaces as crashes or import errors at runtime.

## Doctor pass — gather facts before guessing

When the user reports `ImportError: lib... not found` or a version
mismatch error, run `scripts/doctor.sh` (or capture the equivalent
manually):

```bash
echo "--- versions ---"
cmake --version | head -n 1
ninja --version
python3 --version
gcc --version | head -n 1
nvidia-smi 2>/dev/null | head -n 5

echo "--- pip ---"
pip list | grep -E '^(cuda-quantum|cudaq-(qec|solvers)|cuquantum|tensorrt|torch)'

echo "--- env ---"
env | grep -E '^(CUDAQ|CUDAQX|CUDA|PYTHON|LD_LIBRARY|OMP)'

echo "--- install prefix ---"
ls -la $HOME/.cudaq/  2>/dev/null | head
ls -la $HOME/.cudaq/ 2>/dev/null | head
```

`scripts/doctor.sh --json` prints a JSON snapshot for machine-readable
consumption. For agent workflows prefer `_shared/scripts/preflight.sh
--json`, which is a superset (adds GPU/CPU detail, venv discovery,
submodules, and build state).

## The five most common findings

| Finding                                                           | Cause                                  | Fix                                              |
|-------------------------------------------------------------------|----------------------------------------|--------------------------------------------------|
| `cuda-quantum-cu12` + `cudaq-qec-cu13` (or vice versa)           | Mixed CUDA ABI                         | Reinstall both with matching suffix              |
| `cuquantum-python-cu1x` < 26.03                                   | Old cuQuantum                          | `pip install -U "cuquantum-python-cu1x>=26.03.0"`|
| Stale files in `$HOME/.cudaq/lib/`                               | Branch switch without clean install    | `rm -rf $HOME/.cudaq build && rebuild`          |
| `CUDAQ_DIR` unset, build picks wrong CUDA-Q                       | Missing env / `-DCUDAQ_DIR=`           | Export and pass `-DCUDAQ_DIR=...`                |
| `gfortran` missing                                                | System pkg gap                         | `sudo apt install -y gfortran libblas-dev`       |

## Self-Check Protocol (before reporting "done")

```
[ ] CUDA-Q is installed and CUDAQ_DIR points at it.
[ ] CUDAQX_ENABLE_LIBS is set as intended (qec, solvers, or all).
[ ] System packages present: gfortran, libblas-dev, cmake>=3.28, ninja>=1.10.
[ ] CUDA suffix is consistent across cuda-quantum-cuXX and cudaq-{qec,solvers}-cuXX.
[ ] Built artifacts live under CMAKE_INSTALL_PREFIX (default $HOME/.cudaq).
[ ] Stale $HOME/.cudaq and old build/ removed before a "from scratch" rebuild.
[ ] If wheels: auditwheel show reports manylinux_2_27 (or later) tag.
[ ] If docs: $CUDAQX_INSTALL_PREFIX/docs/index.html opens with non-empty API tables.
[ ] If examples: test_examples.sh exits 0 on the relevant lib(s).
[ ] If validation: validate_container.sh / validate_wheels.sh return 0.
```

When the user reports "build looks fine but import fails", boxes 4 and 6
catch the majority of cases.

## Troubleshooting matrix

| Symptom                                                       | Likely cause                                                | Fix                                                            |
|---------------------------------------------------------------|-------------------------------------------------------------|----------------------------------------------------------------|
| `ImportError: libcustabilizer ...`                            | cuQuantum < 26.03 or wrong CUDA suffix                      | `pip install -U "cuquantum-python-cu1x>=26.03.0"`              |
| `ImportError: libcudart.so.X ...`                             | `nvidia-cuda-runtime-cuXX` missing or major mismatch         | Install matching `nvidia-cuda-runtime-cuXX`                    |
| `cobyla`/`lbfgs` segfault on first call                       | `libgfortran.so.5` missing                                  | `apt install -y libgfortran5` (or `gfortran` package)          |
| C++20 features fail (`<concepts>`, `<ranges>`)                | Old GCC                                                     | Use GCC 11+ (`source /opt/rh/gcc-toolset-11/enable` in manylinux) |
| `cmake: command not found in venv`                            | `pip install cmake` not in `--user` site                    | Use system cmake or `pip install "cmake<4" --user`             |
| `find_package(LLVM)` fails                                    | LLVM headers missing                                        | Set `-DLLVM_DIR=...` to the matching CUDA-Q install            |
| `breathe_projects.cudaq` empty / Doxygen XML missing         | Doxygen failed; check `$logs_dir/doxygen_error.txt`         | Fix header parse error or rerun `build_docs.sh`                |
| Sphinx API tables blank                                       | `CUDAQX_DOCS_GEN_IMPORT_CUDAQ` unset, or `PYTHONPATH` wrong | Use `scripts/build_docs.sh`; ensure install prefix on path     |
| `auditwheel` complains about `libcudaq.so` not in policy      | Excludes list out of date                                   | Add the `lib*.so` to the `--exclude` list in the wheel script  |
| Wheel build container does not pick up source changes         | Stale `cudaqx_wheel_builder` container                      | `docker stop cudaqx_wheel_builder && docker rm cudaqx_wheel_builder`, rerun |
| `ninja install` succeeds but `import cudaq_qec` 404s          | `$CUDAQX_INSTALL_PREFIX` not on `PYTHONPATH`                | Export `PYTHONPATH=$CUDAQX_INSTALL_PREFIX:$PYTHONPATH`         |
| `ImportError: cudaq` after wheel install                      | `cuda-quantum-cu1x` ABI mismatch with installed CUDA-Q      | Reinstall both with matching `-cu1x`                           |
| Tests pass locally, fail in CI on arm64                       | `tensorrt-cu12` not available on arm64                      | `--ignore` `test_trt_decoder.py`; gate by arch                 |
| `docs/sphinx/examples/qec/cpp/*.cpp` link errors              | Wrong `nvq++` flags                                         | Default is `--target=stim -lcudaq-qec`; check leading `// nvq++` |
| Real-time C++ link errors on Quantinuum target                | Missing `-lcudaq-qec-realtime-decoding-quantinuum -Wl,--export-dynamic` | Add to nvq++ link line (see `cudaq-qec-realtime/references/hardware-helios.md`) |
| `cuda-quantum` build patch fails to apply                     | Upstream CUDA-Q moved past the patch                        | Update the inline patch in `scripts/ci/build_cudaq_wheel.sh`   |
| `OMP_NUM_THREADS` warning during chemistry tests              | Docs/build scripts pin `OMP_NUM_THREADS=1`                  | Expected for reproducibility — see `cudaq-solvers-chemistry` (Convention #1) |
