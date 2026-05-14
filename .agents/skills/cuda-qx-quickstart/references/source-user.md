# Source / from-clone path

For contributors and advanced researchers who need to modify
CUDA-QX or recompile against a different CUDA-Q. If the user just
wants to *try* cudaqx, see `pip-user.md` or `docker-user.md` first.

## When to choose source

| Situation | Source build needed? |
|-----------|---------------------|
| Try out a published algorithm | no — pip |
| Run examples from the docs | no — pip or docker |
| Modify a C++ decoder or code | yes |
| Add a new Python plugin (decoder or code) | no — pip + your own file |
| Develop against an unreleased CUDA-Q | yes |
| Cut a wheel for distribution | yes |

When in doubt, start with pip + your own Python file. Most extension
work (custom decoder, custom code, custom operator pool) can be done
without rebuilding the C++.

## Quick clone + build

```bash
git clone --recursive https://github.com/NVIDIA/cudaqx.git
cd cudaqx
# Install CUDA-Q first if not already installed (default: $HOME/.cudaq)
# Build CUDA-QX:
mkdir -p build && cd build
cmake -G Ninja -S .. \
  -DCUDAQ_DIR=$HOME/.cudaq/lib/cmake/cudaq \
  -DCMAKE_INSTALL_PREFIX=$HOME/.cudaqx \
  -DCUDAQX_ENABLE_LIBS=all \
  -DCUDAQX_INCLUDE_TESTS=ON \
  -DCUDAQX_BINDINGS_PYTHON=ON \
  -DCMAKE_BUILD_TYPE=Release
ninja install
```

If any of that fails, delegate to **`cuda-qx-build` SKILL.md** —
that skill owns build issues end-to-end.

After a successful build, set up the environment:

Set the install-path environment block from `cuda-qx-build/SKILL.md`
("Standard environment" section). The minimum two:

```bash
export CUDAQ_INSTALL_PREFIX=$HOME/.cudaq
export CUDAQX_INSTALL_PREFIX=$HOME/.cudaqx
```

See `cuda-qx-build` for the full block (`CUDAQ_DIR`, `PATH`, `PYTHONPATH`).

Sanity check:

```bash
python3 -c "import cudaq_qec; import cudaq_solvers; print('ok')"
```

## Use the dev container instead (recommended)

For Windows or any "works on my machine" situation, the published dev
container removes a lot of pain:

```bash
docker pull ghcr.io/nvidia/cudaqx-dev:latest-amd64-cu12.6
docker run --gpus all -it \
  -v $(pwd):/workspaces/cudaqx \
  ghcr.io/nvidia/cudaqx-dev:latest-amd64-cu12.6
```

Inside the container, CUDA-Q lives at `/usr/local/cudaq` (not
`$HOME/.cudaq`) — see `cuda-qx-build/references/build.md`.

VSCode "Dev Containers" extension picks up `.devcontainer/` if present
and gives you the same setup with one click.

## Run the first example from source

```bash
python3 docs/sphinx/examples/qec/python/code_capacity_noise.py
python3 docs/sphinx/examples/solvers/python/uccsd_vqe.py
```

If `code_capacity_noise.py` runs and prints a logical error rate,
the QEC half of your install is good. If `uccsd_vqe.py` runs and
prints an energy, the solvers half is good.

## What to read next

| Goal | Skill to open |
|------|---------------|
| Build issues, wheels, docs, containers | `cuda-qx-build` |
| QEC workflows (decode, custom code, real-time) | `cuda-qx-qec-decode` |
| Variational solvers (VQE, ADAPT, QAOA, GQE) | `cuda-qx-solvers-algorithms` |
| Add a new decoder or code | `cuda-qx-qec-extending` |
| Add a new operator pool / optimizer | `cuda-qx-solvers-extending` |
| Contribute a PR | `cuda-qx-contributing` |
| Run/extend tests, CI | `cuda-qx-testing-ci` |

## When stuck

1. `bash scripts/doctor.sh` for a structured environment dump.
2. `bash scripts/clean.sh` (when present) wipes `$HOME/.cudaqx` and
   `build/` to start over.
3. `bash .agents/skills/_shared/scripts/preflight.sh --json` reports
   missing pieces in machine-readable form.
4. Open `cuda-qx-build` SKILL.md and walk the workflow index.
