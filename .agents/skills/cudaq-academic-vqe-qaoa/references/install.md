# Install And Smoke Test

Use this for beginner installation questions before VQE or QAOA.

## Recommended Workshop Path

For students in a workshop, start from the provided Brev environment when one
is available. This avoids spending class time on local CUDA, Linux, driver, or
compiler setup and gives everyone the same baseline.

GPU acceleration is useful for larger experiments, but it is not required for
the small VQE and QAOA learning examples in this skill. Students can validate
the install and prototype the examples on CPU, then use the provided Brev/GPU
setup when the class experiment needs acceleration or a standardized runtime.

Inside the Brev environment, install the Solvers package:

```bash
python3 -m pip install cudaq-solvers
```

Then verify the two imports the examples need:

```bash
python3 - <<'PY'
import cudaq
import cudaq_solvers as solvers
print("cudaq:", cudaq.__name__)
print("cudaq_solvers:", solvers.__name__)
PY
```

If those imports work, students are ready to run the small VQE and QAOA
examples.

## Local Linux Path

For students using their own Linux machine, including CPU-only machines:

```bash
python3 -m pip install cudaq-solvers
```

Then verify the two imports the examples need:

```bash
python3 - <<'PY'
import cudaq
import cudaq_solvers as solvers
print("cudaq:", cudaq.__name__)
print("cudaq_solvers:", solvers.__name__)
PY
```

If the user wants both CUDA-QX libraries, use:

```bash
python3 -m pip install cudaq-qec cudaq-solvers
```

Do not suggest `cudaq-solvers[gqe]` for this academic VQE/QAOA path. The
`[gqe]` extra pulls in PyTorch-oriented dependencies for Generative Quantum
Eigensolver workflows, which are out of scope here.

## Common Install Note

CUDA-Q Solvers uses classical optimizers. On Linux, missing `libgfortran` can
break optimizer-backed workflows. On Debian-style systems:

```bash
sudo apt-get install gfortran
```

## Docker Path

For Mac, Windows, or anyone who cannot use Brev but wants a prebuilt
environment:

```bash
docker pull ghcr.io/nvidia/cudaqx
docker run --gpus all -it ghcr.io/nvidia/cudaqx
```

Omit `--gpus all` if the machine has no NVIDIA GPU.

## Quick Decision Guide

- Workshop student: use the provided Brev environment; CPU is fine for the
  small examples, and GPU is helpful for larger class experiments.
- Linux machine: use the local pip path, even on CPU-only machines, and install
  `libgfortran`/`gfortran` if optimizer-backed workflows fail.
- Mac or Windows: prefer Brev; use Docker only if the user is already
  comfortable with containers.
- No GPU: run the small workshop examples on CPU; use Brev when the workshop
  needs the standard class setup.

## Source Paths

- Installation docs: `docs/sphinx/quickstart/installation.rst`
- Solvers package config: `libs/solvers/pyproject.toml.cu12`,
  `libs/solvers/pyproject.toml.cu13`
- Wheel validation: `scripts/ci/test_wheels.sh`
