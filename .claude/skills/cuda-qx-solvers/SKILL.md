---
name: "cuda-qx-solvers"
title: "CUDA-QX Solvers"
description: >-
  CUDA-QX solvers guide for VQE, ADAPT-VQE, QAOA, GQE, molecular Hamiltonians,
  operator pools, optimizers, gradients, and PySCF-based chemistry workflows.
  Use when the user mentions cudaq_solvers, VQE, ADAPT-VQE, QAOA, GQE,
  create_molecule, jordan_wigner, bravyi_kitaev, operator pools,
  UCCSD/UCCGSD/CEO, or chemistry/optimization workflows in CUDA-QX.
version: "0.3.0"
author: "CUDA-QX"
license: "Apache License 2.0"
compatibility: "Python 3.11+, C++ 20, Linux x86_64/aarch64"
tags: [cuda-qx, solvers, chemistry, optimization, vqe, adapt-vqe, qaoa, gqe]
tools: [Read, Glob, Grep, Bash]
metadata:
  author: "CUDA-QX"
  domain: "quantum-applications"
  languages: [python, c++]
  tags: [cuda-qx, solvers, chemistry, optimization, vqe, adapt-vqe, qaoa, gqe]
---

# CUDA-QX Solvers

Operate the `cudaq_solvers` library on this repo. The skill is algorithm-driven:
pick an algorithm, follow its quick-start, then walk the **Self-Check** before
reporting done. Do not invent API names from memory; look them up in the
**Source of Truth** table.

## How to read this skill

1. **First three actions** below: run `_shared/scripts/preflight.sh`,
   `import_smoke.py`, `pick_workflow.py`. They surface common blockers
   (missing `[gqe]` extras, mismatched PyTorch CUDA, stale PySCF server)
   before you spend tokens guessing.
2. Read **Conventions** below. It prevents the most common silent
   correctness mistakes (`OMP_NUM_THREADS`, `MP2` requirement, ADAPT vs
   VQE option keys, `num_qubits` vs `num_orbitals`).
3. Find your task in the **Algorithm Index** and open the matching file
   under `references/`.
4. Walk the per-algorithm **Self-Check** in the reference file before
   declaring done.

If you only have time for one section, read the **Quick Start: H2 VQE**
below plus **Conventions**. Together they cover ~70% of solver tasks.

## Audience

AI coding agents and developers operating `cudaq_solvers` from a
checkout of this repo. Pip-only users (`pip install cudaq-solvers`) get
the same Python API but cannot read the local template files referenced
below; they should consult the published docs at
<https://nvidia.github.io/cudaqx/>.

The artifact is `SKILL.md` (uppercase) inside
`.claude/skills/cuda-qx-solvers/`. Companion files live in
`references/` (see Algorithm Index below). Build/wheel/docs questions
belong to the `cuda-qx-build` skill, not this one.

## First three actions (always, before anything else)

```bash
bash   .claude/skills/_shared/scripts/preflight.sh    --json > /tmp/preflight.json
python .claude/skills/_shared/scripts/import_smoke.py --json > /tmp/import_smoke.json
python .claude/skills/_shared/scripts/pick_workflow.py \
    --intent <pick from list below> \
    --preflight /tmp/preflight.json \
    --imports   /tmp/import_smoke.json
```

Solvers-skill intents: `vqe`, `qaoa`, `gqe`, `chemistry`.
`pick_workflow.py` returns the next reference file to read and the
commands to run. Use it especially for `gqe` runs — the script catches
missing extras and PyTorch/CUDA SM mismatches before a long training
run starts.

## Standard imports

```python
import cudaq
from cudaq import spin
import cudaq_solvers as solvers
import numpy as np
```

## Key Paths

A one-glance map of the solvers library on disk. Full repo map (build, docs,
shared paths) lives at `.claude/skills/_shared/repo_map.md`.

| Area                 | Path                                                                  |
|----------------------|-----------------------------------------------------------------------|
| Library              | `libs/solvers/`                                                       |
| Python package       | `libs/solvers/python/cudaq_solvers/`                                  |
| Python bindings      | `libs/solvers/python/bindings/solvers/`                               |
| C++ headers          | `libs/solvers/include/cudaq/solvers/`                                 |
| C++ implementation   | `libs/solvers/lib/`                                                   |
| PySCF driver (C++)   | `libs/solvers/lib/operators/molecule/drivers/pyscf_driver.cpp`        |
| PySCF tool (Python)  | `libs/solvers/tools/molecule/cudaq-pyscf.py`                          |
| Python tests         | `libs/solvers/python/tests/`                                          |
| C++ tests            | `libs/solvers/unittests/`                                             |
| Examples             | `docs/sphinx/examples/solvers/`, `docs/sphinx/examples_rst/solvers/`  |
| Docs (component/api) | `docs/sphinx/components/solvers/`, `docs/sphinx/api/solvers/`         |

## Source of Truth

Look here before guessing an API name.

| Need to know                              | Authoritative file                                                  |
|-------------------------------------------|---------------------------------------------------------------------|
| Full Python public API                    | `libs/solvers/python/cudaq_solvers/__init__.py`                     |
| C++ VQE / ADAPT / QAOA signatures         | `libs/solvers/include/cudaq/solvers/vqe.h`, `adapt.h`, `qaoa.h`     |
| C++ molecule / operator surface           | `libs/solvers/include/cudaq/solvers/operators/molecule.h`           |
| Optimizer / gradient registry             | `libs/solvers/include/cudaq/solvers/optimizer.h`, `stateprep.h`     |
| Python bindings (definitive Python API)   | `libs/solvers/python/bindings/solvers/py_solvers.cpp`, `py_optim.cpp`|
| GQE algorithm                             | `libs/solvers/python/cudaq_solvers/gqe_algorithm/gqe.py`            |

When the user asks "is there a function for X?", grep
`libs/solvers/python/cudaq_solvers/__init__.py` and the relevant header before
answering.

## Algorithm Index

| If the user wants to                                            | Read                       | `pick_workflow.py` intent |
|-----------------------------------------------------------------|----------------------------|---------------------------|
| VQE, ADAPT-VQE, optimizers/gradients, state prep                | `references/vqe.md`        | `vqe`                     |
| QAOA: MaxCut, clique, QUBO, mixers                              | `references/qaoa.md`       | `qaoa`                    |
| GQE: train a transformer to propose operators (multi-GPU OK)    | `references/gqe.md`        | `gqe`                     |
| Molecule, operator pools, fermion-to-qubit mappings (PySCF)     | `references/chemistry.md`  | `chemistry`               |

Each reference file carries its own Self-Check (the failure modes are
algorithm-specific). `chemistry.md` also has the cross-cutting
troubleshooting table for install/runtime issues.

## Algorithm entry points

| Algorithm | Entry point                                      | Returns                                              |
|-----------|--------------------------------------------------|------------------------------------------------------|
| VQE       | `solvers.vqe(kernel, spin_op, init_params, **opts)` | `(energy, params, list[ObserveIteration])`        |
| ADAPT-VQE | `solvers.adapt_vqe(initial_state, ham, pool, **opts)` | `(energy, params, selected_ops)`                |
| QAOA      | `solvers.qaoa(Hp, [Href,] num_layers, init_params, **opts)` | `QAOAResult(optimal_value, optimal_parameters, optimal_config)` |
| GQE       | `solvers.gqe(cost, pool, config=None, **kwargs)` | `(min_energy, best_op_indices)`                      |

## Quick Start: H2 VQE

```python
import cudaq
from cudaq import spin
import cudaq_solvers as solvers

@cudaq.kernel
def ansatz(theta: float):
    q = cudaq.qvector(2)
    x(q[0])
    ry(theta, q[1])
    x.ctrl(q[1], q[0])

H = (5.907 - 2.1433 * spin.x(0) * spin.x(1)
     - 2.1433 * spin.y(0) * spin.y(1)
     + 0.21829 * spin.z(0) - 6.125 * spin.z(1))

energy, params, data = solvers.vqe(
    lambda thetas: ansatz(thetas[0]),
    H, [0.0],
    optimizer="lbfgs", gradient="parameter_shift", verbose=True)
```

Variant for a real molecule: replace `H` with
`solvers.create_molecule(...).hamiltonian` and the ansatz with one of the
state prep kernels (see `references/vqe.md` "State Preparation Kernels").

## Installation and environment

- `pip install cudaq-solvers` (also `cudaq-solvers-cu12` / `cudaq-solvers-cu13`
  for explicit CUDA pairing). The internal native module is
  `_pycudaqx_solvers_the_suffix_matters_cudaq_solvers`.
- Optional extras: `pip install cudaq-solvers[gqe]` adds `torch>=2.0.0`,
  `lightning>=2.0.0`, `ml_collections`, `mpi4py>=3.1.0`,
  `transformers>=4.30.0`.
- Useful environment variables (set before `import cudaq_solvers`):
  - `OMP_NUM_THREADS=1` for reproducible Hamiltonian coefficients (PySCF).
  - `PMIX_MCA_gds=hash` when launching multi-GPU GQE under `mpiexec`.

## Conventions

These are the recurring mistakes. Code that violates any of them usually runs
but produces wrong numbers, hangs, or crashes during install.

1. **`OMP_NUM_THREADS=1` for reproducible chemistry runs.** PySCF flips signs
   of Hamiltonian coefficients across runs with multiple threads (eigenvalues
   stay correct). If a user reports "different signs every run", this is it.

2. **`natorb=True` and `integrals_natorb=True` require `MP2=True`.** Otherwise
   `create_molecule` raises a `RuntimeError`. `casci=True`, `ccsd=True`, and
   `casscf=True` activate their own energy keys (see "Molecule Workflow" in
   `references/chemistry.md`).

3. **`num_qubits = 2 * num_orbitals` for chemistry codes.** Operator pools
   take *either* `num_qubits` (UCCSD) *or* `num_orbitals` (UCCGSD, UPCCGSD,
   spin-complement-GSD, CEO). Mixing them is the most common pool bug.

4. **ADAPT uses `max_iter`; VQE uses `max_iterations`.** Different keys, same
   intent. Setting the wrong one silently caps iterations at the default
   (30 for ADAPT, no limit for VQE).

5. **`lbfgs` requires gradients; `cobyla` does not.** `solvers.qaoa(...,
   optimizer="lbfgs")` fails with "requires gradients...gradient instance not
   provided" because no gradient is wired into QAOA's optimizer call. Use
   `cobyla` or pass a SciPy optimizer.

6. **Custom optimizers must be `scipy.optimize.minimize`.** Other callables
   raise `RuntimeError("Invalid functional optimizer provided ...")`. The
   wrapper strips `gradient`, `optimizer`, `verbose`, `shots` before calling
   `minimize`; everything else (`method`, `jac`, `tol`, `options`,
   `callback`) is forwarded.

7. **PySCF server lives on `localhost:8000`.** `create_molecule` spawns
   `cudaq-pyscf --server-mode`. If a previous run crashed, the port can stay
   occupied:
   ```bash
   lsof -n -i :8000
   kill -9 <pid>
   ```
   Symptom: a second `create_molecule` call hangs forever.

8. **`libgfortran` is a runtime dependency.** Built-in `cobyla` and `lbfgs`
   crash on Linux without it. Install via the system package manager
   (`apt install libgfortran5` or equivalent). Linux x86_64/aarch64 only;
   macOS and Windows are not supported.

## When stuck

1. Re-run the **First three actions**. Missing extras (`[gqe]`,
   `libgfortran`) and a stale PySCF server are the most common silent
   blockers.
2. Open the matching `references/<algo>.md` and read it end-to-end
   before generating new code.
3. Grep `libs/solvers/python/cudaq_solvers/__init__.py` for the suspected
   API name.
4. Read the C++ header in `libs/solvers/include/cudaq/solvers/` for the
   canonical signature.
5. Reproduce with the H2 quick-start before scaling up — it isolates
   whether the issue is in the optimizer/gradient wiring or in the
   molecule pipeline.
6. If chemistry coefficients look unstable, set `OMP_NUM_THREADS=1` and
   rerun.

## Additional resources

- Algorithm references: `references/vqe.md`, `references/qaoa.md`,
  `references/gqe.md`, `references/chemistry.md`
- Shared diagnostic scripts: `.claude/skills/_shared/scripts/`
- Shared repo map (incl. build/docs paths): `.claude/skills/_shared/repo_map.md`
- Build/wheel/docs questions: `.claude/skills/cuda-qx-build/SKILL.md`
