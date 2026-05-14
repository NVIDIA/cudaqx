---
name: "cuda-qx-solvers-algorithms"
title: "CUDA-QX Solvers — Algorithm Dispatch"
description: >-
  Algorithm dispatch for cudaq_solvers: pick and configure VQE, ADAPT-VQE,
  QAOA, or GQE; wire optimizers and gradients; assemble the variational
  loop. Use when the user mentions cudaq_solvers, VQE, ADAPT-VQE, QAOA,
  GQE, optimizer, gradient, parameter_shift, lbfgs, cobyla, or "which
  algorithm should I use". Do NOT use for: chemistry setup (molecule
  construction, PySCF, basis sets, active spaces, operator pools — use
  `cuda-qx-solvers-chemistry`); writing a new operator pool / optimizer
  (use `cuda-qx-solvers-extending`).
version: "0.3.2"
author: "CUDA-QX"
license: "Apache License 2.0"
compatibility: "Python 3.11+, C++ 20, Linux x86_64/aarch64"
tags: [cuda-qx, solvers, chemistry, optimization, vqe, adapt-vqe, qaoa, gqe]
tools: [Read, Glob, Grep, Bash]
metadata:
  repo: [solvers]
  author: "CUDA-QX"
  domain: "quantum-applications"
  languages: [python, c++]
---

# CUDA-QX Solvers

Operate the `cudaq_solvers` library on this repo. The skill is algorithm-driven:
pick an algorithm, follow its quick-start, then walk the **Self-Check** before
reporting done. Do not invent API names from memory; look them up in the
**Source of Truth** table.

## Inputs

Caller provides:

- A qubit Hamiltonian (`cudaq.SpinOperator`) — typically from
  `cuda-qx-solvers-chemistry`'s output, or constructed directly.
- An ansatz: a `@cudaq.kernel` for VQE/ADAPT, or `(Hp, [Href], num_layers)`
  for QAOA.
- An algorithm choice: `vqe` / `adapt_vqe` / `qaoa` / `gqe`.
- Optimizer + gradient: `lbfgs`/`cobyla`/SciPy callable;
  `parameter_shift`/`finite_difference`/None.
- For ADAPT/GQE: an operator pool (list of `SpinOperator` or pool name).

## Outputs

This skill produces:

- `(energy, params, iteration_log)` for VQE.
- `(energy, params, selected_ops)` for ADAPT-VQE.
- `QAOAResult(optimal_value, optimal_parameters, optimal_config)` for QAOA.
- `(min_energy, best_op_indices)` for GQE.

Does NOT produce: the molecule itself (→ `cuda-qx-solvers-chemistry`);
new operator pools / optimizers (→ `cuda-qx-solvers-extending`).

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
`.agents/skills/cuda-qx-solvers-algorithms/` (mirrored to `.claude/skills/` and
`.cursor/skills/`). Companion files live in `references/` (see
Algorithm Index below). Build/wheel/docs questions belong to the
`cuda-qx-build` skill, not this one.

If the user is new to CUDA-QX or hasn't run an example yet, delegate
to **`cuda-qx-quickstart`** first; come back here once they have a
working install. For deep chemistry setup (PySCF, active spaces,
basis sets, classical baselines), delegate to **`cuda-qx-solvers-chemistry`**.

## First three actions (always, before anything else)

```bash
bash   .agents/skills/_shared/scripts/preflight.sh    --json > /tmp/preflight.json
python .agents/skills/_shared/scripts/import_smoke.py --json > /tmp/import_smoke.json
python .agents/skills/_shared/scripts/pick_workflow.py \
    --intent <pick from list below> \
    --preflight /tmp/preflight.json \
    --imports   /tmp/import_smoke.json
```

Solvers-skill intents: `vqe`, `qaoa`, `gqe` (algorithm dispatch only).
For `chemistry` intent, dispatch to `cuda-qx-solvers-chemistry`.
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
shared paths) lives at `.agents/skills/_shared/repo_map.md`.

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

This skill covers **algorithm dispatch only**. Chemistry setup
(molecules, basis sets, active spaces, operator pools) belongs to
`cuda-qx-solvers-chemistry`. Plugin authoring (custom pools /
optimizers) belongs to `cuda-qx-solvers-extending`.

| If the user wants to                                            | Skill / file                               |
|-----------------------------------------------------------------|--------------------------------------------|
| VQE, ADAPT-VQE, optimizers/gradients, state prep                | **this skill** → `references/vqe.md`       |
| QAOA: MaxCut, clique, QUBO, mixers                              | **this skill** → `references/qaoa.md`      |
| GQE: train a transformer to propose operators (multi-GPU OK)    | **this skill** → `references/gqe.md`       |
| Molecule, operator pools, fermion-to-qubit mappings (PySCF)     | `cuda-qx-solvers-chemistry`                |
| Custom operator pool / optimizer / state-prep                   | `cuda-qx-solvers-extending`                |

Each reference file carries its own Self-Check (the failure modes are
algorithm-specific).

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
  - `OMP_NUM_THREADS=1` — chemistry reproducibility; see
    `cuda-qx-solvers-chemistry` for why.
  - `PMIX_MCA_gds=hash` when launching multi-GPU GQE under `mpiexec`.

## Conventions

These are the recurring algorithm-dispatch mistakes. Code that violates
any of them runs but caps iterations, picks the wrong gradient, or
crashes the optimizer. For chemistry-specific conventions
(`OMP_NUM_THREADS`, `MP2`/`natorb`, `num_qubits` vs `num_orbitals`,
PySCF server), see `cuda-qx-solvers-chemistry` SKILL.md.

1. **ADAPT uses `max_iter`; VQE uses `max_iterations`.** Different keys,
   same intent. Setting the wrong one silently caps iterations at the
   default (30 for ADAPT, no limit for VQE).

2. **`lbfgs` requires gradients; `cobyla` does not.** `solvers.qaoa(...,
   optimizer="lbfgs")` fails with "requires gradients...gradient instance
   not provided" because no gradient is wired into QAOA's optimizer call.
   Use `cobyla` or pass a SciPy optimizer.

3. **Custom optimizers must be `scipy.optimize.minimize`.** Other callables
   raise `RuntimeError("Invalid functional optimizer provided ...")`. The
   wrapper strips `gradient`, `optimizer`, `verbose`, `shots` before
   calling `minimize`; everything else (`method`, `jac`, `tol`, `options`,
   `callback`) is forwarded.

4. **`libgfortran` is a runtime dependency** for `cobyla` and `lbfgs`.
   Quick fix: `apt install gfortran`. Full explanation + symptoms in
   `cuda-qx-build` (Convention #3). Linux x86_64/aarch64 only.

## When stuck

1. Re-run the **First three actions**. Missing extras (`[gqe]`,
   `libgfortran`) are the most common silent blockers.
2. Open the matching `references/<algo>.md` and read it end-to-end
   before generating new code.
3. Grep `libs/solvers/python/cudaq_solvers/__init__.py` for the suspected
   API name.
4. Read the C++ header in `libs/solvers/include/cudaq/solvers/` for the
   canonical signature.
5. Reproduce with the H2 quick-start before scaling up — it isolates
   whether the issue is in the optimizer/gradient wiring.
6. If the bug looks chemistry-related (PySCF, basis sets, active spaces,
   sign flips on Hamiltonian coefficients), delegate to
   `cuda-qx-solvers-chemistry`.

## Additional resources

- Algorithm references: `references/vqe.md`, `references/qaoa.md`,
  `references/gqe.md`
- Shared diagnostic scripts: `.agents/skills/_shared/scripts/`
- Shared repo map (incl. build/docs paths): `.agents/skills/_shared/repo_map.md`

### Related skills

- **`cuda-qx-quickstart`** — onboarding, first example, pip vs source vs Docker.
- **`cuda-qx-solvers-chemistry`** — PySCF, molecule construction, active space,
  operator pools, fermion-to-qubit mappings, classical baselines.
- **`cuda-qx-solvers-extending`** — write a new operator pool, optimizer, or
  state-prep kernel.
- **`cuda-qx-benchmarking`** — VQE / ADAPT / GQE energy and time-to-
  chemical-accuracy comparisons.
- **`cuda-qx-profiling-perf`** — `nsys` / `ncu` for slow VQE / GQE loops.
- **`cuda-qx-build`** — build / wheel / docs / container.
