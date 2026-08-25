---
name: "cudaq-solvers-extending"
title: "CUDA-Q Libraries Solvers Extending (operator pools, optimizers, state-prep)"
description: >-
  Plugin-author skill for the solvers library: define a new operator
  pool plugin (UCCSD-like, GSD-like, custom), wire a new classical
  optimizer plugin, add a state-prep / ansatz kernel plugin, or plug
  in a new gradient method. Use whenever the user mentions "custom
  operator pool", "register an optimizer", "plugin operator pool",
  "custom ansatz", "custom state prep", "custom gradient method", or
  wants to extend the solvers stack beyond VQE / ADAPT / QAOA / GQE
  defaults. Do NOT use for: picking from the stock optimizers
  (cobyla, lbfgs, parameter-shift) — use cudaq-solvers-algorithms;
  configuring the molecule, basis set, or active space (use
  cudaq-solvers-chemistry); QEC code or decoder extensions (use
  cudaq-qec-extending).
version: "0.2.1"
author: "CUDA-Q Libraries"
license: "Apache License 2.0"
compatibility: "Python 3.11+, C++ 20, CMake 3.28+, Linux x86_64/aarch64"
tags: [cudaq, cudaq-solvers, plugin, extension, operator-pool, optimizer, state-prep, gradient]
tools: [Read, Glob, Grep, Bash]
metadata:
  repo: [solvers]
  author: "CUDA-Q Libraries"
  domain: "solvers-extension"
  audience: [researcher, plugin-developer, contributor]
  languages: [python, c++]
---

# CUDA-Q Libraries Solvers Extending

The "I want to add a new operator pool / optimizer / state-prep /
gradient method to cudaq-solvers" skill. Solvers are Apache-2.0
licensed (unlike `libs/qec`), so the bar for upstreaming is lower.

If the user wants to *use* the existing algorithms, delegate to
`cudaq-solvers-algorithms`. For chemistry-specific setup (PySCF, active
spaces, fermion mappings), delegate to `cudaq-solvers-chemistry`.
For QEC code/decoder extensions, see the cudaq-qec-decode package
(`cudaq-qec-extending` skill).

## Inputs

Caller provides:

- Extension type: `operator_pool` / `optimizer` / `gradient` /
  `state_prep`.
- Language: Python (prototype) or C++ (production).
- For operator pools: required kwargs (`num_qubits` for UCCSD-like,
  `num_orbitals` for GSD-like), the operator-construction logic.
- For optimizers: a `scipy.optimize.minimize`-compatible callable.
- For state-prep: a `@cudaq.kernel` / `__qpu__` function.
- Unique registration name.

## Outputs

This skill produces:

- New source files at the right repo paths
  (`libs/solvers/include/cudaq/solvers/operators/operator_pools/` etc.).
- Updated `CMakeLists.txt` for C++ additions.
- A smoke test under `libs/solvers/python/tests/` that confirms H2 ADAPT
  or VQE converges to FCI within sample noise using the new component.
- A short docs stub under `docs/sphinx/components/solvers/` or
  `examples_rst/solvers/`.

Does NOT produce: a chemistry-specific pool wrapper (→
`cudaq-solvers-chemistry`); QEC code/decoder plugins (→
`cudaq-qec-extending`).

## Audience

Researchers and library developers comfortable with Python and (for
C++ plugins) `pybind11`. Pure-Python extension is straightforward;
C++ is for shipping a feature upstream.

## First three actions

```bash
bash   .agents/skills/_shared/scripts/preflight.sh    --json > /tmp/preflight.json
python .agents/skills/_shared/scripts/import_smoke.py --json > /tmp/import_smoke.json
python -c "import cudaq_solvers as s; print([x for x in dir(s) if not x.startswith('_')])"
```

The third command shows what's currently exposed from the Python
API — useful to see whether the registration mechanism you need
already exists.

## Key Paths

| Area | Path |
|------|------|
| Solvers public API (Python) | `libs/solvers/python/cudaq_solvers/__init__.py` |
| Operator pool base (C++) | `libs/solvers/include/cudaq/solvers/operators/operator_pool.h` |
| Built-in pool headers | `libs/solvers/include/cudaq/solvers/operators/operator_pools/` |
| UCCSD / UCCGSD utilities | `libs/solvers/include/cudaq/solvers/operators/uccgsd_excitation_utils.h`, `ceo_excitation_utils.h` |
| Optimizer base | `libs/solvers/include/cudaq/solvers/optimizer.h` |
| Built-in optimizers | `libs/solvers/include/cudaq/solvers/optimizers/` |
| Gradient base | `libs/solvers/include/cudaq/solvers/observe_gradient.h` |
| Built-in gradients | `libs/solvers/include/cudaq/solvers/observe_gradients/` |
| State-prep kernels | `libs/solvers/include/cudaq/solvers/stateprep/` |
| Python bindings | `libs/solvers/python/bindings/solvers/py_solvers.cpp`, `py_optim.cpp` |

## Workflow Index

| If the user wants to | Read |
|----------------------|------|
| Define a new operator pool | `references/operator-pools-authoring.md` |
| Plug in a new classical optimizer or gradient method | `references/optimizers-authoring.md` |
| Add a state-prep / ansatz kernel | `references/state-prep-authoring.md` |

## Conventions

These are the recurring solvers-extension-time bugs.

1. **`num_qubits = 2 * num_orbitals`.** UCCSD pools take
   `num_qubits`; UCCGSD / UPCCGSD / spin-complement-GSD / CEO take
   `num_orbitals`. Mixing produces nonsense pools that silently give
   bad energies.

2. **Prefer prototyping in Python.** For ADAPT or GQE you can pass a
   `list[cudaq.SpinOperator]` directly without registering a pool.
   Register only when shipping upstream.

3. **Custom optimizers must be `scipy.optimize.minimize`-compatible.**
   Other callables raise `RuntimeError("Invalid functional optimizer
   provided ...")`. The wrapper strips `gradient`, `optimizer`,
   `verbose`, `shots` before calling `minimize`; everything else
   (`method`, `jac`, `tol`, `options`, `callback`) is forwarded.

4. **State-prep kernels run inside `@cudaq.kernel`/`__qpu__`.**
   No NumPy / SciPy / Python flow inside; only calls to other
   `@cudaq.kernel` functions.

5. **C++ plugins are Apache 2.0.** Pick the same license for new
   files. Don't drop in code under another license.

6. **Brillouin's theorem sanity check for new pools.** At the HF
   state, the gradient of every well-formed single excitation should
   be ~0 (canonical orbitals). If your new pool produces nonzero
   gradients there, the operator signs are off.

## Quick start: prototype a Python-only pool

```python
import cudaq, cudaq_solvers as solvers
from cudaq import spin

# Build your operator list however you like
ops = [spin.x(0) * spin.y(1) - spin.y(0) * spin.x(1),
       spin.x(2) * spin.y(3) - spin.y(2) * spin.x(3)]

# Use directly in ADAPT
energy, params, selected = solvers.adapt_vqe(initial_state, hamiltonian, ops,
                                             max_iter=10, verbose=True)
```

For shipping a pool upstream as a registered name, see
`references/operator-pools-authoring.md` (C++ registration via
`CUDAQ_REGISTER_OPERATOR_POOL`).

## Self-Check Protocol

```
[ ] Pool kwargs match the convention (num_qubits vs num_orbitals).
[ ] Operator pool size is sane (UCCSD: O(n^2), UCCGSD: O(n^4)).
[ ] At HF state, expectation of each pool operator ~ 0.
[ ] H2 ADAPT-VQE converges to FCI within sample noise using the new component.
[ ] New optimizer is scipy.optimize.minimize-compatible.
[ ] Documentation page added under docs/sphinx/components/solvers/ or examples_rst.
[ ] Unit test added under libs/solvers/python/tests/ or libs/solvers/unittests/.
```

## When stuck

1. Read the matching `references/*-authoring.md` end-to-end:
   `operator-pools-authoring.md`, `optimizers-authoring.md`, or
   `state-prep-authoring.md`.
2. Compare against the equivalent built-in's source under
   `libs/solvers/include/cudaq/solvers/{operators/operator_pools,optimizers,observe_gradients,stateprep}/`.
3. For H2 sanity, run `docs/sphinx/examples/solvers/python/adapt_h2.py`
   first to confirm baseline behavior before swapping in your
   component.
4. For build / registration issues, delegate to `cudaq-build`.

## Additional resources

- `references/operator-pools-authoring.md` — author a new operator pool.
- `references/optimizers-authoring.md` — author a new optimizer or
  gradient method.
- `references/state-prep-authoring.md` — author a new ansatz kernel.
- `cudaq-solvers-algorithms` SKILL.md — algorithm dispatch (where the pool
  gets used).
- `cudaq-solvers-chemistry` SKILL.md — substrate (molecule,
  fermion mappings).
- `cudaq-benchmarking` SKILL.md — measure your new component's
  effect on H2 / LiH / BeH2.
- QEC plugin authoring: see the cudaq-qec-decode package —
  `cudaq-qec-extending` skill when in the cudaq monorepo.
