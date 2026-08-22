# ADAPT-VQE (Minimal Recipe)

Use this reference when the user asks for a minimal ADAPT-VQE example in CUDA-Q
Solvers, especially around **operator pool selection** and **initial state
kernel**. Source of truth: `libs/solvers/python/tests/test_adapt.py` and
`docs/sphinx/examples/solvers/cpp/adapt_h2.cpp` (Python equivalent uses the
same flow).

## API surface

| Symbol | Notes |
| --- | --- |
| `solvers.create_molecule(geometry, basis, charge, spin, casci=True)` | Build a `MolecularHamiltonian` (needs PySCF) |
| `solvers.get_operator_pool(name, **kwargs)` | Pool names include `uccsd`, `spin_complement_gsd`, `uccgsd`, `upccgsd`, `ceo` |
| `solvers.adapt_vqe(initState, hamiltonian, operators, **options)` | Returns `(energy, thetas, ops)` |

## Minimal Python recipe (H2, STO-3G)

```python
import numpy as np
import cudaq
from cudaq import spin
import cudaq_solvers as solvers

geometry = [('H', (0., 0., 0.)), ('H', (0., 0., .7474))]
molecule = solvers.create_molecule(geometry, 'sto-3g', 0, 0, casci=True)

operators = solvers.get_operator_pool(
    "spin_complement_gsd", num_orbitals=molecule.n_orbitals
)

numElectrons = molecule.n_electrons

@cudaq.kernel
def initState(q: cudaq.qview):
    for i in range(numElectrons):
        x(q[i])

energy, thetas, ops = solvers.adapt_vqe(
    initState, molecule.hamiltonian, operators
)
print(f"Energy = {energy}")
```

Expected energy for H2/STO-3G is approximately `-1.137 Ha`.

## Mandatory beginner footguns

- The pool is **not** auto-generated. You must call `solvers.get_operator_pool(...)`
  and pass the result. Calling `solvers.adapt_vqe(initState, hamiltonian)` with
  no pool will fail.
- `initState` must be a `@cudaq.kernel` that prepares a non-trivial reference
  state (typically Hartree-Fock: apply `x(q[i])` for `i in range(n_electrons)`).
  An empty kernel leaves the state at |0...0> and ADAPT will pick uninformative
  operators.
- `get_operator_pool` requires the pool's expected kwargs. `spin_complement_gsd`
  needs `num_orbitals=molecule.n_orbitals`. `uccsd` needs additionally
  `num_electrons=molecule.n_electrons`.
- Pass advanced tuning via `options=` (max_iter, grad_norm_tolerance,
  threshold_energy, initial_theta, verbose, shots). Defaults are sane for a
  workshop demo; do not over-tune.

## When to recommend ADAPT-VQE vs VQE

- VQE with UCCSD: fastest path, fixed ansatz.
- ADAPT-VQE: adaptively picks operators from a pool, deeper but more accurate
  on harder molecules where you don't know the right ansatz.
- GQE: GPT-like generative search over a pool. See `references/gqe.md`.
