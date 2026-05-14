# Authoring a new state-prep / ansatz kernel

Define a parameterized quantum kernel that VQE / ADAPT can optimize.
For *using* existing state-prep kernels (`uccsd`, `uccgsd`,
`upccgsd`, `ceo`, etc.), see
`cuda-qx-solvers-algorithms/references/vqe.md` "State Preparation Kernels".

## Authoritative reference

State-prep kernels live under
`libs/solvers/include/cudaq/solvers/stateprep/`. They are `__qpu__`
functions that prepare a parameterized state on a `qvector`.

## Skeleton (C++)

```cpp
// libs/solvers/include/cudaq/solvers/stateprep/my_ansatz.h
#pragma once
#include <cudaq.h>

namespace cudaq::solvers {

__qpu__ void my_ansatz(std::vector<double> params,
                       std::size_t num_qubits,
                       std::size_t num_electrons);

}
```

Implement in a companion `.cpp` under
`libs/solvers/lib/stateprep/`. Use it from Python through pybind11
bindings (`libs/solvers/python/bindings/solvers/py_solvers.cpp`).

## From Python (prototyping)

For prototyping, define the ansatz as a `@cudaq.kernel` directly in
your script and pass it to `solvers.vqe`:

```python
@cudaq.kernel
def my_ansatz(theta: list[float]):
    q = cudaq.qvector(num_qubits)
    # ... your gates ...

energy, params, _ = solvers.vqe(my_ansatz, hamiltonian, init_params, ...)
```

Skip the C++ work until you want to ship it upstream.

## In-kernel constraints

Inside `@cudaq.kernel` / `__qpu__`:

- No NumPy / SciPy / Python control flow that doesn't lower to Quake MLIR.
- Only calls to other `@cudaq.kernel` / `__qpu__` functions.
- Parameters arrive as `std::vector<double>` (C++) or `list[float]`
  (Python). Treat them as opaque.

## Common pitfalls

| Symptom | Cause |
|---------|-------|
| `compile error: unsupported Python construct` | a generator / lambda / `np.dot` inside the kernel; rewrite using only kernel-eligible operations |
| Wrong qubit count at runtime | `num_qubits` / `num_orbitals` confusion — see `cuda-qx-solvers-chemistry/SKILL.md` Convention #2 |
| Energy stuck at HF | ansatz expressiveness too low; add more excitation terms or switch to a richer pool (`uccgsd`, `ceo`) |
| Energy explodes | parameter scaling — initial params should be near zero |

## Self-check

```
[ ] Kernel compiles (`@cudaq.kernel` raises no `unsupported construct`).
[ ] At init_params = zeros, the ansatz prepares the HF reference state.
[ ] H2 VQE with the new ansatz converges to FCI within sample noise.
[ ] If shipping upstream: source under libs/solvers/lib/stateprep/, header installed.
[ ] Unit test added under libs/solvers/{python/tests,unittests}/.
```

## Where next

- Author an operator pool to pair with the ansatz:
  `operator-pools-authoring.md`.
- Author an optimizer / gradient: `optimizers-authoring.md`.
- Run VQE / ADAPT / GQE with the new ansatz:
  `cuda-qx-solvers-algorithms/references/vqe.md`.
- Benchmark vs built-ins: `cuda-qx-benchmarking`.
