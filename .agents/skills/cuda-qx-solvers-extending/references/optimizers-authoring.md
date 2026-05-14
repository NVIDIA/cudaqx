# Authoring a new optimizer or gradient method

Wire a classical optimizer or gradient evaluator into the cudaq-solvers
algorithm dispatch (VQE / ADAPT / QAOA / GQE). For *using* the
existing optimizers, see `cuda-qx-solvers-algorithms/references/vqe.md`.

## Authoritative reference

- Optimizer base: `libs/solvers/include/cudaq/solvers/optimizer.h`
- Gradient base: `libs/solvers/include/cudaq/solvers/observe_gradient.h`
- Built-in optimizers: `libs/solvers/include/cudaq/solvers/optimizers/`
- Built-in gradients: `libs/solvers/include/cudaq/solvers/observe_gradients/`

## Built-in optimizer names (today)

| Optimizer | Notes |
|-----------|-------|
| `cobyla` | gradient-free; default for QAOA |
| `lbfgs` | needs gradients; default for VQE |
| (scipy)  | pass a `scipy.optimize.minimize` callable for anything else |

## Adding a new optimizer

For Python users it's usually easier to **pass a SciPy callable**
than to subclass: `solvers.vqe(..., optimizer=scipy.optimize.minimize,
method="trust-krylov")`. See `cuda-qx-solvers-algorithms/SKILL.md`
"Custom optimizers must be `scipy.optimize.minimize`."

For a new built-in C++ optimizer:

```cpp
// libs/solvers/include/cudaq/solvers/optimizers/my_optimizer.h
#pragma once
#include "cudaq/solvers/optimizer.h"

namespace cudaq::solvers {

class my_optimizer : public optimizer {
public:
  optimization_result optimize(
      std::size_t dim,
      const optimization_function& f,
      const heterogeneous_map& opts) const override;

  CUDAQ_REGISTER_OPTIMIZER("my-optimizer")
};

}
```

Build glue: source under `libs/solvers/lib/optimizers/`, add to
`libs/solvers/lib/CMakeLists.txt`, rebuild. Verify with
`python3 -c "import cudaq_solvers as s; print(s.list_optimizers())"`
or by passing `optimizer="my-optimizer"` to `solvers.vqe(...)`.

## Built-in gradient methods (today)

- `parameter_shift` — exact for VQE with Pauli sums.
- `central_difference` — symmetric finite difference.
- `forward_difference` — cheaper finite difference.

## Adding a new gradient method

Subclass `observe_gradient` and register similarly:

```cpp
#include "cudaq/solvers/observe_gradient.h"

class my_gradient : public observe_gradient {
public:
  std::vector<double> compute(
      const std::function<double(const std::vector<double>&)>& f,
      const std::vector<double>& x) const override;

  CUDAQ_REGISTER_OBSERVE_GRADIENT("my-gradient")
};
```

## Common pitfalls

| Symptom | Cause |
|---------|-------|
| Optimizer rejects parameters | not registered via `CUDAQ_REGISTER_OPTIMIZER`, or named differently from what the user passed |
| `RuntimeError("Invalid functional optimizer provided ...")` | Python callable passed but not `scipy.optimize.minimize`; the wrapper only forwards SciPy |
| Gradient mismatch with `parameter_shift` baseline | sign convention or shift size wrong in the new gradient class |
| `lbfgs` works but new optimizer claims "no gradient" | new optimizer's `optimize` did not request a gradient via the `optimization_function` interface |

## Self-check

```
[ ] CUDAQ_REGISTER_OPTIMIZER / CUDAQ_REGISTER_OBSERVE_GRADIENT macro present.
[ ] Source added to libs/solvers/lib/{optimizers,observe_gradients}/CMakeLists.txt.
[ ] Appears in cudaq_solvers.list_optimizers() / list_gradients() from Python.
[ ] H2 VQE with the new component converges to FCI within sample noise.
[ ] Documentation page added under docs/sphinx/components/solvers/ or examples_rst.
[ ] Unit test added under libs/solvers/{python/tests,unittests}/.
```

## Where next

- Author a state-prep kernel: `state-prep-authoring.md`.
- Author an operator pool: `operator-pools-authoring.md`.
- Run VQE / ADAPT / GQE with the new component:
  `cuda-qx-solvers-algorithms/references/vqe.md`.
- Benchmark vs built-ins: `cuda-qx-benchmarking`.
