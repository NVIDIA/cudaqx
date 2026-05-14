# Authoring a new operator pool

Define a new ADAPT/GQE-eligible operator pool — the menu of operators
the algorithm can add to the ansatz. For consumers picking an existing
pool (UCCSD vs UCCGSD vs CEO etc.), see the related skills
`cuda-qx-solvers-chemistry/references/operator-pools-using.md`.

## Authoritative reference

- Base header: `libs/solvers/include/cudaq/solvers/operators/operator_pool.h`
- Built-in pools: `libs/solvers/include/cudaq/solvers/operators/operator_pools/`

## Skeleton (C++)

```cpp
// libs/solvers/include/cudaq/solvers/operators/operator_pools/my_pool.h
#pragma once
#include "cudaq/solvers/operators/operator_pool.h"

namespace cudaq::solvers {

class my_pool : public operator_pool {
public:
  std::vector<cudaq::spin_op> generate(
      const heterogeneous_map& config) const override;

  CUDAQ_REGISTER_OPERATOR_POOL("my-pool")
};

}
```

`generate` returns the list of spin operators. The signature is fixed;
check the existing pools for kwarg conventions. UCCSD takes
`num_qubits` + `num_electrons`; UCCGSD / UPCCGSD / spin-complement-GSD
/ CEO all take `num_orbitals`.

## Build glue

1. Header in the `operator_pools/` directory.
2. `.cpp` source under `libs/solvers/lib/operators/operator_pools/`.
3. Add to `libs/solvers/lib/CMakeLists.txt`.
4. Rebuild.
5. Verify: `python3 -c "import cudaq_solvers as s; print(s.list_operator_pools())"`
   (or whichever enumeration call exists in your version).

## From Python (prototyping)

For prototyping, you can pass a `list[cudaq.SpinOperator]` directly to
ADAPT or GQE without registering a pool. Skip the C++ work until you
want to ship it upstream.

## Common pitfalls

| Symptom | Cause |
|---------|-------|
| ADAPT picks the same operator each iteration | sign convention wrong on new pool operators |
| New pool returns an empty list | kwarg name mismatch; `generate(config)` is permissive but returns empty |
| Pool kwarg accepted but VQE diverges | passed `num_qubits` to a pool that wants `num_orbitals` (or vice versa) |

## Self-check

```
[ ] CUDAQ_REGISTER_OPERATOR_POOL("name") macro present, name unique.
[ ] Source added to libs/solvers/lib/CMakeLists.txt.
[ ] Pool appears in `cudaq_solvers.list_operator_pools()` after rebuild.
[ ] H2 ADAPT-VQE converges to FCI within sample noise using the new pool.
[ ] Documentation page added under docs/sphinx/components/solvers/ or examples_rst.
[ ] Unit test added under libs/solvers/python/tests/ or libs/solvers/unittests/.
```

## Where next

- Wire optimizers / gradients: `optimizers-authoring.md`.
- Add a new state-prep kernel: `state-prep-authoring.md`.
- Run VQE / ADAPT / GQE using the new pool:
  `cuda-qx-solvers-algorithms/references/{vqe,gqe}.md`.
- Benchmark against built-ins: `cuda-qx-benchmarking`.
