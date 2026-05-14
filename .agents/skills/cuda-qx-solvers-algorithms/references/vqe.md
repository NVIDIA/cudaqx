# VQE & ADAPT-VQE: ansatz, optimizers, gradients, state prep

How to optimize a parameterized ansatz against a Hamiltonian (VQE), and
how to grow that ansatz iteratively from a pool (ADAPT-VQE). For
chemistry inputs (molecule, operator pools, fermion mapping) see the
related skill `cuda-qx-solvers-chemistry` (start with
`references/molecule-building.md`). For QAOA see `qaoa.md`. For GQE
see `gqe.md`.

## VQE

```python
energy, params, data = solvers.vqe(kernel, spin_op, init_params, **opts)
```

Each `ObserveIteration` exposes `parameters`, `result`
(`cudaq.observe_result`), and `type`
(`ObserveExecutionType.function` or `.gradient`).

| Key              | Default              | Notes                                   |
|------------------|----------------------|-----------------------------------------|
| `optimizer`      | `"cobyla"`           | str or `scipy.optimize.minimize`        |
| `gradient`       | `"parameter_shift"`  | only used if optimizer needs gradients  |
| `shots`          | `-1`                 | `-1` = exact simulation                 |
| `max_iterations` | `-1`                 | `-1` = no limit (built-in optimizers)   |
| `tol`            | `1e-12`              | non-scipy optimizers                    |
| `verbose`        | `False`              |                                         |

## ADAPT-VQE

ADAPT-VQE uses different option keys from VQE (the most common
cross-skill bug). Returns `(energy, params, selected_ops)`.

| Key                                | Default              |
|------------------------------------|----------------------|
| `max_iter` (not `max_iterations`)  | `30`                 |
| `grad_norm_tolerance`              | `1e-5`               |
| `grad_norm_diff_tolerance`         | `1e-5`               |
| `threshold_energy`                 | `1e-6`               |
| `initial_theta`                    | `0.0`                |
| `dynamic_start`                    | `"cold"` or `"warm"` |
| `shots`                            | `-1`                 |
| `tol`                              | `1e-12`              |

`dynamic_start="cold"` re-initializes parameters each iteration;
`"warm"` keeps the previous iteration's parameters as the starting
point (faster convergence on smooth problems).

### Multi-GPU scaling (MQPU)

ADAPT-VQE's per-iteration gradient sweep over the operator pool is
embarrassingly parallel; the cudaqx README cites CUDA-Q's MQPU as the
scaling path. To use it:

```python
cudaq.set_target("nvidia", option="mqpu")
# adapt_vqe partitions the pool-gradient sweep across MQPU devices
energy, params, ops = solvers.adapt_vqe(initial_state, ham, pool, ...)
```

Speed-up is roughly linear up to `min(pool_size, n_gpus)`. Same env
caveat as GQE: `PMIX_MCA_gds=hash` if launching via `mpiexec`. For
multi-GPU GQE specifics see `references/gqe.md`.

## Optimizers and gradients

- Default optimizer: `cobyla` (gradient-free).
- Built-in: `cobyla`, `lbfgs`. `lbfgs` requires gradients.
- SciPy: pass `optimizer=scipy.optimize.minimize` directly. Other
  callables raise `RuntimeError("Invalid functional optimizer
  provided ...")`. Forwarded kwargs include `method`, `jac`, `tol`,
  `options`, `callback`. The wrapper strips `gradient`, `optimizer`,
  `verbose`, `shots` before calling `minimize`.
- Gradients (when optimizer requires them): `parameter_shift`
  (default), `central_difference`, `forward_difference`.
- Direct optimization: `solvers.optim.optimize(function,
  initial_parameters, method="cobyla", **kwargs)`. For gradient-based
  methods the function must return `(value, list[float])`; otherwise
  just `value`. Returns an `OptimizationResult`. `method` must be a
  registered cudaq-x optimizer (e.g. `lbfgs`, `cobyla`); unknown names
  raise `RuntimeError`.

## State Preparation Kernels

| Kernel                                | Signature                                  |
|---------------------------------------|--------------------------------------------|
| `solvers.stateprep.uccsd`             | `(q, thetas, n_electrons, spin)`           |
| `solvers.stateprep.uccgsd`            | `(q, thetas, pauli_words, coeffs)`         |
| `solvers.stateprep.upccgsd`           | `(q, thetas, pauli_words, coeffs)`         |
| `solvers.stateprep.ceo`               | `(q, thetas, pauli_words, coeffs)`         |
| `solvers.stateprep.single_excitation` | `(q, theta, p, q)`                         |
| `solvers.stateprep.double_excitation` | `(q, theta, p, q, r, s)`                   |

Helpers:

- `solvers.stateprep.get_num_uccsd_parameters(n_electrons, n_qubits, spin=0)`
- `solvers.stateprep.get_uccsd_excitations(...)`
- `solvers.stateprep.get_uccgsd_pauli_lists(num_qubits, only_singles=False, only_doubles=False)`
- `solvers.stateprep.get_upccgsd_pauli_lists(num_qubits, only_doubles=False)`
- `solvers.stateprep.get_ceo_pauli_lists(num_orbitals)`

## Self-check (VQE/ADAPT-specific)

```
[ ] Right option key for the algorithm:
      VQE     -> max_iterations
      ADAPT   -> max_iter
[ ] If lbfgs: a gradient is wired in (built-in or scipy with `jac`).
[ ] If custom optimizer: it is scipy.optimize.minimize (not any callable).
[ ] At a small-input smoke run (H2), energy is finite and decreasing.
```

## Canonical tests

| Topic              | File                                                                          |
|--------------------|-------------------------------------------------------------------------------|
| VQE (Python)       | `libs/solvers/python/tests/test_vqe.py`                                       |
| VQE (C++)          | `libs/solvers/unittests/test_vqe.cpp`                                         |
| ADAPT-VQE          | `libs/solvers/python/tests/test_adapt.py`, `unittests/test_adapt.cpp`         |
| ADAPT-VQE MPI      | `libs/solvers/python/tests/test_adapt_mpi.py`, `unittests/test_adapt_mpi.cpp` |
| Optimizers         | `libs/solvers/python/tests/test_optim.py`, `unittests/test_optimizers.cpp`    |
| State prep         | `libs/solvers/python/tests/test_uccsd.py`, `test_uccgsd.py`, `test_upccgsd.py`, `test_ceo.py` |
