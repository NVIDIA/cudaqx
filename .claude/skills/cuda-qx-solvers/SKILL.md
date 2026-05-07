---
name: "cuda-qx-solvers"
title: "CUDA-QX Solvers"
description: "CUDA-QX solvers guide for VQE, ADAPT-VQE, QAOA, GQE, molecular Hamiltonians, operator pools, optimizers, gradients, and PySCF-based chemistry workflows. Use when the user mentions cudaq_solvers, VQE, ADAPT-VQE, QAOA, GQE, create_molecule, jordan_wigner, bravyi_kitaev, operator pools, UCCSD/UCCGSD/CEO, or chemistry/optimization workflows in CUDA-QX."
version: "0.2.0"
author: "CUDA-QX"
tags: [cuda-qx, solvers, chemistry, optimization, vqe, adapt-vqe, qaoa, gqe]
tools: [Read, Glob, Grep, Bash]
license: "Apache License 2.0"
compatibility: "Python 3.11+, C++ 20, Linux x86_64/aarch64"
metadata:
  author: "CUDA-QX"
  tags:
    - cuda-qx
    - solvers
    - chemistry
    - optimization
    - vqe
    - adapt-vqe
    - qaoa
    - gqe
  languages:
    - python
    - c++
  domain: "quantum-applications"
---

# CUDA-QX Solvers Guide

You are a CUDA-QX solvers expert assistant. Guide users through the solvers
library: VQE, ADAPT-VQE, QAOA, GQE, molecular Hamiltonians, fermion-to-qubit
mappings, operator pools, optimizers, gradients, and the PySCF-based chemistry
pipeline.

## Purpose

Help users build and debug CUDA-QX solver workflows correctly the first time,
including the PySCF chemistry pipeline, optimizer/gradient routing, operator
pool usage, and GQE training. Surface non-obvious gotchas (`OMP_NUM_THREADS`,
hung PySCF servers, `natorb` requiring `MP2`, ADAPT vs VQE option keys, QAOA
gradient requirements) before users hit them.

## Algorithms

| Algorithm | Entry point | Returns |
| --- | --- | --- |
| VQE | `solvers.vqe(kernel, spin_op, init_params, **opts)` | `(energy, params, list[ObserveIteration])` |
| ADAPT-VQE | `solvers.adapt_vqe(initial_state, ham, pool, **opts)` | `(energy, params, selected_ops)` |
| QAOA | `solvers.qaoa(Hp, [Href,] num_layers, init_params, **opts)` | `QAOAResult(optimal_value, optimal_parameters, optimal_config)` |
| GQE (optional) | `solvers.gqe(cost, pool, config=None, **kwargs)` | `(min_energy, best_op_indices)` |

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

## Molecule Workflow

`solvers.create_molecule` runs PySCF and returns a `MolecularHamiltonian` with
`hamiltonian` (`cudaq.SpinOperator`), `hpq`, `hpqrs`, `n_electrons`,
`n_orbitals`, and `energies` dict.

```python
geometry = [("H", (0., 0., 0.)), ("H", (0., 0., 0.7474))]
mol = solvers.create_molecule(geometry, "sto-3g", spin=0, charge=0,
                              casci=True, verbose=True)
n_qubits = 2 * mol.n_orbitals     # qubits = 2 * orbitals
n_electrons = mol.n_electrons
```

Energy keys vary by options:

- Always: `nuclear_energy` or `core_energy` (active space).
- `casci=True`: `hf_energy`, `fci_energy`.
- `ccsd=True`: `R-CCSD` (or `UR-CCSD` if `UR=True`).
- `casci=True` with active space: `R-CASCI` (or `UR-CASCI`).
- `casscf=True`: `R-CASSCF`.

Active space:

- Set `nele_cas` and `norb_cas` for CAS calculations.
- `natorb=True` requires `MP2=True`.
- `integrals_natorb=True` requires `MP2=True`.

`create_molecule` also accepts an XYZ filename instead of a geometry list.

### PySCF REST server

`create_molecule` spawns `cudaq-pyscf --server-mode` and talks to it on
`localhost:8000`. If a previous run crashed, the port can stay occupied:

```bash
lsof -n -i :8000
kill -9 <pid>
```

The Python interpreter for the child server is auto-resolved from
`sys.executable` so the active env is reused.

## Operator Pools

```python
ops = solvers.get_operator_pool(name, **config)
```

| Pool name | Required kwargs | Use for |
| --- | --- | --- |
| `uccsd` | `num_qubits`, `num_electrons` | Standard chemistry ADAPT |
| `uccgsd` | `num_orbitals` | More expressive, deeper |
| `upccgsd` | `num_orbitals` | Paired doubles only |
| `spin_complement_gsd` | `num_orbitals` | Spin-symmetric, robust |
| `ceo` | `num_orbitals` | Coupled exchange operators |
| `qaoa` | (see source) | ADAPT-QAOA mixers |

Note: `num_qubits = 2 * num_orbitals`. Mixing them is the most common bug.

## State Preparation Kernels

| Kernel | Signature |
| --- | --- |
| `solvers.stateprep.uccsd` | `(q, thetas, n_electrons, spin)` |
| `solvers.stateprep.uccgsd` | `(q, thetas, pauli_words, coeffs)` |
| `solvers.stateprep.upccgsd` | `(q, thetas, pauli_words, coeffs)` |
| `solvers.stateprep.ceo` | `(q, thetas, pauli_words, coeffs)` |
| `solvers.stateprep.single_excitation` | `(q, theta, p, q)` |
| `solvers.stateprep.double_excitation` | `(q, theta, p, q, r, s)` |

Helpers:

- `solvers.stateprep.get_num_uccsd_parameters(n_electrons, n_qubits, spin=0)`
- `solvers.stateprep.get_uccsd_excitations(...)`
- `solvers.stateprep.get_uccgsd_pauli_lists(num_qubits, only_singles=False, only_doubles=False)`
- `solvers.stateprep.get_upccgsd_pauli_lists(num_qubits, only_doubles=False)`
- `solvers.stateprep.get_ceo_pauli_lists(num_orbitals)`

## Optimizers and Gradients

- Default optimizer: `cobyla` (gradient-free).
- Built-in: `cobyla`, `lbfgs`. `lbfgs` requires gradients.
- SciPy: pass `optimizer=scipy.optimize.minimize` directly. Other callables
  raise `RuntimeError("Invalid functional optimizer provided (only
  scipy.optimize.minimize supported).")`. Forwarded kwargs include `method`,
  `jac`, `tol`, `options`, `callback`. The wrapper strips `gradient`,
  `optimizer`, `verbose`, `shots` before calling `minimize`.
- Gradients (when optimizer requires them):
  `parameter_shift` (default), `central_difference`, `forward_difference`.
- Direct optimization: `solvers.optim.optimize(function, initial_parameters,
  method="cobyla", **kwargs)`. For gradient-based methods the function must
  return `(value, list[float])`; otherwise just `value`. Returns an
  `OptimizationResult`. Method must be a registered cudaq-x optimizer (e.g.
  `lbfgs`, `cobyla`); unknown names raise `RuntimeError`.

## VQE Options

| Key | Default | Notes |
| --- | --- | --- |
| `optimizer` | `"cobyla"` | str or `scipy.optimize.minimize` |
| `gradient` | `"parameter_shift"` | only used if optimizer needs gradients |
| `shots` | `-1` | -1 = exact simulation |
| `max_iterations` | `-1` | -1 = no limit (built-in optimizers) |
| `tol` | `1e-12` | non-scipy optimizers |
| `verbose` | `False` | |

Each `ObserveIteration` exposes `parameters`, `result`
(`cudaq.observe_result`), and `type` (`ObserveExecutionType.function` or
`.gradient`).

## ADAPT-VQE Options (different keys from VQE)

| Key | Default |
| --- | --- |
| `max_iter` (not `max_iterations`) | `30` |
| `grad_norm_tolerance` | `1e-5` |
| `grad_norm_diff_tolerance` | `1e-5` |
| `threshold_energy` | `1e-6` |
| `initial_theta` | `0.0` |
| `dynamic_start` | `"cold"` (or `"warm"`) |
| `shots` | `-1` |
| `tol` | `1e-12` |

## QAOA

- Two overloads:
  - `solvers.qaoa(Hp, Href, num_layers, init_params, **opts)`
  - `solvers.qaoa(Hp, num_layers, init_params, **opts)` (default mixer)
- `optimizer="lbfgs"` fails with QAOA: it requires gradients but no gradient
  instance is provided. Use `cobyla` or pass a SciPy optimizer.
- Empty `init_params` raises `RuntimeError("qaoa initial parameters empty.")`.
- Extra kwargs: `full_parameterization=True`, `counterdiabatic=True`.
- Helpers:
  - `solvers.get_num_qaoa_parameters(H, num_layers, **kwargs)`
  - `solvers.get_maxcut_hamiltonian(nx_graph)`
  - `solvers.get_clique_hamiltonian(nx_graph, penalty=4.0)`
- Graph helpers take NetworkX graphs only and read `weight` attrs from
  nodes/edges.
- `QAOAResult` is tuple-unpackable: `optval, optp, config = result`. `config`
  is a `cudaq.SampleResult`.

## Fermion-to-Qubit Mappings

```python
op = solvers.jordan_wigner(hpq, hpqrs, core_energy=0.0, tolerance=1e-15)
op = solvers.jordan_wigner(hpq_or_hpqrs, core_energy=0.0)  # auto-detect 2D vs 4D
op = solvers.bravyi_kitaev(...)  # same overload pattern
```

`tol` is accepted as an alias for `tolerance`.

## GQE (optional)

Install: `pip install cudaq-solvers[gqe]`. Adds `torch>=2.0.0`,
`lightning>=2.0.0`, `ml_collections`, `mpi4py>=3.1.0`, `transformers>=4.30.0`.

Signature: `solvers.gqe(cost, pool, config=None, **kwargs)`. When `config` is
`None`, kwargs override individual config defaults. Special kwargs:

- `model`: pre-constructed transformer (skip the default GPT-2-style model)
- `optimizer`: pre-constructed `torch.optim` instance (default `AdamW(lr=cfg.lr)`)

```python
from cudaq_solvers.gqe_algorithm.gqe import get_default_config

cfg = get_default_config()
cfg.use_fabric_logging = False
cfg.save_trajectory = False
cfg.verbose = True

min_energy, best_ops = solvers.gqe(cost, op_pool, config=cfg,
                                    max_iters=25, ngates=10)
```

`get_default_config()` keys (with defaults):

| Key | Default |
| --- | --- |
| `num_samples` | `5` (batch size) |
| `max_iters` | `100` |
| `ngates` | `20` |
| `seed` | `3047` |
| `lr` | `5e-7` |
| `energy_offset` | `0.0` |
| `grad_norm_clip` | `1.0` |
| `temperature` | `5.0` |
| `del_temperature` | `0.05` |
| `resid_pdrop`, `embd_pdrop`, `attn_pdrop` | `0.0` |
| `small` | `False` (12-layer/12-head; small=6/6) |
| `use_fabric_logging` | `False` |
| `fabric_logger` | `None` |
| `save_trajectory` | `False` |
| `trajectory_file_path` | `"gqe_logs/gqe_trajectory.json"` |
| `verbose` | `False` |

Validation in `validate_config` rejects non-positive `num_samples`, `max_iters`,
`ngates`, `lr`, `grad_norm_clip`, `temperature`. `del_temperature` must be
non-zero. Dropout values must be in `[0, 1]`.

Schedulers: `DefaultScheduler(start, delta)`, `CosineScheduler(min, max, frequency)`.

GPU sanity check: GQE will `sys.exit(1)` if PyTorch sees a CUDA device but
cannot run kernels on it (newer GPUs need a PyTorch built for CUDA 12.8+).

Multi-GPU:

```python
cudaq.set_target("nvidia", option="mqpu")
cudaq.mpi.initialize()
# ...
cudaq.mpi.finalize()
```

Run: `PMIX_MCA_gds=hash mpiexec -np N python script.py --mpi`.

Determinism block GQE expects:

```python
import os, torch
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.manual_seed(seed)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

## Common Pitfalls

- `OMP_NUM_THREADS=1` is required for bit-for-bit reproducible Hamiltonian
  coefficients. With more threads, signs flip across runs (eigenvalues stay
  correct). PySCF limitation.
- `libgfortran` missing → `cobyla`/`lbfgs` fail. Install via the system
  package manager.
- `lbfgs` with QAOA → "requires gradients...gradient instance not provided".
- `natorb=True` or `integrals_natorb=True` without `MP2=True` → RuntimeError.
- Hung PySCF on `:8000` → `lsof -n -i :8000` and kill.
- ADAPT uses `max_iter`, VQE uses `max_iterations`.
- `num_qubits` vs `num_orbitals` mismatch in operator pool kwargs.
- GQE import error → install `[gqe]` extras.
- macOS/Windows are not supported. Linux x86_64/aarch64 only.

## Key Paths

| Area | Path |
| --- | --- |
| Library | `libs/solvers/` |
| Python package | `libs/solvers/python/cudaq_solvers/` |
| Python bindings | `libs/solvers/python/bindings/solvers/` |
| C++ headers | `libs/solvers/include/cudaq/solvers/` |
| C++ implementation | `libs/solvers/lib/` |
| PySCF driver (C++) | `libs/solvers/lib/operators/molecule/drivers/pyscf_driver.cpp` |
| PySCF tool (Python) | `libs/solvers/tools/molecule/cudaq-pyscf.py` |
| Python tests | `libs/solvers/python/tests/` |
| C++ tests | `libs/solvers/unittests/` |
| Docs | `docs/sphinx/components/solvers/`, `docs/sphinx/api/solvers/` |
| Examples | `docs/sphinx/examples/solvers/`, `docs/sphinx/examples_rst/solvers/` |
| Build | `libs/solvers/CMakeLists.txt`, `libs/solvers/python/CMakeLists.txt` |

## Canonical Test References

| Topic | File |
| --- | --- |
| VQE (Python) | `libs/solvers/python/tests/test_vqe.py` |
| VQE (C++) | `libs/solvers/unittests/test_vqe.cpp` |
| ADAPT-VQE | `libs/solvers/python/tests/test_adapt.py`, `unittests/test_adapt.cpp` |
| ADAPT-VQE MPI | `libs/solvers/python/tests/test_adapt_mpi.py`, `unittests/test_adapt_mpi.cpp` |
| QAOA + graph | `libs/solvers/python/tests/test_qaoa.py`, `unittests/test_qaoa.cpp` |
| Optimizers | `libs/solvers/python/tests/test_optim.py`, `unittests/test_optimizers.cpp` |
| Jordan-Wigner | `libs/solvers/python/tests/test_jordan_wigner.py` |
| Bravyi-Kitaev | `libs/solvers/unittests/test_bravyi_kitaev.cpp` |
| Molecule | `libs/solvers/python/tests/test_molecule.py`, `unittests/test_molecule.cpp` |
| Operator pools | `libs/solvers/python/tests/test_operator_pools.py`, `unittests/test_*_operator_pool.cpp` |
| State prep | `libs/solvers/python/tests/test_uccsd.py`, `test_uccgsd.py`, `test_upccgsd.py`, `test_ceo.py`, `unittests/test_uccsd.cpp` |
| GQE | `libs/solvers/python/tests/test_gqe.py` |

## Build and Validation

- scikit-build-core, CMake ≥ 3.28, Ninja ≥ 1.10.
- Default install prefix: `$HOME/.cudaqx`.
- CMake options: `CUDAQX_SOLVERS_INCLUDE_TESTS`,
  `CUDAQX_SOLVERS_BINDINGS_PYTHON`, `CUDAQX_SOLVERS_INSTALL_PYTHON`.
- Test targets (when configured): `make run_tests`, `make run_python_tests`.
- Wheels: `cudaq-solvers-cu12`, `cudaq-solvers-cu13`. Internal native module:
  `_pycudaqx_solvers_the_suffix_matters_cudaq_solvers`.
- For code changes, run focused Python tests first; broaden to C++ unit tests
  when bindings, headers, or shared APIs are touched.

## Future Sub-skills (planned)

When users ask deep questions in these areas, note that more focused skills
are planned and currently fall back to this guide:

- `cuda-qx-solvers-chemistry` (PySCF, active space, fermion mappings)
- `cuda-qx-solvers-gqe` (GQE training, transformers, MPI)
- `cuda-qx-solvers-qaoa` (graph problems, mixers, counterdiabatic)

## Additional Resources

- Benchmark / eval prompts: [benchmark.md](benchmark.md)
- Scoring helper: `.claude/skills/scripts/score_benchmark.py --skill solvers
  --responses responses.json`
