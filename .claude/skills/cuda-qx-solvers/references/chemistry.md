# Chemistry substrate: molecules, operator pools, fermion mappings

The shared substrate every solver algorithm consumes. Build the
Hamiltonian, get an operator pool, map fermions to qubits — then hand
off to VQE / ADAPT / QAOA / GQE.

## Molecule Workflow

`solvers.create_molecule` runs PySCF and returns a `MolecularHamiltonian`
with `hamiltonian` (`cudaq.SpinOperator`), `hpq`, `hpqrs`,
`n_electrons`, `n_orbitals`, and an `energies` dict.

```python
geometry = [("H", (0., 0., 0.)), ("H", (0., 0., 0.7474))]
mol = solvers.create_molecule(geometry, "sto-3g", spin=0, charge=0,
                              casci=True, verbose=True)
n_qubits = 2 * mol.n_orbitals     # qubits = 2 * orbitals
n_electrons = mol.n_electrons
```

### Energy keys (vary by options)

- Always: `nuclear_energy` or `core_energy` (active space).
- `casci=True`: `hf_energy`, `fci_energy`.
- `ccsd=True`: `R-CCSD` (or `UR-CCSD` if `UR=True`).
- `casci=True` with active space: `R-CASCI` (or `UR-CASCI`).
- `casscf=True`: `R-CASSCF`.

### Active space

- Set `nele_cas` and `norb_cas` for CAS calculations.
- `natorb=True` requires `MP2=True`.
- `integrals_natorb=True` requires `MP2=True`.

`create_molecule` also accepts an XYZ filename instead of a geometry
list.

### PySCF REST server

`create_molecule` spawns `cudaq-pyscf --server-mode` and talks to it on
`localhost:8000`. If a previous run crashed, the port can stay
occupied:

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

| Pool name              | Required kwargs              | Use for                           |
|------------------------|------------------------------|-----------------------------------|
| `uccsd`                | `num_qubits`, `num_electrons`| Standard chemistry ADAPT          |
| `uccgsd`               | `num_orbitals`               | More expressive, deeper           |
| `upccgsd`              | `num_orbitals`               | Paired doubles only               |
| `spin_complement_gsd`  | `num_orbitals`               | Spin-symmetric, robust            |
| `ceo`                  | `num_orbitals`               | Coupled exchange operators        |
| `qaoa`                 | (see source)                 | ADAPT-QAOA mixers                 |

Note: `num_qubits = 2 * num_orbitals`. Mixing the two is the most
common pool bug.

## Fermion-to-Qubit Mappings

```python
op = solvers.jordan_wigner(hpq, hpqrs, core_energy=0.0, tolerance=1e-15)
op = solvers.jordan_wigner(hpq_or_hpqrs, core_energy=0.0)  # auto-detect 2D vs 4D
op = solvers.bravyi_kitaev(...)  # same overload pattern
```

`tol` is accepted as an alias for `tolerance`.

## Self-check (chemistry-specific)

```
[ ] OMP_NUM_THREADS=1 set if reproducibility of coefficients matters.
[ ] If natorb=True or integrals_natorb=True: MP2=True is set.
[ ] num_qubits == 2 * num_orbitals where the pool requires it.
[ ] PySCF server on localhost:8000 is not stale from a previous crash.
[ ] Right pool kwargs:
      uccsd                  -> num_qubits + num_electrons
      uccgsd / upccgsd / ceo -> num_orbitals
```

## Build & validation notes

For full build/wheel/docs guidance see `.claude/skills/cuda-qx-build/`.
Solvers-specific:

- `scikit-build-core`, CMake ≥ 3.28, Ninja ≥ 1.10.
- Default install prefix: `$HOME/.cudaqx`.
- CMake options: `CUDAQX_SOLVERS_INCLUDE_TESTS`,
  `CUDAQX_SOLVERS_BINDINGS_PYTHON`, `CUDAQX_SOLVERS_INSTALL_PYTHON`.
- Test targets (when configured): `make run_tests`,
  `make run_python_tests`.
- Wheels: `cudaq-solvers-cu12`, `cudaq-solvers-cu13`. Internal native
  module: `_pycudaqx_solvers_the_suffix_matters_cudaq_solvers`.
- For code changes, run focused Python tests first; broaden to C++
  unit tests when bindings, headers, or shared APIs are touched.

## Troubleshooting (chemistry & install)

| Symptom                                                     | Cause                                                             | Fix                                                                 |
|-------------------------------------------------------------|-------------------------------------------------------------------|---------------------------------------------------------------------|
| Hamiltonian coefficients change sign across runs            | PySCF threading                                                   | `export OMP_NUM_THREADS=1` before importing                         |
| Second `create_molecule` call hangs                         | Stale `cudaq-pyscf` server on `:8000`                             | `lsof -n -i :8000 && kill -9 <pid>`                                 |
| `RuntimeError` from `create_molecule` mentioning `MP2`      | `natorb=True` or `integrals_natorb=True` without `MP2=True`       | Add `MP2=True`                                                      |
| `cobyla`/`lbfgs` import or call crash on Linux              | Missing `libgfortran`                                             | Install via system package manager                                  |
| `lbfgs` with QAOA: "requires gradients..."                  | QAOA path does not auto-wire a gradient                           | Use `cobyla`, or pass `optimizer=scipy.optimize.minimize` with `jac`|
| ADAPT stops at 30 iterations despite `max_iterations=50`    | ADAPT uses `max_iter`, not `max_iterations`                       | Use `max_iter`                                                      |
| Pool kwarg error                                            | Mixed `num_qubits` and `num_orbitals`                             | UCCSD: `num_qubits` + `num_electrons`. UCCGSD/UPCCGSD/CEO: `num_orbitals` |
| GQE import error                                            | Missing extras                                                    | `pip install cudaq-solvers[gqe]`                                    |
| GQE `sys.exit(1)` on import                                 | PyTorch built for an older CUDA than the GPU                      | Install a CUDA 12.8+ PyTorch (or matching CUDA 13)                  |
| macOS / Windows install fails                               | Not supported                                                     | Use Linux x86_64/aarch64                                            |

## Canonical tests

| Topic              | File                                                                          |
|--------------------|-------------------------------------------------------------------------------|
| Molecule           | `libs/solvers/python/tests/test_molecule.py`, `unittests/test_molecule.cpp`   |
| Operator pools     | `libs/solvers/python/tests/test_operator_pools.py`, `unittests/test_*_operator_pool.cpp` |
| Jordan-Wigner      | `libs/solvers/python/tests/test_jordan_wigner.py`                             |
| Bravyi-Kitaev      | `libs/solvers/unittests/test_bravyi_kitaev.cpp`                               |
