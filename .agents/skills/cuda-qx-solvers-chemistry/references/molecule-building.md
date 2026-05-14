# Building a molecule

The `solvers.create_molecule` call wraps PySCF. Everything starts
here. Get the geometry, basis, and active space right; everything
downstream depends on this object.

## Minimal call

```python
import cudaq_solvers as solvers

geometry = [("H", (0., 0., 0.)), ("H", (0., 0., 0.7474))]
mol = solvers.create_molecule(geometry, "sto-3g",
                              spin=0, charge=0,
                              casci=True, verbose=True)
```

Returns a `MolecularHamiltonian` with `hamiltonian` (a
`cudaq.SpinOperator`), `hpq`, `hpqrs`, `n_electrons`, `n_orbitals`,
and an `energies` dict keyed by the requested methods.

## Geometry

Two forms accepted:

1. Python list of `(element, (x, y, z))` tuples (Ångström).
2. Path to an XYZ file string.

```python
# List form
geometry = [("O", (0.0,  0.000,  0.000)),
            ("H", (0.0,  0.757,  0.587)),
            ("H", (0.0, -0.757,  0.587))]

# XYZ file
mol = solvers.create_molecule("water.xyz", "sto-3g", spin=0, charge=0)
```

Coordinates are **Ångström**. Mixing Bohr silently yields wrong
energies.

## Basis sets

Common choices, in roughly increasing accuracy and cost:

| Basis | Use case |
|-------|----------|
| `sto-3g` | smoke tests, H2, prototyping |
| `6-31g`, `6-31g(d)` | small molecules, qualitative results |
| `cc-pVDZ` | medium accuracy; chemistry standard for QML benchmarks |
| `cc-pVTZ` | high accuracy, expensive for QML |
| `def2-svp`, `def2-tzvp` | Ahlrichs basis family; good for heavier elements |

Any PySCF-supported basis string works. Larger bases → more
orbitals → more qubits. For QML, restrict to an active space (see
below) rather than throw everything in.

## Charge and spin

`spin` is `2S` (= number of unpaired electrons). H2 → `spin=0`,
neutral O atom → `spin=2` (triplet). `charge` is the molecular
charge. Together these determine the electron count:

```
n_electrons = sum(atomic_numbers) - charge
```

Wrong `spin` produces a converged but *wrong* HF state; downstream
VQE will hit chemical garbage.

## Active space

The single most important parameter for keeping qubit counts manageable.
Active space = N electrons in M spatial orbitals → `2*M` qubits.

```python
mol = solvers.create_molecule(
    geometry, "cc-pVDZ", spin=0, charge=0,
    nele_cas=4, norb_cas=4,         # 4 electrons in 4 orbitals -> 8 qubits
    casci=True
)
```

- `nele_cas` — number of electrons in the active space.
- `norb_cas` — number of *spatial* orbitals in the active space.
  Qubits = `2 * norb_cas`.

Selection rules of thumb:

- Include the orbitals around the HOMO/LUMO (frontier orbitals).
- For dissociation curves, include orbitals that change character
  along the curve.
- For magnetic / strongly correlated systems, include the
  partially-occupied 3d/4d orbitals.

`natorb=True` requires `MP2=True`; same for `integrals_natorb=True`.
`casscf=True` activates the CASSCF energy.

## Method flags and their energy keys

| Flag | Energy keys appear |
|------|--------------------|
| (default) | `hf_energy`, `nuclear_energy` (or `core_energy` for active space) |
| `casci=True` | `fci_energy` (no CAS), or `R-CASCI` / `UR-CASCI` (with CAS) |
| `ccsd=True` | `R-CCSD` (or `UR-CCSD` if `UR=True`) |
| `casscf=True` | `R-CASSCF` |
| `MP2=True` | enables MP2-dependent options |

Use these for validation: VQE energy should lie between HF (cheap
upper bound, no correlation) and FCI / FCI-equivalent (best in the
chosen basis / active space).

## Performance options

| Option | Use |
|--------|-----|
| `memory` (MB) | PySCF memory budget; raise for big bases (default 4000) |
| `cycles` | max SCF cycles (default 100) |
| `symmetry=True` | use molecular symmetry for SCF; speeds up for symmetric molecules |
| `UR=True` | unrestricted; needed for open-shell |
| `verbose=True` | log PySCF output to stdout |

## Common pitfalls

| Symptom | Cause | Fix |
|---------|-------|-----|
| Hangs after print "creating molecule" | port 8000 occupied | `lsof -n -i :8000; kill -9 <pid>` |
| `RuntimeError: ... natorb requires MP2` | natorb=True, MP2=False | set `MP2=True` |
| Sign flips across runs | OMP parallelism | `export OMP_NUM_THREADS=1` |
| HF doesn't converge | bad initguess on tricky molecule | try `initguess="atom"` or `initguess="hcore"` |
| Energy way different from literature | wrong charge or spin; or wrong basis | re-check charge/spin/basis vs reference paper |

## Self-check

```
[ ] sum(Z_i) - charge equals n_electrons reported on mol
[ ] 2 * mol.n_orbitals matches expected qubit count
[ ] HF energy within 0.1 Ha of a published reference for the same geometry/basis
[ ] If active space is used, energies are consistent across a small basis perturbation
[ ] If natorb/integrals_natorb=True, MP2=True is set
```

## Where next

- Map to qubit operator: `fermion-mappings.md`.
- Pick a pool for ADAPT/UCCSD/etc: `operator-pools-using.md`.
- Compare to classical methods: `classical-baselines.md`.
- Run the variational loop: `cuda-qx-solvers-algorithms/references/vqe.md`.
