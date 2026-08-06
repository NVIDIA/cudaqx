# Fermion-to-qubit mappings

PySCF gives you a fermionic Hamiltonian (one- and two-body integrals
`hpq` and `hpqrs`). The quantum circuit operates on qubits. Mapping
between the two is non-unique; cudaq-solvers ships Jordan-Wigner and
Bravyi-Kitaev.

## API

```python
import cudaq_solvers as solvers

# Jordan-Wigner
op = solvers.jordan_wigner(hpq, hpqrs, core_energy=0.0, tolerance=1e-15)
# Auto-detect 2D vs 4D
op = solvers.jordan_wigner(hpq_or_hpqrs, core_energy=0.0)

# Bravyi-Kitaev (same overload pattern)
op = solvers.bravyi_kitaev(hpq, hpqrs, core_energy=0.0)
```

`tol` is accepted as an alias for `tolerance`. The output is a
`cudaq.SpinOperator`.

Usually you don't call these directly — `create_molecule` does it
internally via the `fermion_to_spin` option (`"jordan_wigner"` by
default, `"bravyi_kitaev"` as an alternative).

## Picking a mapping

| Mapping | Qubit count | Locality | Best for |
|---------|-------------|----------|----------|
| Jordan-Wigner | `n_qubits = 2 * n_orbitals` | non-local (strings of Z) | most VQE / ADAPT workflows; tooling-friendly |
| Bravyi-Kitaev | same `n_qubits` | more local (log-depth strings) | larger systems where deep Z strings hurt simulation fidelity |

**Recommendation**: start with Jordan-Wigner. Switch to Bravyi-Kitaev
only when:

- You are running on hardware where two-qubit gate count matters
  more than other costs, AND
- You have empirically measured BK to be cheaper for your circuits.

For most simulation work the locality difference is invisible.

## Core energy and constant terms

`core_energy` (the inactive-space contribution + nuclear repulsion)
is added as an identity coefficient in the resulting `SpinOperator`.
For accurate total energies, pass it explicitly:

```python
op = solvers.jordan_wigner(hpq, hpqrs, core_energy=mol.energies["nuclear_energy"])
```

Or, if you got the molecule with an active space, use `core_energy`
from the `energies` dict.

If you forget the core energy, the *minimization* still works (VQE
finds the right θ), but the reported energy is offset by a constant.

## Tolerance

`tolerance` drops terms below the threshold from the operator. For
H2 / `sto-3g`, `1e-15` is fine. For larger molecules, `1e-12` to
`1e-10` is reasonable and substantially reduces operator size. Set
it too high and you discard real terms; set it too low and you carry
numerical noise.

## Common pitfalls

| Symptom | Cause |
|---------|-------|
| VQE converges but energy off by ~constant | forgot `core_energy` |
| Two runs with `[1e-15, 1e-10]` give different ground energies | tolerance too aggressive |
| Switched from JW to BK and got different VQE energy | ansatz not re-derived for the new mapping; can't reuse `θ` blindly |
| Operator has surprising number of terms | likely correct — molecule Hamiltonians are O(n^4) terms before symmetry reduction |

## Self-check

```
[ ] core_energy passed in (or `create_molecule` did it for you)
[ ] Tolerance chosen consciously (not just left at default for large systems)
[ ] If switching JW ↔ BK, ansatz re-prepared from scratch
[ ] Ground-state energy from VQE matches classical baseline within ansatz expressiveness
```

## Where next

- Operator pools that work with these mappings: `operator-pools-using.md`.
- Variational algorithm to consume the qubit Hamiltonian:
  `cudaq-solvers-algorithms/references/vqe.md`.
- For very large molecules, qubit reduction techniques (parity
  mapping, qubit tapering) are not in cudaq-solvers today — see PySCF
  / OpenFermion documentation.
