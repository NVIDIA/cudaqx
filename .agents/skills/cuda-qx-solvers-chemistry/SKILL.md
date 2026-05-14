---
name: "cuda-qx-solvers-chemistry"
title: "CUDA-QX Solvers — Quantum Chemistry"
description: >-
  Quantum chemistry deep-dive for CUDA-QX: molecule construction with
  PySCF, basis sets (sto-3g, 6-31g, cc-pVDZ, ...), geometry input from
  XYZ files or Python lists, charge / spin / symmetry, active-space
  selection (nele_cas / norb_cas / casci / casscf / natorb / MP2),
  fermion-to-qubit mappings (Jordan-Wigner, Bravyi-Kitaev), operator
  pools (UCCSD / UCCGSD / UPCCGSD / spin-complement-GSD / CEO),
  classical baselines (HF / MP2 / CCSD / CASCI / CASSCF / FCI) for
  validation, and the PySCF REST server. Use whenever the user mentions
  chemistry, molecule, create_molecule, PySCF, basis set, geometry,
  active space, CAS, MP2, HF, FCI, CCSD, jordan_wigner, bravyi_kitaev,
  operator pool, fermion-to-qubit mapping, or molecular Hamiltonian.
version: "0.2.1"
author: "CUDA-QX"
license: "Apache License 2.0"
compatibility: "Python 3.11+, Linux x86_64/aarch64, PySCF (auto-installed)"
tags: [cuda-qx, cudaq-solvers, chemistry, pyscf, molecule, basis-set, active-space, fermion-to-qubit, operator-pool, classical-baselines]
tools: [Read, Glob, Grep, Bash]
metadata:
  repo: [solvers]
  author: "CUDA-QX"
  domain: "quantum-chemistry"
  audience: [chemist, researcher, student]
  languages: [python, c++]
---

# CUDA-QX Quantum Chemistry

The chemistry stack underneath every solver algorithm. `cudaq-qx-solvers`
points here when the user is doing chemistry; this skill goes deeper
than the one chemistry page in solvers, with full attention to
classical baselines, active-space selection, and PySCF pitfalls.

If the user is doing pure optimization (QAOA on MaxCut, no chemistry),
delegate to `cuda-qx-solvers-algorithms/references/qaoa.md`. If they are choosing
an algorithm (VQE vs ADAPT vs GQE), delegate to `cuda-qx-solvers-algorithms`. This
skill covers everything **before** the variational loop: getting a
correct Hamiltonian to feed in.

## Inputs

Caller provides:

- A molecule: geometry (`[(element, (x,y,z)), ...]` in Å, or XYZ path),
  basis (`"sto-3g"` / `"cc-pVDZ"` / ...), `spin` (= 2S), `charge`.
- Optional: active space (`nele_cas`, `norb_cas`); classical baselines
  to compute (`MP2`, `casci`, `casscf`, `ccsd`); fermion-to-qubit
  mapping (`jordan_wigner` default, or `bravyi_kitaev`).
- A free `localhost:8000` for the PySCF REST server.
- `OMP_NUM_THREADS=1` if Hamiltonian-coefficient signs must be
  reproducible across runs.

## Outputs

This skill produces:

- A `MolecularHamiltonian` with `hamiltonian` (qubit `SpinOperator`),
  `hpq`, `hpqrs`, `n_electrons`, `n_orbitals`, `energies` dict
  (HF / MP2 / CCSD / CASCI / FCI as requested).
- `n_qubits = 2 * n_orbitals` (after default JW mapping).
- An operator pool ready to hand to ADAPT/GQE
  (`solvers.get_operator_pool(name, ...)`).
- A baseline-energy sanity check: VQE energy must lie between HF and
  the best classical baseline computed.

Does NOT produce: VQE/ADAPT/QAOA/GQE convergence (→ `cuda-qx-solvers-algorithms`);
custom operator pools / optimizers (→ `cuda-qx-solvers-extending`).

## Audience

Quantum chemists, computational chemistry students, and developers
who need molecular Hamiltonians for VQE / ADAPT / GQE. Prior PySCF
experience helps but is not required.

## First three actions

```bash
bash   .agents/skills/_shared/scripts/preflight.sh    --json > /tmp/preflight.json
python .agents/skills/_shared/scripts/import_smoke.py --json > /tmp/import_smoke.json
python -c "import pyscf; print(pyscf.__version__)"
```

`pyscf` is bundled by `cudaq-solvers`; if it is missing, your wheel is
broken. Also check that **port 8000 is free** (PySCF server). A stale
server is the #1 chemistry blocker.

```bash
lsof -n -i :8000      # should be empty before first create_molecule
```

## Key Paths

| Area | Path |
|------|------|
| Molecule API (Python) | `libs/solvers/python/cudaq_solvers/__init__.py` (search `create_molecule`) |
| Molecule API (C++) | `libs/solvers/include/cudaq/solvers/operators/molecule.h` |
| PySCF driver (C++) | `libs/solvers/lib/operators/molecule/drivers/pyscf_driver.cpp` |
| PySCF server tool | `libs/solvers/tools/molecule/cudaq-pyscf.py` |
| Operator pool headers | `libs/solvers/include/cudaq/solvers/operators/operator_pools/` |
| UCCSD / UCCGSD utilities | `libs/solvers/include/cudaq/solvers/operators/uccgsd_excitation_utils.h`, `ceo_excitation_utils.h` |
| Chemistry component doc | `docs/sphinx/components/solvers/introduction.rst` (Molecular Hamiltonian Options section) |
| Molecular Hamiltonian RST | `docs/sphinx/examples_rst/solvers/molecular_hamiltonians.rst` |
| Examples (Python) | `docs/sphinx/examples/solvers/python/generate_molecular_hamiltonians.py`, `uccsd_vqe.py`, `adapt_h2.py`, `gqe_h2.py` |

## Source of Truth

- **`create_molecule` signature**: grep
  `libs/solvers/python/cudaq_solvers/__init__.py` for the function.
- **`molecule_options` C++ struct**:
  `libs/solvers/include/cudaq/solvers/operators/molecule.h`.
- **Component doc table** (all options + defaults):
  `docs/sphinx/components/solvers/introduction.rst`.

## Workflow Index

| If the user wants to | Read |
|----------------------|------|
| Build a molecule, choose basis, set active space | `references/molecule-building.md` |
| Pick a fermion-to-qubit mapping (JW vs BK) | `references/fermion-mappings.md` |
| Choose an operator pool (UCCSD / UCCGSD / CEO) | `references/operator-pools-using.md` |
| Compare against classical baselines (HF / MP2 / CCSD / FCI) | `references/classical-baselines.md` |
| Understand / debug the PySCF REST server (port 8000) | `references/pyscf-server.md` |

## Conventions

These produce silent wrong answers, not visible errors. Read carefully.

1. **`OMP_NUM_THREADS=1` for reproducible coefficients.** PySCF
   flips signs on Hamiltonian coefficients across runs when
   multithreaded. The eigenvalues are still correct, but term-by-
   term comparisons across runs fail.

2. **`num_qubits = 2 * num_orbitals`.** UCCSD operator pools take
   `num_qubits`; UCCGSD / UPCCGSD / spin-complement-GSD / CEO take
   `num_orbitals`. Mixing them produces nonsense pools that silently
   give bad energies.

3. **`natorb=True` and `integrals_natorb=True` require `MP2=True`.**
   Otherwise `create_molecule` raises a `RuntimeError`. The same
   relationship holds for `casci`, `ccsd`, `casscf` and their energy
   keys.

4. **Active-space arithmetic**: `nele_cas` electrons in `norb_cas`
   spatial orbitals → `2 * norb_cas` qubits, `nele_cas` electrons in
   ansatz. Forgetting the factor of two in the qubit count is the
   most common bug.

5. **`create_molecule` spawns PySCF on `localhost:8000`.** If the
   port is occupied (stale server from a previous crashed run),
   `create_molecule` hangs. Symptom: second call after a crash never
   returns. Fix:

   ```bash
   lsof -n -i :8000
   kill -9 <pid>
   ```

6. **`libgfortran` is a runtime dependency** (for `cobyla` / `lbfgs`).
   Quick fix: `apt install gfortran`. See `cuda-qx-build` for the full
   explanation.

7. **PySCF coordinates are Ångström by default.** Bohr is also
   accepted via `unit='Bohr'` to PySCF, but `cudaq_solvers`
   `create_molecule` uses Å. Mixing units silently gives garbage
   energies.

## Quick start: H2

```python
import cudaq_solvers as solvers

geometry = [("H", (0., 0., 0.)), ("H", (0., 0., 0.7474))]
mol = solvers.create_molecule(
    geometry, "sto-3g", spin=0, charge=0,
    casci=True, ccsd=True, verbose=True
)

print("HF:", mol.energies["hf_energy"])
print("FCI:", mol.energies["fci_energy"])    # exact in this basis
print("CCSD:", mol.energies["R-CCSD"])
print("n_qubits =", 2 * mol.n_orbitals)
```

A real molecule replaces `geometry` and chooses a bigger basis. See
`references/molecule-building.md` for the full menu.

## Self-Check Protocol

```
[ ] `OMP_NUM_THREADS=1` set before import.
[ ] No process holding port 8000 before first create_molecule call.
[ ] num_qubits used in pool == 2 * num_orbitals.
[ ] If natorb / integrals_natorb were used, MP2=True was set too.
[ ] Classical baseline (HF, CCSD, or FCI for tiny systems) computed
    and recorded in the same script — VQE energy must lie between HF
    (worst) and FCI (best).
[ ] For runs where signs of coefficients matter, OMP_NUM_THREADS=1.
[ ] At the dissociation limit (large bond length), the active-space
    energies still make physical sense (no spurious binding).
```

## When stuck

1. Re-run the **First three actions**.
2. Read the matching `references/<topic>.md` end-to-end.
3. Reproduce on H2 first — if H2 works and your molecule doesn't,
   it's a basis / active-space / mapping issue, not a CUDA-QX issue.
4. For optimizer / gradient / convergence issues, delegate to
   `cuda-qx-solvers-algorithms/references/vqe.md`.
5. For "I think there's a PySCF bug" → run the equivalent calculation
   with PySCF directly (no cudaq) to isolate.

## Additional resources

- `references/molecule-building.md` — geometry, basis, charge, spin,
  active space, MP2/CASCI/CASSCF gating.
- `references/fermion-mappings.md` — Jordan-Wigner vs Bravyi-Kitaev,
  qubit count, locality.
- `references/operator-pools-using.md` — UCCSD / UCCGSD / UPCCGSD /
  spin-complement-GSD / CEO, when to use which.
- `references/classical-baselines.md` — HF / MP2 / CCSD / CASCI /
  CASSCF / FCI as validation targets.
- `references/pyscf-server.md` — what `cudaq-pyscf --server-mode` is,
  how to debug a hang on port 8000, when to kill an orphan.
- Variational algorithms (VQE, ADAPT, GQE): `cuda-qx-solvers-algorithms/SKILL.md`.
- Benchmarking energies: `cuda-qx-benchmarking` SKILL.md.
