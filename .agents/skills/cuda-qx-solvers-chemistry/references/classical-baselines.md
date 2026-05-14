# Classical baselines (HF, MP2, CCSD, CASCI, CASSCF, FCI)

Always validate VQE against a classical baseline before trusting the
result. PySCF (via cudaq-solvers `create_molecule`) computes these
for you when you set the corresponding flags. This page is a
short guide to which baseline matters for what.

## The hierarchy

Roughly increasing accuracy, increasing cost:

| Method | What it is | Activated by |
|--------|-----------|--------------|
| HF (Hartree-Fock) | mean-field, single Slater determinant | (default; always available) |
| MP2 | second-order perturbation on HF | `MP2=True` |
| CCSD | coupled-cluster with singles+doubles | `ccsd=True` |
| CASCI | full CI within an active space | `casci=True` + `nele_cas`/`norb_cas` |
| CASSCF | CASCI + orbital optimization | `casscf=True` |
| FCI | exact in the chosen basis | `casci=True` with no CAS restriction |

VQE's correct answer is bounded:

```
HF >= VQE >= FCI (lower-bound in the chosen basis)
```

If VQE comes out *below* FCI you have a bug. If it comes out *above*
HF the ansatz is too constrained.

## Reading energies off `mol.energies`

```python
mol = solvers.create_molecule(geom, "sto-3g", spin=0, charge=0,
                              casci=True, ccsd=True, MP2=True,
                              verbose=True)

print(mol.energies)
# {'hf_energy': ..., 'fci_energy': ..., 'R-CCSD': ...,
#  'nuclear_energy': ..., 'mp2_energy': ...}
```

Energy keys depend on the flags:

| Flag | Key(s) appearing |
|------|------------------|
| (always) | `hf_energy`, `nuclear_energy` (or `core_energy` for active space) |
| `MP2=True` | `mp2_energy` |
| `ccsd=True` | `R-CCSD` (or `UR-CCSD` if `UR=True`) |
| `casci=True` no CAS | `fci_energy` |
| `casci=True` with `nele_cas`/`norb_cas` | `R-CASCI` (or `UR-CASCI`) |
| `casscf=True` | `R-CASSCF` |

Source of truth: `pyscf_driver.cpp` + the introduction RST.

## Which baseline matters for which problem

| Problem | Baseline to compare VQE against |
|---------|---------------------------------|
| H2 / LiH minimal basis | FCI (exact) |
| H2O / N2 small basis | CCSD (gold standard for closed-shell) |
| Bond-breaking / dissociation curves | CASCI or CASSCF (single-reference methods like CCSD fail) |
| Transition metals, magnetism | CASSCF (or NEVPT2 if you have access) |
| Open-shell radicals | UR-CCSD or UR-CASCI |

For tiny systems where FCI is feasible, FCI is the right comparison.
For everything else, CCSD-T (when available) is the practical
standard but is not surfaced by `create_molecule` today; CCSD is the
next-best automatically-computed reference.

## Sanity-check pattern

Always print these three before launching VQE:

```python
print(f"HF:   {mol.energies['hf_energy']:.6f}")
print(f"FCI:  {mol.energies.get('fci_energy', 'n/a')}")
print(f"CCSD: {mol.energies.get('R-CCSD', 'n/a')}")
```

If HF and FCI are essentially equal, your basis is too small for VQE
to teach you anything. If they differ by ~`0.05` Ha (chemical
accuracy threshold is ~`1e-3` Ha), you have a system where
correlation matters and VQE has room to shine.

## When VQE doesn't reach the baseline

Possible reasons, in order of frequency:

1. Ansatz too constrained (wrong pool, too few iterations of ADAPT).
2. Optimizer stuck (try `cobyla` instead of `lbfgs`, or a different
   starting point).
3. Statistical noise (shots-based estimation; raise `shots` or use
   exact expectation for development).
4. Wrong fermion-to-qubit mapping (you re-parameterized but kept
   the old `θ`).
5. Wrong active space (the missing correlation is in orbitals you
   excluded).

Note: if `MP2=True` *flipped Hamiltonian signs* across runs, that's
the `OMP_NUM_THREADS` bug — eigenvalues are still right, but
term-by-term debug prints will look like noise.

## Self-check

```
[ ] HF and at least one correlated method (CCSD or FCI) computed and printed.
[ ] HF >= VQE >= correlated-baseline within ansatz expressiveness.
[ ] If the molecule has bond-breaking character, CASCI/CASSCF used as the reference, not CCSD.
[ ] `nuclear_energy` (or `core_energy`) included in the qubit operator.
[ ] OMP_NUM_THREADS=1 if signs of coefficients are being compared.
```

## Where next

- Bigger calculations: increase `memory`, use `symmetry=True`,
  consider tight active space.
- Choose an algorithm: `cuda-qx-solvers-algorithms/references/vqe.md`,
  `gqe.md`.
- Compare against external reference data: literature, NIST
  CCCBDB, or your own PySCF runs.
