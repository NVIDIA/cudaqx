# First Solvers example, walked through

The smallest end-to-end variational example, with output
interpretation.

## Run it

```bash
python docs/sphinx/examples/solvers/python/uccsd_vqe.py
```

It uses `solvers.create_molecule(...)` to build the H2 Hamiltonian
via PySCF, then runs VQE with a UCCSD ansatz to find the ground-state
energy.

## What the example does

1. Defines an H2 geometry (two hydrogens, 0.7474 Å apart).
2. Calls `solvers.create_molecule(geometry, "sto-3g", spin=0,
   charge=0, casci=True)` — this spawns a PySCF server in the
   background, runs Hartree-Fock + CASCI, and returns a
   `MolecularHamiltonian`.
3. Maps the fermionic Hamiltonian to qubits via Jordan-Wigner.
4. Builds the UCCSD operator pool.
5. Runs `solvers.vqe(...)` with `lbfgs` and `parameter_shift`
   gradients.
6. Reports the converged energy.

## Expected output

For H2 in `sto-3g` at 0.7474 Å, you should see something close to:

```
HF energy:    -1.1167 Ha
FCI energy:   -1.1373 Ha
VQE energy:   -1.1373 Ha
```

The VQE energy matches FCI to ~chemical accuracy (1e-3 Ha) because H2
in a minimal basis has only two parameters and UCCSD is exact for
this small problem. For bigger molecules the VQE energy sits between
HF (upper bound, no correlation) and FCI (exact in the chosen
basis).

## What to look at if it doesn't work

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `ImportError: cudaq_solvers` | wheel not installed | `pip install cudaq-solvers` |
| `RuntimeError: libgfortran.so.5` | missing system dep | `apt install gfortran` |
| Hangs after "Creating molecule" | stale PySCF server on port 8000 | `lsof -n -i :8000 && kill -9 <pid>` |
| Coefficient signs flip across runs | multithreaded PySCF | `export OMP_NUM_THREADS=1` |
| Optimizer reports failure | wrong gradient method | use `parameter_shift` for `lbfgs`; `cobyla` needs no gradient |

## Concept check: what is VQE actually doing?

VQE is a hybrid quantum-classical loop:

1. A parameterized quantum circuit (the **ansatz**) prepares some
   trial state |ψ(θ)⟩.
2. The Hamiltonian H is measured on |ψ(θ)⟩ to estimate ⟨ψ(θ)| H |ψ(θ)⟩
   = E(θ).
3. A classical optimizer (here `lbfgs`) updates θ to minimize E(θ).
4. Repeat until E converges.

The minimum of E(θ) is an upper bound on the true ground-state
energy. Whether you hit FCI depends on the ansatz, the optimizer,
and how aggressively you tune them.

## Where to go next

| If the user wants to | Read |
|----------------------|------|
| Try a bigger molecule | `cuda-qx-solvers-chemistry` SKILL.md (geometry, basis, active space) |
| Try ADAPT-VQE (auto-growing ansatz) | `cuda-qx-solvers-algorithms/references/vqe.md` |
| Run QAOA on a MaxCut problem | `cuda-qx-solvers-algorithms/references/qaoa.md` |
| Use GQE (transformer-driven) | `cuda-qx-solvers-algorithms/references/gqe.md` |
| Compare classical baselines | `cuda-qx-benchmarking` SKILL.md |

At this point the user has a working solvers stack. Hand them off to
`cuda-qx-solvers-algorithms` SKILL.md for the algorithm index.
