# CUDA-QX Solvers Skill Benchmark

Evaluation prompts for measuring whether the `cuda-qx-solvers` skill makes
the agent measurably better. Designed so it can be run with the helper at
`scripts/score_benchmark.py`, which only checks for `must_include` /
`must_not_include` substrings in agent output.

## Methodology

Run **three** passes:

1. **With skill** — `cuda-qx-solvers` enabled.
2. **Without skill** — control.
3. **Activation pass** — see "Activation Tests" below; only the skill should
   activate where it should and stay quiet where it shouldn't.

Two scoring layers:

**Human rubric** (per scenario, 0–8):

- Correctness (0–2): facts true, paths/APIs real
- Specificity (0–2): cites files, exact API names, exact kwargs
- Coverage (0–2): hits each "must include" item
- No hallucinations (0–2): no "must not include" items

12 scenarios × 8 + 10 activation = 106 max.

**Substring proxy** (`scripts/score_benchmark.py`):

- Coverage = number of `must_include` substrings hit (max 44 for solvers).
- Purity = `must_not_include` substrings missed (max 15 for solvers; each
  scenario contributes at least 1).
- Activation = correct activation behavior on the 10 activation prompts.
- Substring total max = 69 for solvers. Always pair with a human pass.

---

## Scenario Prompts

### 1. Reproducibility

**prompt:** "My H2 Hamiltonian coefficients have different signs every time
I run `solvers.create_molecule`, but the eigenvalues match. Why?"

- must_include: `OMP_NUM_THREADS=1`, "PySCF"
- must_include: "eigenvalue"
- must_not_include: "CUDA-QX bug", "jordan_wigner bug"

### 2. Hung PySCF Server

**prompt:** "My second `solvers.create_molecule` call hangs forever. What's
going on?"

- must_include: `localhost:8000`, `lsof`, `cudaq-pyscf`
- must_not_include: "reinstall CUDA-Q", "change basis"

### 3. Active Space with Natural Orbitals

**prompt:** "How do I run an active-space N2 calculation with natural
orbitals?"

- must_include: `nele_cas`, `norb_cas`, `casci=True`, `natorb=True`,
  `MP2=True`
- must_not_include: "natorb works without MP2"

### 4. QAOA + L-BFGS Failure

**prompt:** "Why does `solvers.qaoa(..., optimizer='lbfgs')` fail with a
gradient error?"

- must_include: "gradient", "lbfgs", "cobyla"
- must_not_include: "this is an LBFGS bug"

### 5. ADAPT vs VQE Option Keys

**prompt:** "I set `max_iterations=50` on `adapt_vqe` but it still stops
after 30. Bug?"

- must_include: `max_iter`, "30"
- must_not_include: "max_iterations is correct"

### 6. GQE Multi-GPU Setup

**prompt:** "How do I run GQE on H2 across multiple GPUs?"

- must_include: `cudaq-solvers[gqe]`, `mqpu`, `mpiexec`,
  `cudaq.mpi.initialize`, `PMIX_MCA_gds=hash`
- must_not_include: "GQE supports CPU multi-process"

### 7. Operator Pool Kwargs

**prompt:** "How do I generate UCCSD vs UCCGSD operator pools?"

- must_include: `num_qubits`, `num_electrons`, `num_orbitals`, `uccsd`,
  `uccgsd`
- must_not_include: passing `num_orbitals` to UCCSD as the only arg

### 8. SciPy Optimizer Forwarding

**prompt:** "Can I use any SciPy optimizer with `solvers.vqe`? What kwargs
are forwarded?"

- must_include: `scipy.optimize.minimize`, `method`, `jac`, `tol`
- must_include: stripped: `gradient`, `optimizer`, `verbose`, `shots`
- must_not_include: "any callable optimizer is supported"

### 9. MaxCut on a NetworkX Graph

**prompt:** "I want to solve MaxCut on a weighted graph with QAOA. What's
the minimum code?"

- must_include: `networkx`, `get_maxcut_hamiltonian`,
  `get_num_qaoa_parameters`, `solvers.qaoa`
- must_not_include: "use stim for graph problems"

### 10. libgfortran Missing

**prompt:** "I installed `cudaq-solvers` but optimizer calls crash on Linux.
What's missing?"

- must_include: `libgfortran`, "system package"
- must_not_include: "macOS", "Windows"

### 11. solvers.optim.optimize Signature

**prompt:** "What does `solvers.optim.optimize` expect from my objective
function?"

- must_include: `(value, list)`, "gradient-based", "method"
- must_not_include: "always returns just a float"

### 12. ADAPT Dynamic Start

**prompt:** "What does `dynamic_start` mean in `adapt_vqe`?"

- must_include: `cold`, `warm`
- must_not_include: "auto-detect"

---

## Activation Tests

Each prompt is labeled `should_activate` (Y/N). Score 1 point when behavior
matches.

| # | prompt | should_activate |
| --- | --- | --- |
| A1 | "Run VQE on H2 in CUDA-QX" | Y |
| A2 | "Bootstrap a QAOA MaxCut script" | Y |
| A3 | "Configure GQE for H4" | Y |
| A4 | "How do I install CUDA-Q itself?" | N |
| A5 | "What's the best decoder for surface codes?" | N |
| A6 | "Write a Bell state kernel" | N |
| A7 | "Generate a molecular Hamiltonian for LiH" | Y |
| A8 | "Convert hpq/hpqrs to qubit operators" | Y |
| A9 | "Why is my CUDA-Q `nvidia` target slow?" | N |
| A10 | "Use ADAPT-VQE with custom optimizer pools" | Y |

## Sources

- `libs/solvers/python/bindings/solvers/py_solvers.cpp`
- `libs/solvers/python/bindings/solvers/py_optim.cpp`
- `libs/solvers/python/cudaq_solvers/gqe_algorithm/gqe.py`
- `libs/solvers/python/tests/test_*.py`
- `libs/solvers/lib/operators/molecule/drivers/pyscf_driver.cpp`
- `libs/solvers/README.md`, `libs/solvers/pyproject.toml.cu12`
