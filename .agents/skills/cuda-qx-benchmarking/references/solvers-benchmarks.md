# Solvers benchmarks (VQE / ADAPT / GQE)

The "is my variational method actually better?" page. Three things
matter for any solver benchmark:

1. **Energy** vs classical baseline (HF / CCSD / FCI).
2. **Time-to-chemical-accuracy** (or whatever threshold matters).
3. **Resource use** — circuit depth, parameter count, GPU hours.

If any of those is missing, the comparison is incomplete.

## Pick the right benchmark molecule

| Molecule | Why |
|----------|-----|
| H2 / `sto-3g` | smallest non-trivial; FCI is reachable, parameter count is ~1 |
| LiH / `sto-3g` | next step up; tests multi-orbital coupling |
| BeH2 / `sto-3g` | strongly correlated at bond stretching |
| H4 chain | classic benchmark for strong correlation |
| H2O / `cc-pVDZ` | medium-size realistic chemistry |

Avoid jumping to "real" chemistry (e.g. N2, FeMoco) until smaller
benchmarks confirm the method works.

## Authoritative examples

- `docs/sphinx/examples/solvers/python/uccsd_vqe.py` — H2 VQE.
- `docs/sphinx/examples/solvers/python/adapt_h2.py` — H2 ADAPT.
- `docs/sphinx/examples/solvers/python/gqe_h2.py` — H2 GQE.

These are also the "reference run": if your fork gives a different
H2 energy, your fork is broken.

## Three plots for a paper

### 1. Energy convergence

Energy vs iteration (or operator additions for ADAPT). Plot HF, CCSD,
FCI as horizontal lines.

```python
import matplotlib.pyplot as plt

iters, energies = [], []
for it, e in vqe_iteration_log:
    iters.append(it); energies.append(e)
plt.plot(iters, energies, label="VQE")
plt.axhline(hf, ls='--', label="HF")
plt.axhline(fci, ls='-', label="FCI")
plt.fill_between(iters, [fci+1e-3]*len(iters), [fci-1e-3]*len(iters),
                 alpha=0.2, label="chemical accuracy")
```

### 2. Method comparison

VQE vs ADAPT vs GQE on the same molecule:

| Method | Final energy | Iterations / ops | Parameters | Wall-clock |
|--------|--------------|------------------|------------|------------|
| VQE (UCCSD) | -1.137259 | 27 | 3 | 4.1 s |
| ADAPT-VQE | -1.137270 | 5 ops | 5 | 12 s |
| GQE | -1.137260 | (trained) | (learned) | 10 min train + 0.3 s inference |

Always include classical baselines as rows.

### 3. Scaling

For a molecule series (H2 → H4 → H8), plot:

- Energy error vs FCI (or CCSD-T).
- Total wall-clock.
- Best-found parameter count.

This identifies where your method breaks down.

## Reproducibility for variational runs

VQE is very sensitive to:

- Random seeding of initial parameters.
- Optimizer (`cobyla` vs `lbfgs` vs SciPy).
- Gradient method (`parameter_shift` vs `finite_difference`).
- `OMP_NUM_THREADS=1` for sign-stable PySCF coefficients.

Capture all of these in the script's output.

```python
import random, numpy as np, os
random.seed(0); np.random.seed(0); os.environ["OMP_NUM_THREADS"] = "1"
```

For ADAPT, the operator-add ordering depends on gradient evaluation,
which is sensitive to shot noise. For ADAPT comparisons, use exact
expectations (large `shots` or analytic if available).

## Time-to-chemical-accuracy

Chemical accuracy = `|E_VQE - E_FCI| < 1e-3` Ha. Report:

- Wall-clock to reach this gap.
- Iteration count to reach this gap.
- Whether it was ever reached at all.

A method that converges to `1e-2` Ha and stops is a fundamentally
different beast from one that converges to `1e-5` Ha; both should be
reported honestly.

## GQE-specific

GQE is multi-GPU and has a training cost separate from inference.
Report:

- Training time and GPU count.
- Inference time per operator-sequence proposal.
- Best-found energy.
- Token / parameter count of the transformer.

Sources:

- `libs/solvers/python/cudaq_solvers/gqe_algorithm/gqe.py`.
- Examples directory above.

## Common pitfalls

| Symptom | Cause |
|---------|-------|
| ADAPT slower than VQE for H2 | expected — ADAPT pays per-iteration gradient cost; not a regression |
| VQE energy bounces across runs | initial-parameter seed not pinned |
| GQE inference is "instant" — too good to be true | inference time excluded training, which dominates |
| OMP-thread mismatch between runs | OMP_NUM_THREADS=1 not set |
| Energy matches FCI in `sto-3g` but is far from experimental value | basis set too small; not a VQE problem |

## Self-check

```
[ ] Energy reported alongside HF, MP2, CCSD, FCI baselines.
[ ] Random seed and OMP_NUM_THREADS captured in the run log.
[ ] Optimizer + gradient method captured.
[ ] Wall-clock includes warmup / first-call overhead OR is explicitly post-warmup.
[ ] For ADAPT/GQE: training cost and inference cost reported separately.
[ ] At least 3 independent runs reported for stochastic methods.
[ ] Final-energy uncertainty (std across seeds) reported, not just mean.
```

## Where next

- Reproducibility envelope: `reproducibility.md`.
- Profile a slow run: `cuda-qx-profiling-perf`.
- Pick a different operator pool: `cuda-qx-solvers-chemistry/references/operator-pools-using.md`.
