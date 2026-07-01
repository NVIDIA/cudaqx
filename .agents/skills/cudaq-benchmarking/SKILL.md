---
name: "cudaq-benchmarking"
title: "CUDA-Q Libraries Benchmarking and Reproducibility"
description: >-
  Reproducibly compare CUDA-Q Libraries methods: QEC logical-error-rate sweeps,
  pseudo-threshold curves, decoder accuracy vs speed comparisons, VQE /
  ADAPT / GQE energy convergence and time-to-solution comparisons,
  throughput / tail-latency measurement for real-time decoding, with
  proper seeding, OMP_NUM_THREADS reproducibility, statistical error
  bars, plotting conventions, and paper-grade tables. Use whenever the
  user mentions benchmarking, threshold, pseudo-threshold, scaling,
  error bars, statistical significance, reproducibility, "compare
  decoder A vs B", "fair comparison", paper plots, or "what should I
  publish". Do NOT use for: diagnosing why a single piece of code is
  slow (use cudaq-profiling-perf); configuring or training the
  individual algorithm being compared (use cudaq-qec-decode /
  cudaq-qec-ai-decoders / cudaq-solvers-algorithms).
version: "0.1.1"
author: "CUDA-Q Libraries"
license: "Apache License 2.0"
compatibility: "Python 3.11+, Linux x86_64/aarch64 (some GPU-only paths)"
tags: [cudaq, benchmarking, reproducibility, threshold, pseudo-threshold, error-bars, statistics, scaling, paper-results]
tools: [Read, Glob, Grep, Bash]
metadata:
  repo: [qec, solvers]
  author: "CUDA-Q Libraries"
  domain: "evaluation"
  audience: [researcher, ml-researcher, academic]
  languages: [python]
---

# CUDA-Q Libraries Benchmarking and Reproducibility

The "is my decoder actually better?" and "how do I publish this?"
skill. Comparing two decoders, threshold sweeps, or VQE convergence
curves is easy to get wrong: shared random seeds, mismatched DEMs,
misreading wall-clock as compute time. This skill is the
sanity-checking layer.

## Inputs

Caller provides:

- The methods to compare (e.g. two decoders, two solvers, two
  configurations of the same).
- A dataset / DEM / molecule (or instructions to generate one).
- Hardware constraints (GPU model, time budget, shot budget).
- The metric of interest: LER, throughput, energy convergence, etc.

## Outputs

This skill produces:

- A comparison table with point estimates ± SE (binomial for LER;
  std-across-seeds for stochastic methods).
- Plot files (`.png` + a `.npz` raw-data sidecar).
- A reproducibility record (env, seeds, git SHA, hardware).
- A statistical verdict on whether the difference is real (McNemar /
  t-test where appropriate).

Does NOT produce: the actual decode (→ `cudaq-qec-decode`); the actual
variational run (→ `cudaq-solvers-algorithms`); profiling traces (→
`cudaq-profiling-perf`).

## Audience

Researchers preparing benchmark tables and figures for papers,
reports, or grant deliverables. Also useful for engineers verifying
that a change didn't regress accuracy or speed.

## First three actions

```bash
bash   .agents/skills/_shared/scripts/preflight.sh    --json > /tmp/preflight.json
python .agents/skills/_shared/scripts/import_smoke.py --json > /tmp/import_smoke.json
nvidia-smi --query-gpu=name,driver_version,utilization.gpu --format=csv 2>/dev/null || true
```

The `nvidia-smi` snapshot is the start of your reproducibility
record. Capture it in your script's output.

## Key Paths

| Area | Path |
|------|------|
| Pseudo-threshold example | `docs/sphinx/examples/qec/python/pseudo_threshold.py` |
| Per-gate noise example | `docs/sphinx/examples/qec/python/custom_repetition_code_fine_grain_noise.py` |
| Tensor-network exact baseline | `docs/sphinx/examples/qec/python/tensor_network_decoder.py` |
| ADAPT vs VQE | `docs/sphinx/examples/solvers/python/adapt_h2.py`, `uccsd_vqe.py` |
| GQE benchmark | `docs/sphinx/examples/solvers/python/gqe_h2.py` |
| Built-in `doctor.sh` | `scripts/doctor.sh` |

## Workflow Index

| If the user wants to | Read |
|----------------------|------|
| Compare two decoders on the same DEM | `references/decoder-comparison.md` |
| Run a threshold / pseudo-threshold sweep | `references/threshold-sweep.md` |
| Benchmark VQE / ADAPT / GQE energy + time | `references/solvers-benchmarks.md` |
| Capture a reproducibility record for a paper | `references/reproducibility.md` |

## Conventions

These are the recurring "we ran A vs B but the comparison is invalid"
mistakes.

1. **Same DEM for every decoder.** When comparing decoders, *all*
   decoders must consume the same `dem.detector_error_matrix` and
   the same `dem.observables_flips_matrix`. Regenerating the DEM
   between runs changes the noise instance.

2. **Independent random seeds across decoders, same seeds across
   shots.** Each decoder sees the same shots; results differ only in
   how the syndromes are decoded.

3. **`OMP_NUM_THREADS=1` for chemistry benchmarks.** Required for
   reproducible Hamiltonian coefficients across runs. Full
   explanation in `cudaq-solvers-chemistry` (Convention #1).

4. **Wall-clock != compute time.** First-call latency includes
   one-time engine loads, plugin lookups, kernel JIT, and CUDA Graph
   capture. Warm up before timing.

5. **Statistical error bars or it didn't happen.** A "decoder A
   beats B by 5%" claim from 100 shots is noise. Use enough shots to
   distinguish at, ideally, the 3-σ level — see
   `references/decoder-comparison.md`.

6. **Same hardware for every data point in a plot.** Mixing GPU
   models within a benchmark series invalidates the comparison.

7. **Capture the environment in the script's output.** `nvidia-smi`,
   `pip list | grep -E 'cuda-quantum|cudaq-'`, git SHA, host name.

## Quick start: head-to-head decoder

This skill's job is the **comparison harness** (shared inputs, error
bars, statistical test). Get the DEM + shots from `cudaq-qec-decode`, then
run:

```python
# Assumes you already have `dem`, `shots`, `obs` from cudaq-qec-decode.
# (For how to build them, see cudaq-qec-decode/references/decode.md.)
import numpy as np

decoders = {
    "single_error_lut": qec.get_decoder("single_error_lut", dem.detector_error_matrix),
    "multi_error_lut":  qec.get_decoder("multi_error_lut",  dem.detector_error_matrix,
                                        lut_error_depth=2),
}

for name, dec in decoders.items():
    errs = sum(
        1 for s in shots
        if ((dec.decode(s.syndrome).result > 0.5).astype(np.uint8) @ obs.T % 2
            != s.observable).any()
    )
    ler = errs / len(shots)
    se  = np.sqrt(ler * (1 - ler) / len(shots))    # binomial SE
    print(f"{name}: LER = {ler:.4g} ± {se:.4g} (1σ)")
```

The `± 1σ` is binomial standard error. For paper-grade tables show
± 2σ or ± 3σ. For close head-to-head LERs, apply McNemar — see
`references/decoder-comparison.md`.

Full end-to-end (build DEM + shots, then run this harness) lives in
`references/decoder-comparison.md`. Solver-benchmark version (VQE vs
ADAPT vs GQE) lives in `references/solvers-benchmarks.md` — same idea,
domain-specific setup delegated to `cudaq-solvers-algorithms` /
`cudaq-solvers-chemistry`.

## Self-Check Protocol

```
[ ] Same DEM across compared methods.
[ ] Same noise model across simulator and DEM generator.
[ ] OMP_NUM_THREADS=1 (or noted; chemistry only).
[ ] All decoders / methods warmed up before timing.
[ ] Shot count enough to distinguish at the chosen confidence level.
[ ] Error bars reported and visible in plots.
[ ] Hardware, CUDA, driver versions captured in the run log.
[ ] Random seeds reported.
[ ] Git SHA of the cudaq checkout reported.
[ ] At p=0 every method reports zero LER (sanity check).
```

## When stuck

1. Re-run the **First three actions** and capture the environment.
2. Open `references/<topic>.md` and walk it before designing a sweep.
3. If two methods give the same number to many decimals, suspect
   shared random seeds beyond just the noise stream.
4. If timing is unstable, run with `nsys` (`cudaq-profiling-perf`)
   to see the actual GPU timeline.

## Additional resources

- `references/decoder-comparison.md` — head-to-head LER, including
  statistical tests.
- `references/threshold-sweep.md` — pseudo-threshold, varying
  distance and `p`.
- `references/solvers-benchmarks.md` — VQE / ADAPT / GQE energy
  convergence, time-to-chemical-accuracy, parameter count.
- `references/reproducibility.md` — environment capture, seeding,
  publishable data formats.
- Profiling for tail latency: `cudaq-profiling-perf`.
- Authoring a new method then benchmarking it: `cudaq-qec-extending`, `cudaq-solvers-extending`.
