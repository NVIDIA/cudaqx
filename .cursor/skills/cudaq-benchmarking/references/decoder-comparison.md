# Decoder comparison

Comparing two (or more) decoders fairly on the same DEM, with
honest error bars. The pattern below is the one to extend.

## Protocol

1. **Build one DEM.** All decoders consume the same `H` and
   `observables_flips_matrix`.
2. **Sample shots once.** All decoders see the same syndromes.
3. **Time only the `decode` calls.** Construction is one-time; don't
   count it.
4. **Report LER ± SE.** Binomial standard error from N shots: `sqrt(p(1-p)/N)`.
5. **Statistical test only when the methods are close.** Use
   McNemar's test for paired binary outcomes.

## Sampling shots once

```python
shots = qec.sample_memory_circuit(code, op, num_rounds, noise, nShots=N)
# shots is a list/array of (syndrome, observable) pairs
```

`sample_memory_circuit` returns the same noise instance for every
caller in the same run. Save the array if you need to compare across
sessions:

```python
import numpy as np
np.savez("shots.npz",
         syndromes=np.array([s.syndrome for s in shots]),
         observables=np.array([s.observable for s in shots]))
```

Reload in subsequent runs to compare methods that needed different
environments.

## Comparing two decoders on the same shots

```python
import numpy as np

def evaluate(dec, shots, obs):
    errs = 0
    for s in shots:
        r = dec.decode(s.syndrome)
        if not r.converged:
            errs += 1                   # treat as logical error (conservative)
            continue
        pred = (r.result > 0.5).astype(np.uint8)
        if (pred @ obs.T % 2 != s.observable).any():
            errs += 1
    return errs

n = len(shots)
e_a = evaluate(dec_a, shots, obs)
e_b = evaluate(dec_b, shots, obs)

p_a, p_b = e_a/n, e_b/n
se_a = np.sqrt(p_a*(1-p_a)/n)
se_b = np.sqrt(p_b*(1-p_b)/n)
print(f"A: {p_a:.4g} ± {se_a:.4g}")
print(f"B: {p_b:.4g} ± {se_b:.4g}")
```

## When LERs are close: McNemar's test

Paired test on the same shots. Build a 2x2 table:

```
                     B correct   B wrong
A correct              n_cc        n_cw
A wrong                n_wc        n_ww
```

McNemar statistic:

```python
from scipy.stats import binomtest
# n_cw = A correct, B wrong;  n_wc = A wrong, B correct
n_cw = ...  # count from your shots
n_wc = ...
res = binomtest(n_cw, n_cw + n_wc, p=0.5, alternative="two-sided")
print(f"McNemar p-value: {res.pvalue:.4g}")
```

`p < 0.05` (or whatever threshold your field uses) means the
difference is unlikely to be sampling noise.

## Timing decode calls

Warm up first (CUDA kernels JIT, plugin load, allocations):

```python
import time

# Warmup
for s in shots[:10]:
    dec.decode(s.syndrome)

# Time
t0 = time.perf_counter()
for s in shots[:1000]:
    dec.decode(s.syndrome)
t1 = time.perf_counter()
print(f"Per-shot latency: {1e6*(t1-t0)/1000:.2f} us")
```

For GPU decoders, use `cudaDeviceSynchronize()` (via PyTorch:
`torch.cuda.synchronize()`) or `nsys` (`cudaq-profiling-perf`).
Wall-clock without synchronization undercounts GPU work.

## Reporting

Minimum table contents for a paper:

| Decoder | LER (± SE, 1σ) | Median latency | p99 latency | Shots |
|---------|----------------|----------------|-------------|-------|
| A | ... | ... | ... | 10,000 |
| B | ... | ... | ... | 10,000 |

Plus, in the caption or methods section:

- Code, distance, num_rounds.
- Noise model and `p`.
- Hardware (GPU model, driver, CUDA version).
- cudaq git SHA.
- Decoder kwargs.

## Common pitfalls

| Symptom | Cause |
|---------|-------|
| LERs are identical to many decimals | shared random seed beyond noise stream; verify by perturbing `p` and re-running |
| Decoder A wins on one DEM, loses on another | result depends on the random instance — increase shots or seed multiple instances |
| GPU decoder reports lower latency than CPU but throughput is worse | non-synchronized timing; insert `torch.cuda.synchronize` or use `nsys` |
| Decoder B looks worse but I notice many "not converged" shots | reporting LER without convergence rate; cite both |

## Self-check

```
[ ] Same DEM and same shots across compared decoders.
[ ] Binomial SE reported.
[ ] McNemar (or equivalent) for close comparisons.
[ ] Latency is post-warmup; GPU work synchronized.
[ ] Convergence rate reported alongside LER for iterative decoders.
[ ] Plot has visible error bars.
```

## Where next

- Sweep distance / `p`: `threshold-sweep.md`.
- Solvers comparisons (energy, time-to-solution):
  `solvers-benchmarks.md`.
- Reproducibility record: `reproducibility.md`.
