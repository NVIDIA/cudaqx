# Threshold and pseudo-threshold sweeps

The standard figure in QEC papers: LER vs physical error rate `p`,
plotted for multiple code distances. The crossing point is the
threshold (or pseudo-threshold).

## Authoritative example

`docs/sphinx/examples/qec/python/pseudo_threshold.py`. Read it
before writing your own; it shows the right shot-count scaling and
plot conventions.

## What "threshold" means

- **Threshold** — the physical error rate `p*` below which the LER
  decreases as you grow the code distance. Below threshold,
  scaling helps; above, it hurts.
- **Pseudo-threshold** — the crossing point estimated from a finite
  set of distances. Computed as the `p` where LER curves for two
  adjacent distances cross.

For QEC paper plots, you want LER curves for at least 3 distances
(e.g. d=3, 5, 7) over a range of `p` (e.g. `1e-3` to `1e-1` in
~10 log-spaced points).

## Choosing shot counts

The LER scales with `(p/p*)^d`. Near threshold, both numerator and
denominator scale similarly; far from threshold, you need many shots
to see *any* logical errors.

Rule of thumb:

```
shots_needed = 100 / expected_LER
```

(100 errors gives ~10% relative uncertainty.) For a distance-7 code
at `p = 0.5 * p*`, expected LER ~ `(0.5)^7 ≈ 0.008`, so ~12,500
shots. Per data point. Per distance.

This scales fast. Use:

- Importance sampling (Stim's `compile_detector_sampler` doesn't
  directly support this; some research codes add it).
- Adaptive shot count: keep sampling until you've seen ~100 errors
  or hit a max budget.

## Code skeleton

```python
import numpy as np
import cudaq, cudaq_qec as qec

cudaq.set_target("stim")

distances = [3, 5, 7]
ps = np.logspace(-3, -1, 10)
results = {}

for d in distances:
    code = qec.get_code("surface_code", distance=d)
    lers = []
    for p in ps:
        noise = cudaq.NoiseModel()
        noise.add_all_qubit_channel("x", cudaq.Depolarization2(p), 1)
        dem = qec.z_dem_from_memory_circuit(code, qec.operation.prep0, d, noise)
        H, obs = dem.detector_error_matrix, dem.observables_flips_matrix
        dec = qec.get_decoder("nv-qldpc-decoder", H)

        shots_done, errs = 0, 0
        max_shots = max(500, int(100 / max(1e-6, p ** d)))
        while shots_done < max_shots and errs < 100:
            batch = qec.sample_memory_circuit(code, qec.operation.prep0, d, noise,
                                              nShots=min(1000, max_shots - shots_done))
            for s in batch:
                r = dec.decode(s.syndrome)
                pred = (r.result > 0.5).astype(np.uint8)
                if (pred @ obs.T % 2 != s.observable).any():
                    errs += 1
                shots_done += 1

        ler = errs / shots_done
        se = np.sqrt(ler * (1 - ler) / shots_done)
        lers.append((p, ler, se))
    results[d] = lers
```

## Plotting conventions

For paper-quality plots:

- Log-log axes.
- Distinct color *and* marker per distance (color-blind safe).
- Visible error bars (1σ minimum; 2σ preferred).
- Highlight the pseudo-threshold region (vertical band around the
  crossing).
- Label slopes (`LER ∝ p^d`) on each curve as a sanity check.

Matplotlib example:

```python
import matplotlib.pyplot as plt
for d, lers in results.items():
    p, ler, se = zip(*lers)
    plt.errorbar(p, ler, yerr=se, marker='o', label=f"d={d}")
plt.xscale("log"); plt.yscale("log")
plt.xlabel("Physical error rate p")
plt.ylabel("Logical error rate")
plt.legend()
plt.savefig("threshold.png", dpi=150)
```

## Pseudo-threshold extraction

Fit each curve to `LER(p) = a * p^d` in a low-`p` regime; find
crossings analytically. For a more robust estimate, fit all curves
jointly with a finite-size scaling form (see Bombín et al. 2023 or
similar):

```
LER(p, d) = A * (p / p*)^(d * x)
```

Use `scipy.optimize.curve_fit` with sensible parameter bounds.

## Common pitfalls

| Symptom | Cause |
|---------|-------|
| LER curves don't cross | shot count too low; can't resolve LER below ~`1/shots` |
| Threshold estimate wildly different from literature | mismatched noise model — paper uses a different `cudaq.Depolarization` variant |
| LER curves "flatten" at high `p` | already past threshold; scaling stops being polynomial |
| Big systematic offset vs paper | DEM circuit differs from theirs (different stabilizer measurement schedule, different boundary conditions) |
| Wall-clock blows up for big `d` | predictable; budget accordingly. For real-time pipelines, drop to `nv-qldpc-decoder` + sliding window |

## Self-check

```
[ ] At least 3 distances plotted.
[ ] At least 100 logical errors at every data point you publish.
[ ] Error bars visible.
[ ] LER ∝ p^d slope visually consistent with the chosen d in the low-p regime.
[ ] Threshold estimate matches expectations (e.g., ~0.7% for surface code under standard depolarizing).
[ ] Noise model, code, decoder, shot count all reported in the caption.
```

## Where next

- Decoder-vs-decoder threshold deltas: `decoder-comparison.md`.
- Reproducibility record: `reproducibility.md`.
- Real-time deployment of the chosen decoder: `cuda-qx-qec-realtime`.
