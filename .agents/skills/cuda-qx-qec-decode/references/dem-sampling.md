# DEM sampling (`dem_sampling`)

Sample syndromes directly from a parity-check matrix + error
probability vector — no kernel run required. Faster than running the
stabilizer-extraction circuit when you only need shots from a *fixed*
DEM (e.g. for evaluating multiple decoders against identical
syndromes).

## API

```python
import cudaq_qec as qec

syndromes, errors = qec.dem_sampling(
    check_matrix,              # H (numpy uint8, C-order) OR torch CUDA tensor
    num_shots,                 # int
    error_probabilities,       # 1D numpy/torch, length = H.shape[1]
    seed=None,                 # optional int for reproducibility
    backend="auto",            # "auto" | "gpu" | "cpu"
)
```

Source of truth: `libs/qec/python/cudaq_qec/dem_sampling.py` (module
docstring + signature).

Returns `(syndromes, errors)` where:

- `syndromes` shape `(num_shots, H.shape[0])`, dtype `uint8`.
- `errors` shape `(num_shots, H.shape[1])`, dtype `uint8`. The
  per-shot error vectors that produced each syndrome.

## When to use this vs `sample_memory_circuit`

| You want | Use |
|---|---|
| Shots from the DEM of a stabilizer-extraction circuit | `sample_memory_circuit` (runs the kernel) |
| Shots from any arbitrary `H`/`p` (no circuit) | `dem_sampling` |
| Same shots evaluated against multiple decoders | `dem_sampling` (faster, deterministic with `seed=`) |
| Investigating a DEM's structure or a custom noise vector | `dem_sampling` |

For benchmarking decoders on the same DEM (the standard
"head-to-head" comparison pattern in `cuda-qx-benchmarking`),
`dem_sampling` is the right primitive.

## Backend selection

| `backend=` | What runs |
|---|---|
| `"auto"` (default) | GPU via cuStabilizer if available, else CPU |
| `"gpu"` | GPU only; raises `RuntimeError` if cuStabilizer not installed or no CUDA |
| `"cpu"` | CPU fallback |

`backend="auto"` does *not* silently fall back when `"gpu"` was
explicitly requested and unavailable — that's a hard error. Use
`"auto"` when you don't care; `"gpu"` when you want a noisy failure if
the GPU path isn't actually being used.

## Input acceptance

- **NumPy arrays**: the primary path. `check_matrix` must be C-order
  `uint8`; F-order arrays raise.
- **PyTorch CUDA tensors**: accepted when `torch` is installed by the
  user (`pip install torch`). The cudaqx wheel does not bundle torch.
- **PyTorch CPU tensors**: NOT accepted. Convert to NumPy first:
  `check_matrix.cpu().numpy()`.

## Quick start

```python
import numpy as np
import cudaq_qec as qec

# Toy 3-qubit repetition code parity matrix
H = np.array([[1, 1, 0],
              [0, 1, 1]], dtype=np.uint8)
p = np.array([0.01, 0.01, 0.01])

syndromes, errors = qec.dem_sampling(H, num_shots=10_000, error_probabilities=p, seed=0)
assert syndromes.shape == (10_000, 2)
assert errors.shape == (10_000, 3)
print(f"Fraction of nonzero syndromes: {(syndromes.any(axis=1)).mean():.3g}")
```

## Common pitfalls

| Symptom | Cause |
|---|---|
| `RuntimeError: H must be C-contiguous` | F-order numpy; fix with `np.ascontiguousarray(H)` |
| `RuntimeError: backend gpu requested but cuStabilizer not available` | `backend="gpu"` on a system without cuStabilizer — use `"auto"` or install cuStabilizer |
| `TypeError` on a torch tensor | CPU tensor passed; convert to NumPy first |
| Different syndromes across runs at same seed | torch RNG vs numpy RNG; pin `torch.manual_seed` too if your input is a CUDA tensor |
| Slow GPU path on first call | one-time cuStabilizer init; subsequent calls are fast |

## Self-check

```
[ ] check_matrix is numpy uint8 C-order (or a CUDA torch tensor)
[ ] error_probabilities length == check_matrix.shape[1]
[ ] seed set for reproducibility (when comparing decoders)
[ ] At error_probabilities = zeros, output syndromes are all-zero
[ ] backend="gpu" works (or "auto" falls back without error)
```

## Where next

- Compare decoders on the same shots:
  `cuda-qx-benchmarking/references/decoder-comparison.md`.
- Build the DEM from a memory circuit (instead of hand-rolling H):
  `references/decode.md` "Circuit-Level Memory Experiment".
- Wrap the inner decoder in a sliding window for streaming:
  `cuda-qx-qec-realtime/references/sliding-window.md`.
