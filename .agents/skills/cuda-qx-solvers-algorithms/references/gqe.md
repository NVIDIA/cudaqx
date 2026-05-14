# GQE: train a transformer to propose ansatz operators

The heaviest algorithm in the solvers stack: requires `torch`,
`lightning`, `mpi4py`, `transformers`, and is the only solver workflow
that benefits from multiple GPUs. For VQE/ADAPT see `vqe.md`. For QAOA
see `qaoa.md`.

## Install

```bash
pip install cudaq-solvers[gqe]
# Adds: torch>=2.0.0, lightning>=2.0.0, ml_collections, mpi4py>=3.1.0,
#       transformers>=4.30.0
```

The `_shared/scripts/import_smoke.py` test "gqe" feature exists exactly
to flag a missing extra before a long training run starts.

## Signature

```python
solvers.gqe(cost, pool, config=None, **kwargs)
```

When `config` is `None`, kwargs override individual config defaults.
Special kwargs:

- `model`: pre-constructed transformer (skip the default GPT-2-style
  model)
- `optimizer`: pre-constructed `torch.optim` instance (default
  `AdamW(lr=cfg.lr)`)

## Minimum example

```python
from cudaq_solvers.gqe_algorithm.gqe import get_default_config

cfg = get_default_config()
cfg.use_fabric_logging = False
cfg.save_trajectory = False
cfg.verbose = True

min_energy, best_ops = solvers.gqe(cost, op_pool, config=cfg,
                                   max_iters=25, ngates=10)
```

## `get_default_config()` keys

| Key                                       | Default                              |
|-------------------------------------------|--------------------------------------|
| `num_samples`                             | `5` (batch size)                     |
| `max_iters`                               | `100`                                |
| `ngates`                                  | `20`                                 |
| `seed`                                    | `3047`                               |
| `lr`                                      | `5e-7`                               |
| `energy_offset`                           | `0.0`                                |
| `grad_norm_clip`                          | `1.0`                                |
| `temperature`                             | `5.0`                                |
| `del_temperature`                         | `0.05`                               |
| `resid_pdrop`, `embd_pdrop`, `attn_pdrop` | `0.0`                                |
| `small`                                   | `False` (12-layer/12-head; small=6/6)|
| `use_fabric_logging`                      | `False`                              |
| `fabric_logger`                           | `None`                               |
| `save_trajectory`                         | `False`                              |
| `trajectory_file_path`                    | `"gqe_logs/gqe_trajectory.json"`     |
| `verbose`                                 | `False`                              |

`validate_config` rejects non-positive `num_samples`, `max_iters`,
`ngates`, `lr`, `grad_norm_clip`, `temperature`. `del_temperature`
must be non-zero. Dropout values must be in `[0, 1]`.

Schedulers: `DefaultScheduler(start, delta)`,
`CosineScheduler(min, max, frequency)`.

## GPU sanity

GQE will `sys.exit(1)` if PyTorch sees a CUDA device but cannot run
kernels on it (newer GPUs need a PyTorch built for CUDA 12.8+). Catch
this early: run `_shared/scripts/import_smoke.py` against the gqe
feature before kicking off a training run.

## Multi-GPU GQE

```python
cudaq.set_target("nvidia", option="mqpu")
cudaq.mpi.initialize()
# ... run gqe ...
cudaq.mpi.finalize()
```

Run: `PMIX_MCA_gds=hash mpiexec -np N python script.py --mpi`.

The `PMIX_MCA_gds=hash` shell var is required when MPI's PMIx layer
mismatches with the launcher's view of GDS storage; without it
`mpiexec` hangs at startup.

## Determinism

```python
import os, torch
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.manual_seed(seed)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

Even with all of the above, GQE is not bitwise reproducible across
different GPU SMs; pin to a fixed device for cross-run comparison.

## Self-check (GQE-specific)

```
[ ] cudaq-solvers[gqe] installed (import_smoke.py "gqe" feature green).
[ ] PyTorch CUDA matches the GPU SM (cu12.8+ for H100/B100/etc).
[ ] If multi-GPU: PMIX_MCA_gds=hash set in the env before mpiexec.
[ ] If determinism matters: env vars + torch flags above are set.
[ ] Energy is finite and decreasing across iterations.
```

## Canonical test

| Topic | File                                          |
|-------|-----------------------------------------------|
| GQE   | `libs/solvers/python/tests/test_gqe.py`       |
