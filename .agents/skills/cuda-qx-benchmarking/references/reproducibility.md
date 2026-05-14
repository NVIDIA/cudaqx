# Reproducibility record

What to capture so someone else (or you, six months from now) can
reproduce your benchmark exactly.

## The minimum record

Capture these in every benchmark script's output. Don't editorialize
— just dump them.

```python
import subprocess, sys, os, platform

print("=== ENVIRONMENT ===")
print(f"Date:    {__import__('datetime').datetime.now().isoformat()}")
print(f"Host:    {platform.node()}")
print(f"OS:      {platform.platform()}")
print(f"Python:  {sys.version}")
print(f"PWD:     {os.getcwd()}")

# Versions
import cudaq, cudaq_qec, cudaq_solvers
print(f"cudaq:         {cudaq.__version__}")
print(f"cudaq_qec:     {cudaq_qec.__version__}")
print(f"cudaq_solvers: {cudaq_solvers.__version__}")

# Pip list (cudaq + numpy + scipy + torch + tensorrt)
out = subprocess.run(["pip", "list"], capture_output=True, text=True).stdout
for line in out.splitlines():
    if any(k in line.lower() for k in ["cuda-quantum", "cudaq", "numpy", "scipy", "torch", "tensorrt"]):
        print(line)

# Git SHA
sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
print(f"Git SHA:       {sha}")

# nvidia-smi
try:
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout
    print(f"GPU:           {smi.strip()}")
except FileNotFoundError:
    print("GPU:           (no nvidia-smi found, CPU-only run)")
```

That block is enough to reconstruct what ran where. Append it to
the start of every benchmark log.

## Seeding

Pin every source of randomness:

```python
import random, numpy as np, os
random.seed(SEED); np.random.seed(SEED)
os.environ["OMP_NUM_THREADS"] = "1"        # chemistry only; QEC ignores

# PyTorch
try:
    import torch
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True, warn_only=True)
except ImportError:
    pass

# Stim
# stim.Circuit.compile_detector_sampler(seed=SEED)
```

Run *every* benchmark at multiple seeds (`SEED in [0, 1, 2, 3, 4]`)
and report mean + std. A single-seed benchmark is a data point, not
a measurement.

## Publishable data formats

Save raw data alongside derived plots. `.npz` for arrays, `.json` for
metadata:

```python
import numpy as np, json
np.savez("ler_curve.npz", ps=ps, ler=ler, se=se)
with open("ler_curve.json", "w") as f:
    json.dump({
        "git_sha": sha,
        "host": platform.node(),
        "gpu": gpu_string,
        "shots_per_point": shots,
        "code": "surface_code",
        "distance": d,
        "decoder": "nv-qldpc-decoder",
        "decoder_params": params,
    }, f, indent=2)
```

When you share the figure, share both files. Reviewers can re-plot
without re-running.

## What to include in a paper supplement

| Item | Format |
|------|--------|
| Driver script that produced the data | `.py` |
| Environment record (the block above) | `.txt` |
| Raw numbers | `.npz` or `.csv` |
| Run-time logs | `.txt` |
| Container or pip-freeze | `.txt` or `Dockerfile` |
| Git diff / patch if any local edits | `.diff` |

If you ran inside the published `ghcr.io/nvidia/cudaqx` container,
note the exact image SHA (`docker inspect <image> -f '{{.Id}}'`).

## Citing CUDA-QX in a paper

The repo ships a `CITATION.cff` file at the top level. Use it as the
canonical citation source — both for a `BibTeX` export (most
reference managers parse `.cff` directly) and as a `cite-as` link
from preprints. Mention the git SHA of the checkout alongside the
formal citation so readers can reproduce against the exact code.

## What *not* to include

- Trace files larger than a few MB (link to Zenodo / figshare).
- Closed-source plugin internals (the `nv-qldpc-decoder` plugin
  itself is closed; cite it, don't redistribute it).
- Anything that would let a reader infer hardware / API credentials.

## Hardware variability

Even on identical GPUs and drivers, FP16/FP32 ordering on parallel
reductions can flip in the last digit. If your benchmark hinges on
1e-6-level energy differences, run several seeds *per* machine and
report machine variance separately from seed variance.

For real-time / latency benchmarks, run on a quiet system (no other
GPU jobs, fixed clock speeds via `nvidia-smi -lgc <freq>`). Otherwise
tail-latency p99 is noisy.

## Long-term archival

For a paper, you want a frozen artifact. Options:

1. Tag a release on a fork of cudaqx: `git tag v0.1-paper`.
2. Build the wheel from that tag and upload to Zenodo (gets a DOI).
3. Cite the DOI in the paper's "data availability" section.

For internal NVIDIA work, attach the wheel + Dockerfile to the
internal release artifact and reference it.

## Self-check

```
[ ] Environment block printed at the top of every benchmark run.
[ ] Seeds set for all sources of randomness.
[ ] Multi-seed runs and seed-variance reported.
[ ] Raw .npz / .json alongside any figure.
[ ] Git SHA captured.
[ ] Hardware (GPU, driver) captured.
[ ] For paper: DOI'd artifact or pinned Docker image archived.
```

## Where next

- Final decoder comparison plot: `decoder-comparison.md`.
- Threshold curve: `threshold-sweep.md`.
- Solvers benchmark table: `solvers-benchmarks.md`.
