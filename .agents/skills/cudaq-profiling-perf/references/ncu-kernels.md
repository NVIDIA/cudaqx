# Nsight Compute for kernel deep-dives

`ncu` is the per-kernel microscope. Use it after `nsys` has
identified the slow kernel; `ncu` tells you *why* it's slow
(memory-bound? compute-bound? bad occupancy?).

## Capture a single kernel

```bash
ncu \
  --set full \
  --kernel-id ::regex:my_decoder.*:1 \
  --launch-skip 100 \
  --launch-count 1 \
  --export my_kernel \
  python my_script.py
```

- `--set full` collects all metrics (slow capture, lots of data).
  Use `--set basic` for a quick look.
- `--kernel-id` filters which kernel to profile. The `regex` form
  matches by name; the third field is the launch invocation count.
- `--launch-skip 100` skips the warmup launches.

Open `my_kernel.ncu-rep` in Nsight Compute GUI.

## What to look at first

In Nsight Compute, three sections are usually enough:

1. **GPU Speed of Light** — overall achieved compute vs memory
   throughput. Two bars:
   - High SM bar, low DRAM → compute-bound. Look at warp stall
     reasons.
   - Low SM, high DRAM → memory-bound. Look at L1/L2/DRAM traffic.
   - Both low → kernel is launch- or sync-bound; revisit with
     `nsys`.

2. **Warp State Statistics** — why warps are stalled. Top entries:
   - "Wait" → divergent or serialized; profile the diverging
     branch.
   - "Long Scoreboard" → memory latency; consider prefetching or
     coalescing.
   - "Short Scoreboard" → tex/shared latency; check shared-memory
     bank conflicts.

3. **Memory Workload Analysis** — L1 hit rate, L2 hit rate, DRAM
   throughput. Hit rate ≪ 1 usually means access pattern is
   non-coalesced or random.

## Rules of thumb for QEC / solvers kernels

- **Decoder inference kernels** are usually memory-bound. Aim for
  coalesced reads of the syndrome and the parity matrix.
- **Stabilizer-extraction kernels** (Stim) are tiny and not usually
  the bottleneck. Don't waste time here.
- **TensorRT inference** is opaque (TRT generates the kernel). Trust
  TRT, but verify the engine was built with FP16/INT8 if your kernel
  is compute-bound.

## Targeted modes

| `--set` | Use for |
|---------|---------|
| `basic` | quick "is this kernel compute or memory bound?" |
| `roofline` | roofline analysis vs theoretical peak |
| `memory` | memory hierarchy deep-dive |
| `full` | everything; slowest, most data |

## Common pitfalls

| Symptom | Cause |
|---------|-------|
| `ncu` reports "no kernel matched" | regex wrong; check with `nsys` first to learn the kernel name |
| Single kernel takes much longer under `ncu` | `--set full` adds substantial overhead — by design |
| Metrics inconsistent across runs | not pinning clocks; `sudo nvidia-smi -lgc <freq>` |
| Permission denied opening counters | needs `sudo` or kernel module config; see NVIDIA docs |

## Self-check

```
[ ] Kernel chosen based on nsys-identified hot spot, not a guess.
[ ] At least 1 representative launch profiled (after warmup).
[ ] Bottleneck identified: compute / memory / launch.
[ ] Optimization hypothesis has a specific metric attached.
[ ] After optimization, re-profile and confirm the metric improved.
```

## Where next

- Apply changes and re-profile with `nsys` to see whole-pipeline
  effect: `nsys-realtime.md`.
- For PyTorch / training: `pytorch-profiler.md`.
