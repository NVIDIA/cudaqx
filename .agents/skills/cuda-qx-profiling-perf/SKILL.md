---
name: "cuda-qx-profiling-perf"
title: "CUDA-QX Profiling and Performance"
description: >-
  Profile and optimize CUDA-QX code: NVTX ranges for the realtime QEC
  stack, `nsys` and `ncu` for GPU timelines and kernel deep-dives,
  PyTorch profiler for ML decoder training, CUDA Graphs for latency-
  bounded paths, memory bandwidth analysis, and how to read a flame
  graph. Use whenever the user mentions profiling, nsys, ncu, NVTX,
  PyTorch profiler, slow decoder, high latency, memory pressure, GPU
  utilization, CUDA Graph, or "why is my code so slow".
version: "0.1.1"
author: "CUDA-QX"
license: "Apache License 2.0"
compatibility: "Linux x86_64/aarch64 with NVIDIA GPU; NVIDIA Nsight Systems/Compute"
tags: [cuda-qx, profiling, nsys, ncu, nvtx, cuda-graph, performance, latency, throughput]
tools: [Read, Glob, Grep, Bash]
metadata:
  repo: [qec, solvers]
  author: "CUDA-QX"
  domain: "performance"
  audience: [developer, performance-engineer, researcher]
  languages: [python, c++, cuda]
---

# CUDA-QX Profiling and Performance

The "my decoder is too slow" / "VQE is bottlenecked" / "this real-time
pipeline misses its deadline" skill. Covers GPU profiling
(`nsys`/`ncu`), NVTX instrumentation already in the codebase, CUDA
Graph use, and Python-level profiling for the training side.

## Inputs

Caller provides:

- A runnable binary or Python script that exhibits the slowness.
- The perceived bottleneck (or "I don't know").
- A reproduction procedure (or one will be derived).
- Build with debug symbols available (`-DCMAKE_BUILD_TYPE=RelWithDebInfo`)
  for readable kernel names.

## Outputs

This skill produces:

- An identified hot-path with a metric (kernel name + time, or
  CPU-bound vs memory-bound classification).
- A specific fix hypothesis with a target metric to verify.
- A before/after `nsys` or `ncu` report or PyTorch-profiler trace.
- NVTX range additions if the binary lacks instrumentation.

Does NOT produce: algorithm-level changes (→ domain skills); paper-
grade comparisons (→ `cuda-qx-benchmarking`).

## Audience

Performance-minded developers and researchers. CUDA and `nsys`
familiarity helps; no prior NVTX experience required — this skill
covers it.

## First three actions

```bash
bash   .agents/skills/_shared/scripts/preflight.sh    --json > /tmp/preflight.json
nvidia-smi --query-gpu=name,driver_version,utilization.gpu,memory.used --format=csv
which nsys ncu
```

If `nsys` is missing, install NVIDIA Nsight Systems:
<https://developer.nvidia.com/nsight-systems>. `ncu` (Nsight Compute)
is needed only for per-kernel deep-dives.

## Key Paths

| Area | Path |
|------|------|
| Realtime NVTX helpers | `libs/qec/include/cudaq/qec/realtime/nvtx_helpers.h` |
| Autonomous decoder hot path | `libs/qec/include/cudaq/qec/realtime/autonomous_decoder.cuh` |
| AI predecoder hot path | `libs/qec/include/cudaq/qec/realtime/ai_predecoder_service.h` |
| GPU kernels | `libs/qec/include/cudaq/qec/realtime/gpu_kernels.cuh` |
| Build scripts (Release / RelWithDebInfo) | `CMakeLists.txt` → `CMAKE_BUILD_TYPE` |

## Workflow Index

| If the user wants to | Read |
|----------------------|------|
| Profile a real-time decoder pipeline with nsys | `references/nsys-realtime.md` |
| Deep-dive a slow GPU kernel with ncu | `references/ncu-kernels.md` |
| Profile a Python / PyTorch training loop | `references/pytorch-profiler.md` |
| Add NVTX ranges to your own code | `references/nvtx-instrumentation.md` |

## Conventions

These are the "I profiled but learned nothing" mistakes.

1. **Build with debug symbols** (`-DCMAKE_BUILD_TYPE=RelWithDebInfo`).
   Otherwise `nsys` reports inscrutable mangled kernel names.

2. **Warm up before profiling.** First call always includes JIT,
   plugin loading, allocator init. Profile call #10, not call #1.

3. **Sample for at least 100 iterations.** Single-shot profiles are
   pure noise. Use a loop and let `nsys` aggregate.

4. **Pin clocks for latency-sensitive measurements.**
   `sudo nvidia-smi -lgc <freq>` locks the GPU at a fixed clock. The
   default boost behavior makes tail latencies impossible to
   interpret.

5. **NVTX in hot path only**, never in setup. Setup ranges dominate
   the timeline and hide what you're actually measuring.

6. **`cudaDeviceSynchronize` before timing in Python.** Without it,
   `time.perf_counter` measures the host enqueue cost, not the GPU
   work.

7. **Compare like-with-like.** Same shot count, same input sizes,
   same hardware, same driver. Profiling A on H100 against B on RTX
   is meaningless.

## Quick start: nsys profile of a Python script

```bash
nsys profile \
  --trace=cuda,nvtx,osrt \
  --output=my_run \
  --force-overwrite=true \
  python my_benchmark.py
```

Open `my_run.nsys-rep` in Nsight Systems GUI. The timeline shows
CPU, GPU streams, NVTX ranges, and memory transfers.

For headless / remote machines, generate a SQLite export for
analysis:

```bash
nsys export --type=sqlite my_run.nsys-rep --output=my_run.sqlite
```

## The "what's slow?" decision tree

```
Is GPU utilization < 50%?
├─ yes -> CPU-bound or memory-bound. Check NVTX timeline for host
│         fences or cudaMemcpy calls. References: nsys-realtime.md.
└─ no  -> GPU compute bound. Pick the longest kernel from the timeline.
          References: ncu-kernels.md.

Is the same kernel called repeatedly with low utilization?
├─ yes -> kernel launch overhead dominates. Use CUDA Graphs.
└─ no  -> ?

Are CPU<->GPU memcpys taking >20% of wall-clock?
├─ yes -> use pinned memory; check that you're using cudaMemcpyAsync.
└─ no  -> ?

Is `pyscf-driver` showing up?
└─ chemistry-side bottleneck; consider running PySCF once and caching
   the molecule object.
```

## Self-Check Protocol

```
[ ] Build is RelWithDebInfo or Release with line tables.
[ ] Warmup loop runs before profiled loop.
[ ] At least 100 iterations profiled.
[ ] nsys timeline opens and shows NVTX ranges (if instrumented).
[ ] Top-3 kernels identified by total time.
[ ] Hypothesis about the bottleneck has a *number* attached.
[ ] After optimization, re-profile and confirm the win.
```

## When stuck

1. Re-run the **First three actions** to confirm tools are installed.
2. Read the matching `references/<topic>.md`.
3. Run the existing realtime app examples
   (`libs/qec/unittests/realtime/app_examples/`) under `nsys` to see
   what a healthy profile looks like.
4. For Python-only bottlenecks, switch to `cProfile` /
   `py-spy` before reaching for `nsys`.

## Additional resources

- `references/nsys-realtime.md` — full nsys workflow for real-time
  decoding.
- `references/ncu-kernels.md` — Nsight Compute for kernel deep-dives.
- `references/pytorch-profiler.md` — PyTorch profiler for ML decoder
  training.
- `references/nvtx-instrumentation.md` — add ranges to your own code.
- For latency benchmarking methodology: `cuda-qx-benchmarking`.
- For the realtime path being profiled: `cuda-qx-qec-realtime`.
