# Nsight Systems for real-time decoding

`nsys` is the right tool for a real-time / multi-stream pipeline: it
shows you the whole timeline (CPU thread, GPU streams, NVTX ranges,
memory transfers) at once. Use it when "the decoder feels slow" is
the only hypothesis.

## Capture

```bash
nsys profile \
  --trace=cuda,nvtx,osrt,cudnn,cublas \
  --gpu-metrics-device=0 \
  --output=realtime_run \
  --force-overwrite=true \
  python my_realtime_script.py
```

For a fixed-duration capture (skip warmup, capture 5 s of steady
state):

```bash
nsys profile \
  --trace=cuda,nvtx,osrt \
  --delay=10 --duration=5 \
  --output=realtime_run \
  python my_realtime_script.py
```

For a C++ executable, replace `python my_script.py` with your binary.

## What to look for

When the report opens, scan in this order:

1. **CUDA HW** row — are GPU streams busy continuously, or with
   gaps?
2. **NVTX** row — where do the ranges from `nvtx_helpers.h`
   sit? Do they overlap correctly across streams?
3. **CPU** row — is the host blocking on a stream or memcpy?
4. **Memory operations** — are there many `cudaMemcpy` calls? Should
   they be async?

## Common pathologies

| Pattern in timeline | Diagnosis |
|---------------------|-----------|
| GPU idle, CPU active in long ranges | host fence / synchronous memcpy / Python-side bottleneck |
| Many tiny kernel launches | launch overhead dominates; combine via CUDA Graph |
| Long `cudaMallocAsync` ranges | allocator thrashing; preallocate |
| Memcpy H2D right before each kernel | not using pinned memory; or unnecessary copy |
| CPU thread idle, GPU active | wait/sync from CPU; consider overlapping |
| Long single-kernel runs | the kernel itself is slow → use `ncu` (see `ncu-kernels.md`) |

## Profiling the autonomous_decoder pipeline

The autonomous_decoder dispatch system already emits NVTX ranges via
`nvtx_helpers.h`. In Nsight you should see, per syndrome:

- A predecoder kernel range.
- A dispatch / RPC range.
- A correction-writeback range.
- (For the hybrid pipeline) a CPU MWPM range on the host thread.

If any of those is missing, the source you're profiling lacks the
NVTX instrumentation — see `nvtx-instrumentation.md` to add ranges.

## Profiling configure_decoders_from_file

The Phase 3 step in real-time decoding (`cudaq-qec-realtime/references/in-kernel.md`)
runs once per session. If it shows up as a hot spot, you're calling
it on every shot — that's a bug. It should run before `cudaq.run`,
not inside it.

## SQLite export for headless analysis

```bash
nsys export --type=sqlite realtime_run.nsys-rep --output=realtime_run.sqlite

# Top kernels by total time
sqlite3 realtime_run.sqlite \
  "SELECT name, SUM(end - start) AS total_ns
   FROM cupti_activity_kind_kernel
   GROUP BY name ORDER BY total_ns DESC LIMIT 10;"
```

Useful when GUI is not available or you want to script comparisons
across runs.

## Capturing GPU metrics

`--gpu-metrics-device=0` enables on-device counters (SM occupancy,
DRAM throughput, etc.). Only useful on supported GPUs; check
`nsys --help` for `gpu-metrics-set` options.

For deeper per-kernel metrics, use `ncu` (`ncu-kernels.md`).

## Common pitfalls

| Symptom | Cause |
|---------|-------|
| Empty NVTX timeline | source not instrumented; add ranges |
| Capture file huge (>1 GB) | trace too broad; drop `osrt` if not needed |
| `nsys` reports "no CUDA activity" | profiling target didn't actually use CUDA, or wrong process |
| Different timeline shape across runs | system noise; lock GPU clocks |

## Self-check

```
[ ] nsys-rep file opens without errors.
[ ] NVTX ranges visible (if expected).
[ ] GPU streams visible and labeled.
[ ] Identified top-3 longest kernels or ranges.
[ ] After a code change, captured a second run and confirmed the win.
```

## Where next

- Deep-dive a single kernel: `ncu-kernels.md`.
- Add NVTX yourself: `nvtx-instrumentation.md`.
- Compare two implementations methodically: `cudaq-benchmarking/references/decoder-comparison.md`.
