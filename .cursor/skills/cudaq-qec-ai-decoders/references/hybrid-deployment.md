# Deploying a trained engine into the hybrid AI predecoder pipeline

The production-realtime path for surface code decoding: your trained
TensorRT engine runs as a predecoder on GPU; PyMatching MWPM mops up
residuals on CPU. Distance / round combinations supported out of the
box: d=7/T=7, d=13/T=13, d=13/T=104, d=21/T=21, d=31/T=31.

## Authoritative reference

`docs/hybrid_ai_predecoder_pipeline.md`. The design doc covers ring
buffer protocol, dispatcher, ServiceBase classes, CPU worker thread
pool, memory ordering, and backpressure. Don't reinvent any of that.

Headers:

- `libs/qec/include/cudaq/qec/realtime/ai_predecoder_service.h`
- `libs/qec/include/cudaq/qec/realtime/pipeline.h`

RST examples:

- `docs/sphinx/examples_rst/qec/realtime_predecoder_pymatching.rst`
- `docs/sphinx/examples_rst/qec/realtime_predecoder_fpga.rst`

## What the predecoder must output

For the pipeline to consume it, the predecoder model must produce:

1. **Residual detectors** — the syndrome bits that the predecoder
   could *not* trivially explain. Type `uint8`, shape
   `[num_detectors]`. PyMatching consumes these.
2. **Logical frame prediction** — predecoder's guess at the
   observable. Type `uint8`, shape `[num_observables]`. Used as a
   soft prior by PyMatching.

If your trained model only outputs a logical prediction (no residual
detectors), you must wrap it in a model that *also* computes the
residual. See the design doc §4.4 for the standard wrapping pattern.

## Engine compatibility checks

Before deployment:

1. Engine input shape must match the d/T from `PipelineConfig`.
2. Engine output shape must match `(num_detectors,
   num_observables)`.
3. Engine must be FP16 or FP32 (INT8 is not validated for this
   pipeline).

Failures here surface as either "engine input dimensions don't
match" at load time or random predictions at runtime.

## Configuration

```cpp
PipelineConfig cfg;
cfg.d = 13;
cfg.T = 104;
cfg.engine_path = "predecoder_d13_T104.engine";
cfg.num_workers = 8;
cfg.ring_buffer_slots = 32;
cfg.mwpm_thread_pool_size = 8;
cfg.cpu_queue_capacity = 128;
cfg.enable_cpu_handoff = true;     // false → AIDecoderService only
```

Full struct: `pipeline.h`. Defaults are sensible; the most-tuned
parameter is `num_workers` (one CUDA Graph worker per concurrent
syndrome in flight).

## Benchmarking after deployment

Two numbers matter:

1. **End-to-end LER** — measure against the same DEM as your offline
   training; the pipeline should match offline `trt_decoder` LER within
   a few percent.
2. **Throughput** (syndromes/sec) and **tail latency** (p99). The
   design doc reports 98.3% syndrome density reduction at
   d=13/T=104, which translates to specific throughput numbers on
   reference hardware (see §13 of the design doc).

Use `nsys` (`cudaq-profiling-perf`) to capture per-stage timings.
The TensorRT range should be the dominant GPU-side cost; if anything
else dominates, you have a host fence somewhere.

## Common pitfalls

| Symptom | Likely cause |
|---------|--------------|
| Engine loads, but pipeline thinks it has the wrong d/T | engine was built from an ONNX with mismatched detector count |
| MWPM queue grows unboundedly | predecoder not reducing density enough; check residual detector fraction |
| LER higher than offline `trt_decoder` | model not trained with residual output head; see "What the predecoder must output" |
| Throughput much below benchmarks | `num_workers` too low or `ring_buffer_slots` < 4*num_workers |

## Self-check

```
[ ] Engine input/output shapes match the PipelineConfig d/T.
[ ] Engine output includes BOTH residual detectors and logical frame.
[ ] Pipeline runs at p=0: zero residual detectors, all-zero logical frame.
[ ] Throughput benchmark within the design-doc envelope.
[ ] Tail latency p99 within the latency budget.
[ ] LER matches offline trt_decoder on the same DEM.
```

## Where next

- For non-surface-code decoders, autonomous_decoder is the more
  flexible path: `cudaq-qec-realtime/references/autonomous-decoder.md`.
- For benchmarking many decoders side-by-side: `cudaq-benchmarking`.
- For profiling tail latency: `cudaq-profiling-perf`.
