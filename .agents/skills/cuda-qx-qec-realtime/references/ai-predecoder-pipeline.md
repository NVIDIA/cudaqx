# Hybrid AI predecoder + PyMatching pipeline

A realtime hybrid GPU/CPU pipeline: a TensorRT neural network
"predecodes" most syndromes on GPU; residual hard cases are sent to a
PyMatching MWPM thread pool on CPU. Surface code distances d=7, 13,
21, 31 with round counts T matching d (and a special d=13/T=104
configuration).

## Authoritative reference

`docs/hybrid_ai_predecoder_pipeline.md` is the design document.
Read it end-to-end before changing this pipeline; it covers ring
buffer protocol, host dispatcher logic, AIDecoderService base class,
AIPredecoderService extension, decoupled CPU worker architecture,
CUDA Graph structure, memory ordering, and backpressure.

Source headers:

- `libs/qec/include/cudaq/qec/realtime/ai_decoder_service.h` —
  base class for AI decoder services.
- `libs/qec/include/cudaq/qec/realtime/ai_predecoder_service.h` —
  predecoder + CPU handoff implementation.
- `libs/qec/include/cudaq/qec/realtime/pipeline.h` — pipeline config
  struct.
- `libs/qec/include/cudaq/qec/realtime/graph_resources.h` — CUDA
  graph + worker pool.

RST examples:

- `docs/sphinx/examples_rst/qec/realtime_predecoder_pymatching.rst`
- `docs/sphinx/examples_rst/qec/realtime_predecoder_fpga.rst`

## Pipeline at a glance

```
Test harness or FPGA DMA
        │  syndrome data (uint8 detectors)
        ▼
Ring buffer (mapped pinned memory, 16 slots)
        │
        ▼
Host-side dispatcher (spin polls rx_flags[])
        │  pick free worker from idle_mask
        ▼
Per-worker CUDA Graph
   ├─ TensorRT inference (predecoder)
   ├─ DMA output to host
   └─ Signal ready kernel
        │
        ▼
Decoupled PyMatching thread pool (CPU)
   └─ MWPM on residual detectors
        │
        ▼
Ring buffer (writeback) + ack
```

Key claim from the design doc: the predecoder achieves ~98.3%
syndrome density reduction for d=13/T=104, making MWPM tractable on
CPU under the latency budget.

## Pipeline configurations

`PipelineConfig` struct supports the following defaults:

| `d` | `T` | Use case |
|----|----|---------|
| 7 | 7 | smoke test |
| 13 | 13 | typical short experiment |
| 13 | 104 | long experiment with many rounds |
| 21 | 21 | medium-distance benchmarking |
| 31 | 31 | high-distance benchmarking |

Use a value not listed → write a new `PipelineConfig` entry or read
the pipeline.h file for the full struct.

## Building the TensorRT engine

The predecoder uses a TensorRT engine compiled from an ONNX file:

```bash
python scripts/build_engine_from_onnx.py \
    --onnx my_predecoder.onnx \
    --engine my_predecoder.engine \
    --fp16   # if supported
```

Engines are cached on disk and reloaded on subsequent runs. To train
the underlying neural network, see `cuda-qx-qec-ai-decoders` SKILL.md.

## Configuration parameters (most-used)

| Param | Meaning |
|-------|---------|
| `num_workers` | concurrent CUDA Graph workers; ≥ 4 typical |
| `worker_stream_priority` | high for predecoder, normal for housekeeping |
| `ring_buffer_slots` | 16 by default; bigger for higher-throughput streams |
| `enable_cpu_handoff` | if false, falls back to AIDecoderService (no MWPM stage) |
| `mwpm_thread_pool_size` | one per CPU core; benchmark on target system |
| `cpu_queue_capacity` | lock-free queue size to MWPM workers |

Full table: `docs/hybrid_ai_predecoder_pipeline.md` §12.

## When to use this pipeline

| Situation | Use this | Use something else |
|-----------|---------|--------------------|
| d ≤ 31 surface code, long-stream decoding with strict latency | yes | — |
| Tiny code (Steane, repetition) | no | `multi_error_lut` is faster + simpler |
| Need exact ML decoding | no | `tensor_network_decoder` (offline) |
| QLDPC (non-surface) code | no, currently | `nv-qldpc-decoder`, possibly with sliding window |
| Pure GPU decoder, no CPU fallback | use AIDecoderService base, not the predecoder + MWPM combo | — |

## Self-check

```
[ ] TensorRT engine built from a fresh ONNX file (not stale).
[ ] PipelineConfig matches the trained model's d and T.
[ ] Ring buffer slot count ≥ 4 * num_workers.
[ ] MWPM thread pool size <= physical CPU cores.
[ ] At p=0 the pipeline reports zero residual detectors.
[ ] Throughput benchmark beats the offline baseline by the documented margin.
```

## Common pitfalls

| Symptom | Likely cause |
|---------|--------------|
| Engine loads but predictions look random | ONNX exported from a different surface code distance |
| Hangs after first batch | MWPM thread pool starved; raise `cpu_queue_capacity` |
| Throughput much lower than docs | host fence inside the dispatcher; check for memcpy without `cudaMemcpyAsync` |
| Engine rebuilds on every launch | `.engine` cache path is in a tempdir; pin it |

## When stuck

1. Read `docs/hybrid_ai_predecoder_pipeline.md` §4 ("Component
   Deep-Dive") and §10 ("Pipeline Configurations").
2. Profile with `nsys` (see `cuda-qx-profiling-perf`); the
   predecoder kernel and MWPM stage should each be visible as
   distinct ranges.
3. If retraining the predecoder, delegate to `cuda-qx-qec-ai-decoders`.
4. For FPGA integration, see
   `docs/sphinx/examples_rst/qec/realtime_predecoder_fpga.rst`.
