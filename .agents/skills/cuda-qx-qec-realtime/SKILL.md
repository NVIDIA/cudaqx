---
name: "cuda-qx-qec-realtime"
title: "CUDA-QX QEC Real-Time Decoding (on hardware and in simulation)"
description: >-
  In-depth skill for real-time / latency-bounded QEC decoding: in-kernel
  decoder API (reset_decoder, enqueue_syndromes, get_corrections),
  decoder_config / multi_decoder_config YAML configuration,
  configure_decoders_from_file, sliding window decoding, autonomous
  GPU-resident decoders (autonomous_decoder CRTP), the hybrid AI
  predecoder + PyMatching pipeline, FPGA ring-buffer dispatch, CUDA
  Graphs, NVTX instrumentation, and running real-time decoding on
  Quantinuum Helios. Use whenever the user mentions real-time decoding,
  in-kernel decoding, autonomous_decoder, AIDecoderService,
  AIPreDecoderService, CUDA Graph QEC, Helios, Quantinuum, FPGA, ring
  buffer, decoder dispatch, RPC, or "low latency QEC".
version: "0.1.2"
author: "CUDA-QX"
license: "Apache License 2.0 (parts under LicenseRef-NVIDIA-Proprietary, see libs/qec/LICENSE)"
compatibility: "Python 3.11+, C++ 20, Linux x86_64 with NVIDIA GPU"
tags: [cuda-qx, cudaq-qec, real-time, in-kernel, autonomous-decoder, cuda-graph, helios, quantinuum, fpga, predecoder]
tools: [Read, Glob, Grep, Bash]
metadata:
  repo: [qec]
  author: "CUDA-QX"
  domain: "quantum-error-correction"
  audience: [researcher, hardware-engineer, developer]
  languages: [python, c++, cuda]
---

# CUDA-QX Real-Time Decoding

The latency-bounded QEC corner of the repo. Decoding while the kernel
is still running, on simulators or hardware, with zero-CPU paths and
sub-millisecond budgets. The base QEC skill (`cuda-qx-qec-decode`) covers
offline decoding; **this** skill covers everything where time matters.

If the user is doing code-capacity or batch circuit-level decoding,
delegate to `cuda-qx-qec-decode`. If they are training an ML decoder, hand
off to `cuda-qx-qec-ai-decoders`. This skill assumes the model is fixed
and you need it to run *fast and in line with the kernel*.

## Inputs

Caller provides:

- A QEC `code` and `num_rounds`.
- A noise model (or `None` for hardware).
- A target: `"stim"` for simulation, `"quantinuum"` for hardware/emulate.
- Decoder choice from the real-time eligible set: `nv-qldpc-decoder`,
  `single_error_lut`, `multi_error_lut`. (Other decoders are NOT
  real-time eligible.)
- Quantinuum credentials path if targeting hardware.

## Outputs

This skill produces:

- A `config.yaml` consumable by `qec.configure_decoders_from_file`.
- A working `@cudaq.kernel` calling `reset_decoder` /
  `enqueue_syndromes` / `get_corrections`.
- LER + per-shot corrections from `cudaq.run`.
- For autonomous_decoder builds: a registered CRTP subclass + RPC handler.

Does NOT produce: trained neural decoders (→ `cuda-qx-qec-ai-decoders`);
new code/decoder plugins (→ `cuda-qx-qec-extending`).

## Audience

Hardware researchers, FPGA / quantum-control engineers, and
performance-minded developers. C++ familiarity is recommended; the
fastest paths in this skill are GPU-side device code.

## First three actions

```bash
bash   .agents/skills/_shared/scripts/preflight.sh    --json > /tmp/preflight.json
python .agents/skills/_shared/scripts/import_smoke.py --json > /tmp/import_smoke.json
python .agents/skills/_shared/scripts/pick_workflow.py \
    --intent qec-realtime \
    --preflight /tmp/preflight.json \
    --imports   /tmp/import_smoke.json
```

`pick_workflow.py` reports which decoder backends and extras are
present. Real-time eligibility requires at least one of:
`nv-qldpc-decoder`, `single_error_lut`, `multi_error_lut`. (As of
writing, `tensor_network_decoder`, `trt_decoder`, and `sliding_window`
are **not** real-time eligible.)

## Key Paths

| Area | Path |
|------|------|
| In-kernel API (Python) | `libs/qec/python/bindings/py_decoding.cpp` |
| Real-time C++ API | `libs/qec/include/cudaq/qec/realtime/` |
| Autonomous decoder CRTP | `libs/qec/include/cudaq/qec/realtime/autonomous_decoder.cuh` |
| AI decoder service | `libs/qec/include/cudaq/qec/realtime/ai_decoder_service.h` |
| AI predecoder service | `libs/qec/include/cudaq/qec/realtime/ai_predecoder_service.h` |
| Pipeline + config | `libs/qec/include/cudaq/qec/realtime/pipeline.h`, `decoding_config.h` |
| Ring buffer / GPU kernels | `libs/qec/include/cudaq/qec/realtime/gpu_kernels.cuh`, `graph_resources.h` |
| NVTX helpers | `libs/qec/include/cudaq/qec/realtime/nvtx_helpers.h` |
| Minimal end-to-end (Python) | `docs/sphinx/examples/qec/python/real_time_complete.py` |
| Minimal end-to-end (C++) | `docs/sphinx/examples/qec/cpp/real_time_complete.cpp` |
| Production-shaped app examples | `libs/qec/unittests/realtime/app_examples/` |
| Autonomous decoder dev guide | `docs/autonomous_decoder_guide.md` |
| Hybrid AI predecoder pipeline | `docs/hybrid_ai_predecoder_pipeline.md` |
| Predecoder example (RST) | `docs/sphinx/examples_rst/qec/realtime_predecoder_pymatching.rst`, `realtime_predecoder_fpga.rst` |
| Sequential Relay BP | `docs/sphinx/examples_rst/qec/realtime_relay_bp.rst` |
| Decoders RST page | `docs/sphinx/examples_rst/qec/decoders.rst` |
| Python API page | `docs/sphinx/api/qec/python_realtime_decoding_api.rst` |
| C++ API page | `docs/sphinx/api/qec/cpp_realtime_decoding_api.rst` |

## Source of Truth

- **In-kernel API**: `docs/sphinx/api/qec/python_realtime_decoding_api.rst`
  and `py_decoding.cpp`. These names (`reset_decoder`,
  `enqueue_syndromes`, `get_corrections`, `finalize_decoders`) are
  *not* re-bound in `__init__.py`; do not grep there.
- **Autonomous decoder**: `autonomous_decoder.cuh` and
  `docs/autonomous_decoder_guide.md` are authoritative.
- **Hybrid AI predecoder**: `docs/hybrid_ai_predecoder_pipeline.md`
  is the design doc; `ai_predecoder_service.h` is the code.

## Workflow Index

| If the user wants to | Read |
|----------------------|------|
| Run in-kernel decoding in simulation (4-phase procedure) | `references/in-kernel.md` |
| Decode long syndrome streams with a latency budget | `references/sliding-window.md` |
| Write a new GPU-resident decoder (autonomous_decoder) | `references/autonomous-decoder.md` |
| Wire an AI predecoder + global decoder pipeline | `references/ai-predecoder-pipeline.md` |
| Deploy real-time decoding on Quantinuum Helios | `references/hardware-helios.md` |

## Conventions

These are the recurring real-time-specific traps. Code that violates
any of them compiles and runs, but reports wrong corrections or hangs.

1. **`cudaq.set_target("stim")` for simulation,
   `cudaq.set_target("quantinuum", ...)` for hardware/emulation.**
   The default state-vector target cannot simulate QEC-sized circuits
   under noise.

2. **Configure decoders *before* `cudaq.run` (Phase 3 of the four-
   phase procedure).** `qec.configure_decoders_from_file("config.yaml")`
   must be called before the kernel is launched; otherwise
   in-kernel calls will see an unconfigured decoder.

3. **Always call `qec.finalize_decoders()` after the run.** Skipping
   this leaks GPU resources and breaks subsequent runs in the same
   process.

4. **Match `decoder_config.id` across `reset_decoder`,
   `enqueue_syndromes`, `get_corrections`, and `decoder_config`.**
   One unique `id` per logical qubit; mismatches silently route
   corrections to the wrong logical qubit.

5. **`block_size = H.shape[1]`, `syndrome_size = H.shape[0]`.** Off-by-
   one here is silent and produces nonsense corrections.

6. **`D_sparse` accounts for an extra round.** Compute as
   `num_syndromes_per_round * num_rounds_plus_one`. The example
   templates contain the right idiom; copy them, do not derive.

7. **Real-time API is in-kernel only.** `qec.reset_decoder`,
   `qec.enqueue_syndromes`, `qec.get_corrections` must be called
   inside `@cudaq.kernel` or `__qpu__` — not at Python top level.

8. **NVTX ranges must wrap the hot path,** not the configuration
   path. See `nvtx_helpers.h`; misuse confuses `nsys` profiles.

9. **CUDA Graph capture happens once.** Re-capturing per shot
   destroys the speedup; pre-allocate buffers in pinned/mapped memory
   and reuse the graph.

## Quick start: 3-qubit repetition code real-time

The minimal real-time example. Copy this verbatim and modify, don't
write from scratch.

```python
import os
os.environ["CUDAQ_DEFAULT_SIMULATOR"] = "stim"

import cudaq
import cudaq_qec as qec
```

The full source is in
`docs/sphinx/examples/qec/python/real_time_complete.py`. Walk that
file before adding hardware-specific code.

## Four-phase procedure (always in this order)

```
Phase 1: DEM         dem = qec.z_dem_from_memory_circuit(code, op, num_rounds, noise)
Phase 2: Configure   build qec.decoder_config + qec.multi_decoder_config; write YAML
Phase 3: Load        qec.configure_decoders_from_file("config.yaml")  (BEFORE cudaq.run)
Phase 4: In-kernel   qec.reset_decoder / qec.enqueue_syndromes / qec.get_corrections
                     then qec.finalize_decoders() at the end
```

Per-phase details and the `decoder_config` cheat sheet live in
`references/in-kernel.md`.

## Self-Check Protocol

```
[ ] Target set: stim (sim) or quantinuum (hardware).
[ ] H.shape[0] == syndrome_size, H.shape[1] == block_size.
[ ] D_sparse computed with num_rounds_plus_one, not num_rounds.
[ ] All decoder_config.id values are unique across logical qubits.
[ ] qec.configure_decoders_from_file called *before* cudaq.run.
[ ] qec.finalize_decoders called after the run.
[ ] reset_decoder / enqueue_syndromes / get_corrections live inside @cudaq.kernel.
[ ] Same noise object passed to sample_memory_circuit and dem_from_memory_circuit.
[ ] At p=0, the decoder reports zero corrections.
[ ] At small p, LER is reasonable (compare to offline decoding of the same DEM).
```

## When stuck

1. Re-run the **First three actions**. Missing decoder extras and a
   stale `_pycudaqx_*.so` are the most common silent blockers.
2. Read `references/in-kernel.md` end-to-end.
3. Reproduce in simulation (`stim`) before targeting hardware
   (`quantinuum`).
4. Set `CUDAQ_QEC_DEBUG_DECODER=1` to log decoder uploads.
5. Run the matching `surface_code_1.py` /
   `surface_code-{1,2,3}.cpp` in `libs/qec/unittests/realtime/app_examples/`
   and diff your YAML config against the one in those tests.
6. For autonomous_decoder development, read
   `docs/autonomous_decoder_guide.md` end-to-end before writing CUDA
   code — the CRTP / dispatch contract is easy to get wrong silently.

## Additional resources

- Workflow references: `references/in-kernel.md`,
  `references/sliding-window.md`, `references/autonomous-decoder.md`,
  `references/ai-predecoder-pipeline.md`,
  `references/hardware-helios.md`.
- Base QEC workflows (offline decoding, custom codes):
  `cuda-qx-qec-decode/SKILL.md`.
- Training neural decoders and ONNX → TensorRT:
  `cuda-qx-qec-ai-decoders/SKILL.md`.
- GPU profiling, NVTX, `nsys`: `cuda-qx-profiling-perf/SKILL.md`.
