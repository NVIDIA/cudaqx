---
name: "cuda-qx-qec"
title: "CUDA-QX QEC"
description: >-
  Build, run, and verify CUDA-Q QEC (cudaq_qec) workflows: choose and configure
  decoders (nv-qldpc-decoder, tensor_network_decoder, trt_decoder,
  sliding_window, single/multi_error_lut), construct codes (Steane, repetition,
  surface_code), set up cudaq.NoiseModel for QEC, generate Detector Error
  Models, run code-capacity or circuit-level memory experiments, sample
  syndromes from a DEM, wire a predecoder, and run real-time decoding on Stim
  or Quantinuum Helios. Use whenever the user mentions cudaq-qec, cudaq_qec,
  QEC, quantum error correction, syndrome decoder, parity check matrix,
  surface code, Steane code, repetition code, detector error model, DEM,
  dem_sampling, sliding window, predecoder, real-time decoding, Helios, or
  Quantinuum.
version: "0.1.0"
author: "CUDA-QX"
license: "Apache License 2.0"
compatibility: "Linux x86_64/aarch64, Python 3.10+, C++ 20"
tags: [cuda-qx, cudaq-qec, quantum-error-correction, decoders, surface-code, real-time-decoding, dem-sampling, nvidia]
tools: [Read, Glob, Grep, Bash]
metadata:
  author: "CUDA-QX"
  domain: "quantum-error-correction"
  languages:
    - python
    - c++
  tags:
    - cuda-qx
    - cudaq-qec
    - quantum-error-correction
    - decoders
    - surface-code
    - real-time-decoding
    - dem-sampling
---

# CUDA-Q QEC Skill

Operate the `cudaq_qec` library on this repo. The skill is workflow-driven:
pick a workflow, follow it end-to-end, then run the **Self-Check Protocol**
before reporting done. Do not invent API names from memory; look them up in
the **Source of Truth** table.

## How to read this skill

1. Read the **Conventions** section. It is short and prevents the
   most common silent-correctness mistakes.
2. Find your task in the **Workflow Index**.
3. Open the matching workflow in `REFERENCE.md` and follow it.
4. Walk the **Self-Check Protocol** in `REFERENCE.md` before declaring done.

If you only have time for one workflow, read **W2: Circuit-Level Memory
Experiment** in `REFERENCE.md`. With the conventions, it covers roughly 80%
of QEC tasks: build a code, decode under noise, measure the logical error
rate.

## Audience

AI coding agents and developers operating `cudaq_qec` from a checkout of this
repo. Pip-only users (`pip install cudaq-qec`) get the same Python API but
cannot read the template files referenced below; they should consult the
published docs at <https://nvidia.github.io/cudaqx/> instead.

The artifact is `SKILL.md` (uppercase) inside `.claude/skills/cuda-qx-qec/`.
People sometimes call it "skills.md"; it is the same file. The companion
file is `REFERENCE.md` in the same directory.

## Standard imports

Every workflow assumes:

```python
import cudaq
import cudaq_qec as qec   # the qec. alias used throughout this skill
import numpy as np
```

## Key Paths

A one-glance map of the QEC library on disk.

| Area                   | Path                                                                    |
|------------------------|--------------------------------------------------------------------------|
| QEC library            | `libs/qec/`                                                              |
| Python package         | `libs/qec/python/cudaq_qec/`                                             |
| Python bindings        | `libs/qec/python/bindings/`                                              |
| C++ public headers     | `libs/qec/include/cudaq/qec/`                                            |
| C++ implementation     | `libs/qec/lib/`                                                          |
| Decoder plugins        | `libs/qec/lib/decoders/`, `libs/qec/python/cudaq_qec/plugins/decoders/`  |
| Python tests           | `libs/qec/python/tests/`                                                 |
| C++ tests              | `libs/qec/unittests/`                                                    |
| Real-time app examples | `libs/qec/unittests/realtime/app_examples/`                              |
| QEC docs               | `docs/sphinx/components/qec/`, `docs/sphinx/api/qec/`                    |
| QEC examples           | `docs/sphinx/examples/qec/`, `docs/sphinx/examples_rst/qec/`             |

## Source of Truth

Look here before guessing an API name.

| Need to know                                  | Authoritative file                                                          |
|-----------------------------------------------|------------------------------------------------------------------------------|
| Full Python public API (one-page enumeration) | `libs/qec/python/cudaq_qec/__init__.py`                                      |
| C++ decoder base class and realtime API       | `libs/qec/include/cudaq/qec/decoder.h`                                       |
| C++ code base class and `patch` type          | `libs/qec/include/cudaq/qec/code.h`, `libs/qec/include/cudaq/qec/patch.h`    |
| DEM types and functions                       | `libs/qec/include/cudaq/qec/detector_error_model.h`                          |
| PCM utilities                                 | `libs/qec/include/cudaq/qec/pcm_utils.h`                                     |
| Built-in code headers                         | `libs/qec/include/cudaq/qec/codes/`                                          |
| Library overview and conventions              | `docs/sphinx/components/qec/introduction.rst`                                |
| Per-decoder API pages                         | `docs/sphinx/api/qec/nv_qldpc_decoder_api.rst`, `tensor_network_decoder_api.rst`, `trt_decoder_api.rst`, `sliding_window_api.rst` (note: no `_decoder` infix on the last one) |
| Real-time in-kernel API (Python)              | `docs/sphinx/api/qec/python_realtime_decoding_api.rst`                       |
| Real-time in-kernel API (C++)                 | `docs/sphinx/api/qec/cpp_realtime_decoding_api.rst`                          |

When the user asks "is there a function for X?", grep `__init__.py` and the
relevant header before answering. **One caveat**: a few names are not
re-bound at the top of `__init__.py`. The in-kernel real-time API
(`reset_decoder`, `enqueue_syndromes`, `get_corrections`) flows through
the `from ._pycudaqx_qec_the_suffix_matters_cudaq_qec import *` wildcard
and is defined in `libs/qec/python/bindings/py_decoding.cpp`. For that
API, treat `docs/sphinx/api/qec/python_realtime_decoding_api.rst` as
authoritative.

## Workflow Index

Match the user's intent to a workflow, then open its full description in
`REFERENCE.md`.

| If the user wants to                                                | Use                  |
|----------------------------------------------------------------------|----------------------|
| Decode random bit-flips on a parity-check matrix                     | **W1: Code-Capacity**|
| Run a memory experiment with circuit noise and measure LER           | **W2: Circuit-Level**|
| Define a new code in Python or C++                                   | **W3: Custom Code**  |
| Define a new decoder in Python or C++                                | **W4: Custom Decoder**|
| Decode many syndrome rounds with a latency budget                    | **W5: Sliding Window**|
| Decode in-kernel during execution (Stim, Quantinuum emulate, Helios) | **W6: Real-Time**    |
| Sample errors and syndromes directly from a DEM (CPU/GPU)            | **W7: DEM Sampling** |
| Wire a predecoder in front of a main decoder (PyMatching, FPGA)      | **W8: Predecoder**   |

Cross-cutting topics in `REFERENCE.md`: **Decoder Selection** decision tree,
**`nv-qldpc-decoder` parameters**, **Noise Model Patterns**, **Self-Check
Protocol**, and **Troubleshooting**.

## Installation and environment

- `pip install cudaq-qec`. Optional extras:
  `cudaq-qec[tensor-network-decoder]`, `cudaq-qec[trt-decoder]`.
- The `nv-qldpc-decoder` is a closed-source plugin distributed separately;
  see `libs/qec/README.md`.
- Useful environment variables (set before `import cudaq_qec`):
  `CUDAQ_DEFAULT_SIMULATOR=stim`, `CUDAQ_QEC_DEBUG_DECODER=1`,
  `CUDAQ_QUANTINUUM_CREDENTIALS=...`.

---

## Conventions

These are the recurring mistakes. Code that violates any of them usually
runs but reports the wrong logical error rate.

1. **Use `cudaq.set_target("stim")` for any workflow that runs a CUDA-Q
   kernel.** That covers W2 (circuit-level), W3 (custom code), W6
   (real-time in simulation), and W8 (predecoder pipelines). The default
   state-vector simulator does not scale to QEC sizes. Two exceptions:
   pure code-capacity work (W1) and DEM sampling (W7) operate directly on
   parity-check matrices and never launch a kernel, so they need no
   target. For real-time decoding on Quantinuum (W6 hardware path), set
   `cudaq.set_target("quantinuum", ...)` instead.

2. **CSS layout is block-diagonal.** Every built-in CSS code uses:

   - `H_CSS = diag(H_Z, H_X)`. Z-stabilizers detect X-errors and vice versa.
   - Concatenated syndrome `S = S_X | S_Z`, error `E = E_X | E_Z`.
   - For a `prep0` (Z-basis) experiment, the X half of the syndrome is
     meaningless and must be sliced off before decoding:

     ```python
     syndromes = syndromes.reshape((nShots, nRounds, -1))
     syndromes = syndromes[:, :, :syndromes.shape[2] // 2]  # keep Z half
     syndromes = syndromes.reshape((nShots, -1))
     ```

   The same pattern applies for `prep1`. For X-basis experiments
   (`prepp`/`prepm`), keep the X half. The general non-CSS path uses
   `dem_from_memory_circuit`, with no slice.

3. **For circuit-level decoding, decode against
   `dem.detector_error_matrix`, not `code.get_parity()`.** The DEM's PCM
   has the right column ordering and weights for the actual circuit; the
   code's bare parity does not.

4. **Pass the same `noise` object to both `sample_memory_circuit` and the
   matching `*_dem_from_memory_circuit` call.** When the simulator and the
   decoder disagree about the noise model, the LER is meaningless.

5. **In-kernel restrictions.** Inside `@cudaq.kernel` (Python) or
   `__qpu__` (C++), do not use NumPy or SciPy, and do not use Python
   control flow that does not lower to Quake MLIR. A kernel can call
   only other `@cudaq.kernel` functions. The `qec.patch` type holds
   three views: `data`, `ancx`, `ancz`.

6. **The real-time API is in-kernel only.** `qec.reset_decoder(id)`,
   `qec.enqueue_syndromes(id, syndromes, offset)`, and
   `qec.get_corrections(id, num_obs, blocking)` are called from inside
   a `@cudaq.kernel`, not from Python top level.

---

## When stuck

1. Read the matching workflow's template file end-to-end before generating
   new code. Every workflow in `REFERENCE.md` points at one.
2. Grep `libs/qec/python/cudaq_qec/__init__.py` for the suspected API name.
3. Read the C++ header in `libs/qec/include/cudaq/qec/` for the canonical
   signature.
4. Reproduce with `nShots=10` and an explicit `p=0` run. The decoder
   should report zero LER without noise; this catches structural bugs
   before noise-related ones.
5. If the symptom is "LER looks wrong", go to the
   **Troubleshooting** section in `REFERENCE.md`. The first three
   causes there account for roughly 90% of cases.
