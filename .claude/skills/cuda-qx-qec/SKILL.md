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
version: "0.2.0"
author: "CUDA-QX"
license: "LicenseRef-NVIDIA-Proprietary"
compatibility: "Python 3.11+, C++ 20, Linux x86_64/aarch64"
tags: [cuda-qx, cudaq-qec, qec, quantum-error-correction, decoders, surface-code, real-time-decoding, dem-sampling, nvidia]
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

1. **First three actions** below: run `_shared/scripts/preflight.sh`,
   `import_smoke.py`, `pick_workflow.py`. They surface common blockers
   (missing decoder extras, stale install, ABI mismatch) before you spend
   tokens guessing.
2. Read the **Conventions** section. It is short and prevents the most
   common silent-correctness mistakes.
3. Find your task in the **Workflow Index** and open the matching file
   under `references/`.
4. Walk the **Self-Check Protocol** in `references/triage.md` before
   declaring done.

If you only have time for one file, read `references/decode.md`. The
"Circuit-Level Memory Experiment" section there, plus the conventions,
covers roughly 80% of QEC tasks: build a code, decode under noise,
measure the logical error rate.

## Audience

AI coding agents and developers operating `cudaq_qec` from a checkout of
this repo. Pip-only users (`pip install cudaq-qec`) get the same Python
API but cannot read the template files referenced below; they should
consult the published docs at <https://nvidia.github.io/cudaqx/>
instead.

The artifact is `SKILL.md` (uppercase) inside `.claude/skills/cuda-qx-qec/`.
Companion files live in `references/` (see Workflow Index below).

## First three actions (always, before anything else)

```bash
bash   .claude/skills/_shared/scripts/preflight.sh    --json > /tmp/preflight.json
python .claude/skills/_shared/scripts/import_smoke.py --json > /tmp/import_smoke.json
python .claude/skills/_shared/scripts/pick_workflow.py \
    --intent <pick from list below> \
    --preflight /tmp/preflight.json \
    --imports   /tmp/import_smoke.json
```

QEC-skill intents: `qec-decode`, `qec-custom`, `qec-realtime`,
`qec-debug`. `pick_workflow.py` returns the next reference file to read
and the commands to run. Do that, then come back here for conventions.

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

| If the user wants to                                                | Read                       | `pick_workflow.py` intent |
|---------------------------------------------------------------------|----------------------------|---------------------------|
| Decode (code-capacity, circuit-level, pick a decoder)               | `references/decode.md`     | `qec-decode`              |
| Define a new code or decoder (Python or C++)                        | `references/extend.md`     | `qec-custom`              |
| Sliding window, real-time in-kernel, DEM sampling, predecoder       | `references/realtime.md`   | `qec-realtime`            |
| Diagnose "LER looks wrong" / decoder gotchas / install failures     | `references/triage.md`     | `qec-debug`               |

`triage.md` is also where the **Self-Check Protocol**, the
**`nv-qldpc-decoder` parameters** table, and the **Noise Model Patterns**
live. Read it any time the workflow runs but produces suspicious
numbers.

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

1. Re-run the **First three actions**. Decoder extras (`tensor-network-decoder`,
   `trt-decoder`, `gqe`) often appear missing only on first use.
2. Read the matching workflow's template file end-to-end before generating
   new code. Every workflow in `references/` points at one.
3. Grep `libs/qec/python/cudaq_qec/__init__.py` for the suspected API name.
4. Read the C++ header in `libs/qec/include/cudaq/qec/` for the canonical
   signature.
5. Reproduce with `nShots=10` and an explicit `p=0` run. The decoder
   should report zero LER without noise; this catches structural bugs
   before noise-related ones.
6. If the symptom is "LER looks wrong", open
   `references/triage.md` → "Troubleshooting: 'LER looks wrong'". The
   first three causes account for roughly 90% of cases.

## Additional resources

- Workflow references: `references/decode.md`, `references/extend.md`,
  `references/realtime.md`, `references/triage.md`
- Shared diagnostic scripts: `.claude/skills/_shared/scripts/`
- Shared repo map (incl. build/docs paths): `.claude/skills/_shared/repo_map.md`
- Build/wheel/docs questions: `.claude/skills/cuda-qx-build/SKILL.md`
