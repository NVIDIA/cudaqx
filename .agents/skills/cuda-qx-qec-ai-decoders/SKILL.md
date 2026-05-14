---
name: "cuda-qx-qec-ai-decoders"
title: "CUDA-QX QEC AI Decoders (training, ONNX, TensorRT)"
description: >-
  End-to-end skill for neural-network QEC decoders: dataset generation
  from Stim circuits, MLP / transformer training in PyTorch, ONNX
  export, TensorRT engine build (scripts/build_engine_from_onnx.py),
  trt_decoder integration with cudaq-qec, and deployment into the
  hybrid AI predecoder pipeline. Use whenever the user mentions
  training a decoder, neural decoder, MLP decoder, transformer
  decoder, ONNX, TensorRT, trt_decoder, build_engine_from_onnx,
  AIDecoderService, AIPreDecoderService, or hybrid predecoder
  training.
version: "0.1.1"
author: "CUDA-QX"
license: "Apache License 2.0"
compatibility: "Python 3.11+, PyTorch 2.0+, TensorRT (matching CUDA), NVIDIA GPU"
tags: [cuda-qx, cudaq-qec, ai-decoder, machine-learning, pytorch, onnx, tensorrt, trt-decoder, hybrid-predecoder, mlp]
tools: [Read, Glob, Grep, Bash]
metadata:
  repo: [qec]
  author: "CUDA-QX"
  domain: "quantum-error-correction"
  audience: [ml-researcher, qec-researcher, developer]
  languages: [python, c++]
---

# CUDA-QX AI Decoders

Train, export, and deploy neural network decoders for QEC. Three
distinct stages, each with its own pitfalls:

```
Train (PyTorch) ─► Export (ONNX) ─► Compile (TensorRT) ─► Run (trt_decoder or hybrid pipeline)
```

If the user is *running* a pre-trained decoder, delegate to
`cuda-qx-qec-realtime` (`ai-predecoder-pipeline.md`). This skill
covers everything **upstream** of that — getting a model that's worth
deploying.

## Inputs

Caller provides:

- A target code (e.g. surface code distance `d` + rounds `T`).
- Training data spec: Stim circuit + sampler seed + `error_prob`.
- Model class (default: MLP per `train_mlp_decoder.py`; or custom).
- Output target: `.engine` (TensorRT), `.onnx`, or `.pt` checkpoint.
- Acceptance threshold: target LER vs a chosen baseline decoder.

## Outputs

This skill produces:

- A trained `.pt` checkpoint.
- An ONNX export (`.onnx`).
- A TensorRT engine (`.engine`) compiled for a specific GPU + driver.
- A validation LER on a held-out test set.
- Wiring snippet for `qec.get_decoder("trt_decoder", ...,
  engine_path=...)` or for the hybrid predecoder pipeline.

Does NOT produce: a deployed real-time pipeline (→
`cuda-qx-qec-realtime`); a registered decoder plugin (→
`cuda-qx-qec-extending`).

## Audience

ML-savvy QEC researchers and developers. Comfort with PyTorch and a
CUDA-capable GPU are assumed. TensorRT familiarity is helpful but not
required for the example workflow.

## First three actions

```bash
bash   .agents/skills/_shared/scripts/preflight.sh    --json > /tmp/preflight.json
python .agents/skills/_shared/scripts/import_smoke.py --json > /tmp/import_smoke.json
python -c "import torch, tensorrt; print(torch.__version__, tensorrt.__version__)"
```

The third command catches the most common blocker: PyTorch and
TensorRT versions mismatched against installed CUDA. If
`import tensorrt` fails, install the matching NVIDIA-published
wheel: <https://developer.nvidia.com/tensorrt>.

## Key Paths

| Area | Path |
|------|------|
| MLP decoder training example | `docs/sphinx/examples/qec/python/train_mlp_decoder.py` |
| TRT decoder example (Python) | `docs/sphinx/examples/qec/python/tensor_network_decoder.py` (TN baseline, *not* TRT) — for TRT see API + RST docs |
| ONNX → TensorRT script | `scripts/build_engine_from_onnx.py` |
| TRT decoder header (internal) | `libs/qec/include/cudaq/qec/trt_decoder_internal.h` |
| trt_decoder API doc | `docs/sphinx/api/qec/trt_decoder_api.rst` |
| Hybrid AI predecoder pipeline | `docs/hybrid_ai_predecoder_pipeline.md` |
| AI decoder service (C++) | `libs/qec/include/cudaq/qec/realtime/ai_decoder_service.h` |
| AI predecoder service (C++) | `libs/qec/include/cudaq/qec/realtime/ai_predecoder_service.h` |
| Decoders RST page | `docs/sphinx/examples_rst/qec/decoders.rst` |
| `deploying-ai-decoders` doc | `docs/sphinx/quickstart/installation.rst` (linked section) |

## Workflow Index

| If the user wants to | Read |
|----------------------|------|
| Train an MLP decoder on Stim-generated data | `references/training.md` |
| Export to ONNX, build a TensorRT engine | `references/onnx-tensorrt.md` |
| Plug the engine into `trt_decoder` in cudaq-qec | `references/trt-decoder-integration.md` |
| Deploy into the hybrid AI predecoder pipeline | `references/hybrid-deployment.md` |

## Conventions

Common silent-correctness bugs in ML decoder pipelines:

1. **Stim's detector ordering is opaque.** Train on detector outputs
   from the same `stim.Circuit.compile_detector_sampler()` that you
   will run at inference. If you regenerate the circuit with
   different `after_clifford_depolarization` or different seed, the
   *labels* of detectors can shift and your model "works" on one
   dataset but not another.

2. **Match `error_prob` across train / val / test / deployment.** A
   model trained on `p=0.005` does not gracefully degrade to
   `p=0.01`. Hold out independent test sets sampled at the *exact*
   physical error rate you will run at.

3. **Class imbalance is severe.** Most syndromes are zero. Use
   `pos_weight` in `BCEWithLogitsLoss`, or weighted sampling, or
   filter out the all-zero syndromes from the training set.

4. **ONNX export must use the exact input shape used at inference.**
   Set `dynamic_axes` only for the batch dimension; everything else
   must be static or TensorRT optimization profiles get unwieldy.

5. **TensorRT is precision-sensitive.** FP16 is usually fine for QEC
   inference; INT8 needs calibration and rarely pays off. Validate
   accuracy after engine build before benchmarking latency.

6. **Engine files are GPU + driver specific.** A `.engine` built on
   one machine may not load on another with a different driver
   version. Cache by `(gpu, driver, model-hash)` not just by model
   hash.

7. **`trt_decoder` expects a specific input/output layout.** See
   `docs/sphinx/api/qec/trt_decoder_api.rst`. Models trained outside
   that contract will load but produce garbage.

8. **ARM64 (aarch64) skips stim.** Multiple examples (and the MLP
   trainer template) early-exit on aarch64 because the upstream
   `stim` wheel is x86_64-only. Train on x86_64 and copy the engine
   file to ARM64 hosts.

## Quick start: MLP decoder for a distance-3 surface code

The smallest end-to-end. Walk this before scaling up.

1. Generate syndromes from a stim surface code:
   `docs/sphinx/examples/qec/python/train_mlp_decoder.py`.
2. Train the MLP (PyTorch). The example uses 5,000 train + 1,000 val
   + 1,000 test samples at `error_prob=0.005`.
3. Export to ONNX with `torch.onnx.export(model, ...)`.
4. Build the engine:
   ```bash
   python scripts/build_engine_from_onnx.py \
       --onnx my_mlp.onnx --engine my_mlp.engine --fp16
   ```
5. Plug into `trt_decoder` via `qec.get_decoder("trt_decoder",
   H, ..., engine_path="my_mlp.engine")` (exact kwargs in
   `references/trt-decoder-integration.md`).

## Self-Check Protocol

```
[ ] Stim circuit + sampler used at training is the SAME object used at inference (or seeded reproducibly).
[ ] Class imbalance handled (pos_weight / weighted sampler / filter zero-syndromes).
[ ] Held-out test set at p = deployment p reports loss + LER, not just training accuracy.
[ ] ONNX exported with the right input shape; dynamic only on batch.
[ ] Engine built with FP16 (or fallback to FP32) and validated against PyTorch outputs.
[ ] trt_decoder loads the engine without warnings.
[ ] At p=0 the decoder reports zero corrections.
[ ] Decoder LER beats single_error_lut baseline on the same DEM.
```

## When stuck

1. Re-run the **First three actions**. PyTorch + TensorRT version
   mismatch is the most common silent blocker.
2. Train *and* deploy on the same machine first; eliminate driver /
   GPU mismatch before debugging accuracy.
3. Profile with `nsys` (see `cuda-qx-profiling-perf`): if engine
   inference dominates, FP16/INT8 may help; if dataloading dominates,
   you trained badly.
4. For surface-code-specific dataset generation, see
   `references/training.md`.

## Additional resources

- `references/training.md` — dataset generation, model design,
  training-time pitfalls.
- `references/onnx-tensorrt.md` — export, engine build, calibration.
- `references/trt-decoder-integration.md` — wire into cudaq-qec at
  inference time.
- `references/hybrid-deployment.md` — drop into the predecoder +
  PyMatching pipeline.
- Real-time deployment: `cuda-qx-qec-realtime` SKILL.md.
- Benchmarking the decoder: `cuda-qx-benchmarking` SKILL.md.
- Profiling: `cuda-qx-profiling-perf` SKILL.md.
