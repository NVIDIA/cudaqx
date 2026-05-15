# Integrating a TensorRT engine into `trt_decoder`

Once you have a `.engine` file that matches your PyTorch model, plug
it into cudaq-qec via the `trt_decoder` plugin.

## Prerequisites

```bash
pip install cudaq-qec[trt-decoder]
```

This pulls in TensorRT bindings and the `trt_decoder` plugin. The
plugin must be present *before* `qec.get_decoder("trt_decoder", ...)`
is called; otherwise the call falls back to "decoder not found".

## Authoritative API

`docs/sphinx/api/qec/trt_decoder_api.rst`. Read it before
generating new code; the exact parameter set evolves between
releases.

C++ header (private): `libs/qec/include/cudaq/qec/trt_decoder_internal.h`.

## Usage

```python
import cudaq_qec as qec

decoder = qec.get_decoder("trt_decoder", H,
                          engine_path="my_decoder.engine",
                          threshold=0.5)  # logit -> hard prediction
```

The engine must accept inputs in the same layout your training
pipeline produced. Typical layout: `[batch, num_detectors]` of
float32, logits out of shape `[batch, num_observables]`.

## Real-time eligibility

`trt_decoder` is currently **not** eligible for in-kernel real-time
decoding. To use a TensorRT model at hardware latency, you have two
options:

1. **Hybrid AI predecoder pipeline** — see
   `hybrid-deployment.md` and `cudaq-qec-realtime/references/ai-predecoder-pipeline.md`.
   The engine drives the GPU stage; PyMatching handles residuals on
   CPU. This is the validated production path for surface codes.
2. **Custom autonomous_decoder with TensorRT** — write a CRTP
   subclass that calls TensorRT inside the device-side dispatch loop.
   Advanced; see
   `cudaq-qec-realtime/references/autonomous-decoder.md`.

For offline circuit-level decoding with `trt_decoder`, no special
real-time setup is needed; it behaves like any other decoder.

## Validating the integration

Run the trained engine through the standard cudaq-qec decoding loop
and compare to your PyTorch test-set LER:

```python
H = dem.detector_error_matrix          # whatever PCM you trained on
decoder = qec.get_decoder("trt_decoder", H, engine_path="my_decoder.engine")

logical_errors = 0
for shot in test_set:
    result = decoder.decode(shot.syndrome)
    prediction = (result.result > 0.5).astype(np.uint8)
    if (prediction @ obs % 2) != (shot.label):
        logical_errors += 1

print(f"trt_decoder test LER: {logical_errors / len(test_set):.4g}")
```

The number should match your PyTorch test LER to within sampling
noise. A large divergence indicates an export problem (re-read
`onnx-tensorrt.md`).

## Common pitfalls

| Symptom | Likely cause |
|---------|--------------|
| `decoder not found: trt_decoder` | extras not installed: `pip install 'cudaq-qec[trt-decoder]'` |
| Engine loads but accuracy is way worse than PyTorch | input layout mismatch (transpose, batch axis, dtype) |
| Latency much higher than expected | engine using full optimization profile, not pinned to the actual batch size |
| Random crashes after many calls | TensorRT context leak; reuse one context across calls |

## Self-check

```
[ ] cudaq-qec[trt-decoder] installed.
[ ] trt_decoder loaded without "decoder not found".
[ ] Test-set LER matches PyTorch test LER within sampling noise.
[ ] Latency per syndrome benchmarked and recorded.
[ ] If deploying real-time: chosen one of (predecoder pipeline, autonomous_decoder) — trt_decoder is offline only.
```
