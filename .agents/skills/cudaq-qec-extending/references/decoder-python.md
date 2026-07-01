# Custom decoder in Python

The fastest way to prototype a new decoding algorithm. No rebuild
required; the plugin loader picks it up at import.

## Authoritative templates

- `libs/qec/python/cudaq_qec/plugins/decoders/example.py` — minimal
  template used by loader tests.
- `libs/qec/python/cudaq_qec/plugins/decoders/tensor_network_decoder.py`
  — full real Python decoder. Shows the result-object pattern.

## Skeleton

```python
import numpy as np
import cudaq_qec as qec

class MyDecoder:
    def __init__(self, H, **kwargs):
        self.H = H
        self.params = kwargs

    def decode(self, syndrome):
        # syndrome: shape (H.shape[0],), uint8
        # returns: qec.DecoderResult with .result, .converged, .error
        prediction = self._run_my_algorithm(syndrome)
        return qec.DecoderResult(result=prediction,
                                 converged=True,
                                 error=0.0)

qec.register_decoder("my-decoder", MyDecoder)
```

After import, users get the decoder via `qec.get_decoder("my-decoder",
H, **kwargs)`.

## Decoder result object

```python
qec.DecoderResult(
    result=np.zeros(H.shape[1]),    # soft prediction; threshold downstream
    converged=True,                  # bool
    error=0.0,                       # optional: decoder-internal error estimate
)
```

Conventions:

- `result` length **must** equal `H.shape[1]` (one entry per
  column / error mechanism). Not per data qubit.
- `result` is "soft": float in `[0, 1]`. Users threshold at 0.5 for
  hard predictions.
- `converged=False` flags shots where the decoder bailed out;
  benchmark code typically reports both LER and per-shot
  convergence.

## Where to put your decoder file

For prototyping: anywhere on `PYTHONPATH`. As long as the module is
imported before `qec.get_decoder` is called, registration takes
effect.

For shipping with cudaq-qec: drop it under
`libs/qec/python/cudaq_qec/plugins/decoders/` and follow the loader
contract used by built-in plugins. The decoder tests at
`libs/qec/python/tests/test_decoder.py` exercise the registration path
and show what's expected.

## Plugin loader behavior

`qec.list_decoders()` enumerates registered decoders. The loader
runs at import time of `cudaq_qec`. If your plugin lives elsewhere
and isn't auto-imported, you need to:

```python
import my_decoder_module     # triggers register_decoder
import cudaq_qec as qec
qec.get_decoder("my-decoder", H)
```

## Smoke test

```python
import numpy as np
import cudaq_qec as qec
import my_decoder_module

# Build a tiny H (3 qubits, 2 stabilizers — repetition-like)
H = np.array([[1, 1, 0],
              [0, 1, 1]], dtype=np.uint8)

dec = qec.get_decoder("my-decoder", H)
result = dec.decode(np.zeros(2, dtype=np.uint8))   # syndrome all zeros
assert (result.result == 0).all(), "p=0 should yield zero correction"
```

If `p=0` doesn't yield zero correction, the decoder is mis-defined.

## Performance notes

Python decoders run on the CPU, single-threaded by default. For
throughput:

- Vectorize over shots inside `decode` (accept a 2D syndrome array).
- Use `numba`, `cython`, or `numpy` BLAS for hot inner loops.
- For GPU acceleration, prototype in PyTorch or CuPy.
- For production-scale latency, port to C++ (`decoder-cpp.md`).

## Common pitfalls

| Symptom | Cause |
|---------|-------|
| `qec.get_decoder("my-decoder", H)` not found | `register_decoder` not called; or module not imported |
| `result.result` length mismatch | returned per data qubit instead of per column |
| Decoder reports "converged" even when it didn't | sentinel never updated; downstream LER misleading |
| Memory blowup on big H | inadvertently dense LU/QR; consider sparse |

## Self-check

```
[ ] register_decoder fires at module import.
[ ] result.result length == H.shape[1].
[ ] result.result values in [0, 1].
[ ] Decoder returns zero corrections at p=0.
[ ] Smoke test against single_error_lut on a tiny H matches in low-noise limit.
[ ] Decoder appears in qec.list_decoders() after import.
```

## Where next

- Performance-critical version in C++: `decoder-cpp.md`.
- Make it a real-time decoder: not in Python; see
  `cudaq-qec-realtime/references/autonomous-decoder.md`.
- Benchmark vs built-ins: `cudaq-benchmarking`.
