# Custom QEC code in Python

The fastest way to prototype a new error-correcting code. Define
the stabilizers, the encoding kernels, and the parity matrices in
pure Python; the plugin loader picks it up at import time.

## Authoritative template

- `docs/sphinx/examples/qec/python/my_steane.py` — full Python
  Steane definition.
- `docs/sphinx/examples/qec/python/my_steane_test.py` — companion
  smoke test.
- `libs/qec/python/cudaq_qec/plugins/codes/example.py` — plugin
  example used by the loader tests.

Copy one of these and modify; do not write a code from scratch.

## Minimal skeleton

```python
import cudaq
import cudaq_qec as qec

class MyCode(qec.Code):
    def __init__(self, options=None):
        super().__init__()
        # define stabilizers, register encodings ...

    def get_num_data_qubits(self) -> int:
        return ...

    def get_num_ancilla_x_qubits(self) -> int:
        return ...

    def get_num_ancilla_z_qubits(self) -> int:
        return ...

    def get_parity_x(self):
        return ...     # H_X parity matrix (numpy)

    def get_parity_z(self):
        return ...     # H_Z parity matrix (numpy)

    def get_observables_x(self):
        return ...

    def get_observables_z(self):
        return ...

# Register
qec.register_code("my-code", MyCode)
```

## Required components

A complete Python code provides:

1. **Stabilizer measurement kernels** (`@cudaq.kernel`) for each
   round.
2. **Encoding kernels** (`prep0`, `prep1`, `prepp`, `prepm`) — at
   minimum, one matching the basis you intend to test.
3. **Parity-check matrices** (`get_parity_x`, `get_parity_z`).
4. **Logical observables** (`get_observables_x`,
   `get_observables_z`).
5. **Optional: logical-gate kernels** (`x`, `y`, `z`, `h`, `s`, `cx`,
   `cy`, `cz`) for fault-tolerant computation experiments.

For a smoke test you can omit the logical gates; only the
stabilizer-measurement + prep + observables are needed for memory
experiments.

## In-kernel constraints

Inside `@cudaq.kernel`:

- No NumPy or SciPy.
- No Python control flow that does not lower to Quake MLIR (no
  generator expressions, no `lambda` in flow positions).
- Function calls only to other `@cudaq.kernel` functions.
- `qec.patch` has three views: `data`, `ancx`, `ancz`.

When in doubt, look at the built-in code Python wrappers and copy
their kernel shape.

## Registration and discovery

`qec.register_code("name", MyCode)` adds your code to the registry.
After this, users can do:

```python
import my_code_module     # triggers registration
code = qec.get_code("my-code", **options)
```

The registration *must* happen at import time, before any
`get_code` call. The simplest way: put the `qec.register_code` line
at module top level.

## Smoke test pattern

```python
import cudaq, cudaq_qec as qec
import numpy as np
from my_code_module import MyCode

cudaq.set_target("stim")

code = qec.get_code("my-code")

# 1. Structural sanity
H_z = code.get_parity_z()
H_x = code.get_parity_x()
print("H_z shape:", H_z.shape)
print("H_x shape:", H_x.shape)

# 2. Code-capacity smoke test at p=0 (should be zero LER)
shots = qec.sample_code_capacity(H_z, nShots=10, p=0.0)
print("code-capacity p=0 ok:", all(s.sum() == 0 for s in shots))

# 3. Circuit-level smoke test at p=0
noise = cudaq.NoiseModel()
dem = qec.z_dem_from_memory_circuit(code, qec.operation.prep0, 2, noise)
print("DEM shape:", dem.detector_error_matrix.shape)
```

If any of these fails at p=0, the code is mis-defined.

## Common pitfalls

| Symptom | Cause |
|---------|-------|
| `qec.get_code("my-code")` raises "unknown code" | `register_code` not called at import; module not imported before `get_code` |
| LER > 0 at p=0 | parity matrix wrong, or observables don't commute with stabilizers |
| DEM shape doesn't match expectation | stabilizer kernels emit measurements in a different order than parity matrix indicates |
| Kernel compilation fails | Python control flow or numpy ref inside `@cudaq.kernel` |

## Self-check

```
[ ] H_z @ obs_z.T == 0 (mod 2) — observables commute with stabilizers
[ ] H_x @ obs_x.T == 0 (mod 2)
[ ] Code-capacity at p=0 → zero LER over many shots
[ ] DEM at p=0 → all zero detectors
[ ] Smoke test runs in under a few seconds
[ ] Module documented (one paragraph + minimal example)
```

## Where next

- C++ port if performance demands: `code-cpp.md`.
- Test the code under multiple decoders: `cudaq-qec-decode/references/decode.md`.
- Add to the docs: `cudaq-build/references/docs.md`.
