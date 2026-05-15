# Python tests (pytest)

Python tests live under `libs/qec/python/tests/` and
`libs/solvers/python/tests/`. The default pytest discovery picks
them up.

## File and function naming

- File: `test_<feature>.py`.
- Function: `def test_<what_it_checks>():`.

## Standard imports

```python
import os, sys, platform
import numpy as np
import pytest

import cudaq
import cudaq_qec as qec        # or import cudaq_solvers as solvers
```

For QEC tests that launch a kernel:

```python
os.environ.setdefault("CUDAQ_DEFAULT_SIMULATOR", "stim")
```

## Fixtures

Keep fixtures local to the test file unless they're shared across
several files (then put in a `conftest.py` in the same dir).

```python
@pytest.fixture
def steane_code():
    return qec.get_code("steane")
```

## Marks

```python
@pytest.mark.slow                         # takes >30s
@pytest.mark.gpu                          # needs CUDA
@pytest.mark.skipif(platform.machine().lower() in ("arm64", "aarch64"),
                    reason="stim not available on aarch64")
```

CI selects marks via `-m`. Local: `pytest -m "not slow"` for a fast
sanity loop.

## Decoder smoke pattern

For new decoders, the minimum useful test:

```python
def test_my_decoder_zero_input():
    H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    dec = qec.get_decoder("my-decoder", H)
    r = dec.decode(np.zeros(2, dtype=np.uint8))
    assert (r.result == 0).all(), "expected zero corrections at p=0"
    assert r.converged

def test_my_decoder_matches_lut():
    """Cross-check against single_error_lut at low noise."""
    code = qec.get_code("repetition", distance=3)
    H = code.get_parity_z()
    my_dec  = qec.get_decoder("my-decoder", H)
    lut_dec = qec.get_decoder("single_error_lut", H)
    rng = np.random.default_rng(0)
    for _ in range(200):
        e = (rng.random(H.shape[1]) < 0.01).astype(np.uint8)
        s = (H @ e) % 2
        r_my  = my_dec.decode(s)
        r_lut = lut_dec.decode(s)
        # Predictions agree (or both wrong in the same way)
        assert ((r_my.result > 0.5) == (r_lut.result > 0.5)).any()
```

The "matches LUT at low noise" pattern catches an enormous fraction
of decoder bugs.

## Solver pattern

```python
def test_h2_vqe_matches_fci():
    os.environ["OMP_NUM_THREADS"] = "1"
    np.random.seed(0)

    geom = [("H", (0.,0.,0.)), ("H", (0.,0.,0.7474))]
    mol = solvers.create_molecule(geom, "sto-3g", spin=0, charge=0, casci=True)

    @cudaq.kernel
    def ansatz(theta: float):
        q = cudaq.qvector(2); x(q[0]); ry(theta, q[1]); x.ctrl(q[1], q[0])

    H = (5.907 - 2.1433 * spin.x(0) * spin.x(1)
         - 2.1433 * spin.y(0) * spin.y(1)
         + 0.21829 * spin.z(0) - 6.125 * spin.z(1))
    energy, _, _ = solvers.vqe(lambda t: ansatz(t[0]), H, [0.0],
                               optimizer="lbfgs", gradient="parameter_shift")
    assert abs(energy - mol.energies["fci_energy"]) < 1e-3
```

## Cleanup

Tests that spawn the PySCF server should clean up:

```python
@pytest.fixture(autouse=True)
def _kill_pyscf_server():
    yield
    # in some envs the server lingers; if so:
    # subprocess.run(["pkill", "-f", "cudaq-pyscf"], check=False)
```

For most tests this is unnecessary; only add if your test exposes
the issue.

## Common pitfalls

| Symptom | Cause |
|---------|-------|
| Test passes alone, fails when full suite runs | global state (env vars, file leftovers); use fixtures |
| `import cudaq_qec` fails | `PYTHONPATH` not set; do `export PYTHONPATH=$CUDAQX_INSTALL_PREFIX` |
| Hangs at `create_molecule` | PySCF server occupied; see chemistry skill |
| Test takes 5 minutes | should be `@pytest.mark.slow`; CI runs slow tests separately |
| Random failure 1-in-10 | RNG not seeded |

## Self-check

```
[ ] Test file follows `test_<feature>.py` naming.
[ ] Each `test_*` function checks one thing.
[ ] RNGs seeded.
[ ] No reliance on test ordering.
[ ] Slow tests marked.
[ ] GPU-only tests skipped on CPU-only runners.
[ ] Test runs in <30s (or is marked slow).
```

## Where next

- C++ tests: `cpp-tests.md`.
- Mirror CI locally: `run-locally.md`.
