# GQE (Generative Quantum Eigensolver)

Use this reference when the user asks about the GPT-style / transformer-based
quantum eigensolver in CUDA-Q Solvers. Source of truth:
`libs/solvers/python/tests/test_gqe.py`, `docs/sphinx/examples/solvers/python/gqe_h2.py`,
and `libs/solvers/python/cudaq_solvers/__init__.py` (ImportError fallback).

## Install

GQE is an **optional extra**. You must install the `[gqe]` extra to pull in
torch / transformers / lightning:

```bash
pip install cudaq-solvers[gqe]
```

Without it, `solvers.gqe(...)` raises a stub ImportError pointing at the same
command (see `cudaq_solvers/__init__.py`). The `cudaq-solvers` install on its
own (no extras) does **not** include GQE.

GQE training benefits from CUDA-enabled PyTorch. Without it, several tests are
marked `requires_cuda_kernels` and are skipped. The workshop demo can still
import GQE on a CPU box, but training will be slow.

## API surface

| Symbol | Notes |
| --- | --- |
| `from cudaq_solvers.gqe_algorithm.gqe import get_default_config` | Returns a config dataclass with sane defaults |
| `solvers.gqe(cost, pool, config=cfg, ...)` | Returns `(energy, indices)`; `indices` are pool positions of the chosen ansatz |
| `cost(sampled_ops, **kwargs)` | User-defined callback; should return a real expectation value |

## Minimal recipe (Z0 + Z1)

```python
import cudaq
from cudaq import spin
import cudaq_solvers as solvers
from cudaq_solvers.gqe_algorithm.gqe import get_default_config

qubit_count = 2
ham = spin.z(0) + spin.z(1)

def ops_pool(n):
    pool = []
    for i in range(n):
        pool.append(cudaq.SpinOperator(spin.x(i)))
        pool.append(cudaq.SpinOperator(spin.y(i)))
        pool.append(cudaq.SpinOperator(spin.z(i)))
    for i in range(n - 1):
        pool.append(cudaq.SpinOperator(spin.z(i) * spin.z(i + 1)))
    return pool

pool = ops_pool(qubit_count)

def term_coefficients(op): return [t.evaluate_coefficient() for t in op]
def term_words(op): return [t.get_pauli_word(qubit_count) for t in op]

@cudaq.kernel
def kernel(qcount: int, coeffs: list[float], words: list[cudaq.pauli_word]):
    q = cudaq.qvector(qcount)
    h(q)
    for i in range(len(coeffs)):
        exp_pauli(coeffs[i], q, words[i])

def cost(sampled_ops, **kwargs):
    full_coeffs, full_words = [], []
    for op in sampled_ops:
        full_coeffs += [c.real for c in term_coefficients(op)]
        full_words += term_words(op)
    return cudaq.observe(kernel, ham, qubit_count, full_coeffs, full_words).expectation()

cfg = get_default_config()
cfg.num_samples = 5
cfg.max_iters = 25
cfg.ngates = 4
cfg.seed = 3047
cfg.lr = 1e-6

energy, indices = solvers.gqe(cost, pool, config=cfg)
print(energy, indices)
```

## Mandatory beginner footguns

- Without `pip install cudaq-solvers[gqe]`, `solvers.gqe(...)` raises
  `ImportError: Failed to load GQE solver due to missing dependencies.` Do
  **not** tell users plain `pip install cudaq-solvers` is enough.
- `cost` receives `sampled_ops: list[SpinOperator]` plus `**kwargs`. You must
  accept `**kwargs` or older training paths will fail when they pass
  `qpu_id` for MQPU.
- `config` is a dataclass from `cudaq_solvers.gqe_algorithm.gqe.get_default_config()`
  — do not invent your own. Required positive-valued fields:
  `num_samples > 0`, `lr > 0`, `temperature > 0` (validated, see
  `test_gqe.py::test_invalid_inputs`).
- Return tuple is `(energy, indices)`, where `indices` are positions inside
  the pool you passed. Not parameters, not operators.
- Optional `loss="gflow"` switches to GFlow-style loss.
