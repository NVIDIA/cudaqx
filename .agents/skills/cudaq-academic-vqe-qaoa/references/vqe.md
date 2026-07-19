# Minimal VQE Path

Use this for a first VQE example. Keep the answer focused on the algorithm
shape, not chemistry setup.

## Teaching Example

```python
import cudaq
from cudaq import spin
import cudaq_solvers as solvers


@cudaq.kernel
def ansatz(theta: float):
    q = cudaq.qvector(2)
    x(q[0])
    ry(theta, q[1])
    x.ctrl(q[1], q[0])


hamiltonian = (
    5.907
    - 2.1433 * spin.x(0) * spin.x(1)
    - 2.1433 * spin.y(0) * spin.y(1)
    + 0.21829 * spin.z(0)
    - 6.125 * spin.z(1)
)

energy, params, history = solvers.vqe(
    lambda thetas: ansatz(thetas[0]),
    hamiltonian,
    [0.0],
    optimizer="lbfgs",
    gradient="parameter_shift",
    tol=1e-7,
)

print("energy:", energy)
print("params:", params)
```

## Beginner Defaults

- Use a non-empty initial parameter list.
- Use `optimizer="lbfgs"` with `gradient="parameter_shift"`.
- Or omit optimizer/gradient and let the default optimizer path run.
- Return value is `(energy, params, history)`.

## Self Check

- The ansatz argument count matches how `solvers.vqe` calls it.
- Initial parameters are not empty.
- Gradient-based optimizers have a gradient setting.
- The answer mentions `cudaq.kernel`, `spin`, and `solvers.vqe`.

## Source Paths

- Python example: `docs/sphinx/examples/solvers/python/uccsd_vqe.py`
- Python tests: `libs/solvers/python/tests/test_vqe.py`
- C++ API: `libs/solvers/include/cudaq/solvers/vqe.h`
- Python bindings: `libs/solvers/python/bindings/solvers/py_solvers.cpp`
