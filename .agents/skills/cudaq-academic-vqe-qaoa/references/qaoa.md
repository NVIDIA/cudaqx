# Minimal QAOA / MaxCut Path

Use this for a first QAOA example. Keep the answer anchored on MaxCut because
the helper API is easy to explain and easy to test.

## Teaching Example

```python
import numpy as np
import networkx as nx
import cudaq_solvers as solvers

graph = nx.Graph()
graph.add_weighted_edges_from([
    (0, 1, 1.0),
    (1, 2, 2.0),
    (0, 2, 0.5),
])

hamiltonian = solvers.get_maxcut_hamiltonian(graph)
num_layers = 1
num_parameters = solvers.get_num_qaoa_parameters(hamiltonian, num_layers)
initial_parameters = np.zeros(num_parameters)

result = solvers.qaoa(
    hamiltonian,
    num_layers,
    initial_parameters,
    optimizer="cobyla",
)

optimal_value, optimal_parameters, sample_result = result
print("MaxCut value:", -optimal_value)
print("Best bitstring:", sample_result.most_probable())
print("Parameters:", optimal_parameters)
```

## Beginner Defaults

- Use a NetworkX graph for MaxCut.
- Use `solvers.get_maxcut_hamiltonian(graph)`.
- Use `solvers.get_num_qaoa_parameters(...)` instead of guessing parameter
  count.
- Use non-empty initial parameters.
- Use `optimizer="cobyla"` as the safe beginner default.
- `QAOAResult` can be tuple-unpacked as
  `(optimal_value, optimal_parameters, sample_result)`.
- QAOA minimizes the Hamiltonian. For MaxCut, print `-optimal_value` as the
  cut value.

## Pitfall

`optimizer="lbfgs"` requires gradients. The QAOA path does not provide a
gradient instance by default, so beginners should use `cobyla` unless they are
passing a compatible SciPy optimizer with a `jac=`.

## Source Paths

- Python example: `docs/sphinx/examples/solvers/python/molecular_docking_qaoa.py`
- Python tests: `libs/solvers/python/tests/test_qaoa.py`
- C++ API: `libs/solvers/include/cudaq/solvers/qaoa.h`
- Python bindings: `libs/solvers/python/bindings/solvers/py_solvers.cpp`
