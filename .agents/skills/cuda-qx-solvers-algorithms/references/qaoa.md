# QAOA: graphs, MaxCut, QUBO, mixers

How to bootstrap a QAOA workflow on a graph problem and the recurring
configuration mistakes. For VQE/ADAPT see `vqe.md`; for chemistry inputs
see the related skill `cuda-qx-solvers-chemistry`
(`references/molecule-building.md`).

## Two overloads

```python
solvers.qaoa(Hp, Href, num_layers, init_params, **opts)
solvers.qaoa(Hp, num_layers, init_params, **opts)  # default mixer
```

Common pitfalls:

- `optimizer="lbfgs"` fails: it requires gradients but no gradient
  instance is provided to the QAOA path. Use `cobyla` or pass a SciPy
  optimizer with a `jac=`.
- Empty `init_params` raises
  `RuntimeError("qaoa initial parameters empty.")`.
- Extra kwargs: `full_parameterization=True`, `counterdiabatic=True`.
- `QAOAResult` is tuple-unpackable: `optval, optp, config = result`.
  `config` is a `cudaq.SampleResult`.

## Helpers

- `solvers.get_num_qaoa_parameters(H, num_layers, **kwargs)`
- `solvers.get_maxcut_hamiltonian(nx_graph)`
- `solvers.get_clique_hamiltonian(nx_graph, penalty=4.0)`

Graph helpers take **NetworkX graphs only** and read `weight` attrs from
nodes/edges.

## Minimum MaxCut script

```python
import networkx as nx
import cudaq_solvers as solvers

g = nx.Graph()
g.add_weighted_edges_from([(0, 1, 1.0), (1, 2, 2.0), (0, 2, 0.5)])

H = solvers.get_maxcut_hamiltonian(g)
num_layers = 1
n_params = solvers.get_num_qaoa_parameters(H, num_layers)
init = [0.0] * n_params

result = solvers.qaoa(H, num_layers, init, optimizer="cobyla")
optval, optp, config = result
print("MaxCut value:", -optval)         # qaoa minimizes; cut is the negative
print("Best string :", config.most_probable())
```

## Self-check (QAOA-specific)

```
[ ] init_params is non-empty (RuntimeError otherwise).
[ ] optimizer is cobyla, OR a scipy.optimize.minimize with `jac=`.
[ ] If clique: penalty argument is set (default 4.0; tune for problem).
[ ] If counterdiabatic=True: read solvers.qaoa source to confirm
    parameter count expected by your ansatz.
```

## Canonical tests

| Topic         | File                                                  |
|---------------|-------------------------------------------------------|
| QAOA + graph  | `libs/solvers/python/tests/test_qaoa.py`              |
| QAOA (C++)    | `libs/solvers/unittests/test_qaoa.cpp`                |
