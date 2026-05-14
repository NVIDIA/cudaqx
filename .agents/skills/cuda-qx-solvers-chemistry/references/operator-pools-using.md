# Operator pools

The menu of operators ADAPT-VQE (and GQE) picks from. Choosing the
wrong pool — or passing `num_qubits` where `num_orbitals` is expected
— is the single most common silent-correctness bug in chemistry
workflows.

## API

```python
ops = solvers.get_operator_pool(name, **config)
```

Returns a list of `cudaq.SpinOperator` instances.

## The five chemistry pools

| Pool | Required kwargs | Notes |
|------|-----------------|-------|
| `uccsd` | `num_qubits`, `num_electrons` | Standard chemistry ADAPT pool. Singles + doubles. |
| `uccgsd` | `num_orbitals` | Generalized: includes non-particle-conserving excitations. More expressive, deeper. |
| `upccgsd` | `num_orbitals` | Paired doubles only. Compact pool, often near-optimal for small molecules. |
| `spin_complement_gsd` | `num_orbitals` | Spin-symmetric, robust under noise. |
| `ceo` | `num_orbitals` | Coupled exchange operators. Compact, often outperforms UCCSD for ADAPT in moderate-size molecules. |

(Source: `libs/solvers/include/cudaq/solvers/operators/operator_pools/`.)

## The optimization pool

| Pool | Required kwargs | Use for |
|------|-----------------|---------|
| `qaoa` | (see source) | ADAPT-QAOA mixer pool for combinatorial optimization. |

For chemistry, this is **not** what you want.

## `num_qubits` vs `num_orbitals` — the silent bug

`num_qubits = 2 * num_orbitals` (Jordan-Wigner or Bravyi-Kitaev with
no qubit reduction). Mixing them up:

```python
# WRONG: UCCSD takes num_qubits, you passed num_orbitals
ops = solvers.get_operator_pool("uccsd",
                                num_qubits=mol.n_orbitals,  # bug
                                num_electrons=mol.n_electrons)

# WRONG: UCCGSD takes num_orbitals, you passed num_qubits
ops = solvers.get_operator_pool("uccgsd",
                                num_orbitals=2*mol.n_orbitals)  # bug
```

Both compile, both produce *some* operator list, neither is the
right pool. Always check:

```python
n_qubits = 2 * mol.n_orbitals
assert n_qubits == op.num_qubits          # `op` is the SpinOperator from jordan_wigner
```

## Choosing a pool

For small molecules and prototyping:

- `uccsd` — works, well-understood, big enough for H2 / LiH / BeH2.
- `upccgsd` — smaller, often as accurate as UCCSD for small systems.

For moderate-size molecules (8-16 qubits, ADAPT):

- `ceo` — usually competitive with UCCSD at fewer ADAPT iterations.
- `spin_complement_gsd` — best when spin symmetry must be preserved
  exactly.
- `uccgsd` — most expressive; converges in fewer iterations but each
  iteration is more expensive.

For GQE (transformer):

- The transformer learns to *propose* operator sequences, so the pool
  size affects the action space. UCCSD is the canonical starting
  pool.

## Common pitfalls

| Symptom | Cause |
|---------|-------|
| ADAPT converges to wrong energy | `num_qubits` vs `num_orbitals` mismatch |
| Operator pool empty / too small | wrong kwargs for the chosen pool |
| ADAPT picks the same operator repeatedly | gradient calculation bug; check sign convention of pool operators |
| Pool size explodes for big molecule | use UPCCGSD or CEO instead of UCCGSD |

## Self-check

```
[ ] Pool kwargs match the table (num_qubits vs num_orbitals).
[ ] Operator pool size is sane (UCCSD: O(n^2), UCCGSD: O(n^4)).
[ ] Spot-check one pool operator: assert it acts on the right qubits.
[ ] At HF state, expectation of each pool operator ~ 0 (Brillouin's theorem
    holds for canonical orbitals; gradients flag the active excitations).
```

## Where next

- Use the pool in ADAPT: `cuda-qx-solvers-algorithms/references/vqe.md` "ADAPT-VQE".
- Use the pool in GQE: `cuda-qx-solvers-algorithms/references/gqe.md`.
- Compare ADAPT depth/accuracy across pools: `cuda-qx-benchmarking`.
