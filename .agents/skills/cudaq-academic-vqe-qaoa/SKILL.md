---
name: cudaq-academic-vqe-qaoa
description: Academic workshop workflow for CUDA-QX Solvers. Use when the user asks for beginner-friendly CUDA-Q Solvers installation, VQE examples, QAOA examples, MaxCut with QAOA, a first ADAPT-VQE example, or a first GQE (generative / GPT-style eigensolver) example. Do not use for QEC, Ising-specific material, advanced chemistry active-space setup, custom operator pools, or CUDA-QX source-build debugging.
---

# CUDA-QX Academic VQE/QAOA

Use this skill to answer workshop-style questions that move from installing
CUDA-QX Solvers to running simple VQE and QAOA examples. Keep responses short,
teachable, and grounded in repo APIs.

## Workflow

1. Identify the user intent.
2. Load exactly one reference file unless the user asks for comparison.
3. Answer with a minimal runnable path first, then mention the source files for
   users who want to inspect the implementation.
4. When evaluating before/after behavior, use the deterministic evaluator in
   `.agents/evals/academic-vqe-qaoa/`.

## Recognizing the problem

If the user describes a problem in domain terms (without naming an algorithm),
first map it to a quantum problem class, then route below:

| User describes... | Problem class | Map to |
| --- | --- | --- |
| Splitting / partitioning / grouping a set where pairwise costs matter (dividing delivery stops, clustering, team assignment) | Graph partition → **MaxCut** | QAOA (`references/qaoa.md`) |
| Finding the lowest-energy / most stable configuration of a molecule | Ground state | VQE / ADAPT-VQE (`references/vqe.md`, `references/adapt.md`) |
| Any other "best discrete choice among many options" problem | Custom Hamiltonian | QAOA (`references/qaoa.md`) |

Do not assume every optimization problem is MaxCut. If the problem does not fit
a class above, say so rather than forcing a fit.

## Intent Routing

| User intent | Read |
| --- | --- |
| Install or smoke test CUDA-QX Solvers | `references/install.md` |
| Build a first VQE example | `references/vqe.md` |
| Build a first QAOA or MaxCut example | `references/qaoa.md` |
| Build a first ADAPT-VQE example | `references/adapt.md` |
| Build a first GQE (GPT-style eigensolver) example | `references/gqe.md` |

## Response Contract

For install questions, include:

- the provided Brev environment as the recommended workshop path
- a note that CPU execution is acceptable for the small VQE/QAOA learning examples
- `pip install cudaq-solvers`
- an import smoke test for `cudaq` and `cudaq_solvers`
- the `libgfortran` note for classical optimizers

For VQE questions, include:

- `cudaq.kernel`
- `cudaq.spin`
- `solvers.vqe`
- initial parameters
- optimizer/gradient guidance

For QAOA questions, include:

- `networkx`
- `solvers.get_maxcut_hamiltonian`
- `solvers.get_num_qaoa_parameters`
- `solvers.qaoa`
- non-empty initial parameters
- `optimizer="cobyla"` as the beginner-safe default

For ADAPT-VQE questions, include:

- `solvers.create_molecule`
- `solvers.get_operator_pool`
- a named pool such as `spin_complement_gsd` with `num_orbitals=`
- a `@cudaq.kernel` `initState` that prepares Hartree-Fock
- the unpacked return `energy, thetas, ops = solvers.adapt_vqe(...)`

For GQE questions, include:

- `pip install cudaq-solvers[gqe]` as the required extra
- the ImportError fallback note (bare `cudaq-solvers` is not enough)
- `from cudaq_solvers.gqe_algorithm.gqe import get_default_config`
- a `cost(sampled_ops, **kwargs)` callback signature
- the unpacked return `energy, indices = solvers.gqe(cost, pool, config=cfg)`

## Source Of Truth

Prefer these repo files for API details:

- `docs/sphinx/quickstart/installation.rst`
- `docs/sphinx/examples/solvers/python/uccsd_vqe.py`
- `docs/sphinx/examples/solvers/python/molecular_docking_qaoa.py`
- `libs/solvers/python/tests/test_vqe.py`
- `libs/solvers/python/tests/test_qaoa.py`
- `libs/solvers/python/bindings/solvers/py_solvers.cpp`
