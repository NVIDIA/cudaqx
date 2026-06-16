---
name: cudaq-academic-vqe-qaoa
description: Academic workshop workflow for CUDA-QX Solvers. Use when the user asks for beginner-friendly CUDA-Q Solvers installation, VQE examples, QAOA examples, MaxCut with QAOA, or before/after skill metrics for VQE/QAOA agent responses. Do not use for QEC, GQE, ADAPT-VQE, Ising-specific material, advanced chemistry active-space setup, custom operator pools, or CUDA-QX source-build debugging.
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
4. When evaluating before/after behavior, use the metrics reference and the
   deterministic evaluator.

## Intent Routing

| User intent | Read |
| --- | --- |
| Install or smoke test CUDA-QX Solvers | `references/install.md` |
| Build a first VQE example | `references/vqe.md` |
| Build a first QAOA or MaxCut example | `references/qaoa.md` |
| Compare with-skill vs without-skill answers | `references/metrics.md` |

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

For evaluation questions, report:

- pass rate against deterministic assertions
- required concept coverage
- forbidden concept hits
- context files loaded
- runtime when available

## Source Of Truth

Prefer these repo files for API details:

- `docs/sphinx/quickstart/installation.rst`
- `docs/sphinx/examples/solvers/python/uccsd_vqe.py`
- `docs/sphinx/examples/solvers/python/molecular_docking_qaoa.py`
- `libs/solvers/python/tests/test_vqe.py`
- `libs/solvers/python/tests/test_qaoa.py`
- `libs/solvers/python/bindings/solvers/py_solvers.cpp`
