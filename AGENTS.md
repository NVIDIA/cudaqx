# CUDA-QX Agent Skills

This branch carries a small academic-facing skill slice for CUDA-QX Solvers.
It is intentionally independent from the larger skills architecture PR.

## Skills

| Skill | Purpose |
| --- | --- |
| `cudaq-academic-vqe-qaoa` | Basic install, VQE, and QAOA workflows for academic workshop examples |

The skill source lives under `.agents/skills/`. Evaluation prompts, assertions,
and lightweight metrics tooling live under `.agents/evals/academic-vqe-qaoa/`.

## Scope

Keep this branch focused on:

- installing and smoke-testing `cudaq-solvers`
- a minimal VQE workflow
- a minimal QAOA / MaxCut workflow
- objective before/after metrics for agent responses

Avoid expanding this PR into QEC, GQE, chemistry active-space design, custom
operators, or the full multi-agent mirror infrastructure.
