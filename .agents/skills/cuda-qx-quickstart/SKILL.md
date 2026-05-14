---
name: "cuda-qx-quickstart"
title: "CUDA-QX Quickstart and First-Time Setup"
description: >-
  First-time orientation for CUDA-QX. Use whenever a user is new to the
  repo, asks "how do I get started", "I just pip installed cudaq-qec what
  next", "hello world for cudaqx", "what's the simplest example", or
  needs to choose between pip / Docker / source install. Walks the new
  user end-to-end through a minimal QEC and a minimal solvers example,
  explains what they see, and then delegates to the deeper workflow
  skills (cuda-qx-qec-decode, cuda-qx-solvers-algorithms, cuda-qx-build).
version: "0.1.2"
author: "CUDA-QX"
license: "Apache License 2.0"
compatibility: "Python 3.11+, Linux x86_64/aarch64 (pip), macOS (Docker only)"
tags: [cuda-qx, quickstart, onboarding, tutorial, hello-world, students, install]
tools: [Read, Glob, Grep, Bash]
metadata:
  repo: [qec, solvers]
  author: "CUDA-QX"
  domain: "onboarding"
  audience: [student, researcher, first-time-user]
  languages: [python]
---

# CUDA-QX Quickstart

The "you just discovered this repo, now what?" entry point. The three
workflow skills (`cuda-qx-build`, `cuda-qx-qec-decode`, `cuda-qx-solvers-algorithms`)
assume you already know what you want to do. This skill takes a user
from zero to a working example in under 15 minutes, then routes them
into the right workflow skill.

## Inputs

Caller provides:

- The user's environment state: do they have a cloned repo? Docker?
  Mac vs Linux? GPU vs CPU-only?
- Interest area: QEC, solvers, both, or undecided.

## Outputs

This skill produces:

- An identified install path (pip / Docker / source) and confirmed
  working install.
- One worked example with interpreted output (LER for QEC, energy for
  solvers).
- A delegation to the right specialist skill for the user's next step.

Does NOT produce: deep workflows, decoder tuning, ansatz construction
(→ specialist skills). This skill is a router + first example, not a
full workflow.

## Audience

Three concentric audiences, in order of likely arrival:

1. **Pip-only user** (student, researcher who wants to try cudaqx):
   `pip install cudaq-qec` or `pip install cudaq-solvers`, run a Python
   example, read the published docs at <https://nvidia.github.io/cudaqx/>.
   They cannot read local repo files.
2. **Docker user** (Mac, locked-down workstation, or "I want it to just
   work"): `docker run --gpus all -it ghcr.io/nvidia/cudaqx`.
3. **Source user** (contributor, advanced researcher who needs to
   recompile): clone the repo, follow `Building.md`, delegate to
   `cuda-qx-build`.

Detect which the user is in and route accordingly. The single biggest
mistake is throwing source-build commands at a pip-only user.

## Decision tree (run this first)

```
Q1: Has the user cloned the cudaqx repo? (look for `libs/qec` in cwd)
    yes -> source user, see references/source-user.md
    no  -> Q2

Q2: Is this a Mac, or do they say "I just want to try it"?
    yes -> Docker user, see references/docker-user.md
    no  -> Q3

Q3: Do they have NVIDIA GPU drivers? (`nvidia-smi` works)
    yes -> Pip user with GPU, see references/pip-user.md
    no  -> Pip user, CPU only — still works for code-capacity QEC and
           small solver examples. See references/pip-user.md anyway.
```

Always confirm the user's path before running install commands. Pip
install on a system that already has a source build creates two copies
of `cudaq_qec` on `PYTHONPATH` and confuses imports.

## The 15-minute walkthrough

After install, hand the user **one** working example for the area they
care about. Resist the temptation to dump three.

| Interest | Run this | Reference |
|----------|----------|-----------|
| Quantum error correction | `python docs/sphinx/examples/qec/python/code_capacity_noise.py` | `references/first-qec-example.md` |
| Variational solvers / chemistry | `python docs/sphinx/examples/solvers/python/uccsd_vqe.py` | `references/first-solvers-example.md` |
| Both / undecided | Start with QEC code-capacity — it has no chemistry deps | `references/first-qec-example.md` |

Pip-only users do not have a clone of the repo. For them, paste the
example file from the published docs page at
<https://nvidia.github.io/cudaqx/examples_rst/qec/code_capacity_noise.html>
into a local `.py` file and run that. (There is no top-level
`examples/` directory in the cudaqx repo — only `docs/sphinx/examples/`.)

## What success looks like

After the walkthrough the user should be able to answer three questions
without re-reading anything:

1. Which library am I using — `cudaq_qec`, `cudaq_solvers`, or both?
2. Where do I find more examples? (`docs/sphinx/examples/{qec,solvers}/`
   in source, or <https://nvidia.github.io/cudaqx/> for pip users.)
3. Which workflow skill do I open next? (`cuda-qx-qec-decode`,
   `cuda-qx-solvers-algorithms`, or `cuda-qx-build` for build issues.)

If they cannot, the example did not land — back up and re-explain
output, not commands.

## Key concepts (the one-paragraph version)

A user who has never seen QEC or VQE may not know what they are looking
at. Have these one-liners ready and offer them when output is confusing:

- **QEC, logical error rate (LER)**: a quantum error-correcting code
  protects "logical" qubits behind many physical qubits. The LER is the
  probability that, despite correction, the logical qubit still flips.
  Lower is better; the goal is LER lower than the physical error rate.
- **Stabilizer / syndrome**: stabilizers are extra measurements that
  detect (but do not collapse) errors. A "syndrome" is what those
  measurements report; the decoder turns it into a correction.
- **Decoder**: the classical algorithm that converts a syndrome into a
  correction. `single_error_lut`, `nv-qldpc-decoder`,
  `tensor_network_decoder`, etc. — different speed/accuracy tradeoffs.
- **VQE (variational quantum eigensolver)**: a quantum circuit with
  free parameters; a classical optimizer adjusts the parameters until
  the circuit's expectation value matches a target Hamiltonian's ground
  energy. Used heavily in quantum chemistry.
- **ADAPT-VQE**: VQE that grows the ansatz one operator at a time,
  picking from an operator pool. More accurate, more expensive.
- **QAOA**: VQE-shaped but for combinatorial optimization (MaxCut,
  QUBO, etc.).
- **GQE (generative quantum eigensolver)**: a transformer learns to
  *propose* operator sequences. Multi-GPU, optional extra
  (`pip install cudaq-solvers[gqe]`).

These belong in `references/concepts.md` in expanded form.

## When to delegate

| If the user wants to                              | Delegate to                                |
|---------------------------------------------------|--------------------------------------------|
| Pick a decoder, run a circuit-level experiment    | `cuda-qx-qec-decode` SKILL.md                     |
| Tune VQE, ADAPT, QAOA, GQE, or chemistry          | `cuda-qx-solvers-algorithms` SKILL.md                 |
| Fix an install / build / docker / wheel issue     | `cuda-qx-build` SKILL.md                   |
| Write a new code or decoder                       | `cuda-qx-qec-extending` SKILL.md           |
| Write a new operator pool / optimizer             | `cuda-qx-solvers-extending` SKILL.md       |
| Set up real-time decoding on hardware             | `cuda-qx-qec-realtime` SKILL.md       |

Do not duplicate workflow content here. This skill is a router.

## Additional resources

- `references/pip-user.md` — pip install path, CPU/GPU, `libgfortran`,
  CUDA 12 vs 13 wheels.
- `references/docker-user.md` — `ghcr.io/nvidia/cudaqx`, `--gpus all`,
  notebook port forwarding, common Mac caveats.
- `references/source-user.md` — clone, dev container, `Building.md`
  pointers; delegates to `cuda-qx-build`.
- `references/first-qec-example.md` — code-capacity walkthrough with
  output interpretation.
- `references/first-solvers-example.md` — H2 VQE walkthrough.
- `references/concepts.md` — one-page glossary.
