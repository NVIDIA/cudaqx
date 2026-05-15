# Agent Instructions

This repository ships skills for AI coding agents that work with CUDA-Q Libraries.
The skills cover building/packaging the libraries, quantum error correction
(QEC) workflows, and solver (VQE / ADAPT-VQE / QAOA / GQE) workflows.

This file is the entry point for any agent. Agent-specific loaders find
the skills in their conventional location (see "Where skills live" below);
the actual skill content is identical across locations.

## Where skills live

| Agent       | Discovery path                       | Notes                                    |
|-------------|--------------------------------------|------------------------------------------|
| Codex       | `.agents/skills/<name>/SKILL.md`     | Canonical — edit here. Open AGENTS.md convention. |
| Claude Code | `.claude/skills/<name>/SKILL.md`     | Generated mirror                         |
| Cursor      | `.cursor/skills/<name>/SKILL.md`     | Generated mirror                         |

The canonical content lives at **`.agents/skills/`** — the open
AGENTS.md convention that Codex (and other tools) discover natively. The
`.claude/skills/` and `.cursor/skills/` trees are generated mirrors and
tracked so a fresh clone is immediately usable by Claude Code and Cursor.
Run the sync after editing canonical content:

```bash
bash scripts/sync_agents_skills.sh
```

Edit `.agents/skills/` and re-run the sync whenever you change a skill —
`.claude/skills/` and `.cursor/skills/` are regenerated each time.

To add a fourth agent (Copilot, Gemini, Windsurf, etc.), append one
entry to `TARGETS` in `scripts/sync_agents_skills.sh`. No skill
content needs to be duplicated by hand.

## Available skills

Skills are grouped by domain:

- **Cross-cutting** — apply to both libraries or to the repo as a whole.
- **QEC family** — sub-domains of `cudaq-qec-decode`. All carry the `cudaq-qec-*` prefix.
- **Solvers family** — sub-domains of `cudaq-solvers-algorithms`. All carry the `cudaq-solvers-*` prefix.

The filesystem is flat (every skill lives at `.agents/skills/<name>/SKILL.md`)
because that's what each agent's discovery expects; the hierarchy is expressed
in names, not in folder nesting.

### Cross-cutting

| Skill                       | What it covers                                                                       |
|-----------------------------|--------------------------------------------------------------------------------------|
| `cudaq-quickstart`        | onboarding, first example, pip vs Docker vs source, concept glossary                 |
| `cudaq-build`             | cmake / ninja / wheels / docs / dev container / release validation                   |
| `cudaq-benchmarking`      | LER sweeps, pseudo-thresholds, solver energy comparisons, reproducibility            |
| `cudaq-profiling-perf`    | NVTX, `nsys`, `ncu`, PyTorch profiler, CUDA Graphs                                   |
| `cudaq-contributing`      | DCO sign-off, PR workflow, formatting (clang-format, yapf), issue reporting, pytest/GoogleTest/ctest, CI subsets, container/wheel validation |
| `cudaq-skills-authoring`  | meta-skill: editing the .agents/skills/ tree, sync, evals, target agents             |

### QEC family

| Skill                       | What it covers                                                                       |
|-----------------------------|--------------------------------------------------------------------------------------|
| `cudaq-qec-decode`               | decoders, codes (Steane / surface / repetition), DEMs, real-time decoding            |
| `cudaq-qec-realtime`      | in-kernel API, autonomous_decoder, AI predecoder pipeline, Quantinuum Helios         |
| `cudaq-qec-ai-decoders`   | train neural decoders, ONNX export, TensorRT engine, `trt_decoder` + hybrid deployment |
| `cudaq-qec-extending`     | add a new QEC code or decoder (Python or C++)                                        |

### Solvers family

| Skill                          | What it covers                                                                       |
|--------------------------------|--------------------------------------------------------------------------------------|
| `cudaq-solvers-algorithms`              | VQE, ADAPT-VQE, QAOA, GQE algorithm dispatch                                         |
| `cudaq-solvers-chemistry`    | PySCF, basis sets, active spaces, fermion-to-qubit, operator pools, baselines        |
| `cudaq-solvers-extending`    | add a new operator pool, optimizer, state-prep kernel, or gradient method            |

Each `SKILL.md` carries its own activation triggers in the YAML
frontmatter, a workflow index, conventions to follow, and a self-check
protocol. Start there.

## Skills as context boundaries

Each skill is designed as a **context boundary**: it has explicit
inputs (what a caller must provide), explicit outputs (what it
produces), and stays on its own concerns. Cross-skill content is
delegated by reference, not embedded inline. This makes skills safe
to invoke as **sub-agents** with isolated context windows.

### What this means for orchestration

1. **One skill per dispatch.** The orchestrator picks one skill based
   on the user's intent, loads only that skill's `SKILL.md` + the
   relevant `references/` file. Cross-cutting concerns (build,
   benchmarking) are separate dispatches.

2. **Read each skill's `## Inputs` and `## Outputs` sections.** They
   define the contract. If the work needs inputs the skill doesn't
   take, dispatch a different skill first to produce them.

3. **Don't dump raw notes between skills.** When you finish one skill
   and start another, pass only the curated output (e.g., a
   `MolecularHamiltonian` object from `cudaq-solvers-chemistry`
   → `cudaq-solvers-algorithms`), not the full transcript.

### Sub-agent invocation (Claude Code Agent tool)

For workflows where multiple skills participate, prefer dispatching
sub-agents (Claude Code's `Agent` tool with a custom `subagent_type`
per skill) over loading multiple skills into the main context. Each
sub-agent:

- Loads only its skill into a fresh context window.
- Reads only the relevant `references/<workflow>.md`.
- Returns a concise summary to the orchestrator (not its working
  notes).

Example: a "benchmark this new decoder" request becomes:

```
1. Dispatch cudaq-qec-extending  -> returns: registered plugin name
2. Dispatch cudaq-qec-decode            -> returns: shots + DEM
3. Dispatch cudaq-benchmarking   -> returns: comparison table + plots
```

The main agent only sees the three summaries, not the 1000+ lines of
skill content that produced them.

### Personas and entry points

| Persona | Start here |
|---------|-----------|
| Student / first-time user | `cudaq-quickstart` |
| QEC researcher | `cudaq-qec-decode` → `cudaq-qec-realtime` / `cudaq-qec-ai-decoders` |
| Quantum chemist / VQE practitioner | `cudaq-solvers-chemistry` → `cudaq-solvers-algorithms` |
| Hardware / real-time engineer | `cudaq-qec-realtime` → `cudaq-profiling-perf` |
| Algorithm researcher (any) | domain skill → `cudaq-benchmarking` |
| QEC plugin author (codes / decoders) | `cudaq-qec-extending` |
| Solvers plugin author (pools / optimizers) | `cudaq-solvers-extending` |
| External contributor | `cudaq-contributing` |
| Skill author / maintainer | `cudaq-skills-authoring` |

## Shared resources

| Need                                            | Path                                           |
|-------------------------------------------------|-------------------------------------------------|
| Preflight / import-smoke / workflow picker      | `.agents/skills/_shared/scripts/`               |
| Repo map (where things live in the source tree) | `.agents/skills/_shared/repo_map.md`            |
| Evaluation harness (prompts, assertions, graders)| `.agents/evals/`                               |
| Top-level build / format / validation scripts   | `scripts/`                                      |

## Editing the skills

1. Edit files under `.agents/skills/<name>/`.
2. Run `bash scripts/sync_agents_skills.sh` to regenerate the
   `.claude/skills/` and `.cursor/skills/` mirrors.
3. Commit the canonical `.agents/skills/` changes and the regenerated
   mirrors together so users do not need to run sync manually after clone.
