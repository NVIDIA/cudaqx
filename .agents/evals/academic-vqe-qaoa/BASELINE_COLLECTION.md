# Academic VQE/QAOA Baseline Collection

Use this guide to collect `codex_without_skill.json` responses without
contaminating the baseline with the `cudaq-academic-vqe-qaoa` skill context.

## Goal

Create the local run templates:

```bash
python3 .agents/evals/academic-vqe-qaoa/compare_runs.py init \
  --agent codex \
  --model gpt-5
```

Then fill:

```text
.agents/evals/academic-vqe-qaoa/runs/codex_without_skill.json
```

with real answers from a clean no-skill run. Make sure the paired with-skill
file also contains real with-skill responses before using the comparison as a
before/after result:

```text
.agents/evals/academic-vqe-qaoa/runs/codex_with_skill.json
```

## Clean Baseline Rules

- Use a fresh Codex thread or another model session that has not read
  `.agents/skills/cudaq-academic-vqe-qaoa/SKILL.md`.
- Do not paste the assertions, metrics reference, or with-skill answers into
  that no-skill session.
- Ask each prompt exactly once.
- Paste the model's full answer into the matching `response` field.
- Leave duration fields as `null` unless the tool exposes exact values.
- Keep `context_files` empty unless the no-skill agent explicitly read files.

## Prompts

### INSTALL01

```text
I am new to CUDA-QX Solvers. What should I install and how do I quickly check that I can run VQE or QAOA examples?
```

### VQE01

```text
Show me a minimal CUDA-Q Solvers VQE example with a small ansatz, a SpinOperator Hamiltonian, and the right optimizer and gradient settings.
```

### QAOA01

```text
Show me the simplest QAOA MaxCut workflow in CUDA-Q Solvers using a small weighted NetworkX graph.
```

### QAOA02

```text
Why does solvers.qaoa(..., optimizer='lbfgs') fail with a gradient error, and what should a beginner use instead?
```

## Score The Baseline

The `runs/` directory is ignored by git because these files are local
validation artifacts. If the run files do not exist, create them with the
`init` command above.

To avoid hand-editing JSON, capture pasted answers into the no-skill run file:

```bash
python3 .agents/evals/academic-vqe-qaoa/record_responses.py \
  .agents/evals/academic-vqe-qaoa/runs/codex_without_skill.json \
  --only-empty
```

For each prompt, paste the no-skill model's answer and end it with:

```text
<<<END>>>
```

From the repo root:

```bash
python3 .agents/evals/academic-vqe-qaoa/evaluate_metrics.py \
  .agents/evals/academic-vqe-qaoa/runs/codex_with_skill.json \
  .agents/evals/academic-vqe-qaoa/runs/codex_without_skill.json
```

Then write the paired summary:

```bash
python3 .agents/evals/academic-vqe-qaoa/compare_runs.py compare \
  .agents/evals/academic-vqe-qaoa/runs/codex_with_skill.json \
  .agents/evals/academic-vqe-qaoa/runs/codex_without_skill.json \
  --out .agents/evals/academic-vqe-qaoa/runs/codex-comparison-summary.json
```

## Local Validation Example

When the with-skill responses are populated from this workflow, one local
validation run scored:

```text
[codex:with_skill] pass_rate=100% coverage=21/21 forbidden=0 context_files=8
```

Treat this as an example validation result, not a clean-checkout guarantee. The
no-skill score should only be reported after `codex_without_skill.json` contains
real no-skill responses.
