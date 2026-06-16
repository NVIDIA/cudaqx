# Evaluation Metrics

Use this when comparing with-skill and without-skill answers for the academic
VQE/QAOA workshop.

## What To Measure

Minimum objective metrics:

- `agent` and `model`: recorded labels for the tool/model used
- `config`: `with_skill` or `without_skill`
- `pass_rate`: percent of prompts passing deterministic assertions
- `coverage_rate`: required concepts present in the answer
- `forbidden_hits`: known wrong or out-of-scope advice
- `context_files`: files loaded by the agent, if available
- `duration_ms`: end-to-end runtime, if available

Compare Codex with-skill against Codex without-skill, Claude against Claude,
and Cursor against Cursor before making cross-tool claims.

## Step-By-Step Evaluation

1. Create templates for one agent/model:

```bash
python3 .agents/evals/academic-vqe-qaoa/compare_runs.py init \
  --agent codex \
  --model gpt-5
```

2. Fill both generated files under `.agents/evals/academic-vqe-qaoa/runs/`.
   This directory is ignored by git because run files are local validation
   artifacts:

- `codex_with_skill.json`: run each prompt with this skill available.
- `codex_without_skill.json`: run each prompt without loading this skill.

3. For every prompt response, paste the answer into `response`. Add
   `duration_ms` and `context_files` when the agent exposes them. Leave
   unavailable fields as `null`.

4. Score and compare:

```bash
python3 .agents/evals/academic-vqe-qaoa/compare_runs.py compare \
  .agents/evals/academic-vqe-qaoa/runs/codex_with_skill.json \
  .agents/evals/academic-vqe-qaoa/runs/codex_without_skill.json
```

5. Use the generated summary under `.agents/evals/academic-vqe-qaoa/runs/` for
   local workshop/report notes. Force-add run artifacts only when a PR
   intentionally needs to preserve a validation snapshot.

## Response File Shape

The evaluator accepts either a simple mapping:

```json
{
  "INSTALL01": "answer text",
  "VQE01": "answer text",
  "QAOA01": "answer text"
}
```

or a metric-bearing structure:

```json
{
  "agent": "codex",
  "model": "gpt-5",
  "config": "with_skill",
  "responses": [
    {
      "id": "QAOA01",
      "response": "answer text",
      "context_files": [
        ".agents/skills/cudaq-academic-vqe-qaoa/SKILL.md",
        ".agents/skills/cudaq-academic-vqe-qaoa/references/qaoa.md"
      ],
      "duration_ms": 3100
    }
  ]
}
```

## Run

```bash
python3 .agents/evals/academic-vqe-qaoa/compare_runs.py init \
  --agent codex \
  --model gpt-5

python3 .agents/evals/academic-vqe-qaoa/compare_runs.py compare \
  .agents/evals/academic-vqe-qaoa/runs/codex_with_skill.json \
  .agents/evals/academic-vqe-qaoa/runs/codex_without_skill.json
```

Use the summary to talk about context efficiency:

- fewer context files loaded
- similar or better answer pass rate
- fewer forbidden hits

## Prompt Set

Prompts live at:

`.agents/evals/academic-vqe-qaoa/prompts.json`

Assertions live at:

`.agents/evals/academic-vqe-qaoa/assertions.json`
