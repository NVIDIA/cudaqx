# CUDA-QX skill evaluation harness

This tree is **not** a skill. It evaluates the skills under
`.claude/skills/`. Nothing in this tree is referenced from any `SKILL.md`
or any reference file, so an agent loading a CUDA-QX skill will not see
the prompts, assertions, runners, or graders that live here.

## Layout

```
.claude/evals/
├── prompts/           <skill>.evals.json           ← user-facing prompts only
├── assertions/        <skill>.json                 ← THE answer key (substring rules)
├── runners/
│   └── runner.py                                   ← prompt dump + iteration aggregate
├── graders/
│   ├── programmatic.py                             ← substring + activation grader
│   ├── executable.py                               ← runs code blocks in a sandbox
│   └── judge.py                                    ← LLM-as-judge (BYO client)
├── viewer/
│   └── generate_review.py                          ← HTML report
├── aggregate.py                                    ← cross-grader agreement (Cohen's κ)
├── workspaces/                                     ← per-iteration outputs (gitignored)
└── README.md
```

`prompts/` and `assertions/` are split deliberately: an agent answering an
eval prompt must never read the answer key. Keeping the two on opposite
sides of the directory boundary, and keeping both *outside* the skill
folder, is the cheapest way to enforce that.

## Adding a new skill

1. Add the skill alias to `SKILL_DIRS` in `runners/runner.py` and in every
   grader (each grader is standalone and carries its own copy of the map).
2. Drop `<skill>.evals.json` into `prompts/` and `<skill>.json` into
   `assertions/`. Both files use the same `Sxx`/`Axx` ids.
3. Smoke-test:

   ```bash
   python .claude/evals/runners/runner.py prompts --skill <alias>
   ```

## End-to-end loop (per iteration)

```bash
WORKSPACE=.claude/evals/workspaces/$(date +%Y-%m-%d)-iter-1

# 1. Dump prompts for your evaluator to consume.
python .claude/evals/runners/runner.py prompts --skill qec --kind all > $WORKSPACE/qec.prompts.jsonl

# 2. Your evaluator runs the agent (with and without the skill loaded) and
#    writes responses.json + timing.json into:
#       $WORKSPACE/with_skill/    and    $WORKSPACE/without_skill/

# 3. Grade. Each grader is independent; run as many as you have.
python .claude/evals/graders/programmatic.py --skill qec --responses $WORKSPACE/with_skill/responses.json
python .claude/evals/graders/programmatic.py --skill qec --responses $WORKSPACE/without_skill/responses.json
# (executable.py and judge.py optional; same calling convention.)

# 4. Aggregate per-iteration. Computes deltas between configurations.
python .claude/evals/runners/runner.py aggregate $WORKSPACE

# 5. Cross-grader agreement (Cohen's κ between programmatic / judge / etc).
python .claude/evals/aggregate.py $WORKSPACE

# 6. Render the HTML viewer for human review.
python .claude/evals/viewer/generate_review.py $WORKSPACE --out $WORKSPACE/report.html
```

## Why three graders

| Grader | Catches | Cost | Reliability |
| --- | --- | --- | --- |
| `programmatic.py` | API names, exact paths, exact flags | free | high precision, low recall |
| `executable.py` | "does the suggested code actually run and produce the gold answer" | medium (sandbox) | highest signal for buildable skills |
| `judge.py` | correctness, specificity, hallucinations | $$ (LLM call) | best for subjective dimensions |

Run all three and use `aggregate.py` to compute Cohen's κ between
programmatic and judge. Disagreement is exactly the place where the rubric
or the skill needs work.

## Scoring rubric (human pass)

When a human grader looks at the viewer, score each scenario 0–8:

- **Correctness** (0–2): facts true, paths/APIs real
- **Specificity** (0–2): cites files, exact API names, exact kwargs
- **Coverage** (0–2): hits each `must_include` item
- **No hallucinations** (0–2): no `must_not_include` items

For build-skill responses, add an optional fifth dimension:

- **Action quality** (0–2): did the response identify the right script /
  invocation / fix? (Build prompts often want "run this script" rather
  than a long explanation, and substring scorers undercount that.)

12 scenarios × 8 + 10 activation = 106 max (130 with action quality).

## Token-cost tracking

A skill that delivers higher accuracy at much higher token cost may not be
worth shipping. The evaluator should write `timing.json` alongside
`responses.json`:

```json
{"total_tokens": 84852, "duration_ms": 23332}
```

The HTML viewer surfaces these alongside the grader scores.

## Sources by skill

- **build**: `CMakeLists.txt`, `Building.md`, `scripts/build_*.sh`,
  `scripts/ci/*.sh`, `scripts/validation/*`, `docker/build_env/`,
  `libs/{qec,solvers}/pyproject.toml.cu{12,13}`.
- **qec**: `libs/qec/python/cudaq_qec/__init__.py`,
  `libs/qec/python/bindings/`, `libs/qec/include/cudaq/qec/`,
  `docs/sphinx/components/qec/`, `docs/sphinx/examples/qec/`.
- **solvers**: `libs/solvers/python/cudaq_solvers/__init__.py`,
  `libs/solvers/python/bindings/solvers/`, `libs/solvers/include/`,
  `libs/solvers/python/cudaq_solvers/gqe_algorithm/gqe.py`.
