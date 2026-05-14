# Skill evals

`.agents/evals/` is the evaluation harness for skills. Use it to
keep regressions out: a skill that *says* it does X should
demonstrably *produce* X when an agent invokes it.

## Layout

```
.agents/evals/
├── README.md          # how to run
├── REPORT.md          # last report
├── aggregate.py       # produces REPORT.md from runner outputs
├── prompts/           # eval prompts (user-facing prompts to test routing + execution)
├── assertions/        # post-run checks (file exists, output contains, etc.)
├── graders/           # graders (LLM-based or scripted)
├── runners/           # per-agent runners (Codex, Claude Code, Cursor, ...)
└── viewer/            # local viewer for reports
```

Start by reading `.agents/evals/README.md` — it documents the
runner contract and how to add an eval.

## When to add an eval

| Skill change | Eval needed? |
|--------------|--------------|
| New top-level skill | yes |
| New reference in an existing skill | not required, but recommended for new workflows |
| Wording / clarification edits | no |
| Path or API name updates | yes (catches regressions if a path moves) |
| Trigger-phrase changes in `description` | yes (catches routing regressions) |

## Anatomy of an eval

1. **Prompt** (`prompts/<id>.txt` or `.md`) — what the agent is
   asked. Phrased as a user would phrase it.
2. **Assertion** (`assertions/<id>.py` or `.json`) — script or
   declarative check on what the agent produced.
3. **Grader** (`graders/<id>.*`) — optional LLM-as-judge stage that
   scores subjective things (e.g., "did the answer cite the right
   reference?").
4. **Runner** (`runners/<agent>/`) — per-agent wiring; runs the
   prompt through that agent, captures outputs, hands to
   assertion + grader.

The exact contract is in `.agents/evals/README.md`; read it before
adding files.

## Writing a useful prompt

- Phrase it the way the user actually would (typos and all, when
  realistic).
- Test the skill's *routing* (the description should activate this
  skill) and its *content* (the skill should produce the right
  output).
- Avoid contrived prompts that no real user would write.

Good prompt example:

```
my decoder hangs after the first round of stabilizer measurement.
i'm running real-time on Quantinuum. what should i check?
```

(Tests routing to `cuda-qx-qec-realtime` and the in-kernel
procedure.)

Bad prompt example:

```
explain QEC
```

(Too vague; doesn't test any specific skill.)

## Assertions

Two flavors:

1. **Hard checks** — file `X` exists, output contains string `Y`,
   command `Z` exits 0. Pythonic asserts; deterministic.
2. **Soft checks** — graded by an LLM ("did the response cite the
   right reference?"). Less deterministic; combine with hard checks
   to detect drift.

Mix both. Hard checks catch failures cheaply; graders catch
regressions in style and routing.

## Running evals

See `.agents/evals/README.md`. Typical:

```bash
cd .agents/evals
python runners/claude-code/run.py --prompts prompts/qec-realtime/
python aggregate.py
# Open viewer/index.html
```

## CI integration

Evals are not part of the standard PR CI (they're slow and call
external LLMs). They run on a nightly or pre-release schedule.

For routine PRs, the format/sync/build/test gates are enough.

## Common pitfalls

| Symptom | Cause |
|---------|-------|
| Eval passes locally, fails on the runner | environment / API key issue on the runner |
| Eval flaky across runs | LLM-as-judge nondeterminism; tighten the grader prompt |
| New skill not getting routed in evals | weak `description` field; add more trigger phrases |
| Hard assertion failing for the wrong reason | check the captured output; adjust the assertion to match real output, not idealized |

## Self-check

```
[ ] Eval prompt written from a real-user POV.
[ ] At least one hard assertion (deterministic).
[ ] If using a grader: prompt is tight and consistent.
[ ] Eval ID linked from the skill's SKILL.md (optional but useful).
[ ] Runner script for at least one agent (Claude Code or Codex).
```

## Where next

- Skill authoring: `new-skill.md`, `edit-skill.md`.
- Onboarding a new agent: `add-target-agent.md`.
