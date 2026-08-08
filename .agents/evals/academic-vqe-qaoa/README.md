# CUDA-QX Academic VQE/QAOA Workshop

This directory holds the prompts, assertions, and scorer for a small
workshop comparing skill-equipped vs skill-free coding-agent behavior
on CUDA-QX Solvers questions (VQE, QAOA, ADAPT-VQE, GQE).

The skill itself lives at:
`../../skills/cudaq-academic-vqe-qaoa/`

## Running the demo

The demo is a controlled A/B on one prompt: run it **without** the skill,
then **with** it, and compare three things you can read off directly —
*recognition* (which algorithm the agent recommends), *exploration cost*
(files read / tool calls, shown in the agent UI), and *tokens*.

1. **Pick a prompt** from `prompts.json` (P1–P8). P7/P8 best show the
   recognition benefit; P1–P5 best show the cost benefit.
2. **Toggle the skill — this is the *only* thing you change between the
   two runs.** Skill discovery is model-invoked and differs per tool, so
   the reliable control is to *explicitly invoke* the skill in the
   with-skill run and *not mention it* (and keep it out of reach) in the
   without-skill run:

   | Runtime | Without skill | With skill |
   | --- | --- | --- |
   | Cursor | Fresh chat; don't reference the skill (or disable skills in settings) | Start your message with `Use the cudaq-academic-vqe-qaoa skill.` then the prompt |
   | Claude Code | Run from a checkout with `.claude/skills/` removed (or a scratch dir) | Run from the repo root, start with `Use the cudaq-academic-vqe-qaoa skill.` |
   | Codex | Fresh session in a dir without `.agents/skills/` | Run from the repo root so `AGENTS.md` discovers `.agents/skills/cudaq-academic-vqe-qaoa/` |

3. **Paste the same prompt in both runs.** Record three numbers per run:
   the algorithm it recommended, the files-read / tool-call counts from
   the agent UI, and the token total.
4. **Compare against the Results tables below.** P1–P5 should get cheaper
   (fewer files/calls/tokens); P7 should flip from a classical answer to
   a CUDA-Q one.

### Two ways to get the token numbers

- **Offline, exact — `context_size.py`** (the "Ctx in" column).
  Tokenizes the files+prompt each path pulls in with `tiktoken`; no API
  key and no agent CLI needed (just `pip install tiktoken` once), so it
  runs anywhere: `python3 context_size.py`. This is the easiest thing to
  demo live inside Cursor.
- **Live API meter — `measure_tokens.py`.** If an agent CLI is installed
  (Claude Code / Codex / Cursor), this captures the *real* per-turn token
  usage and cost from the agent itself. See **Measure real token usage**.

## Results (live agent runs)

Every cell below is measured, not guessed. Context and response tokens
are real `tiktoken` (`cl100k_base`) counts of actual bytes; files read,
tool calls, and which algorithm the agent recommended come from live
subagent runs of each prompt in both conditions.

### Table A — token accounting

*Ctx in* = the files+prompt each path pulls into context (deterministic,
`context_size.py`). *Resp out* = tokens of the agent's actual answer.
*Total* = ctx in + resp out. Ratio is no-skill ÷ with-skill (>1 means the
skill is cheaper).

| Prompt | Type | Maps to | Ctx in: skill / no | Resp out: skill / no | Total: skill / no | Total ratio |
| --- | --- | --- | --- | --- | --- | --- |
| P1 | technical | QAOA | 1,676 / 3,797 | 844 / 1,248 | 2,520 / 5,045 | **2.00×** |
| P2 | technical | QAOA | 1,648 / 3,769 | 727 / 1,084 | 2,375 / 4,853 | **2.04×** |
| P3 | technical | QAOA | 1,647 / 3,768 | 382 / 901 | 2,029 / 4,669 | **2.30×** |
| P4 | technical | ADAPT-VQE | 1,779 / 1,302 | 914 / 1,844 | 2,693 / 3,146 | **1.17×** |
| P5 | technical | GQE | 2,035 / 3,517 | 931 / 1,923 | 2,966 / 5,440 | **1.83×** |
| P6 | technical | install | 1,797 / 815 | 441 / 702 | 2,238 / 1,517 | 0.68× |
| P7 | domain | MaxCut/QAOA | 1,639 / 3,760 | 892 / 1,381 | 2,531 / 5,141 | **2.03×** |
| P8 | domain | VQE | 1,485 / 1,305 | 1,222 / 2,162 | 2,707 / 3,467 | **1.28×** |
| **All 8** | — | — | **13,706 / 22,033** | **6,353 / 11,245** | **20,059 / 33,278** | **1.66×** |

### Table B — agent behavior

Files read, tool calls, and recognition from live runs (one run per
condition for P1–P6 and P8; P5 and P7 corroborated by an earlier 5-run
batch, whose means are shown).

| Prompt | Files: no→skill | Calls: no→skill | Recognition: no→skill |
| --- | --- | --- | --- |
| P1 | 3→4 | 11→7 | quantum→quantum |
| P2 | 3→3 | 10→7 | quantum→quantum |
| P3 | 3→2 | 6→3 | quantum→quantum |
| P4 | 9→2 | 12→5 | quantum→quantum |
| P5 | 4→2 | 8→5 | quantum→quantum |
| P6 | 0→2 | 4→5 | n/a (install) |
| P7 | 4.2→5.0 | 10.8→10.2 | **classical→quantum** (0/5→5/5) |
| P8 | 3→3 | 8→5 | quantum→quantum (no gap) |

### What the data says

- **Total tokens are the cleanest story: 1.66× fewer with the skill**
  (20.1k vs 33.3k across all 8). Counting the response as well as the
  context *helps* the skill even on rows where context-in alone looked
  unfavorable (P4 → 1.17×, P8 → 1.28×), because no-skill answers run
  consistently ~2× longer.
- **Only P6 (install) favors no-skill** (0.68×). It's a one-line apt fix,
  so loading a ~1.8k-token skill is overkill — keep it as the honest
  counter-example, not a result to hide.
- **Cost wins are consistent on 7 of 8**: fewer or equal tool calls and
  shorter answers. Largest reductions are P3 (calls 6→3, response
  901→382) and P4 (files 9→2, calls 12→5).
- **Recognition is the skill's headline benefit, but it is narrower than
  it looks.** P7 (delivery/fleet split) is the clean win: every no-skill
  run reframed it as classical vehicle routing (OR-Tools) and never
  surfaced QAOA, while every with-skill run produced runnable
  `get_maxcut_hamiltonian` + `solvers.qaoa` code. **P8's gap effectively
  vanished**: a no-skill agent exploring inside this repo recognized
  "stability = ground-state energy → VQE" on its own. Being *inside the
  cudaqx repo* nudges even a skill-free agent toward CUDA-Q, so P8 is a
  cost story, not a recognition story.

### How each number is produced

- **Ctx in** — deterministic. `context_size.py` tokenizes `SKILL.md` +
  the one routed reference + the prompt (with-skill) vs the minimal repo
  source files + the prompt (no-skill). Reproduce with
  `python3 context_size.py` (or `--json`).
- **Resp out** — the agent's full answer for each run was saved and
  tokenized with the same encoder (`context_size.py --response-tokens`).
- **Files / calls / recognition** — read from live subagent runs. The
  recognition column is verifiable from the answer text itself (which
  algorithm was recommended), not self-reported.

### Caveats (read before quoting these numbers)

- The *no-skill* `Ctx in` column is a **conservative lower bound** — it
  counts only the minimal source files, whereas live no-skill runs opened
  more (P7 averaged ~4 files / ~11 calls). Real no-skill context runs
  higher than shown, so the token ratios understate the skill's edge.
- Table B behavioral rows are **N=1** except P5/P7 (5-run means). LLM
  output varies; re-run with `measure_tokens.py` to tighten. The P7
  recognition outcome (0/5 vs 5/5) is the most robust signal.
- `SKILL.md` carries a fixed ~1.0k-token routing overhead. It pays off
  against a bulky repo alternative (QAOA examples+tests, GQE) and loses
  on cheap topics (install, ADAPT/VQE context), exactly as P4/P6 show.
- The no-skill answers are not "wrong" in isolation — for tiny molecules
  classical quantum chemistry is genuinely the better practical tool. The
  point is that, absent the skill, P7-style problems never surface the
  CUDA-Q path; the workshop's premise is learning quantum algorithms, so
  that recognition is the intended win.

## Measure real token usage

`measure_tokens.py` is a real recorder: it invokes an installed agent
CLI in headless mode and parses the **actual** token usage out of its
structured output — no guessing, no hand-typing.

> **Important — `--config` is only a label.** The script sends the raw
> prompt from `prompts.json` and writes the result to
> `runs/<config>.json`; it does **not** itself enable or disable the
> skill. *You* control the skill the same way as in the manual demo: run
> the `without_skill` recording from a directory where the skill isn't
> discoverable (a scratch dir or a checkout with the skills folders
> removed), and the `with_skill` recording from the repo root. Then pass
> the matching `--config` so the output files are labelled correctly.

| Runtime | Command it runs | Meter field parsed |
| --- | --- | --- |
| Claude Code | `claude -p "<prompt>" --output-format json` | `usage.input_tokens/output_tokens`, `total_cost_usd` |
| Codex | `codex exec --json "<prompt>"` | last `turn.completed.usage` |
| Cursor | `cursor-agent -p --output-format json "<prompt>"` | `usage` (recent CLI versions) |

```bash
EVAL="$PWD/.agents/evals/academic-vqe-qaoa"   # run this from the repo root

# WITH skill: launch from the repo root so the skill is discoverable.
# Auto-detects whichever CLI is on PATH. Output -> runs/with_skill.json
python3 "$EVAL/measure_tokens.py" --config with_skill --prompt-ids P5

# WITHOUT skill: launch from a scratch dir with no skills present.
# Output still lands in the eval dir's runs/without_skill.json
mkdir -p /tmp/no-skill && cd /tmp/no-skill
python3 "$EVAL/measure_tokens.py" --config without_skill --prompt-ids P5
cd -

# preview the exact command without running it (no CLI needed)
python3 "$EVAL/measure_tokens.py" --runtime codex --prompt-ids P5 --dry-run

# verify the parsers (no CLI needed)
python3 "$EVAL/measure_tokens.py" --selftest
```

Each record carries `tokens_source`: `meter` (real API usage), or
`manual` (typed in, only when no CLI is installed — never silently
treated as a meter reading). Output lands in `runs/<config>.json`, ready
for the scorer.

## Optional: score your runs

```bash
python3 evaluate_metrics.py runs/with_skill.json runs/without_skill.json
```

The scorer reports per-run pass-rate, coverage, forbidden-hit count,
context-file count, and token totals (summed from the recorded `tokens`
field). To score by hand without the recorder, write a run-JSON of this
shape — one file per run:

```json
{
  "agent": "claude-code",
  "model": "claude-opus-4-7",
  "config": "with_skill",
  "responses": [
    {
      "id": "P1",
      "response": "<paste the agent's full answer here>",
      "tokens": {"input": 0, "output": 0},
      "context_files": ["SKILL.md", "references/qaoa.md"]
    }
  ]
}
```

## Files

- `prompts.json` — the eight workshop prompts (P1–P6 technical, P7–P8
  domain-translation).
- `measure_tokens.py` — real recorder: invokes the detected agent CLI
  (Claude Code / Codex / Cursor) headlessly and parses actual token
  usage from its JSON output; manual entry only when no CLI is present.
  `--selftest` verifies the parsers offline.
- `context_size.py` — offline context-size measurement: tokenizes the
  files each path pulls into context with `tiktoken` (needs
  `pip install tiktoken`; no API key/runtime). Produces the "Ctx in"
  column of Table A; `--response-tokens <files>` tokenizes saved answers
  for the "Resp out" column.
- `assertions.json` — per-prompt `must_include` / `must_not_include`
  substring checks.
- `evaluate_metrics.py` — deterministic substring scorer + token
  summer.
