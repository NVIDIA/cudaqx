# Filing a useful bug report

A bug report without a minimal reproducer is half a report. This
page is what to include.

## Where to file

- Bugs and feature requests:
  <https://github.com/NVIDIA/cuda-qx/issues>
- General questions / "is this a bug?":
  <https://github.com/NVIDIA/cuda-qx/discussions>

## Anatomy of a good report

```
Title:  Short verb-noun: "tensor_network_decoder crashes when H is empty"

Body:

## Environment
(paste the env block from cudaq-benchmarking/references/reproducibility.md)

## Reproducer
```python
import cudaq_qec as qec
H = numpy.zeros((0, 5), dtype=numpy.uint8)
dec = qec.get_decoder("tensor_network_decoder", H)   # crashes here
```

## Expected
Decoder constructs and `decode` returns zero corrections for any
syndrome.

## Actual
Segfault during construction. Stack trace:
```
...
```

## Workaround
(optional) Skip empty-H case in user code.

## Notes
(optional) related issue numbers, links to PRs, papers, etc.
```

## What `scripts/doctor.sh` gives you

The bundled `scripts/doctor.sh` prints an environment snapshot
formatted for bug reports:

```bash
bash scripts/doctor.sh > doctor.txt
```

Paste `doctor.txt` into the "Environment" section. Maintainers can
reproduce your setup much faster with this.

## Minimal reproducer guidelines

A reproducer must be:

- **Self-contained** — runnable with only `pip install cudaq-qec`
  (or the appropriate package).
- **Small** — fewer than ~30 lines is the goal. If your reproducer
  is 200 lines, you have not minimized.
- **Deterministic** — same inputs, same crash. Seed RNGs.
- **Stripped of unrelated complexity** — if the bug is in the
  decoder, the molecule / chemistry setup is noise.

Reproducer minimization process:

1. Start with your real script.
2. Comment out everything until the bug stops reproducing.
3. Add back the smallest piece that brings the bug back.
4. Iterate until you can't remove anything more.

## When the bug is in a closed plugin

Some plugins (`nv-qldpc-decoder`) are closed source. You still get a
useful report by:

- Reporting the exact plugin version (`pip show cudaq-qec | grep
  Version`, plus the plugin's bundled version if available).
- Including the input shapes, parameters, and PCM characteristics.
- Confirming the bug reproduces with a different decoder swapped in
  (e.g. `multi_error_lut`) — narrows the cause.

## Feature requests

For a feature request:

- State the problem, not the solution. "I want X" is harder to
  triage than "I'm trying to do Y, and X would help".
- Identify the user persona (researcher, ML practitioner,
  hardware engineer, ...) so reviewers know which audience benefits.
- Note workarounds you've tried.

## What goes in discussions vs issues

| Topic | Where |
|-------|-------|
| Crash, wrong output, broken install | Issue |
| "Is this expected?" | Discussion first; promote to issue if confirmed bug |
| "How do I use X?" | Discussion |
| "I built something cool" | Discussion |
| Feature request | Issue |
| Documentation gap | Issue (typically) |

## Self-check

```
[ ] Title is one short sentence, verb-noun.
[ ] Environment block included (use `scripts/doctor.sh`).
[ ] Minimal reproducer under ~30 lines.
[ ] Expected / Actual sections present.
[ ] If applicable: cudaq git SHA / pip-list versions / stack trace.
[ ] Linked from / to related issues if any.
```

## Where next

- Once a bug is filed and accepted, fix it via PR:
  `pr-workflow.md`.
- For environment capture details: `cudaq-benchmarking/references/reproducibility.md`.
