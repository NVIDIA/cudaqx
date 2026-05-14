---
name: "cuda-qx-qec-extending"
title: "CUDA-QX QEC Extending (custom codes and decoders)"
description: >-
  Plugin-author skill for the QEC library: define a new QEC code (Python
  or C++), register a new decoder (Python or C++), hook into the plugin
  loader, expose it through cudaq_qec, and add unit tests. Use whenever
  the user mentions "custom code", "custom QEC code", "new decoder",
  "plugin", "subclass code", "register decoder", or "plugin_loader" in
  the context of QEC. For real-time / autonomous_decoder development,
  delegate to cuda-qx-qec-realtime. For solvers extensions (operator
  pools, optimizers, state-prep), see the cuda-qx-solvers-algorithms package
  (`cuda-qx-solvers-extending` skill).
version: "0.1.2"
author: "CUDA-QX"
license: "LicenseRef-NVIDIA-Proprietary (libs/qec/LICENSE)"
compatibility: "Python 3.11+, C++ 20, CMake 3.28+, Linux x86_64/aarch64"
tags: [cuda-qx, cudaq-qec, qec, plugin, extension, custom-code, custom-decoder, plugin-loader]
tools: [Read, Glob, Grep, Bash]
metadata:
  repo: [qec]
  author: "CUDA-QX"
  domain: "qec-extension"
  audience: [researcher, plugin-developer, contributor]
  languages: [python, c++]
---

# CUDA-QX QEC Extending

The "I want to add a new code or decoder to cudaq-qec" skill. Two
extension surfaces (code, decoder), two languages each (Python, C++).
Pick the right surface for your research idea before writing code.

If the user wants to *use* the existing decoders or codes, delegate
to `cuda-qx-qec-decode`. If they want to write a new real-time / autonomous
decoder (GPU-resident, zero-CPU), delegate to
`cuda-qx-qec-realtime/references/autonomous-decoder.md` — the
device-side path has its own contract. For solvers operator pools,
state-prep kernels, or optimizers, see the cuda-qx-solvers-algorithms package
(`cuda-qx-solvers-extending` skill).

## Inputs

Caller provides:

- Extension type: `code` or `decoder`.
- Language: Python (prototype, no rebuild) or C++ (production).
- For codes: stabilizer set, parity matrices, encoding kernels.
- For decoders: a `decode(syndrome) -> DecoderResult` body.
- Unique registration name (string).

## Outputs

This skill produces:

- New source files at the right repo paths
  (`libs/qec/python/cudaq_qec/plugins/{codes,decoders}/` for Python,
  `libs/qec/{lib,include}/...` for C++).
- Updated `CMakeLists.txt` for C++ additions.
- A smoke test under `libs/qec/{python/tests,unittests}/` that
  confirms the new name appears in `qec.list_decoders()` /
  `qec.get_code(name)` and decodes zero corrections at p=0.
- A short docs stub under `docs/sphinx/components/qec/` or
  `examples_rst/qec/`.

Does NOT produce: a real-time / GPU-resident decoder (→
`cuda-qx-qec-realtime/references/autonomous-decoder.md`); a trained
neural decoder (→ `cuda-qx-qec-ai-decoders`).

## Audience

Researchers and library developers comfortable with Python *and*
C++ (at least at the "I can read pybind11 binding code" level). For
pure prototyping a Python-only plugin is fine; for production-grade
deployment the C++ path is faster but heavier.

## First three actions

```bash
bash   .agents/skills/_shared/scripts/preflight.sh    --json > /tmp/preflight.json
python .agents/skills/_shared/scripts/import_smoke.py --json > /tmp/import_smoke.json
python .agents/skills/_shared/scripts/pick_workflow.py \
    --intent qec-custom \
    --preflight /tmp/preflight.json \
    --imports   /tmp/import_smoke.json
```

`pick_workflow.py` with `--intent qec-custom` returns the next
reference file to read.

## Key Paths

| Area | Path |
|------|------|
| QEC C++ code base class | `libs/qec/include/cudaq/qec/code.h` |
| QEC C++ decoder base class | `libs/qec/include/cudaq/qec/decoder.h` |
| Built-in code headers | `libs/qec/include/cudaq/qec/codes/` |
| Plugin loader | `libs/qec/include/cudaq/qec/plugin_loader.h` |
| Decoder plugins (C++) | `libs/qec/lib/decoders/` |
| Decoder plugins (Python) | `libs/qec/python/cudaq_qec/plugins/decoders/` |
| Code plugin example (Python) | `libs/qec/python/cudaq_qec/plugins/codes/example.py` |
| Decoder plugin example (Python) | `libs/qec/python/cudaq_qec/plugins/decoders/example.py` |
| Python steane example | `docs/sphinx/examples/qec/python/my_steane.py`, `my_steane_test.py` |

## Workflow Index

| If the user wants to | Read |
|----------------------|------|
| Define a new QEC code in Python | `references/code-python.md` |
| Define a new QEC code in C++ | `references/code-cpp.md` |
| Define a new decoder in Python (offline) | `references/decoder-python.md` |
| Define a new decoder in C++ (incl. real-time hooks) | `references/decoder-cpp.md` |

## Conventions

These are the recurring extension-time bugs.

1. **Pick Python first if you can.** Python plugins ship without a
   rebuild, deploy as a single file, and integrate with `plugin_loader`
   at runtime. Move to C++ only when the hot path makes Python
   unworkable.

2. **`get_code` / `get_decoder` look up by string name.** Your
   plugin's registered name must match what users will type. Choose a
   unique name; collisions silently shadow built-ins.

3. **C++ codes register via the `CUDAQ_REGISTER_TYPE`-family macro.**
   Python codes register by subclassing `qec.Code` and being
   import-able. Mixing the registration mechanism with the language
   is a common confusion source.

4. **Decoder `result.result` is per-column (i.e. per-error-mechanism),
   not per-data-qubit.** When you write a new decoder, return a
   length-`H.shape[1]` vector. Thresholding at 0.5 yields a hard
   prediction; users threshold downstream.

5. **`converged` is a real field, not decoration.** Set it
   honestly: BP-style decoders may not converge; LUT decoders always
   do. Downstream LER analyses ignore non-converged shots.

6. **For C++ codes/decoders, the build glue matters.** Add your
   sources to the right `CMakeLists.txt`, expose your header in
   `include/cudaq/qec/`, and (if a Python binding) extend the
   pybind11 module.

7. **Test that your plugin appears in the registry.** A quick smoke
   test:

   ```python
   import cudaq_qec as qec
   print(qec.list_decoders())   # should include your name
   ```

   If your plugin doesn't appear, the registration didn't fire.

## Quick start: pick your extension

```
Q1: Adding a code or a decoder?
    code     -> Q2
    decoder  -> Q3

Q2: Will you write the stabilizer circuits in CUDA-Q quantum kernels?
    yes -> Python is easier. references/code-python.md
    no  -> C++ for performance. references/code-cpp.md

Q3: Real-time / latency-bounded?
    yes -> C++ + autonomous_decoder. cuda-qx-qec-realtime/references/autonomous-decoder.md
    no, but high-throughput / GPU-resident -> C++ decoder plugin. references/decoder-cpp.md
    offline prototype -> Python. references/decoder-python.md
```

## Self-Check Protocol

```
[ ] Plugin name unique; checked with qec.list_decoders() / get_code.
[ ] Registration mechanism matches language (subclass for Python; CUDAQ_REGISTER_* for C++).
[ ] If C++: source added to libs/qec/lib/CMakeLists.txt; header in libs/qec/include/.
[ ] If C++ Python bindings: pybind11 binding updated.
[ ] Smoke test from a clean Python session lists the new plugin.
[ ] Unit test added under libs/qec/python/tests/ or libs/qec/unittests/.
[ ] At p=0, the new decoder returns zero corrections (or the new code reports zero LER).
[ ] Plugin documented (one page in docs/sphinx/components/qec/ or examples_rst).
```

## When stuck

1. Open the matching `references/*.md` and walk it end-to-end.
2. Read the example plugin under
   `libs/qec/python/cudaq_qec/plugins/{codes,decoders}/example.py` and
   diff against your file.
3. For C++, build with `CUDAQX_INCLUDE_TESTS=ON` and run `ctest -R
   <your_test>` after every change.
4. For build issues, delegate to `cuda-qx-build`.

## Additional resources

- Workflow references: `references/code-python.md`,
  `references/code-cpp.md`, `references/decoder-python.md`,
  `references/decoder-cpp.md`.
- Built-in decoders / codes as living references:
  `libs/qec/include/cudaq/qec/codes/`,
  `libs/qec/python/cudaq_qec/plugins/decoders/`.
- Testing your plugin: `cuda-qx-testing-ci` SKILL.md.
- Real-time eligible decoders: `cuda-qx-qec-realtime` SKILL.md.
- Solvers extensions (operator pools etc.): see the cuda-qx-solvers-algorithms
  package — `cuda-qx-solvers-extending` skill when in the cudaqx monorepo.
