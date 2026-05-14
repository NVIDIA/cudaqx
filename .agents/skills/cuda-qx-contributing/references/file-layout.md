# File layout: where new code goes

Pick the right directory before writing the file. Moving things later
breaks tests, install sets, and IDE indexers.

## QEC contributions

| What you're adding | Where it goes |
|--------------------|---------------|
| New C++ code header (public) | `libs/qec/include/cudaq/qec/codes/<name>.h` |
| New C++ code source | `libs/qec/lib/codes/<name>.cpp` (and add to `lib/CMakeLists.txt`) |
| New decoder header (public) | `libs/qec/include/cudaq/qec/decoders/<name>.h` |
| New decoder plugin (C++) | `libs/qec/lib/decoders/<name>/` |
| New Python plugin (decoder or code) | `libs/qec/python/cudaq_qec/plugins/{codes,decoders}/<name>.py` |
| Real-time / autonomous decoder | `libs/qec/include/cudaq/qec/realtime/` + `libs/qec/lib/realtime/` |
| Python public API addition | `libs/qec/python/cudaq_qec/__init__.py` (re-export) |
| Python bindings (new pybind11) | `libs/qec/python/bindings/<name>.cpp` |
| C++ test | `libs/qec/unittests/<name>.cpp` (and add to `unittests/CMakeLists.txt`) |
| Python test | `libs/qec/python/tests/test_<name>.py` |
| Example (Python) | `docs/sphinx/examples/qec/python/<name>.py` |
| Example (C++) | `docs/sphinx/examples/qec/cpp/<name>.cpp` |
| Long-form RST example | `docs/sphinx/examples_rst/qec/<name>.rst` |
| API page (RST) | `docs/sphinx/api/qec/<name>.rst` |
| Component docs | `docs/sphinx/components/qec/<name>.rst` (link from `introduction.rst`) |

## Solvers contributions

| What you're adding | Where it goes |
|--------------------|---------------|
| New operator pool (C++) | `libs/solvers/include/cudaq/solvers/operators/operator_pools/<name>.h` |
| New operator pool source | `libs/solvers/lib/operators/operator_pools/<name>.cpp` |
| New state-prep | `libs/solvers/include/cudaq/solvers/stateprep/<name>.h` |
| New optimizer | `libs/solvers/include/cudaq/solvers/optimizers/<name>.h` |
| New gradient method | `libs/solvers/include/cudaq/solvers/observe_gradients/<name>.h` |
| PySCF integration | `libs/solvers/lib/operators/molecule/drivers/` (rare; usually no change needed) |
| Python public API | `libs/solvers/python/cudaq_solvers/__init__.py` |
| Python bindings | `libs/solvers/python/bindings/solvers/py_solvers.cpp` |
| C++ test | `libs/solvers/unittests/test_<name>.cpp` |
| Python test | `libs/solvers/python/tests/test_<name>.py` |
| Example (Python) | `docs/sphinx/examples/solvers/python/<name>.py` |
| Example (C++) | `docs/sphinx/examples/solvers/cpp/<name>.cpp` |
| API page (RST) | `docs/sphinx/api/solvers/<name>.rst` |

## Shared / cross-cutting

| What you're adding | Where it goes |
|--------------------|---------------|
| Helper used by both libs | `libs/core/` (C++ helpers only) |
| Build / format / validation script | `scripts/` |
| Docker / container setup | `docker/build_env/` or `docker/release/` |
| Skill (agent docs) | `.agents/skills/<name>/` (then run sync) |
| Eval prompt / grader | `.agents/evals/` |
| Top-level doc | repo root (`README.md`, `Contributing.md`, `Building.md`) |

### About `libs/core/`

`libs/core/` is the C++ helper layer shared by both QEC and Solvers
libraries. Public headers live under
`libs/core/include/cuda-qx/core/` and include `extension_point.h`,
`graph.h`, `heterogeneous_map.h`, `kwargs_utils.h`, `library_utils.h`,
`tear_down.h`, `tensor.h`, `tensor_impl.h`, `tuple_utils.h`,
`type_traits.h`. It has its own `CMakeLists.txt`, `lib/`, `python/`,
and `unittests/` trees.

**Who edits it**: anyone adding a helper used by *both* QEC and
solvers. Don't put QEC-only or solvers-only code here.

**Why it matters for the future QEC/solvers repo split**: `libs/core/`
is the single invisible coupling between the two libraries. After a
split, it must either be:

1. Extracted to its own repo (`cudaqx-core` or similar), with both
   QEC and solvers depending on it as an external package.
2. Vendored into both repos (duplicated; only viable if `libs/core/`
   churn is near-zero).

Today no skill owns `libs/core/`; changes to it should be paired with
sanity tests in *both* libraries (`bash scripts/test_libs_builds.sh`
covers this — see `cuda-qx-testing-ci`).

## License placement

| Path | License |
|------|---------|
| `libs/qec/**` | `LicenseRef-NVIDIA-Proprietary` (see `libs/qec/LICENSE`) |
| `libs/solvers/**`, `libs/core/**`, top-level | Apache 2.0 |
| Examples, docs | Apache 2.0 |

Every file should start with the appropriate copyright header.
Copy from existing files in the same directory to be sure.

## Naming conventions

- Lowercase + underscore for C++ files (`my_decoder.h`, `my_decoder.cpp`).
- Lowercase + underscore for Python files and tests.
- Snake_case for Python public APIs.
- snake_case for C++ namespace functions; PascalCase for class names.
- Decoder / code / pool registration names follow the existing
  convention — lowercase, dashes (`my-decoder`, `nv-qldpc-decoder`).

## Adding to install / build

C++ additions usually require *two* CMake edits:

1. Add to the library's `lib/CMakeLists.txt` (or
   `lib/decoders/CMakeLists.txt`, etc.) to be compiled.
2. Add to the install set (so users get the header on install).

Look at the most recent built-in addition for the pattern; mimic
exactly.

## Documentation expectations

For a user-visible addition:

| Component | Documentation expected |
|-----------|------------------------|
| New decoder | API page (`api/qec/<name>.rst`), short example, mention in `components/qec/introduction.rst` |
| New code | API page, short example, mention in `components/qec/introduction.rst` |
| New operator pool | API page, mention in `components/solvers/` |
| New realtime feature | mention in `components/qec/introduction.rst` real-time section + a how-to in `examples_rst/qec/` |

For an internal-only addition: a doc-string and a unit test are
enough. Reviewers will ask if more is warranted.

## Self-check

```
[ ] New file is in the right place per the tables above.
[ ] Copyright header matches the license of that directory.
[ ] CMakeLists updated if C++.
[ ] Test placed alongside other tests for the same component.
[ ] Doc page or example added if the addition is user-visible.
[ ] No file leaks across the libs/qec ↔ libs/solvers license boundary.
```

## Where next

- Format the new files: `formatting.md`.
- Write the PR: `pr-workflow.md`.
- Plugin-author specifics: `cuda-qx-qec-extending`, `cuda-qx-solvers-extending`.
