# C++ tests (GoogleTest)

C++ tests live under `libs/qec/unittests/` and
`libs/solvers/unittests/`. They use GoogleTest, are configured via
CMake, and run via `ctest`.

## File and target

```cpp
// libs/qec/unittests/test_my_decoder.cpp
#include <gtest/gtest.h>
#include "cudaq/qec/decoders/my_decoder.h"

TEST(MyDecoder, ConstructsAndZeroInputDecodes) {
    cudaq::qec::heterogeneous_map opts;
    // Build a tiny H
    std::vector<std::vector<bool>> H = {{1, 1, 0}, {0, 1, 1}};
    cudaq::qec::my_decoder dec(H, opts);
    auto r = dec.decode({0.0, 0.0});
    for (auto v : r.result) EXPECT_FLOAT_EQ(v, 0.0);
}
```

## CMake registration

Add to `libs/qec/unittests/CMakeLists.txt`:

```cmake
add_qec_unittest(test_my_decoder
    SOURCES test_my_decoder.cpp
    LIBRARIES cudaq-qec ${MY_DECODER_LIB})
```

(Look at how an existing entry is wired; the macro names and
argument shape vary between subdirectories.)

After CMake regenerates, `ctest --test-dir build` includes the new
test.

## Running tests

```bash
# Run all
ctest --test-dir build -V

# Run by name pattern
ctest --test-dir build -R MyDecoder -V

# List without running
ctest --test-dir build --show-only=human
```

`-V` shows full output (useful for failing tests). `-VV` is even
more verbose.

## GoogleTest tips

- `EXPECT_*` records failure and continues; `ASSERT_*` aborts the
  test. Use `ASSERT_*` for preconditions (e.g. construction
  succeeded) and `EXPECT_*` for the actual checks.
- `EXPECT_FLOAT_EQ` is exact-equal for float (within 4 ULPs);
  `EXPECT_NEAR(a, b, eps)` for explicit tolerance.
- `TEST_P` for parameterized tests over multiple inputs.

## Real-time / app_examples tests

The `libs/qec/unittests/realtime/app_examples/` directory holds
production-shaped end-to-end tests for the real-time stack. They use
the same CTest infrastructure but are slower and depend on optional
plugins.

If your change touches the real-time stack:

```bash
ctest --test-dir build -R realtime -V
```

## Common pitfalls

| Symptom | Cause |
|---------|-------|
| Test target not built | source not added to `CMakeLists.txt` |
| `gtest.h` not found | gtest not pulled in by the macro; check macro usage |
| Test runs but reports zero assertions | TEST signature wrong (missing `()`?) |
| Linker error: undefined reference | library not in the `LIBRARIES` list of `add_qec_unittest` |
| Test passes locally, fails in CI | CI builds a different config (`CMAKE_BUILD_TYPE`, CUDA arch); rebuild with the same and re-run |

## Coverage

`-DCUDAQX_ENABLE_COVERAGE=ON` (when supported) instruments tests
with gcov. After running:

```bash
lcov --capture --directory build --output-file coverage.info
genhtml coverage.info --output-directory coverage_html
```

Open `coverage_html/index.html`. Not typically part of the PR check
flow; useful for spotting untested branches.

## Self-check

```
[ ] Test compiles and runs locally.
[ ] CMakeLists wired so `ctest --show-only` lists the new test.
[ ] Test covers the happy path and at least one edge case.
[ ] ASSERT_* used for preconditions, EXPECT_* for checks.
[ ] No reliance on host-specific paths or env vars (or guarded with
    `GTEST_SKIP()` if absent).
```

## Where next

- Add a Python test counterpart: `python-tests.md`.
- Run the same subset CI runs: `run-locally.md`.
