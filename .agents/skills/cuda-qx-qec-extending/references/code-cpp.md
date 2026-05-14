# Custom QEC code in C++

The performance path. Choose when you need:

- High-throughput simulation of large codes.
- Tight integration with the realtime decoding stack.
- Reuse from C++ user code without paying for Python startup.

Otherwise, prototype in Python first (`code-python.md`).

## Authoritative template

- Built-in codes: `libs/qec/include/cudaq/qec/codes/repetition.h`,
  `steane.h`, `surface_code.h`. Read all three; the `surface_code.h`
  is the most complex and shows every pattern.
- Base class contract: `libs/qec/include/cudaq/qec/code.h`.

## Skeleton

```cpp
// libs/qec/include/cudaq/qec/codes/my_code.h
#pragma once
#include "cudaq/qec/code.h"

namespace cudaq::qec {

class my_code : public code {
protected:
  std::size_t get_num_data_qubits() const override;
  std::size_t get_num_ancilla_qubits() const override;
  std::size_t get_num_ancilla_x_qubits() const override;
  std::size_t get_num_ancilla_z_qubits() const override;
  std::size_t get_num_x_stabilizers() const override;
  std::size_t get_num_z_stabilizers() const override;
  // ... and the parity / observable accessors
public:
  my_code(const heterogeneous_map& options);
  CUDAQ_REGISTER_CODE("my-code")
};

}
```

The `CUDAQ_REGISTER_CODE("my-code")` macro hooks your class into the
runtime registry. Pair it with the macro's `.cpp` companion as the
existing built-in codes do.

## Stabilizer kernels

The code's `operation_encodings` map sends each `operation` enum
value to a `cudaq::qkernel`. Define your encoding kernels in a
companion `.cpp`:

```cpp
__qpu__ void my_code_prep0(patch p) { ... }
__qpu__ std::vector<cudaq::measure_result> my_code_stab_round(
    patch p, const std::vector<std::size_t>& dx, const std::vector<std::size_t>& dz) {
  ...
}
```

Register them in the constructor:

```cpp
my_code::my_code(const heterogeneous_map&) {
  operation_encodings[operation::prep0] = my_code_prep0;
  operation_encodings[operation::stabilizer_round] = my_code_stab_round;
}
```

The kernel type aliases are in `code.h`:
- `one_qubit_encoding = cudaq::qkernel<void(patch)>`
- `two_qubit_encoding = cudaq::qkernel<void(patch, patch)>`
- `stabilizer_round = cudaq::qkernel<std::vector<cudaq::measure_result>(patch, const std::vector<std::size_t>&, const std::vector<std::size_t>&)>`

## Build glue

1. Add the source file to `libs/qec/lib/CMakeLists.txt` (look at how
   `surface_code.cpp` is added; mimic exactly).
2. Header lives under `libs/qec/include/cudaq/qec/codes/my_code.h`.
3. If you want a Python binding, extend
   `libs/qec/python/bindings/py_code.cpp` to expose your class.
4. Rebuild: `ninja install` inside the `build/` directory.
5. Run the C++ test: `ctest -R my_code`.

For the build path end-to-end, see `cuda-qx-build/references/build.md`.

## Python visibility

Once the C++ class is registered, it appears in `qec.get_code` from
Python *if* the Python bindings expose it. The simplest path: do
*not* add a separate Python binding; the existing
`qec.get_code("my-code", **options)` works once your C++ side is
registered, because the loader looks up by string.

Caveat: kwargs in Python map to `heterogeneous_map` in C++; make sure
your constructor accepts all the names users might pass.

## Smoke test

```cpp
// libs/qec/unittests/test_my_code.cpp
#include "cudaq/qec/codes/my_code.h"
#include <gtest/gtest.h>

TEST(MyCode, ParityShapes) {
  cudaq::qec::heterogeneous_map opts;
  cudaq::qec::my_code code(opts);
  EXPECT_EQ(code.get_num_data_qubits(), 7);   // for example
  // ... structural checks
}
```

Add it to the test list in `libs/qec/unittests/CMakeLists.txt`.

## Common pitfalls

| Symptom | Cause |
|---------|-------|
| `qec.get_code("my-code")` fails: not found | `CUDAQ_REGISTER_CODE` macro missing, or library not loaded |
| Build fails: unresolved external | source not added to library's CMakeLists.txt |
| Header in `include/cudaq/qec/codes/` not visible | not exported in the install set |
| Python doesn't see the new code, even though C++ tests pass | shared library not loaded by `cudaq_qec` Python module — confirm install prefix |

## Self-check

```
[ ] CUDAQ_REGISTER_CODE("name") present and unique.
[ ] Source added to libs/qec/lib/CMakeLists.txt.
[ ] Header in libs/qec/include/cudaq/qec/codes/.
[ ] ctest -R <my-code> passes.
[ ] From Python: qec.get_code("name") works after rebuild.
[ ] Parity / observable structural checks pass.
[ ] Smoke decode at p=0 → zero corrections.
```

## Where next

- Decoder to pair: `decoder-cpp.md` or use a built-in decoder via
  `cuda-qx-qec-decode/references/decode.md`.
- For real-time targeting: `cuda-qx-qec-realtime`.
- Add docs for the new code: `cuda-qx-build/references/docs.md`.
