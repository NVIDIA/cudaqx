# Custom decoder in C++

The performance path for a new decoder. Pick this when:

- You need GPU acceleration that PyTorch / CuPy can't reach.
- You need integration with `nv-qldpc-decoder` style plugins.
- You need a code path that links cleanly into the real-time / FPGA
  stack.

For real-time GPU-resident decoders (zero-CPU), see
`cuda-qx-qec-realtime/references/autonomous-decoder.md`. That
is a different contract (CRTP + device dispatch); not this skill.

## Authoritative reference

- Base class: `libs/qec/include/cudaq/qec/decoder.h`.
- Built-in plugins to copy: `libs/qec/lib/decoders/`.
- `nv_qldpc_decoder_api.rst`, `tensor_network_decoder_api.rst`,
  `trt_decoder_api.rst`, `sliding_window_api.rst` for example API
  shapes.

## Skeleton

```cpp
// libs/qec/include/cudaq/qec/decoders/my_decoder.h
#pragma once
#include "cudaq/qec/decoder.h"

namespace cudaq::qec {

class my_decoder : public decoder {
public:
  my_decoder(const std::vector<std::vector<bool>>& H,
             const heterogeneous_map& params);

  decoder_result decode(const std::vector<float_t>& syndrome) override;

  CUDAQ_REGISTER_DECODER("my-decoder")
};

}
```

Match the existing pattern in
`libs/qec/lib/decoders/<existing>/` — the constructor signature, the
`decoder_result` type, and the `CUDAQ_REGISTER_DECODER` macro are all
defined in `decoder.h`.

## Build glue

1. Drop `my_decoder.cpp` (and any private headers) directly into
   `libs/qec/lib/decoders/` alongside the existing decoder sources
   (or add a subdir under `libs/qec/lib/decoders/plugins/` if you
   prefer the plugin layout — see `libs/qec/lib/decoders/plugins/example/`).
2. Wire the source into `libs/qec/lib/CMakeLists.txt` next to the
   other built-in decoders.
3. If your decoder needs an external dependency (e.g. LAPACK, CUDA),
   declare it in `libs/qec/lib/CMakeLists.txt` next to your `add_library`.
4. Place any public headers under `libs/qec/include/cudaq/qec/` and
   add them to that include directory's CMake install set (see
   `libs/qec/CMakeLists.txt`) if downstream users should `#include`
   them.
5. Rebuild: `ninja install` in `build/`.
6. Verify with `ctest -R my_decoder`.

## Python visibility

Decoders registered via `CUDAQ_REGISTER_DECODER` show up in
`qec.get_decoder("name", H, **kwargs)` from Python automatically.
kwargs are forwarded into the C++ `heterogeneous_map`. No extra
pybind11 work needed unless you want a richer Python interface
(e.g. methods beyond `decode`).

## Real-time eligibility

To make your decoder real-time eligible (callable from
`@cudaq.kernel`), it must additionally:

1. Implement the real-time decoder interface (see
   `libs/qec/include/cudaq/qec/realtime/decoding.h`).
2. Provide a `decoder_config`-compatible config struct.
3. Be uploadable to the dispatch system (if targeting Quantinuum) or
   wired into the autonomous_decoder CRTP path (if GPU-resident).

For full details: `cuda-qx-qec-realtime/references/autonomous-decoder.md`.

## GPU integration

If your decoder uses CUDA:

- Use `nvtx_helpers.h` for profiling ranges on hot paths.
- Allocate device memory once at construction; reuse across calls.
- Avoid host fences inside `decode()`; they kill latency.
- For multi-stream parallelism, take a CUDA stream parameter via
  `heterogeneous_map` (see `nv-qldpc-decoder` for the pattern).

## Common pitfalls

| Symptom | Cause |
|---------|-------|
| Decoder not found in `qec.list_decoders()` | macro missing, or shared library not loaded |
| `ctest` passes but Python `get_decoder` fails | wrong install prefix; `pip` install shadowing source build |
| Decoder works at small sizes, OOM at big | per-call allocation; preallocate at construction |
| Real-time use produces wrong corrections | non-eligible decoder type; check the eligibility table in `decode.md` |

## Self-check

```
[ ] CUDAQ_REGISTER_DECODER("name") present, name unique.
[ ] CMakeLists wired: source compiled, header installed.
[ ] ctest -R <my_decoder> passes.
[ ] qec.list_decoders() (Python) includes the new name after rebuild.
[ ] result.result length == H.shape[1].
[ ] At p=0, decoder returns zero corrections.
[ ] LER on built-in code (Steane / surface) within sanity of expected.
```

## Where next

- Real-time deployment: `cuda-qx-qec-realtime`.
- AI-decoder integration (TensorRT under the hood): `cuda-qx-qec-ai-decoders`.
- Benchmark against existing decoders: `cuda-qx-benchmarking`.
- Profile: `cuda-qx-profiling-perf`.
