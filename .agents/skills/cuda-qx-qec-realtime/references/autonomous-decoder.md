# Autonomous decoder (GPU-resident, zero-CPU)

CRTP-based interface for decoders that run entirely on GPU,
integrated with CUDA-Q's realtime dispatch kernel. Use when the
syndrome arrives directly to GPU memory (FPGA DMA or kernel
measurement) and the decoder must produce corrections in the same
GPU stream, without ever waking the CPU.

## Authoritative reference

`docs/autonomous_decoder_guide.md` is the developer guide. Read it
end-to-end before writing CUDA code; the CRTP / RPC contract is easy
to get wrong silently.

Source headers:

- `libs/qec/include/cudaq/qec/realtime/autonomous_decoder.cuh`
- `libs/qec/include/cudaq/qec/realtime/decoder_context.h`
- `libs/qec/include/cudaq/qec/realtime/decoding.h`,
  `decoding_config.h`
- `libs/qec/include/cudaq/qec/realtime/gpu_kernels.cuh`,
  `graph_resources.h`, `sparse_to_csr.h`

## The CRTP base

```cpp
template <typename Derived>
class autonomous_decoder {
public:
  __device__ void decode(const uint8_t* measurements,
                         uint8_t* corrections,
                         std::size_t num_measurements,
                         std::size_t num_observables) {
    static_cast<Derived*>(this)->decode_impl(
        measurements, corrections, num_measurements, num_observables);
  }
};
```

Your decoder inherits from `autonomous_decoder<YourDecoder>` and
implements `decode_impl` as a `__device__` member. No runtime
polymorphism, no virtual calls — the dispatch happens at compile
time.

## Minimal custom decoder

1. **Define a context struct** in a new header
   (`libs/qec/include/cudaq/qec/realtime/my_decoder_context.h`):

   ```cpp
   struct my_decoder_context : public decoder_context_base {
     // your matrices, params, LUTs in device-accessible memory
   };
   ```

2. **Implement the decoder** in a `.cu` file:

   ```cpp
   class my_decoder : public autonomous_decoder<my_decoder> {
   public:
     __device__ void decode_impl(const uint8_t* measurements,
                                 uint8_t* corrections,
                                 std::size_t num_meas,
                                 std::size_t num_obs);
   };
   ```

3. **Register a host-side factory + RPC handler.** The dispatch
   kernel calls your handler when an RPC message with your
   `function_id` arrives.

4. **Build it as a CUDA library and link against the realtime
   dispatch infrastructure.**

The mock decoder example in `docs/autonomous_decoder_guide.md`
("Example: Mock Decoder") is the smallest end-to-end version and the
correct starting point for a copy-modify workflow.

## Architecture (one-page)

```
FPGA/Hardware  ─►  Mapped pinned ring buffer (host-visible, GPU-resident)
                     │
                     ▼
              Host-side spin-polling dispatcher  ─►  cudaGraphLaunch(worker[i])
                     │
                     ▼
              Persistent dispatch kernel (GPU)
                     │  routes by function_id
                     ▼
              Your RPC handler (__device__)
                     │  calls
                     ▼
              autonomous_decoder<T>::decode  (CRTP → your decode_impl)
                     │
                     ▼
              Write corrections back into ring buffer; signal ready_flag
```

The CPU never touches the decode path after setup. Latency budget is
hardware-defined; aim for "single-digit microseconds per shot" on
small codes with current-gen NVIDIA GPUs.

## Conventions

1. **No host memory access in `decode_impl`.** All matrices and LUTs
   must be in device memory or mapped pinned memory. Touching host
   memory from a device kernel triggers a CPU round-trip and breaks
   the latency budget.
2. **No dynamic allocation.** Every buffer must be preallocated.
3. **CUDA Graph capture is one-shot.** Capture the dispatch graph
   once at setup; reuse forever. Re-capture is fatally slow.
4. **NVTX ranges in `decode_impl` and the dispatch loop only.**
   Setup should not be NVTX-instrumented; it confuses `nsys`
   timelines.
5. **Use `sparse_to_csr.h` helpers** to upload PCMs / observables in
   the layout the dispatch kernel expects.

## Testing your decoder

```bash
# Build with the QEC realtime app examples enabled (see cuda-qx-build).
# Then run the autonomous decoder unit tests:
ctest -R autonomous_decoder
```

Source: `libs/qec/unittests/realtime/`. Add a new test alongside
`mock_decoder_test.cpp` (when present) using your registered
`function_id`.

## Self-check

```
[ ] decode_impl is __device__ only; no host accesses.
[ ] Matrices live in device or mapped pinned memory.
[ ] CUDA Graph captured once; never re-captured.
[ ] function_id is unique across decoders in the dispatch table.
[ ] Mock-decoder regression test passes after your registration.
[ ] At p=0 the decoder returns all-zero corrections.
```

## When stuck

1. Read `docs/autonomous_decoder_guide.md` end-to-end.
2. Diff your context struct against the mock decoder's.
3. Profile with `nsys` (see `cuda-qx-profiling-perf`) — a long tail
   in the dispatch kernel almost always means a host fence somewhere.
4. Set `CUDAQ_QEC_DEBUG_DECODER=1` for runtime logging.
5. For AI-decoder variants (TensorRT inside `autonomous_decoder`),
   see `references/ai-predecoder-pipeline.md`.
