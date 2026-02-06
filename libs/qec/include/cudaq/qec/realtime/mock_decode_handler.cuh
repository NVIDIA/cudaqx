/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

#include "cudaq/qec/realtime/decoder_context.h"

// cudaq_function_entry_t for function table initialization
#include "cudaq/nvqlink/daemon/dispatcher/cudaq_realtime.h"
#include "cudaq/nvqlink/daemon/dispatcher/dispatch_kernel_launch.h"

namespace cudaq::qec::realtime {

//==============================================================================
// Global Device Context
//==============================================================================

/// @brief Global device pointer to mock decoder context.
/// Must be set via set_mock_decoder_context() before kernel launch.
extern __device__ mock_decoder_context *g_mock_decoder_ctx;

/// @brief Set the mock decoder context from host.
/// @param ctx Device pointer to mock_decoder_context
inline void set_mock_decoder_context(mock_decoder_context *ctx) {
  cudaMemcpyToSymbol(g_mock_decoder_ctx, &ctx, sizeof(mock_decoder_context *));
}

//==============================================================================
// Core Mock Decode Handler
//==============================================================================

/// @brief Mock decode handler for CI testing.
///
/// This handler looks up the input measurements in a pre-recorded table
/// and returns the corresponding expected corrections. It does NOT perform
/// actual decoding - it simply verifies that the infrastructure can correctly
/// route data through the decode pipeline.
///
/// @param ctx Mock decoder context with lookup table pointers
/// @param input_measurements Input raw measurements from RX buffer
/// @param output_corrections Output buffer for corrections (TX buffer)
/// @param num_measurements Size of input measurements
/// @param num_observables Number of observables (corrections to output)
__device__ void
mock_decode_handler(const mock_decoder_context &ctx,
                    const uint8_t *__restrict__ input_measurements,
                    uint8_t *__restrict__ output_corrections,
                    std::size_t num_measurements, std::size_t num_observables);

//==============================================================================
// DeviceRPCFunction-Compatible Wrapper
//==============================================================================

/// @brief RPC-compatible wrapper for the mock decoder.
///
/// This function matches the DeviceRPCFunction signature expected by the
/// cuda-quantum dispatch kernel.
///
/// @param buffer Input measurements / output corrections buffer
/// @param arg_len Length of input measurement data
/// @param max_result_len Maximum space available for results
/// @param result_len Output: actual result length written
/// @return 0 on success, non-zero on error
__device__ int mock_decode_rpc(void *buffer, std::uint32_t arg_len,
                               std::uint32_t max_result_len,
                               std::uint32_t *result_len);

/// @brief Get device function pointer for mock_decode_rpc.
/// @return Device function pointer to mock_decode_rpc
__device__ auto get_mock_decode_rpc_ptr();

//==============================================================================
// Legacy Direct-Call Wrapper
//==============================================================================

/// @brief Wrapper for direct device-to-device calls (non-RPC path).
__device__ void mock_decode_dispatch(void *ctx_ptr, const uint8_t *input,
                                     std::size_t input_size, uint8_t *output,
                                     std::size_t output_size);

//==============================================================================
// Mock Decoder Context GPU Setup
//==============================================================================

/// @brief GPU resources for mock decoder context.
///
/// Holds device pointers allocated during setup. Call cleanup() to free.
struct MockDecoderGpuResources {
  uint8_t *d_lookup_measurements = nullptr;
  uint8_t *d_lookup_corrections = nullptr;
  mock_decoder_context *d_ctx = nullptr;

  void cleanup() {
    if (d_lookup_measurements)
      cudaFree(d_lookup_measurements);
    if (d_lookup_corrections)
      cudaFree(d_lookup_corrections);
    if (d_ctx)
      cudaFree(d_ctx);
    d_lookup_measurements = nullptr;
    d_lookup_corrections = nullptr;
    d_ctx = nullptr;
  }
};

/// @brief Build lookup tables and upload mock decoder context to the GPU.
///
/// After this call, the global device context is set and the dispatch kernel
/// can invoke mock_decode_rpc.
///
/// @param measurements Flat array of measurements (num_entries * syndrome_size)
/// @param corrections  Array of expected corrections (num_entries)
/// @param num_entries  Number of lookup entries
/// @param syndrome_size Number of measurements per entry
/// @param[out] resources Device pointers (caller must call cleanup())
/// @return cudaSuccess on success
cudaError_t setup_mock_decoder_on_gpu(
    const uint8_t *measurements, const uint8_t *corrections,
    std::size_t num_entries, std::size_t syndrome_size,
    MockDecoderGpuResources &resources);

//==============================================================================
// Function Table Initialization
//==============================================================================

/// @brief Function ID for mock decoder (FNV-1a hash of "mock_decode").
constexpr std::uint32_t MOCK_DECODE_FUNCTION_ID =
    cudaq::nvqlink::fnv1a_hash("mock_decode");

/// @brief Device kernel to initialize a cudaq function table entry for the
///        mock decoder RPC handler.
///
/// Must be called as: init_mock_decode_function_table<<<1,1>>>(d_entries);
/// @param entries Device pointer to pre-allocated cudaq_function_entry_t array
__global__ void init_mock_decode_function_table(
    cudaq_function_entry_t *entries);

/// @brief Host-callable wrapper that launches init_mock_decode_function_table
///        and synchronizes.
///
/// This allows callers compiled by a C++ compiler (not nvcc) to set up the
/// function table without needing CUDA kernel launch syntax.
/// @param d_entries Device pointer to pre-allocated cudaq_function_entry_t
void setup_mock_decode_function_table(cudaq_function_entry_t *d_entries);

} // namespace cudaq::qec::realtime
