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

#include "cudaq/qec/realtime/autonomous_decoder.cuh"
#include "cudaq/qec/realtime/decoder_context.h"

namespace cudaq::qec::realtime {

//==============================================================================
// Mock Decoder Class (CRTP)
//==============================================================================

/// @brief Mock decoder for CI testing, inheriting from autonomous_decoder.
///
/// This decoder uses a lookup table to map input measurements to expected
/// corrections. It demonstrates the autonomous_decoder CRTP pattern and
/// provides a testable implementation for verifying the real-time dispatch
/// infrastructure.
class mock_decoder : public autonomous_decoder<mock_decoder> {
public:
  /// @brief Constructor taking a reference to the decoder context.
  ///
  /// @param ctx Reference to mock_decoder_context with lookup tables
  __device__ __host__ explicit mock_decoder(mock_decoder_context &ctx)
      : ctx_(ctx) {}

  /// @brief Core decode implementation (required by CRTP base).
  ///
  /// Searches the lookup table for matching measurements and returns the
  /// corresponding corrections. This is called by the base class's decode()
  /// method.
  ///
  /// @param measurements Input raw measurements from quantum circuit
  /// @param corrections Output buffer for computed corrections
  /// @param num_measurements Number of input measurements
  /// @param num_observables Number of observable corrections to output
  __device__ void decode_impl(const uint8_t *__restrict__ measurements,
                              uint8_t *__restrict__ corrections,
                              std::size_t num_measurements,
                              std::size_t num_observables);

  /// @brief Get the decoder context.
  /// @return Reference to the mock_decoder_context
  __device__ __host__ mock_decoder_context &context() { return ctx_; }

  /// @brief Get the decoder context (const version).
  __device__ __host__ const mock_decoder_context &context() const {
    return ctx_;
  }

private:
  mock_decoder_context &ctx_; ///< Reference to decoder context
};

//==============================================================================
// Global Device Decoder Instance
//==============================================================================

/// @brief Global device pointer to mock decoder instance.
/// Must be set via set_mock_decoder() before kernel launch.
extern __device__ mock_decoder *g_mock_decoder;

/// @brief Set the mock decoder instance from host.
/// @param decoder Device pointer to mock_decoder instance
inline void set_mock_decoder(mock_decoder *decoder) {
  cudaMemcpyToSymbol(g_mock_decoder, &decoder, sizeof(mock_decoder *));
}

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
// Graph-Compatible RPC Handler (for CUDAQ_DISPATCH_GRAPH_LAUNCH)
//==============================================================================

/// @brief Graph kernel using pointer indirection pattern.
///
/// This kernel reads the buffer address from a device pointer, allowing
/// the launching kernel to update which buffer to process before each launch.
///
/// @param buffer_ptr Pointer to current RPC buffer address (device pointer)
__global__ void mock_decode_graph_kernel(void **buffer_ptr);

} // namespace cudaq::qec::realtime
