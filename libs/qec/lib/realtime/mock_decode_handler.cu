/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/qec/realtime/mock_decode_handler.cuh"

namespace cudaq::qec::realtime {

//==============================================================================
// Global Device Context
//==============================================================================

__device__ mock_decoder_context *g_mock_decoder_ctx = nullptr;

//==============================================================================
// Core Mock Decode Handler
//==============================================================================

__device__ void
mock_decode_handler(const mock_decoder_context &ctx,
                    const uint8_t *__restrict__ input_measurements,
                    uint8_t *__restrict__ output_corrections,
                    std::size_t num_measurements, std::size_t num_observables) {

  // Thread 0 does the lookup (simple single-threaded implementation for CI)
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    // Default: if no match found, output zeros
    for (std::size_t i = 0; i < num_observables; ++i) {
      output_corrections[i] = 0;
    }

    // Search for matching measurements in lookup table
    for (std::size_t entry = 0; entry < ctx.num_lookup_entries; ++entry) {
      const uint8_t *entry_measurements =
          ctx.lookup_measurements + entry * num_measurements;

      // Check if this entry matches the input
      bool match = true;
      for (std::size_t i = 0; i < num_measurements; ++i) {
        if (entry_measurements[i] != input_measurements[i]) {
          match = false;
          break;
        }
      }

      if (match) {
        // Found match - copy expected corrections to output
        const uint8_t *entry_corrections =
            ctx.lookup_corrections + entry * num_observables;
        for (std::size_t i = 0; i < num_observables; ++i) {
          output_corrections[i] = entry_corrections[i];
        }
        break;
      }
    }
  }

  // Single-threaded handler; no block-wide sync required.
}

//==============================================================================
// DeviceRPCFunction-Compatible Wrapper
//==============================================================================

__device__ int mock_decode_rpc(void *buffer, std::uint32_t arg_len,
                               std::uint32_t max_result_len,
                               std::uint32_t *result_len) {

  // Get global context
  mock_decoder_context *ctx = g_mock_decoder_ctx;
  if (ctx == nullptr) {
    *result_len = 0;
    return -1; // Error: context not set
  }

  // Input is measurements, output is corrections (written in-place)
  uint8_t *measurements = static_cast<uint8_t *>(buffer);
  uint8_t *corrections = static_cast<uint8_t *>(buffer);

  // Perform decode
  mock_decode_handler(*ctx, measurements, corrections, ctx->num_measurements,
                      ctx->num_observables);

  *result_len = static_cast<std::uint32_t>(ctx->num_observables);
  return 0; // Success
}

__device__ auto get_mock_decode_rpc_ptr() { return &mock_decode_rpc; }

//==============================================================================
// Legacy Direct-Call Wrapper
//==============================================================================

__device__ void mock_decode_dispatch(void *ctx_ptr, const uint8_t *input,
                                     std::size_t input_size, uint8_t *output,
                                     std::size_t output_size) {

  const mock_decoder_context *ctx =
      static_cast<const mock_decoder_context *>(ctx_ptr);

  mock_decode_handler(*ctx, input, output, ctx->num_measurements,
                      ctx->num_observables);
}

} // namespace cudaq::qec::realtime
