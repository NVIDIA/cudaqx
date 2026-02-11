/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/nvqlink/daemon/dispatcher/dispatch_kernel_launch.h"
#include "cudaq/qec/realtime/mock_decode_handler.cuh"

namespace cudaq::qec::realtime {

//==============================================================================
// Global Device Decoder Instance
//==============================================================================

__device__ mock_decoder *g_mock_decoder = nullptr;

//==============================================================================
// Mock Decoder Class Implementation
//==============================================================================

__device__ void mock_decoder::decode_impl(
    const uint8_t *__restrict__ measurements, uint8_t *__restrict__ corrections,
    std::size_t num_measurements, std::size_t num_observables) {
  // Thread 0 does the lookup (simple single-threaded implementation for CI)
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    // Default: if no match found, output zeros
    for (std::size_t i = 0; i < num_observables; ++i) {
      corrections[i] = 0;
    }

    // Search for matching measurements in lookup table
    for (std::size_t entry = 0; entry < ctx_.num_lookup_entries; ++entry) {
      const uint8_t *entry_measurements =
          ctx_.lookup_measurements + entry * num_measurements;

      // Check if this entry matches the input
      bool match = true;
      for (std::size_t i = 0; i < num_measurements; ++i) {
        if (entry_measurements[i] != measurements[i]) {
          match = false;
          break;
        }
      }

      if (match) {
        // Found match - copy expected corrections to output
        const uint8_t *entry_corrections =
            ctx_.lookup_corrections + entry * num_observables;
        for (std::size_t i = 0; i < num_observables; ++i) {
          corrections[i] = entry_corrections[i];
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

  if (g_mock_decoder != nullptr) {
    uint8_t *measurements = static_cast<uint8_t *>(buffer);
    uint8_t *corrections = static_cast<uint8_t *>(buffer);

    const auto &ctx = g_mock_decoder->context();
    g_mock_decoder->decode(measurements, corrections, ctx.num_measurements,
                           ctx.num_observables);

    *result_len = static_cast<std::uint32_t>(ctx.num_observables);
    return 0; // Success
  } else {
    // Error: decoder not set
    *result_len = 0;
    return -1;
  }
}

__device__ auto get_mock_decode_rpc_ptr() { return &mock_decode_rpc; }

//==============================================================================
// Graph-Compatible RPC Handler (for CUDAQ_DISPATCH_GRAPH_LAUNCH)
//==============================================================================

__global__ void mock_decode_graph_kernel(void **buffer_ptr) {
  void *data_buffer = (buffer_ptr != nullptr) ? *buffer_ptr : nullptr;
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    if (data_buffer == nullptr)
      return;

    // Parse RPC header
    auto *header = static_cast<cudaq::nvqlink::RPCHeader *>(data_buffer);
    void *arg_buffer = static_cast<void *>(header + 1);

    auto *response = static_cast<cudaq::nvqlink::RPCResponse *>(data_buffer);

    if (g_mock_decoder != nullptr) {
      uint8_t *measurements = static_cast<uint8_t *>(arg_buffer);
      uint8_t *corrections = static_cast<uint8_t *>(arg_buffer);

      const auto &ctx = g_mock_decoder->context();
      g_mock_decoder->decode(measurements, corrections, ctx.num_measurements,
                             ctx.num_observables);

      // Write response
      response->magic = cudaq::nvqlink::RPC_MAGIC_RESPONSE;
      response->status = 0;
      response->result_len = static_cast<std::uint32_t>(ctx.num_observables);
    } else {
      // Error: decoder not set
      response->magic = cudaq::nvqlink::RPC_MAGIC_RESPONSE;
      response->status = -1;
      response->result_len = 0;
    }
  }
}

} // namespace cudaq::qec::realtime
