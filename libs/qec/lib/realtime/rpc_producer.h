/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace cudaq::qec::realtime {
class qec_realtime_session;
} // namespace cudaq::qec::realtime

namespace cudaq::qec::decoding::rpc_producer {

struct dispatcher_unresponsive_error : std::runtime_error {
  using std::runtime_error::runtime_error;
};

constexpr int kAcquireSlotTimeoutMs = 5000;
constexpr int kResponseTimeoutMs = 5000;

__attribute__((visibility("default"))) void
enqueue_syndromes(cudaq::qec::realtime::qec_realtime_session &session,
                  std::size_t decoder_id, const std::uint8_t *syndromes,
                  std::uint64_t num_syndromes, std::uint64_t counter,
                  std::uint64_t syndrome_mapping_id);

__attribute__((visibility("default"))) void
get_corrections(cudaq::qec::realtime::qec_realtime_session &session,
                std::size_t decoder_id, std::uint8_t *corrections,
                std::uint64_t correction_length, std::uint64_t reset);

__attribute__((visibility("default"))) void
reset_decoder(cudaq::qec::realtime::qec_realtime_session &session,
              std::size_t decoder_id);

} // namespace cudaq::qec::decoding::rpc_producer
