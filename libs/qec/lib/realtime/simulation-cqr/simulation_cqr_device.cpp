/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Backend-neutral device-side wrappers for the QEC realtime decoding API.
//
// These mirror the simulation backend's __qpu__ wrappers, but the cudaq::
// device_call targets use the *generic* callee names defined by
// cudaq-spec/proposals/decoder_server_runtime.md -- enqueue_syndromes /
// get_corrections / reset_decoder -- rather than the simulation backend's
// simulation_* / C++-mangled symbols. The realtime function_id is
// fnv1a_32(<callee name>), so these names are what the host-dispatch service
// (decoding_server_cqr.cpp) registers.
//
// This translation unit is compiled with -frealtime-lowering (see the
// cudaq-qec-realtime-decoding-simulation-cqr library), which rewrites each
// device_call to the realtime frame ABI and dispatches it by function_id
// through the host-dispatch channel. The callee names below are only hashed
// into the function_id; the symbols themselves are defined in
// simulation_cqr_host.cpp with HIDDEN visibility (purely to satisfy the link),
// so the very generic names are never exported and cannot collide with other
// libraries.

#include <cstdint>
#include <vector>

#include "cudaq.h"
#include "cudaq/qec/realtime/decoding.h"

// device_call targets. Named to match the spec function_ids. Defined
// (hidden) in simulation_cqr_host.cpp.
extern "C" {
void enqueue_syndromes(std::uint64_t decoder_id,
                       const std::vector<bool> &syndromes, std::uint64_t tag);

void get_corrections(std::uint64_t decoder_id, std::vector<bool> &corrections,
                     bool reset);

void reset_decoder(std::uint64_t decoder_id);
}

namespace cudaq::qec::decoding {

__qpu__ void
enqueue_syndromes(std::uint64_t decoder_id,
                  const std::vector<cudaq::measure_result> &syndromes,
                  std::uint64_t tag) {
  // Discriminate the measurement handles to bits before crossing the boundary.
  cudaq::device_call(::enqueue_syndromes, decoder_id,
                     cudaq::to_bools(syndromes), tag);
}

__qpu__ void enqueue_syndromes_test(std::uint64_t decoder_id,
                                    const std::vector<bool> &syndromes,
                                    std::uint64_t tag) {
  cudaq::device_call(::enqueue_syndromes, decoder_id, syndromes, tag);
}

__qpu__ std::vector<bool> get_corrections(std::uint64_t decoder_id,
                                          std::uint64_t return_size,
                                          bool reset) {
  std::vector<bool> result(return_size);
  cudaq::device_call(::get_corrections, decoder_id, result, reset);
  return result;
}

__qpu__ void reset_decoder(std::uint64_t decoder_id) {
  cudaq::device_call(::reset_decoder, decoder_id);
}

} // namespace cudaq::qec::decoding
