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
// The wrappers also perform the spec's on-wire rewrite themselves (until the
// device_call lowering can): syndrome/correction bits cross the wire
// bit-packed (LSB-first) in a std::vector<uint8_t> placed last, with the
// explicit num_syndromes / return_size scalars of the spec's
// enqueue_syndromes / get_corrections wire layouts. The stdvec<u8>
// serialization's uint64 byte-count prefix is the spec's ARRAY_UINT8 prefix.
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

// device_call targets, in the spec's on-wire argument order (variable-length
// byte vector last). Named to match the spec function_ids. Defined (hidden) in
// simulation_cqr_host.cpp.
extern "C" {
void enqueue_syndromes(std::uint64_t decoder_id, std::uint64_t counter,
                       std::uint64_t syndrome_mapping_id,
                       std::uint64_t num_syndromes,
                       const std::vector<std::uint8_t> &syndrome_bits);

void get_corrections(std::uint64_t decoder_id, std::uint64_t return_size,
                     std::vector<std::uint8_t> &corrections, bool reset);

void reset_decoder(std::uint64_t decoder_id);
}

namespace cudaq::qec::decoding {

__qpu__ void
enqueue_syndromes(std::uint64_t decoder_id,
                  const std::vector<cudaq::measure_result> &syndromes,
                  std::uint64_t tag) {
  // No syndrome mapping table yet; id 0 is the identity mapping.
  constexpr std::uint64_t kSyndromeMappingId = 0;
  // Discriminating the measurement handles to bits (element reads below) and
  // packing LSB-first before crossing the boundary.
  std::size_t num_bits = syndromes.size();
  std::size_t num_bytes = (num_bits + 7) / 8;
  std::vector<std::uint8_t> packed(num_bytes);
  for (std::size_t byte = 0; byte < num_bytes; ++byte) {
    std::uint8_t value = 0;
    for (std::size_t bit = 0; bit < 8; ++bit) {
      std::size_t index = byte * 8 + bit;
      if (index < num_bits) {
        bool is_set = syndromes[index];
        if (is_set)
          value = value | static_cast<std::uint8_t>(1 << bit);
      }
    }
    packed[byte] = value;
  }
  cudaq::device_call(::enqueue_syndromes, decoder_id, tag, kSyndromeMappingId,
                     static_cast<std::uint64_t>(num_bits), packed);
}

__qpu__ void enqueue_syndromes_test(std::uint64_t decoder_id,
                                    const std::vector<bool> &syndromes,
                                    std::uint64_t tag) {
  constexpr std::uint64_t kSyndromeMappingId = 0;
  std::size_t num_bits = syndromes.size();
  std::size_t num_bytes = (num_bits + 7) / 8;
  std::vector<std::uint8_t> packed(num_bytes);
  for (std::size_t byte = 0; byte < num_bytes; ++byte) {
    std::uint8_t value = 0;
    for (std::size_t bit = 0; bit < 8; ++bit) {
      std::size_t index = byte * 8 + bit;
      if (index < num_bits) {
        std::uint8_t bit_val = syndromes[index];
        value = value | static_cast<std::uint8_t>(bit_val << bit);
      }
    }
    packed[byte] = value;
  }
  cudaq::device_call(::enqueue_syndromes, decoder_id, tag, kSyndromeMappingId,
                     static_cast<std::uint64_t>(num_bits), packed);
}

__qpu__ std::vector<bool> get_corrections(std::uint64_t decoder_id,
                                          std::uint64_t return_size,
                                          bool reset) {
  std::size_t num_bytes = (return_size + 7) / 8;
  std::vector<std::uint8_t> packed(num_bytes);
  for (std::size_t byte = 0; byte < num_bytes; ++byte)
    packed[byte] = 0;
  cudaq::device_call(::get_corrections, decoder_id, return_size, packed, reset);
  std::vector<bool> result(return_size);
  for (std::size_t i = 0; i < return_size; ++i) {
    std::uint32_t byte_val = packed[i / 8];
    std::uint32_t shift = static_cast<std::uint32_t>(i % 8);
    std::uint32_t bit_val = (byte_val >> shift) & 1u;
    result[i] = bit_val == 1u;
  }
  return result;
}

__qpu__ void reset_decoder(std::uint64_t decoder_id) {
  cudaq::device_call(::reset_decoder, decoder_id);
}

} // namespace cudaq::qec::decoding
