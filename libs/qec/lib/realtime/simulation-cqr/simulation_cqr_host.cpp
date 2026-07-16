/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Definitions for the generic device_call targets referenced by
// simulation_cqr_device.cpp. These exist for two reasons:
//   1. To satisfy the link: the -frealtime-lowering'd device stubs still emit
//   an
//      undefined reference to the callee symbol that the final link must
//      resolve.
//   2. To service the direct/emulate channel, where the device_call resolves
//      straight to these (the host-dispatch channel instead routes by
//      function_id to the service in decoding_server_cqr.cpp and never calls
//      these).
//
// They are given HIDDEN visibility: the device stubs resolve them within this
// library, but the deliberately generic names (enqueue_syndromes, ...) are NOT
// exported, so they cannot collide with symbols in other libraries.

#include "../realtime_decoding.h"

#include <cstdint>

namespace {
// nvq++ lowers a std::vector<bool> device_call argument to a {pointer, length}
// span; the trampolines must accept that layout (same trick as the simulation
// backend's boolean_span).
struct boolean_span {
  std::uint8_t *buffer;
  std::uint64_t length;
};
} // namespace

extern "C" __attribute__((visibility("hidden"))) void
enqueue_syndromes(std::uint64_t decoder_id, boolean_span syndromes,
                  std::uint64_t tag) {
  cudaq::qec::decoding::host::enqueue_syndromes(decoder_id, syndromes.buffer,
                                                syndromes.length, tag);
}

extern "C" __attribute__((visibility("hidden"))) void
get_corrections(std::uint64_t decoder_id, boolean_span corrections,
                bool reset) {
  cudaq::qec::decoding::host::get_corrections(decoder_id, corrections.buffer,
                                              corrections.length, reset);
}

extern "C" __attribute__((visibility("hidden"))) void
reset_decoder(std::uint64_t decoder_id) {
  cudaq::qec::decoding::host::reset_decoder(decoder_id);
}
