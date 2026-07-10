/****************************************************************-*- C++ -*-****
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <cstdint>

namespace cudaq::qec::decoding::simulation {

struct boolean_span {
  uint8_t *buffer;
  uint64_t length;
};

__attribute__((visibility("default"))) void
enqueue_syndromes(std::uint64_t decoder_id, uint8_t *syndromes,
                  std::uint64_t syndrome_length, std::uint64_t tag);

__attribute__((visibility("default"))) void
get_corrections(std::uint64_t decoder_id, uint8_t *corrections,
                std::uint64_t correction_length, bool reset);

__attribute__((visibility("default"))) void
reset_decoder(std::uint64_t decoder_id);

} // namespace cudaq::qec::decoding::simulation
