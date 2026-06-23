/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq.h"
#include "cudaq/qec/realtime/decoding.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <vector>

extern "C" std::uint64_t
qec_enqueue_syndromes_ui64([[maybe_unused]] std::uint64_t decoder_id,
                           [[maybe_unused]] std::uint64_t syndrome_size,
                           [[maybe_unused]] std::uint64_t syndrome,
                           [[maybe_unused]] std::uint64_t tag) {
  std::abort();
}

extern "C" std::uint64_t
qec_get_corrections_ui64([[maybe_unused]] std::uint64_t decoder_id,
                         [[maybe_unused]] std::uint64_t return_size,
                         [[maybe_unused]] std::uint64_t reset) {
  std::abort();
}

extern "C" std::uint64_t
qec_reset_decoder_ui64([[maybe_unused]] std::uint64_t decoder_id) {
  std::abort();
}

namespace cudaq::qec::decoding {

__qpu__ void
enqueue_syndromes(std::uint64_t decoder_id,
                  const std::vector<cudaq::measure_result> &syndromes,
                  std::uint64_t tag) {
  const std::uint64_t syndrome_size = syndromes.size();
  const std::uint64_t syndrome = cudaq::to_integer(cudaq::to_bools(syndromes));
  // The ignored ack keeps host-dispatch from treating this as fire-and-forget.
  (void)cudaq::device_call(0, qec_enqueue_syndromes_ui64, decoder_id,
                           syndrome_size, syndrome, tag);
}

__qpu__ void enqueue_syndromes_test(std::uint64_t decoder_id,
                                    const std::vector<bool> &syndromes,
                                    std::uint64_t tag) {
  const std::uint64_t syndrome_size = syndromes.size();
  const std::uint64_t syndrome = cudaq::to_integer(syndromes);
  // The ignored ack keeps host-dispatch from treating this as fire-and-forget.
  (void)cudaq::device_call(0, qec_enqueue_syndromes_ui64, decoder_id,
                           syndrome_size, syndrome, tag);
}

__qpu__ std::vector<bool> get_corrections(std::uint64_t decoder_id,
                                          std::uint64_t return_size,
                                          bool reset) {
  std::vector<bool> result(return_size);
  const auto packed =
      cudaq::device_call(0, qec_get_corrections_ui64, decoder_id, return_size,
                         static_cast<std::uint64_t>(reset));
  for (std::size_t i = 0; i < return_size; ++i)
    result[i] = (packed >> i) & 1;
  return result;
}

__qpu__ void reset_decoder(std::uint64_t decoder_id) {
  // The ignored ack keeps host-dispatch from treating this as fire-and-forget.
  (void)cudaq::device_call(0, qec_reset_decoder_ui64, decoder_id);
}

} // namespace cudaq::qec::decoding
