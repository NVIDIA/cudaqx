/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// The translation unit that compiles the `memory_circuit` device kernels into
// libcudaq-qec. This TU must exist: it is what turns the header-defined kernels
// into linkable symbols + JIT kernel registrations for the sample/DEM entry
// points in experiments.cpp (and for applications, which only include the
// declarations in cudaq/qec/device/memory_circuit.h).
#include "cudaq/qec/device/memory_circuit.h"

namespace cudaq::qec::decoding {

// Default no-op implementations of the realtime decoding API (see
// cudaq/qec/realtime/decoding.h), so that binaries which do not link a target
// decoding shim still link and run: enqueues and resets do nothing, and
// corrections come back all-zero.
//
// Deliberately defined in the SAME translation unit as the kernels above:
// nvq++ inlines these empty bodies into the library's registered copy of the
// kernels, making that copy deterministically decoding-free (it serves the
// sample/DEM entry points). Realtime applications compile their own kernel
// copy (by including memory_circuit.h in one nvq++ TU) and link a
// decoding shim, whose implementations their copy binds to instead.

__qpu__ void
enqueue_syndromes(std::uint64_t decoder_id,
                  const std::vector<cudaq::measure_result> &syndromes,
                  std::uint64_t tag) {}

__qpu__ void enqueue_syndromes_test(std::uint64_t decoder_id,
                                    const std::vector<bool> &syndromes,
                                    std::uint64_t tag) {}

__qpu__ std::vector<bool> get_corrections(std::uint64_t decoder_id,
                                          std::uint64_t return_size,
                                          bool reset) {
  std::vector<bool> result(return_size);
  return result;
}

__qpu__ void reset_decoder(std::uint64_t decoder_id) {}

} // namespace cudaq::qec::decoding
