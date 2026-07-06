/****************************************************************-*- C++ -*-****
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/qec/realtime/decoding_config.h"
#include <cstdint>

// Note: none of these are intended to be user-facing functions.

namespace cudaq::qec::realtime {
class qec_realtime_session;
} // namespace cudaq::qec::realtime

namespace cudaq::qec::decoding::host {

/// @brief Accessor for the per-process realtime session. Returns nullptr
/// unless CUDAQ_QEC_REALTIME_MODE=inproc_rpc has initialized the shared-ring
/// dispatch session.
__attribute__((visibility("default")))
cudaq::qec::realtime::qec_realtime_session *
get_realtime_session();

__attribute__((visibility("default"))) void
enqueue_syndromes(std::size_t decoder_id, uint8_t *syndromes,
                  std::uint64_t syndrome_length, std::uint64_t tag);

/// @brief Zero-copy staging for transport handlers: reserve the decoder's
/// execution-worker ring slot and return its payload buffer (blocks while
/// the ring is full -- natural backpressure). The caller writes the
/// unpacked byte-per-bool syndrome directly into it and calls
/// commit_syndromes; decoder execution then happens on the decoder's own
/// worker thread, so multiple decoders (logical qubits) run concurrently.
/// Per-decoder FIFO order is the stage/commit order (single staging
/// producer: the transport dispatcher thread).
__attribute__((visibility("default"))) uint8_t *
stage_syndromes(std::size_t decoder_id, std::uint64_t max_length);

/// @brief Commit the syndrome staged by stage_syndromes for asynchronous
/// execution on the decoder's worker thread. A decoder exception during
/// deferred execution becomes sticky and is rethrown at that decoder's
/// next get_corrections (reset_decoder clears it).
__attribute__((visibility("default"))) void
commit_syndromes(std::size_t decoder_id, std::uint64_t length,
                 std::uint64_t tag);

/// @brief High-water mark of concurrently-busy decoder execution workers
/// since decoder initialization (concurrency evidence for tests/stats).
__attribute__((visibility("default"))) std::uint64_t
max_concurrent_decoder_workers();

__attribute__((visibility("default"))) cudaqx::heterogeneous_map
prepare_decoder_params(
    const cudaq::qec::decoding::config::decoder_config &decoder_config);

__attribute__((visibility("default"))) void
get_corrections(std::size_t decoder_id, uint8_t *corrections,
                std::uint64_t correction_length, bool reset);

__attribute__((visibility("default"))) void
reset_decoder(std::size_t decoder_id);

int configure_decoders(
    cudaq::qec::decoding::config::multi_decoder_config &config);
int configure_decoders_from_file(const char *config_file);
int configure_decoders_from_str(const char *config_str);
void finalize_decoders();

/// @brief Set a callback to capture syndrome data as it's enqueued.
/// Used by --save_syndrome feature to record syndromes to file.
/// @param callback Function pointer that receives packed syndrome bytes.
///                 Set to nullptr to disable capture.
__attribute__((visibility("default"))) void
set_syndrome_capture_callback(void (*callback)(const uint8_t *, size_t));

} // namespace cudaq::qec::decoding::host
