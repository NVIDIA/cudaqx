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

namespace cudaq::qec::decoding::host {

__attribute__((visibility("default"))) void
enqueue_syndromes(std::size_t decoder_id, uint8_t *syndromes,
                  std::uint64_t syndrome_length, std::uint64_t tag);

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
