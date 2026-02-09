/*******************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "realtime_decoding.h"
#include "common/FmtCore.h"
#include "cudaq/qec/decoder.h"
#include "cudaq/qec/pcm_utils.h"
#include "cudaq/qec/realtime/decoding_config.h"
#include "cudaq/runtime/logger/logger.h"
#include <set>

// Optional syndrome capture callback for --save_syndrome feature
namespace {
using SyndromeCaptureCallback = void (*)(const uint8_t *, size_t);
SyndromeCaptureCallback g_syndrome_capture_callback = nullptr;
} // namespace

std::vector<std::unique_ptr<cudaq::qec::decoder>> g_decoders;

// Helper to pack syndrome bits into bytes (8 bits per byte, MSB first for
// readability)
static std::vector<uint8_t> pack_syndrome_bits(const uint8_t *syndromes,
                                               size_t length) {
  size_t num_bytes = (length + 7) / 8; // Round up
  std::vector<uint8_t> packed(num_bytes, 0);

  for (size_t i = 0; i < length; i++) {
    if (syndromes[i]) {
      size_t byte_idx = i / 8;
      size_t bit_idx = 7 - (i % 8); // MSB first
      packed[byte_idx] |= (1 << bit_idx);
    }
  }

  return packed;
}

namespace cudaq::qec::decoding::host {

int configure_decoders(
    cudaq::qec::decoding::config::multi_decoder_config &config) {
  CUDAQ_INFO("Initializing decoders...");

  const auto &decoder_configs = config.decoders;

  // First validate that the there are no duplicate decoder IDs.
  std::set<int64_t> decoder_ids;
  auto min_decoder_id = std::numeric_limits<int64_t>::max();
  auto max_decoder_id = std::numeric_limits<int64_t>::min();
  for (auto &decoder_config : decoder_configs) {
    if (decoder_ids.count(decoder_config.id) > 0) {
      CUDAQ_WARN("Duplicate decoder ID found: {}", decoder_config.id);
      return 1;
    }
    decoder_ids.insert(decoder_config.id);
    min_decoder_id = std::min(min_decoder_id, decoder_config.id);
    max_decoder_id = std::max(max_decoder_id, decoder_config.id);
  }

  // Then check that the maximum decoder ID is less than the number of decoders.
  if (max_decoder_id >= decoder_configs.size()) {
    CUDAQ_WARN(
        "Maximum decoder ID is greater than the number of decoders: {} >= {}",
        max_decoder_id, decoder_configs.size());
    return 2;
  }
  if (min_decoder_id < 0) {
    CUDAQ_WARN("Minimum decoder ID is less than 0: {}", min_decoder_id);
    return 3;
  }

  // Create the decoders based on the decoder configs.
  try {
    g_decoders.clear();
    g_decoders.resize(max_decoder_id + 1);
    for (const auto &decoder_config : decoder_configs) {
      // Form the PCM from the sparse vector.
      auto t0 = std::chrono::high_resolution_clock::now();
      CUDAQ_INFO("Creating decoder {} of type {}", decoder_config.id,
                 decoder_config.type);
      auto pcm = cudaq::qec::pcm_from_sparse_vec(decoder_config.H_sparse,
                                                 decoder_config.syndrome_size,
                                                 decoder_config.block_size);
      auto new_decoder = cudaq::qec::get_decoder(
          decoder_config.type, pcm,
          decoder_config.decoder_custom_args_to_heterogeneous_map());
      new_decoder->set_decoder_id(decoder_config.id);
      // Count the number of -1's in the O_sparse vector. That is the number of
      // rows (observables) in the observable matrix.
      auto num_observables = std::count(decoder_config.O_sparse.begin(),
                                        decoder_config.O_sparse.end(), -1);
      // Populate the ***real-time*** fields of the decoder.
      auto observable_matrix = cudaq::qec::pcm_from_sparse_vec(
          decoder_config.O_sparse, num_observables, decoder_config.block_size);
      new_decoder->set_O_sparse(decoder_config.O_sparse);
      if (!decoder_config.D_sparse.empty()) {
        new_decoder->set_D_sparse(decoder_config.D_sparse);
      } else {
        throw std::runtime_error(
            "D_sparse must be provided in decoder configuration");
      }

      // Invoke a dummy decoding operation to force the decoder to be
      // initialized.
      auto t1 = std::chrono::high_resolution_clock::now();
      std::vector<cudaq::qec::float_t> syndrome(decoder_config.syndrome_size,
                                                0.0);
      new_decoder->decode(syndrome);
      auto t2 = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> duration1 = t1 - t0;
      std::chrono::duration<double> duration2 = t2 - t1;
      CUDAQ_INFO(
          "Done initializing decoder {} in {:.6f} seconds (creation: {:.6f}s, "
          "initial decoding dry run: {:.6f}s)",
          decoder_config.id, duration1.count() + duration2.count(),
          duration1.count(), duration2.count());

      g_decoders[decoder_config.id] = std::move(new_decoder);
    }
  } catch (const std::exception &e) {
    CUDAQ_WARN("Error initializing decoders: {}", e.what());
    return 4;
  }
  return 0;
}

void finalize_decoders() {
  CUDAQ_INFO("Finalizing the realtime decoding library.");
  g_decoders.clear();
}

__attribute__((visibility("default"))) void
set_syndrome_capture_callback(void (*callback)(const uint8_t *, size_t)) {
  g_syndrome_capture_callback = callback;
}

void enqueue_syndromes(std::size_t decoder_id, uint8_t *syndromes,
                       std::uint64_t syndrome_length, std::uint64_t tag) {
  if (decoder_id >= g_decoders.size()) {
    throw std::invalid_argument(
        fmt::format("Decoder {} not found", decoder_id));
  }
  auto *decoder = g_decoders[decoder_id].get();
  if (!decoder) {
    throw std::invalid_argument(
        fmt::format("Decoder {} not found", decoder_id));
  }
  if (syndrome_length == 0) {
    throw std::invalid_argument("syndrome_length must be greater than 0");
  }
  if (!syndromes) {
    throw std::invalid_argument("syndromes buffer is null");
  }
  const auto max_syndromes = decoder->get_num_msyn_per_decode();
  if (max_syndromes == 0) {
    throw std::invalid_argument(
        "Decoder has no measurement syndromes configured");
  }
  if (syndrome_length > max_syndromes) {
    throw std::invalid_argument(
        fmt::format("syndrome_length ({}) exceeds configured measurement count "
                    "({})",
                    syndrome_length, max_syndromes));
  }

  // Invoke syndrome capture callback if registered (for --save_syndrome
  // feature)
  if (g_syndrome_capture_callback) {
    auto packed_syndrome = pack_syndrome_bits(syndromes, syndrome_length);
    g_syndrome_capture_callback(packed_syndrome.data(), packed_syndrome.size());
  }

  std::vector<uint8_t> syndrome_u8(syndrome_length);
  bool did_decode = false;
  for (std::size_t i = 0; i < syndrome_length; i++) {
    syndrome_u8[i] = syndromes[i];
  }
  std::chrono::duration<double> duration{};
  if (decoder) {
    auto t0 = std::chrono::high_resolution_clock::now();
    did_decode =
        decoder->enqueue_syndrome(syndrome_u8.data(), syndrome_u8.size());
    auto t1 = std::chrono::high_resolution_clock::now();
    duration = t1 - t0;
  } else {
    throw std::invalid_argument(
        fmt::format("Decoder {} not found", decoder_id));
  }

  // Consider demoting this to a lower log level.
  // Also consider logging the syndrome (at a lower log level).
  CUDAQ_INFO("[decoder={}][tag={}] enqueue_syndrome took {:.3f} us, "
             "syndrome_length={}, did_decode={}",
             decoder_id, tag, duration.count() * 1e6, syndrome_length,
             did_decode ? 'Y' : 'N');
}

void get_corrections(std::size_t decoder_id, uint8_t *corrections,
                     std::uint64_t correction_length, bool reset) {
  CUDAQ_INFO("Entered get_corrections function decoder_id={}, "
             "correction_length={}, reset={}",
             decoder_id, correction_length, reset);
  if (decoder_id >= g_decoders.size()) {
    throw std::invalid_argument(
        fmt::format("Decoder {} not found", decoder_id));
  }
  auto *decoder = g_decoders[decoder_id].get();
  if (decoder) {
    auto num_observables = decoder->get_num_observables();
    auto ret = decoder->get_obs_corrections();
    if (correction_length == 0) {
      throw std::invalid_argument("correction_length must be greater than 0");
    }
    if (!corrections) {
      throw std::invalid_argument("corrections buffer is null");
    }
    if (correction_length != num_observables) {
      throw std::invalid_argument(
          fmt::format("correction_length ({}) does not match number of "
                      "observables ({})",
                      correction_length, num_observables));
    }
    for (std::size_t i = 0; i < correction_length; ++i) {
      corrections[i] = ret[i];
    }
    if (reset)
      decoder->clear_corrections();
  } else {
    throw std::invalid_argument(
        fmt::format("Decoder {} not found", decoder_id));
  }
  return;
}

void reset_decoder(std::size_t decoder_id) {
  CUDAQ_INFO("Entered reset_decoder for decoder_id={}", decoder_id);
  if (decoder_id >= g_decoders.size()) {
    throw std::invalid_argument(
        fmt::format("Decoder {} not found", decoder_id));
  }
  auto *decoder = g_decoders[decoder_id].get();
  if (decoder) {
    decoder->reset_decoder();
  } else {
    throw std::invalid_argument(
        fmt::format("Decoder {} not found", decoder_id));
  }
}

} // namespace cudaq::qec::decoding::host
