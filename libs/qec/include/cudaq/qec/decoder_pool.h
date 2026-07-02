/*******************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "cudaq/qec/decoder.h"
#include <future>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace cudaq::qec {

/// @brief One decoder's placement + construction inputs for a decoder_pool.
struct pool_decoder_spec {
  int id;                            ///< Caller-chosen routing id.
  std::string name;                  ///< Registered decoder name.
  cudaqx::tensor<uint8_t> H;         ///< Parity-check matrix.
  cudaqx::heterogeneous_map options; ///< get() options (cuda_device_id, ...).
};

/// @brief Runs a set of decoders concurrently, each on its own worker thread
/// bound (bind_current_thread) to that decoder's CUDA device / NUMA node. The
/// decoder is constructed on its worker thread so its resources land on-node.
class decoder_pool {
public:
  explicit decoder_pool(std::vector<pool_decoder_spec> specs);
  ~decoder_pool();
  decoder_pool(const decoder_pool &) = delete;
  decoder_pool &operator=(const decoder_pool &) = delete;

  /// @brief Decode every id's syndromes on that id's pinned worker, with all
  /// decoders running concurrently; blocks until they finish and returns the
  /// results grouped by id. Each id maps to that decoder's syndromes (the
  /// per-decoder batched decode happens inside its worker). Throws if an id has
  /// no matching decoder, or rethrows a worker's decode exception.
  std::unordered_map<int, std::vector<decoder_result>> decode_all(
      const std::unordered_map<int, std::vector<std::vector<float_t>>> &work);

  /// @brief Non-blocking streaming submit: enqueue `syndromes` on `id`'s pinned
  /// worker and return immediately with a future for that chunk's results.
  /// Submit chunks over time and consume the futures as they resolve to stream
  /// per id, concurrently; a size-1 chunk streams a single syndrome. Throws if
  /// `id` has no matching decoder; the future rethrows a worker decode
  /// exception.
  std::future<std::vector<decoder_result>>
  submit(int id, std::vector<std::vector<float_t>> syndromes);

private:
  struct worker;
  std::vector<std::unique_ptr<worker>> workers_;
  std::unordered_map<int, worker *> by_id_;
};

} // namespace cudaq::qec
