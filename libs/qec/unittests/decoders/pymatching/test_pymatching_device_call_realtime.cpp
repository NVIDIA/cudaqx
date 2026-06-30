/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq.h"
#include "cudaq/qec/realtime/decoding.h"
#include "cudaq/qec/realtime/decoding_config.h"
#include "cudaq/realtime.h"
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <variant>
#include <vector>

extern "C" void cudaqx_qec_realtime_device_call_service_force_link();

namespace {

namespace config = cudaq::qec::decoding::config;

constexpr std::uint64_t kDecoderId = 0;
constexpr std::uint64_t kBlockSize = 3;
constexpr std::uint64_t kSyndromeSize = 3;
constexpr std::uint64_t kSyndromeTag = 1;
constexpr std::uint64_t kRunShots = 1;
constexpr std::size_t kActiveSyndromeIndex = 1;
constexpr std::size_t kSparseEntriesPerRow = 2;
constexpr std::int64_t kSparseRowEnd = -1;
constexpr double kUniformErrorRate = 0.1;
constexpr std::int64_t kExpectedPackedCorrection = std::int64_t{1}
                                                   << kActiveSyndromeIndex;

std::vector<std::int64_t> make_identity_sparse_matrix() {
  std::vector<std::int64_t> sparse_matrix;
  sparse_matrix.reserve(kSparseEntriesPerRow * kSyndromeSize);
  for (std::uint64_t column = {}; column < kSyndromeSize; ++column) {
    sparse_matrix.push_back(static_cast<std::int64_t>(column));
    sparse_matrix.push_back(kSparseRowEnd);
  }
  return sparse_matrix;
}

config::multi_decoder_config make_config() {
  config::decoder_config decoder_config;
  decoder_config.id = kDecoderId;
  decoder_config.type = "pymatching";
  decoder_config.block_size = kBlockSize;
  decoder_config.syndrome_size = kSyndromeSize;

  const auto identity_sparse_matrix = make_identity_sparse_matrix();
  decoder_config.H_sparse = identity_sparse_matrix;
  decoder_config.O_sparse = identity_sparse_matrix;
  decoder_config.D_sparse = identity_sparse_matrix;

  decoder_config.decoder_custom_args = config::pymatching_config();
  auto &pymatching_config =
      std::get<config::pymatching_config>(decoder_config.decoder_custom_args);
  pymatching_config.error_rate_vec =
      std::vector<double>(kBlockSize, kUniformErrorRate);
  pymatching_config.merge_strategy = "smallest_weight";

  config::multi_decoder_config multi_config;
  multi_config.decoders.push_back(decoder_config);
  return multi_config;
}

struct DecoderGuard {
  bool armed = false;
  ~DecoderGuard() {
    if (armed)
      config::finalize_decoders();
  }
};

struct RealtimeGuard {
  bool armed = false;
  ~RealtimeGuard() {
    if (armed)
      cudaq::realtime::finalize();
  }
};

__qpu__ std::int64_t pymatching_device_call_kernel() {
  constexpr std::uint64_t kKernelDecoderId = 0;
  constexpr std::uint64_t kKernelBlockSize = 3;
  constexpr std::uint64_t kKernelSyndromeSize = 3;
  constexpr std::uint64_t kKernelSyndromeTag = 1;
  constexpr std::size_t kKernelActiveSyndromeIndex = 1;

  cudaq::qec::decoding::reset_decoder(/*decoder_id=*/kKernelDecoderId);

  std::vector<bool> syndrome(kKernelSyndromeSize);
  for (std::size_t i = 0; i < kKernelSyndromeSize; ++i)
    syndrome[i] = false;
  syndrome[kKernelActiveSyndromeIndex] = true;
  cudaq::qec::decoding::enqueue_syndromes_test(
      /*decoder_id=*/kKernelDecoderId, syndrome, /*tag=*/kKernelSyndromeTag);

  auto corrections = cudaq::qec::decoding::get_corrections(
      /*decoder_id=*/kKernelDecoderId, /*return_size=*/kKernelBlockSize,
      /*reset=*/true);
  std::int64_t packed = {};
  for (std::size_t i = 0; i < kKernelBlockSize; ++i)
    if (corrections[i])
      packed |= (std::int64_t{1} << i);
  return packed;
}

void initialize_realtime() {
  int argc = 1;
  char program[] = "test_pymatching_device_call_realtime";
  char *argv[] = {program, nullptr};
  cudaq::realtime::initialize(argc, argv);
}

} // namespace

TEST(PyMatchingDeviceCallRealtime, HostDispatch) {
  // Keep the service library loaded so CUDA-Q can discover its
  // cudaqGetDeviceCallServicePluginInfo symbol via dlsym(RTLD_DEFAULT).
  cudaqx_qec_realtime_device_call_service_force_link();

  auto decoder_config = make_config();
  ASSERT_EQ(config::configure_decoders(decoder_config), 0);
  DecoderGuard decoder_guard{true};

  initialize_realtime();
  RealtimeGuard realtime_guard{true};

  const auto results = cudaq::run(kRunShots, pymatching_device_call_kernel);
  ASSERT_EQ(results.size(), kRunShots);
  EXPECT_EQ(results[0], kExpectedPackedCorrection);
}
