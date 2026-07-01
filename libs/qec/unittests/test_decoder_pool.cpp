/*******************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "cudaq/qec/decoder_pool.h"
#include <cuda_runtime.h>
#include <gtest/gtest.h>

static cudaqx::tensor<uint8_t> makeH() {
  cudaqx::tensor<uint8_t> H({2, 3});
  return H;
}

// Orchestration (no GPU needed): each id's syndromes are routed to its own
// decoder and aggregated back under that id.
TEST(DecoderPool, RoutesAndAggregatesPerId) {
  using cudaq::qec::float_t;
  std::vector<cudaq::qec::pool_decoder_spec> specs;
  specs.push_back({7, "multi_error_lut", makeH(), {}});
  specs.push_back({9, "multi_error_lut", makeH(), {}});
  cudaq::qec::decoder_pool pool(std::move(specs));

  std::unordered_map<int, std::vector<std::vector<float_t>>> work;
  work[7] = {{0.1f, 0.1f}, {0.2f, 0.2f}};
  work[9] = {{0.3f, 0.3f}};
  auto res = pool.decode_all(work);
  ASSERT_EQ(res.size(), 2u);
  EXPECT_EQ(res[7].size(), 2u);
  EXPECT_EQ(res[9].size(), 1u);
}

TEST(DecoderPool, UnknownIdThrows) {
  using cudaq::qec::float_t;
  std::vector<cudaq::qec::pool_decoder_spec> specs;
  specs.push_back({1, "multi_error_lut", makeH(), {}});
  cudaq::qec::decoder_pool pool(std::move(specs));
  std::unordered_map<int, std::vector<std::vector<float_t>>> work;
  work[42] = {{0.1f, 0.1f}};
  EXPECT_THROW(pool.decode_all(work), std::runtime_error);
}

// Placement: each worker decodes on its assigned GPU. Uses the probe decoder,
// which reports (via result.result[0]) the device its allocation landed on.
TEST(DecoderPool, EachWorkerRunsOnAssignedGpu) {
  using cudaq::qec::float_t;
  int n = 0;
  cudaGetDeviceCount(&n);
  if (n < 2)
    GTEST_SKIP() << "needs >= 2 GPUs";
  cudaSetDevice(0);
  std::vector<cudaq::qec::pool_decoder_spec> specs;
  cudaqx::heterogeneous_map o0;
  o0.insert("cuda_device_id", 0);
  cudaqx::heterogeneous_map o1;
  o1.insert("cuda_device_id", 1);
  specs.push_back({0, "device_probe_decoder", makeH(), o0});
  specs.push_back({1, "device_probe_decoder", makeH(), o1});
  cudaq::qec::decoder_pool pool(std::move(specs));
  std::unordered_map<int, std::vector<std::vector<float_t>>> work;
  work[0] = {{0.1f, 0.1f}};
  work[1] = {{0.1f, 0.1f}};
  auto res = pool.decode_all(work);
  ASSERT_EQ(res[0].size(), 1u);
  ASSERT_EQ(res[1].size(), 1u);
  EXPECT_EQ(static_cast<int>(res[0][0].result[0]), 0);
  EXPECT_EQ(static_cast<int>(res[1][0].result[0]), 1);
}

// Validates the real closed nv-qldpc decoder, when compiled against this
// base, running two decoders concurrently on distinct GPUs through the pool.
// Skips wherever nv-qldpc-decoder isn't installed (e.g. this dev build).
TEST(DecoderPool, NvQldpcRunsOnDistinctGpusWhenAvailable) {
  using cudaq::qec::float_t;
  std::vector<uint8_t> H_vec = {1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1,
                                1, 0, 1, 0, 0, 1, 0, 1, 1, 1};
  cudaqx::tensor<uint8_t> H;
  H.copy(H_vec.data(), {3, 7});
  try {
    auto d = cudaq::qec::decoder::get("nv-qldpc-decoder", H);
  } catch (const std::exception &) {
    GTEST_SKIP() << "nv-qldpc-decoder not available";
  }

  int n = 0;
  cudaGetDeviceCount(&n);
  if (n < 2)
    GTEST_SKIP() << "needs >= 2 GPUs";

  std::vector<cudaq::qec::pool_decoder_spec> specs;
  cudaqx::heterogeneous_map o0;
  o0.insert("cuda_device_id", 0);
  cudaqx::heterogeneous_map o1;
  o1.insert("cuda_device_id", 1);
  specs.push_back({0, "nv-qldpc-decoder", H, o0});
  specs.push_back({1, "nv-qldpc-decoder", H, o1});
  cudaq::qec::decoder_pool pool(std::move(specs));

  std::unordered_map<int, std::vector<std::vector<float_t>>> work;
  work[0] = {{1.0f, 0.0f, 1.0f}};
  work[1] = {{1.0f, 0.0f, 1.0f}};
  auto res = pool.decode_all(work);
  ASSERT_EQ(res[0].size(), 1u);
  ASSERT_EQ(res[1].size(), 1u);
  EXPECT_TRUE(res[0][0].converged);
  EXPECT_TRUE(res[1][0].converged);
}

// Placement at construction time: each worker's decoder allocates GPU memory
// in its constructor (like production GPU decoders), and the construct-time
// device guard must already have it on the right device before any decode.
TEST(DecoderPool, EachWorkerConstructsOnAssignedGpu) {
  using cudaq::qec::float_t;
  int n = 0;
  cudaGetDeviceCount(&n);
  if (n < 2)
    GTEST_SKIP() << "needs >= 2 GPUs";
  cudaSetDevice(0);
  std::vector<cudaq::qec::pool_decoder_spec> specs;
  cudaqx::heterogeneous_map o0;
  o0.insert("cuda_device_id", 0);
  cudaqx::heterogeneous_map o1;
  o1.insert("cuda_device_id", 1);
  specs.push_back({0, "eager_device_probe_decoder", makeH(), o0});
  specs.push_back({1, "eager_device_probe_decoder", makeH(), o1});
  cudaq::qec::decoder_pool pool(std::move(specs));
  std::unordered_map<int, std::vector<std::vector<float_t>>> work;
  work[0] = {{0.1f, 0.1f}};
  work[1] = {{0.1f, 0.1f}};
  auto res = pool.decode_all(work);
  ASSERT_EQ(res[0].size(), 1u);
  ASSERT_EQ(res[1].size(), 1u);
  EXPECT_EQ(static_cast<int>(res[0][0].result[0]), 0);
  EXPECT_EQ(static_cast<int>(res[1][0].result[0]), 1);
}
