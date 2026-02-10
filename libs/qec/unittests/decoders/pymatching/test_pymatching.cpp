/*******************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/qec/decoder.h"
#include <cmath>
#include <gtest/gtest.h>
#include <vector>

TEST(PyMatchingDecoder, checkRegularEdges) {
  using cudaq::qec::float_t;

  std::size_t block_size = 2;
  std::size_t syndrome_size = 3;
  cudaqx::heterogeneous_map custom_args;

  // clang-format off
  std::vector<uint8_t> H_vec = {1, 0,
                                1, 1,
                                0, 1};
  // clang-format on
  cudaqx::tensor<uint8_t> H;
  H.copy(H_vec.data(), {syndrome_size, block_size});
  auto d = cudaq::qec::decoder::get("pymatching", H, custom_args);

  // Activate error in column 0 and verify that the error is detected.
  std::vector<float_t> syndrome = {1, 1, 0};
  auto result = d->decode(syndrome);
  EXPECT_EQ(result.result[0], 1.0);
  EXPECT_EQ(result.result[1], 0.0);

  // Activate error in column 1 and verify that the error is detected.
  syndrome = {0, 1, 1};
  result = d->decode(syndrome);
  EXPECT_EQ(result.result[0], 0.0);
  EXPECT_EQ(result.result[1], 1.0);

  // Activate errors in columns 0 and 1 and verify that the errors are detected.
  syndrome = {1, 0, 1};
  result = d->decode(syndrome);
  EXPECT_EQ(result.result[0], 1.0);
  EXPECT_EQ(result.result[1], 1.0);
}

TEST(PyMatchingDecoder, checkBoundaryEdges) {
  using cudaq::qec::float_t;

  std::size_t block_size = 3;
  std::size_t syndrome_size = 3;
  cudaqx::heterogeneous_map custom_args;

  // clang-format off
  std::vector<uint8_t> H_vec = {1, 0, 0,
                                0, 1, 0,
                                0, 0, 1};
  // clang-format on
  cudaqx::tensor<uint8_t> H;
  H.copy(H_vec.data(), {syndrome_size, block_size});
  auto d = cudaq::qec::decoder::get("pymatching", H, custom_args);

  // Activate error in column 0 and verify that the error is detected.
  std::vector<float_t> syndrome = {1, 0, 0};
  auto result = d->decode(syndrome);
  EXPECT_EQ(result.result[0], 1.0);
  EXPECT_EQ(result.result[1], 0.0);
  EXPECT_EQ(result.result[2], 0.0);

  // Activate error in column 1 and verify that the error is detected.
  syndrome = {0, 1, 0};
  result = d->decode(syndrome);
  EXPECT_EQ(result.result[0], 0.0);
  EXPECT_EQ(result.result[1], 1.0);
  EXPECT_EQ(result.result[2], 0.0);

  // Activate error in column 2 and verify that the error is detected.
  syndrome = {0, 0, 1};
  result = d->decode(syndrome);
  EXPECT_EQ(result.result[0], 0.0);
  EXPECT_EQ(result.result[1], 0.0);
  EXPECT_EQ(result.result[2], 1.0);
}
