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

  syndrome = {0.5, 0, 0};
  result = d->decode(syndrome);
  EXPECT_EQ(result.result[0], 1.0);
  EXPECT_EQ(result.result[1], 0.0);
  EXPECT_EQ(result.result[2], 0.0);
}

TEST(PyMatchingDecoder, rejectsDuplicateSparseInput) {
  using index_type = cudaq::qec::sparse_binary_matrix::index_type;

  // Duplicate sparse entries are not allowed: callers that want GF(2)
  // duplicate-collapse semantics must canonicalize before constructing the
  // decoder.
  std::vector<std::vector<index_type>> nested = {{0, 0, 0}, {0, 1}, {1}};
  auto H = cudaq::qec::sparse_binary_matrix::from_nested_csc(
      /*num_rows=*/2, /*num_cols=*/3, nested);

  cudaqx::heterogeneous_map custom_args;
  EXPECT_THROW((void)cudaq::qec::decoder::get("pymatching", H, custom_args),
               std::invalid_argument);
}

TEST(PyMatchingDecoder, preservesCallerColumnOrderUnderNonCanonicalOrdering) {
  using cudaq::qec::float_t;

  std::size_t block_size = 4;
  std::size_t syndrome_size = 4;
  cudaqx::heterogeneous_map custom_args;

  // Column c is a boundary edge at row: col0->0, col1->1, col2->3, col3->2.
  // Topological order by row content is [0,1,3,2], i.e. columns 2 and 3 swap.
  // clang-format off
  std::vector<uint8_t> H_vec = {1, 0, 0, 0,
                                0, 1, 0, 0,
                                0, 0, 0, 1,
                                0, 0, 1, 0};
  // clang-format on
  cudaqx::tensor<uint8_t> H;
  H.copy(H_vec.data(), {syndrome_size, block_size});
  auto d = cudaq::qec::decoder::get("pymatching", H, custom_args);

  // Detector at row 3 is only touched by column 2. If columns were reordered to
  // [0,1,3,2], this would wrongly light up column 3.
  std::vector<float_t> syndrome = {0, 0, 0, 1};
  auto result = d->decode(syndrome);
  ASSERT_TRUE(result.converged);
  ASSERT_EQ(result.result.size(), 4u);
  EXPECT_EQ(result.result[0], 0.0);
  EXPECT_EQ(result.result[1], 0.0);
  EXPECT_EQ(result.result[2], 1.0);
  EXPECT_EQ(result.result[3], 0.0);

  // Detector at row 2 is only touched by column 3.
  syndrome = {0, 0, 1, 0};
  result = d->decode(syndrome);
  EXPECT_EQ(result.result[0], 0.0);
  EXPECT_EQ(result.result[1], 0.0);
  EXPECT_EQ(result.result[2], 0.0);
  EXPECT_EQ(result.result[3], 1.0);

  // Columns 0 and 1 are already in canonical position; verify they're
  // unaffected.
  syndrome = {1, 0, 0, 0};
  result = d->decode(syndrome);
  EXPECT_EQ(result.result[0], 1.0);
  EXPECT_EQ(result.result[1], 0.0);
  EXPECT_EQ(result.result[2], 0.0);
  EXPECT_EQ(result.result[3], 0.0);
}

TEST(PyMatchingDecoder, AcceptsAllMergeStrategiesAndRejectsUnknown) {
  cudaqx::tensor<uint8_t> H;
  std::vector<uint8_t> H_vec = {1};
  H.copy(H_vec.data(), {1, 1});

  for (const std::string &strategy :
       {"disallow", "independent", "smallest_weight", "keep_original",
        "replace"}) {
    cudaqx::heterogeneous_map params;
    params.insert("merge_strategy", strategy);
    auto d = cudaq::qec::decoder::get("pymatching", H, params);
    ASSERT_NE(d, nullptr) << strategy;
    auto result = d->decode(std::vector<cudaq::qec::float_t>{1.0});
    ASSERT_TRUE(result.converged) << strategy;
    ASSERT_EQ(result.result.size(), 1u) << strategy;
  }

  cudaqx::heterogeneous_map params;
  params.insert("merge_strategy", std::string("not_a_strategy"));
  EXPECT_THROW((void)cudaq::qec::decoder::get("pymatching", H, params),
               std::runtime_error);
}

TEST(PyMatchingDecoder, RejectsObservableMatrixWithWrongBlockSize) {
  cudaqx::tensor<uint8_t> H;
  std::vector<uint8_t> H_vec = {1, 0, 0, 1};
  H.copy(H_vec.data(), {2, 2});

  cudaqx::tensor<uint8_t> O({1, 3});
  O.at({0, 0}) = 1;
  cudaqx::heterogeneous_map params;
  params.insert("O", O);

  EXPECT_THROW((void)cudaq::qec::decoder::get("pymatching", H, params),
               std::runtime_error);
}

TEST(PyMatchingDecoder, DecodesHighObservableIndicesAcrossPaths) {
  using cudaq::qec::float_t;

  // Exercise the packed-mask path at bit 32 and retain the vector-path
  // coverage at 64 observables; each identity edge flips only its matching bit.
  for (const std::size_t num_observables : {33u, 64u}) {
    cudaqx::tensor<uint8_t> H({num_observables, num_observables});
    cudaqx::tensor<uint8_t> O({num_observables, num_observables});
    for (std::size_t i = 0; i < num_observables; ++i) {
      H.at({i, i}) = 1;
      O.at({i, i}) = 1;
    }

    cudaqx::heterogeneous_map params;
    params.insert("O", O);
    auto d = cudaq::qec::decoder::get("pymatching", H, params);
    // ASSERT: valid graph-like identity matrices must construct a decoder.
    ASSERT_NE(d, nullptr);

    std::vector<float_t> syndrome(num_observables, 0.0);
    syndrome.back() = 1.0;
    auto result = d->decode(syndrome);
    // ASSERT: both observable decoding paths must successfully converge.
    ASSERT_TRUE(result.converged) << "num_observables=" << num_observables;
    // ASSERT: observable-aware decoding returns one result per O row.
    ASSERT_EQ(result.result.size(), num_observables);
    // ASSERT: the high bit must not alias another bit through a narrow mask.
    for (std::size_t i = 0; i < num_observables; ++i)
      EXPECT_EQ(result.result[i], i == num_observables - 1 ? 1.0 : 0.0)
          << "num_observables=" << num_observables << ", index=" << i;
  }
}
