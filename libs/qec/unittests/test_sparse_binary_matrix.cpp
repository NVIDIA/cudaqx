/*******************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/qec/pcm_utils.h"
#include "cudaq/qec/sparse_binary_matrix.h"
#include <gtest/gtest.h>
#include <random>
#include <stdexcept>

namespace cudaq::qec {
namespace {

using index_type = sparse_binary_matrix::index_type;

bool dense_pcm_equal(const cudaqx::tensor<std::uint8_t> &a,
                     const cudaqx::tensor<std::uint8_t> &b) {
  if (a.rank() != 2 || b.rank() != 2)
    return false;
  if (a.shape()[0] != b.shape()[0] || a.shape()[1] != b.shape()[1])
    return false;
  for (std::size_t r = 0; r < a.shape()[0]; ++r)
    for (std::size_t c = 0; c < a.shape()[1]; ++c)
      if (a.at({r, c}) != b.at({r, c}))
        return false;
  return true;
}

// -----------------------------------------------------------------------------
// Dense <-> sparse_binary_matrix (CSC)
// -----------------------------------------------------------------------------

TEST(SparseBinaryMatrix, DenseToCscToDense_Small) {
  // 3x4 matrix: rows x cols
  std::vector<std::uint8_t> data = {
      1, 0, 1, 0, // row 0
      0, 1, 1, 0, // row 1
      1, 1, 0, 1  // row 2
  };
  cudaqx::tensor<std::uint8_t> dense({3, 4});
  dense.copy(data.data(), {3, 4});

  sparse_binary_matrix sp(dense, sparse_binary_matrix_layout::csc);
  EXPECT_EQ(sp.layout(), sparse_binary_matrix_layout::csc);
  EXPECT_EQ(sp.num_rows(), 3);
  EXPECT_EQ(sp.num_cols(), 4);
  EXPECT_EQ(sp.num_nnz(), 7);

  auto back = sp.to_dense();
  EXPECT_TRUE(dense_pcm_equal(dense, back));
}

TEST(SparseBinaryMatrix, DenseToCsrToDense_Small) {
  std::vector<std::uint8_t> data = {1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1};
  cudaqx::tensor<std::uint8_t> dense({3, 4});
  dense.copy(data.data(), {3, 4});

  sparse_binary_matrix sp(dense, sparse_binary_matrix_layout::csr);
  EXPECT_EQ(sp.layout(), sparse_binary_matrix_layout::csr);
  EXPECT_EQ(sp.num_rows(), 3);
  EXPECT_EQ(sp.num_cols(), 4);
  EXPECT_EQ(sp.num_nnz(), 7);

  auto back = sp.to_dense();
  EXPECT_TRUE(dense_pcm_equal(dense, back));
}

TEST(SparseBinaryMatrix, FromCscToDense) {
  // 2x3 matrix: col0 has rows {0,1}, col1 has {}, col2 has {1}
  index_type num_rows = 2, num_cols = 3;
  std::vector<index_type> col_ptrs = {0, 2, 2, 3};
  std::vector<index_type> row_indices = {0, 1, 1};

  auto sp =
      sparse_binary_matrix::from_csc(num_rows, num_cols, col_ptrs, row_indices);
  auto dense = sp.to_dense();

  EXPECT_EQ(dense.shape()[0], 2);
  EXPECT_EQ(dense.shape()[1], 3);
  EXPECT_EQ(dense.at({0, 0}), 1);
  EXPECT_EQ(dense.at({1, 0}), 1);
  EXPECT_EQ(dense.at({0, 1}), 0);
  EXPECT_EQ(dense.at({1, 1}), 0);
  EXPECT_EQ(dense.at({0, 2}), 0);
  EXPECT_EQ(dense.at({1, 2}), 1);
}

TEST(SparseBinaryMatrix, FromCsrToDense) {
  // 2x3 matrix: row0 has cols {0}, row1 has cols {0, 2}
  index_type num_rows = 2, num_cols = 3;
  std::vector<index_type> row_ptrs = {0, 1, 3};
  std::vector<index_type> col_indices = {0, 0, 2};

  auto sp =
      sparse_binary_matrix::from_csr(num_rows, num_cols, row_ptrs, col_indices);
  auto dense = sp.to_dense();

  EXPECT_EQ(dense.shape()[0], 2);
  EXPECT_EQ(dense.shape()[1], 3);
  EXPECT_EQ(dense.at({0, 0}), 1);
  EXPECT_EQ(dense.at({0, 1}), 0);
  EXPECT_EQ(dense.at({0, 2}), 0);
  EXPECT_EQ(dense.at({1, 0}), 1);
  EXPECT_EQ(dense.at({1, 1}), 0);
  EXPECT_EQ(dense.at({1, 2}), 1);
}

// -----------------------------------------------------------------------------
// CSC <-> CSR conversion round-trip
// -----------------------------------------------------------------------------

TEST(SparseBinaryMatrix, CscToCsrToCsc_RoundTrip) {
  std::vector<std::uint8_t> data = {1, 0, 1, 0, 1, 1, 0, 1, 0, 1};
  cudaqx::tensor<std::uint8_t> dense({2, 5});
  dense.copy(data.data(), {2, 5});

  sparse_binary_matrix csc(dense, sparse_binary_matrix_layout::csc);
  sparse_binary_matrix csr = csc.to_csr();
  sparse_binary_matrix csc2 = csr.to_csc();

  EXPECT_EQ(csc2.layout(), sparse_binary_matrix_layout::csc);
  auto back = csc2.to_dense();
  EXPECT_TRUE(dense_pcm_equal(dense, back));
}

TEST(SparseBinaryMatrix, CsrToCscToCsr_RoundTrip) {
  std::vector<std::uint8_t> data = {1, 1, 0, 0, 1, 0, 1, 1};
  cudaqx::tensor<std::uint8_t> dense({2, 4});
  dense.copy(data.data(), {2, 4});

  sparse_binary_matrix csr(dense, sparse_binary_matrix_layout::csr);
  sparse_binary_matrix csc = csr.to_csc();
  sparse_binary_matrix csr2 = csc.to_csr();

  EXPECT_EQ(csr2.layout(), sparse_binary_matrix_layout::csr);
  auto back = csr2.to_dense();
  EXPECT_TRUE(dense_pcm_equal(dense, back));
}

// -----------------------------------------------------------------------------
// Edge cases: empty, 1x1, all zeros
// -----------------------------------------------------------------------------

TEST(SparseBinaryMatrix, EmptyMatrix) {
  cudaqx::tensor<std::uint8_t> dense({0, 0});
  sparse_binary_matrix sp(dense, sparse_binary_matrix_layout::csc);
  EXPECT_EQ(sp.num_rows(), 0);
  EXPECT_EQ(sp.num_cols(), 0);
  EXPECT_EQ(sp.num_nnz(), 0);
  auto back = sp.to_dense();
  EXPECT_EQ(back.shape()[0], 0);
  EXPECT_EQ(back.shape()[1], 0);
}

TEST(SparseBinaryMatrix, SingleElementZero) {
  cudaqx::tensor<std::uint8_t> dense({1, 1});
  dense.at({0, 0}) = 0;
  sparse_binary_matrix sp(dense, sparse_binary_matrix_layout::csc);
  EXPECT_EQ(sp.num_nnz(), 0);
  auto back = sp.to_dense();
  EXPECT_EQ(back.at({0, 0}), 0);
}

TEST(SparseBinaryMatrix, SingleElementOne) {
  cudaqx::tensor<std::uint8_t> dense({1, 1});
  dense.at({0, 0}) = 1;
  sparse_binary_matrix sp(dense, sparse_binary_matrix_layout::csc);
  EXPECT_EQ(sp.num_nnz(), 1);
  auto back = sp.to_dense();
  EXPECT_EQ(back.at({0, 0}), 1);
}

TEST(SparseBinaryMatrix, Small2x2) {
  std::vector<std::uint8_t> data = {1, 1, 1, 0};
  cudaqx::tensor<std::uint8_t> dense({2, 2});
  dense.copy(data.data(), {2, 2});

  for (auto layout :
       {sparse_binary_matrix_layout::csc, sparse_binary_matrix_layout::csr}) {
    sparse_binary_matrix sp(dense, layout);
    auto back = sp.to_dense();
    EXPECT_TRUE(dense_pcm_equal(dense, back));
  }
}

// -----------------------------------------------------------------------------
// Random PCM via generate_random_pcm
// -----------------------------------------------------------------------------

TEST(SparseBinaryMatrix, RandomPcm_CscRoundTrip) {
  std::mt19937_64 rng(12345);
  auto dense = generate_random_pcm(2, 3, 4, 2, std::move(rng));
  ASSERT_EQ(dense.rank(), 2);

  sparse_binary_matrix sp(dense, sparse_binary_matrix_layout::csc);
  auto back = sp.to_dense();
  EXPECT_TRUE(dense_pcm_equal(dense, back));
}

TEST(SparseBinaryMatrix, RandomPcm_CsrRoundTrip) {
  std::mt19937_64 rng(67890);
  auto dense = generate_random_pcm(3, 2, 3, 2, std::move(rng));
  ASSERT_EQ(dense.rank(), 2);

  sparse_binary_matrix sp(dense, sparse_binary_matrix_layout::csr);
  auto back = sp.to_dense();
  EXPECT_TRUE(dense_pcm_equal(dense, back));
}

TEST(SparseBinaryMatrix, RandomPcm_CscToCsrToDense) {
  std::mt19937_64 rng(42);
  auto dense = generate_random_pcm(2, 4, 3, 2, std::move(rng));
  sparse_binary_matrix csc(dense, sparse_binary_matrix_layout::csc);
  sparse_binary_matrix csr = csc.to_csr();
  auto back = csr.to_dense();
  EXPECT_TRUE(dense_pcm_equal(dense, back));
}

TEST(SparseBinaryMatrix, RandomPcm_CsrToCscToDense) {
  std::mt19937_64 rng(99);
  auto dense = generate_random_pcm(4, 3, 2, 2, std::move(rng));
  sparse_binary_matrix csr(dense, sparse_binary_matrix_layout::csr);
  sparse_binary_matrix csc = csr.to_csc();
  auto back = csc.to_dense();
  EXPECT_TRUE(dense_pcm_equal(dense, back));
}

// -----------------------------------------------------------------------------
// Nested CSC / CSR
// -----------------------------------------------------------------------------

TEST(SparseBinaryMatrix, ToNestedCsc_FromCsc) {
  // 2x3 matrix: col0 rows {0,1}, col1 rows {}, col2 rows {1}
  index_type num_rows = 2, num_cols = 3;
  std::vector<index_type> col_ptrs = {0, 2, 2, 3};
  std::vector<index_type> row_indices = {0, 1, 1};

  auto sp =
      sparse_binary_matrix::from_csc(num_rows, num_cols, col_ptrs, row_indices);
  auto nested = sp.to_nested_csc();

  ASSERT_EQ(nested.size(), 3);
  EXPECT_EQ(nested[0], (std::vector<index_type>{0, 1}));
  EXPECT_TRUE(nested[1].empty());
  EXPECT_EQ(nested[2], (std::vector<index_type>{1}));
}

TEST(SparseBinaryMatrix, ToNestedCsr_FromCsr) {
  // 2x3 matrix: row0 cols {0}, row1 cols {0, 2}
  index_type num_rows = 2, num_cols = 3;
  std::vector<index_type> row_ptrs = {0, 1, 3};
  std::vector<index_type> col_indices = {0, 0, 2};

  auto sp =
      sparse_binary_matrix::from_csr(num_rows, num_cols, row_ptrs, col_indices);
  auto nested = sp.to_nested_csr();

  ASSERT_EQ(nested.size(), 2);
  EXPECT_EQ(nested[0], (std::vector<index_type>{0}));
  EXPECT_EQ(nested[1], (std::vector<index_type>{0, 2}));
}

TEST(SparseBinaryMatrix, ToNestedCsc_RoundTrip) {
  std::vector<std::uint8_t> data = {1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1};
  cudaqx::tensor<std::uint8_t> dense({3, 4});
  dense.copy(data.data(), {3, 4});

  for (auto layout :
       {sparse_binary_matrix_layout::csc, sparse_binary_matrix_layout::csr}) {
    sparse_binary_matrix sp(dense, layout);
    auto nested = sp.to_nested_csc();
    ASSERT_EQ(nested.size(), sp.num_cols());
    index_type nnz = 0;
    std::vector<index_type> col_ptrs(sp.num_cols() + 1);
    col_ptrs[0] = 0;
    std::vector<index_type> row_indices;
    for (index_type j = 0; j < sp.num_cols(); ++j) {
      nnz += static_cast<index_type>(nested[j].size());
      col_ptrs[j + 1] = col_ptrs[j] + static_cast<index_type>(nested[j].size());
      row_indices.insert(row_indices.end(), nested[j].begin(), nested[j].end());
    }
    EXPECT_EQ(nnz, sp.num_nnz());
    auto sp2 = sparse_binary_matrix::from_csc(sp.num_rows(), sp.num_cols(),
                                              std::move(col_ptrs),
                                              std::move(row_indices));
    EXPECT_TRUE(dense_pcm_equal(dense, sp2.to_dense()));
  }
}

TEST(SparseBinaryMatrix, ToNestedCsr_RoundTrip) {
  std::vector<std::uint8_t> data = {1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1};
  cudaqx::tensor<std::uint8_t> dense({3, 4});
  dense.copy(data.data(), {3, 4});

  for (auto layout :
       {sparse_binary_matrix_layout::csc, sparse_binary_matrix_layout::csr}) {
    sparse_binary_matrix sp(dense, layout);
    auto nested = sp.to_nested_csr();
    ASSERT_EQ(nested.size(), sp.num_rows());
    index_type nnz = 0;
    std::vector<index_type> row_ptrs(sp.num_rows() + 1);
    row_ptrs[0] = 0;
    std::vector<index_type> col_indices;
    for (index_type i = 0; i < sp.num_rows(); ++i) {
      nnz += static_cast<index_type>(nested[i].size());
      row_ptrs[i + 1] = row_ptrs[i] + static_cast<index_type>(nested[i].size());
      col_indices.insert(col_indices.end(), nested[i].begin(), nested[i].end());
    }
    EXPECT_EQ(nnz, sp.num_nnz());
    auto sp2 = sparse_binary_matrix::from_csr(sp.num_rows(), sp.num_cols(),
                                              std::move(row_ptrs),
                                              std::move(col_indices));
    EXPECT_TRUE(dense_pcm_equal(dense, sp2.to_dense()));
  }
}

TEST(SparseBinaryMatrix, FromNestedCsc_MatchesFromCsc) {
  index_type num_rows = 2, num_cols = 3;
  std::vector<std::vector<index_type>> nested = {{0, 1}, {}, {1}};
  auto sp = sparse_binary_matrix::from_nested_csc(num_rows, num_cols, nested);
  EXPECT_EQ(sp.num_rows(), num_rows);
  EXPECT_EQ(sp.num_cols(), num_cols);
  EXPECT_EQ(sp.num_nnz(), 3);
  std::vector<index_type> col_ptrs = {0, 2, 2, 3};
  std::vector<index_type> row_indices = {0, 1, 1};
  auto sp_ref =
      sparse_binary_matrix::from_csc(num_rows, num_cols, col_ptrs, row_indices);
  EXPECT_TRUE(dense_pcm_equal(sp.to_dense(), sp_ref.to_dense()));
}

TEST(SparseBinaryMatrix, FromNestedCsr_MatchesFromCsr) {
  index_type num_rows = 2, num_cols = 3;
  std::vector<std::vector<index_type>> nested = {{0}, {0, 2}};
  auto sp = sparse_binary_matrix::from_nested_csr(num_rows, num_cols, nested);
  EXPECT_EQ(sp.num_rows(), num_rows);
  EXPECT_EQ(sp.num_cols(), num_cols);
  EXPECT_EQ(sp.num_nnz(), 3);
  std::vector<index_type> row_ptrs = {0, 1, 3};
  std::vector<index_type> col_indices = {0, 0, 2};
  auto sp_ref =
      sparse_binary_matrix::from_csr(num_rows, num_cols, row_ptrs, col_indices);
  EXPECT_TRUE(dense_pcm_equal(sp.to_dense(), sp_ref.to_dense()));
}

TEST(SparseBinaryMatrix, FromNestedCsc_RoundTrip) {
  std::vector<std::uint8_t> data = {1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1};
  cudaqx::tensor<std::uint8_t> dense({3, 4});
  dense.copy(data.data(), {3, 4});
  sparse_binary_matrix sp(dense, sparse_binary_matrix_layout::csc);
  const auto nested = sp.to_nested_csc();
  auto sp2 = sparse_binary_matrix::from_nested_csc(sp.num_rows(), sp.num_cols(),
                                                   nested);
  EXPECT_TRUE(dense_pcm_equal(dense, sp2.to_dense()));
}

TEST(SparseBinaryMatrix, FromNestedCsr_RoundTrip) {
  std::vector<std::uint8_t> data = {1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1};
  cudaqx::tensor<std::uint8_t> dense({3, 4});
  dense.copy(data.data(), {3, 4});
  sparse_binary_matrix sp(dense, sparse_binary_matrix_layout::csr);
  const auto nested = sp.to_nested_csr();
  auto sp2 = sparse_binary_matrix::from_nested_csr(sp.num_rows(), sp.num_cols(),
                                                   nested);
  EXPECT_TRUE(dense_pcm_equal(dense, sp2.to_dense()));
}

TEST(SparseBinaryMatrix, FromNestedCsc_InvalidSizeThrows) {
  std::vector<std::vector<index_type>> nested = {{0}, {1}};
  EXPECT_THROW(sparse_binary_matrix::from_nested_csc(2, 3, nested),
               std::invalid_argument);
}

TEST(SparseBinaryMatrix, FromNestedCsr_InvalidSizeThrows) {
  std::vector<std::vector<index_type>> nested = {{0}, {1}, {0}};
  EXPECT_THROW(sparse_binary_matrix::from_nested_csr(2, 2, nested),
               std::invalid_argument);
}

} // namespace
} // namespace cudaq::qec
