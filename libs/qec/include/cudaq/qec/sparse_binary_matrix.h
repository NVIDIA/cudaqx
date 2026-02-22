/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "cuda-qx/core/tensor.h"
#include <cstdint>
#include <vector>

namespace cudaq::qec {

/// @brief Storage layout for the sparse PCM: column-major (CSC) or row-major
/// (CSR). All non-zero entries are assumed to be 1; values are not stored.
enum class sparse_binary_matrix_layout { csc, csr };

/// @brief Sparse parity-check matrix in either CSC or CSR form.
/// Index types are uint32_t; non-zero values are always 1 and are not stored.
class sparse_binary_matrix {
public:
  using index_type = std::uint32_t;

  /// @brief Construct a sparse PCM in CSC form.
  /// @param num_rows Number of rows.
  /// @param num_cols Number of columns.
  /// @param col_ptrs Column pointer array (length num_cols + 1); column j has
  /// indices in \p row_indices[col_ptrs[j] .. col_ptrs[j+1]-1].
  /// @param row_indices Row indices of non-zeros (length nnz).
  static sparse_binary_matrix from_csc(index_type num_rows, index_type num_cols,
                                       std::vector<index_type> col_ptrs,
                                       std::vector<index_type> row_indices);

  /// @brief Construct a sparse PCM in CSR form.
  /// @param num_rows Number of rows.
  /// @param num_cols Number of columns.
  /// @param row_ptrs Row pointer array (length num_rows + 1); row i has
  /// indices in \p col_indices[row_ptrs[i] .. row_ptrs[i+1]-1].
  /// @param col_indices Column indices of non-zeros (length nnz).
  static sparse_binary_matrix from_csr(index_type num_rows, index_type num_cols,
                                       std::vector<index_type> row_ptrs,
                                       std::vector<index_type> col_indices);

  /// @brief Construct from nested CSC: \p nested[j] is the list of row indices
  /// for column j; \p nested.size() must equal \p num_cols.
  static sparse_binary_matrix
  from_nested_csc(index_type num_rows, index_type num_cols,
                  const std::vector<std::vector<index_type>> &nested);

  /// @brief Construct from nested CSR: \p nested[i] is the list of column
  /// indices for row i; \p nested.size() must equal \p num_rows.
  static sparse_binary_matrix
  from_nested_csr(index_type num_rows, index_type num_cols,
                  const std::vector<std::vector<index_type>> &nested);

  /// @brief Construct a sparse PCM from a dense PCM tensor (rows x columns).
  /// Any non-zero entry is treated as 1.
  /// @param dense Dense parity-check matrix; must have rank 2.
  /// @param layout Storage layout for the sparse representation (default CSC).
  sparse_binary_matrix(
      const cudaqx::tensor<std::uint8_t> &dense,
      sparse_binary_matrix_layout layout = sparse_binary_matrix_layout::csc);

  // Default constructor
  sparse_binary_matrix() = default;

  // Copy constructor
  sparse_binary_matrix(const sparse_binary_matrix &) = default;

  // Move constructor
  sparse_binary_matrix(sparse_binary_matrix &&) noexcept = default;

  // Copy assignment operator
  sparse_binary_matrix &operator=(const sparse_binary_matrix &) = default;

  // Move assignment operator
  sparse_binary_matrix &operator=(sparse_binary_matrix &&) noexcept = default;

  sparse_binary_matrix_layout layout() const { return layout_; }
  index_type num_rows() const { return num_rows_; }
  index_type num_cols() const { return num_cols_; }
  index_type num_nnz() const {
    return indices_.empty() ? 0 : static_cast<index_type>(indices_.size());
  }

  /// @brief For CSC: ptr has length num_cols+1; for CSR: ptr has length
  /// num_rows+1.
  const std::vector<index_type> &ptr() const { return ptr_; }
  /// @brief For CSC: row indices; for CSR: column indices.
  const std::vector<index_type> &indices() const { return indices_; }

  /// @brief Return a copy of this matrix in CSC layout. No-op if already CSC.
  sparse_binary_matrix to_csc() const;

  /// @brief Return a copy of this matrix in CSR layout. No-op if already CSR.
  sparse_binary_matrix to_csr() const;

  /// @brief Convert to a dense PCM tensor (rows x columns). Non-zero entries
  /// are set to 1.
  cudaqx::tensor<std::uint8_t> to_dense() const;

  /// @brief Nested CSC: outer vector has size num_cols; inner vector for
  /// column j lists row indices of non-zeros in that column.
  std::vector<std::vector<index_type>> to_nested_csc() const;

  /// @brief Nested CSR: outer vector has size num_rows; inner vector for row i
  /// lists column indices of non-zeros in that row.
  std::vector<std::vector<index_type>> to_nested_csr() const;

private:
  sparse_binary_matrix(sparse_binary_matrix_layout layout, index_type num_rows,
                       index_type num_cols, std::vector<index_type> ptr,
                       std::vector<index_type> indices);

  sparse_binary_matrix_layout layout_ = sparse_binary_matrix_layout::csc;
  index_type num_rows_ = 0;
  index_type num_cols_ = 0;
  std::vector<index_type> ptr_;
  std::vector<index_type> indices_;
};

} // namespace cudaq::qec
