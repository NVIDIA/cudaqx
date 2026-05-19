/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/qec/sparse_binary_matrix.h"
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace cudaq::qec {

namespace {

/// CSC/CSR pointers and per-column/per-row runs use \c index_type (uint32_t);
/// the total nnz must fit in that type (std::vector::size may exceed it).
inline void throw_if_nnz_exceeds_index_range(std::size_t nnz,
                                             const char *context) {
  if (nnz >
      static_cast<std::size_t>(
          std::numeric_limits<sparse_binary_matrix::index_type>::max()))
    throw std::invalid_argument(std::string(context) +
                                ": nnz exceeds index_type (uint32_t) range");
}

} // namespace

sparse_binary_matrix::sparse_binary_matrix(sparse_binary_matrix_layout layout,
                                           index_type num_rows,
                                           index_type num_cols,
                                           std::vector<index_type> ptr,
                                           std::vector<index_type> indices)
    : layout_(layout), num_rows_(num_rows), num_cols_(num_cols),
      ptr_(std::move(ptr)), indices_(std::move(indices)) {}

sparse_binary_matrix
sparse_binary_matrix::from_csc(index_type num_rows, index_type num_cols,
                               std::vector<index_type> col_ptrs,
                               std::vector<index_type> row_indices) {
  const auto expected_ptr = static_cast<std::size_t>(num_cols) + 1;
  if (col_ptrs.size() != expected_ptr) {
    throw std::invalid_argument(
        "sparse_binary_matrix::from_csc: col_ptrs must have size num_cols + 1");
  }
  if (col_ptrs.front() != 0U) {
    throw std::invalid_argument(
        "sparse_binary_matrix::from_csc: col_ptrs[0] must be zero");
  }
  for (index_type j = 1; j <= num_cols; ++j) {
    if (col_ptrs[j - 1] > col_ptrs[j]) {
      throw std::invalid_argument(
          "sparse_binary_matrix::from_csc: col_ptrs must be non-decreasing");
    }
  }
  throw_if_nnz_exceeds_index_range(row_indices.size(),
                                   "sparse_binary_matrix::from_csc");
  const auto nnz = static_cast<index_type>(row_indices.size());
  if (col_ptrs.back() != nnz) {
    throw std::invalid_argument(
        "sparse_binary_matrix::from_csc: last col_ptr must equal nnz "
        "(row_indices.size())");
  }
  for (index_type idx : row_indices) {
    if (idx >= num_rows) {
      throw std::invalid_argument(
          "sparse_binary_matrix::from_csc: row index out of range");
    }
  }
  return sparse_binary_matrix(sparse_binary_matrix_layout::csc, num_rows,
                              num_cols, std::move(col_ptrs),
                              std::move(row_indices));
}

sparse_binary_matrix
sparse_binary_matrix::from_csr(index_type num_rows, index_type num_cols,
                               std::vector<index_type> row_ptrs,
                               std::vector<index_type> col_indices) {
  const auto expected_ptr = static_cast<std::size_t>(num_rows) + 1;
  if (row_ptrs.size() != expected_ptr) {
    throw std::invalid_argument(
        "sparse_binary_matrix::from_csr: row_ptrs must have size num_rows + 1");
  }
  if (row_ptrs.front() != 0U) {
    throw std::invalid_argument(
        "sparse_binary_matrix::from_csr: row_ptrs[0] must be zero");
  }
  for (index_type i = 1; i <= num_rows; ++i) {
    if (row_ptrs[i - 1] > row_ptrs[i]) {
      throw std::invalid_argument(
          "sparse_binary_matrix::from_csr: row_ptrs must be non-decreasing");
    }
  }
  throw_if_nnz_exceeds_index_range(col_indices.size(),
                                   "sparse_binary_matrix::from_csr");
  const auto nnz = static_cast<index_type>(col_indices.size());
  if (row_ptrs.back() != nnz) {
    throw std::invalid_argument(
        "sparse_binary_matrix::from_csr: last row_ptr must equal nnz "
        "(col_indices.size())");
  }
  for (index_type idx : col_indices) {
    if (idx >= num_cols) {
      throw std::invalid_argument(
          "sparse_binary_matrix::from_csr: column index out of range");
    }
  }
  return sparse_binary_matrix(sparse_binary_matrix_layout::csr, num_rows,
                              num_cols, std::move(row_ptrs),
                              std::move(col_indices));
}

sparse_binary_matrix sparse_binary_matrix::from_nested_csc(
    index_type num_rows, index_type num_cols,
    const std::vector<std::vector<index_type>> &nested) {
  if (nested.size() != static_cast<std::size_t>(num_cols)) {
    throw std::invalid_argument(
        "sparse_binary_matrix::from_nested_csc: nested.size() must equal "
        "num_cols");
  }
  std::vector<index_type> col_ptrs(num_cols + 1);
  col_ptrs[0] = 0;
  std::size_t nnz_accum = 0;
  for (const auto &col : nested)
    nnz_accum += col.size();
  throw_if_nnz_exceeds_index_range(nnz_accum,
                                   "sparse_binary_matrix::from_nested_csc");
  std::vector<index_type> row_indices;
  row_indices.reserve(nnz_accum);
  for (index_type j = 0; j < num_cols; ++j) {
    for (index_type r : nested[j]) {
      if (r >= num_rows) {
        throw std::invalid_argument(
            "sparse_binary_matrix::from_nested_csc: row index out of range");
      }
      row_indices.push_back(r);
    }
    col_ptrs[j + 1] = static_cast<index_type>(row_indices.size());
  }
  return sparse_binary_matrix(sparse_binary_matrix_layout::csc, num_rows,
                              num_cols, std::move(col_ptrs),
                              std::move(row_indices));
}

sparse_binary_matrix sparse_binary_matrix::from_nested_csr(
    index_type num_rows, index_type num_cols,
    const std::vector<std::vector<index_type>> &nested) {
  if (nested.size() != static_cast<std::size_t>(num_rows)) {
    throw std::invalid_argument(
        "sparse_binary_matrix::from_nested_csr: nested.size() must equal "
        "num_rows");
  }
  std::vector<index_type> row_ptrs(num_rows + 1);
  row_ptrs[0] = 0;
  std::size_t nnz_accum = 0;
  for (const auto &rw : nested)
    nnz_accum += rw.size();
  throw_if_nnz_exceeds_index_range(nnz_accum,
                                   "sparse_binary_matrix::from_nested_csr");
  std::vector<index_type> col_indices;
  col_indices.reserve(nnz_accum);
  for (index_type i = 0; i < num_rows; ++i) {
    for (index_type c : nested[i]) {
      if (c >= num_cols) {
        throw std::invalid_argument(
            "sparse_binary_matrix::from_nested_csr: column index out of range");
      }
      col_indices.push_back(c);
    }
    row_ptrs[i + 1] = static_cast<index_type>(col_indices.size());
  }
  return sparse_binary_matrix(sparse_binary_matrix_layout::csr, num_rows,
                              num_cols, std::move(row_ptrs),
                              std::move(col_indices));
}

sparse_binary_matrix::sparse_binary_matrix(
    const cudaqx::tensor<std::uint8_t> &dense,
    sparse_binary_matrix_layout layout) {
  if (dense.rank() != 2) {
    throw std::invalid_argument(
        "sparse_binary_matrix: dense PCM tensor must have rank 2");
  }
  const std::size_t nrows = dense.shape()[0];
  const std::size_t ncols = dense.shape()[1];
  if (nrows >
          static_cast<std::size_t>(std::numeric_limits<index_type>::max()) ||
      ncols >
          static_cast<std::size_t>(std::numeric_limits<index_type>::max())) {
    throw std::invalid_argument(
        "sparse_binary_matrix: dense PCM dimensions exceed index_type range");
  }
  num_rows_ = static_cast<index_type>(nrows);
  num_cols_ = static_cast<index_type>(ncols);
  layout_ = layout;

  if (layout_ == sparse_binary_matrix_layout::csc) {
    std::vector<index_type> col_nnz(num_cols_, 0);
    for (index_type r = 0; r < num_rows_; ++r) {
      const auto *row_ptr =
          &dense.at({static_cast<std::size_t>(r), static_cast<std::size_t>(0)});
      for (index_type c = 0; c < num_cols_; ++c)
        if (row_ptr[c])
          ++col_nnz[c];
    }
    std::size_t total_nnz = 0;
    for (index_type c = 0; c < num_cols_; ++c)
      total_nnz += static_cast<std::size_t>(col_nnz[c]);
    throw_if_nnz_exceeds_index_range(total_nnz,
                                     "sparse_binary_matrix(dense,to CSC)");
    ptr_.resize(num_cols_ + 1);
    ptr_[0] = 0;
    for (index_type c = 0; c < num_cols_; ++c)
      ptr_[c + 1] = ptr_[c] + col_nnz[c];
    indices_.resize(ptr_.back());
    std::fill(col_nnz.begin(), col_nnz.end(), 0);
    for (index_type r = 0; r < num_rows_; ++r) {
      const auto *row_ptr =
          &dense.at({static_cast<std::size_t>(r), static_cast<std::size_t>(0)});
      for (index_type c = 0; c < num_cols_; ++c) {
        if (row_ptr[c]) {
          index_type slot = ptr_[c] + col_nnz[c];
          indices_[slot] = r;
          ++col_nnz[c];
        }
      }
    }
    return;
  }

  std::vector<index_type> row_nnz(num_rows_, 0);
  for (index_type r = 0; r < num_rows_; ++r) {
    const auto *row_ptr =
        &dense.at({static_cast<std::size_t>(r), static_cast<std::size_t>(0)});
    for (index_type c = 0; c < num_cols_; ++c)
      if (row_ptr[c])
        ++row_nnz[r];
  }
  std::size_t total_nnz = 0;
  for (index_type r = 0; r < num_rows_; ++r)
    total_nnz += static_cast<std::size_t>(row_nnz[r]);
  throw_if_nnz_exceeds_index_range(total_nnz,
                                   "sparse_binary_matrix(dense,to CSR)");
  ptr_.resize(num_rows_ + 1);
  ptr_[0] = 0;
  for (index_type r = 0; r < num_rows_; ++r)
    ptr_[r + 1] = ptr_[r] + row_nnz[r];
  indices_.resize(ptr_.back());
  std::fill(row_nnz.begin(), row_nnz.end(), 0);
  for (index_type r = 0; r < num_rows_; ++r) {
    const auto *row_ptr =
        &dense.at({static_cast<std::size_t>(r), static_cast<std::size_t>(0)});
    for (index_type c = 0; c < num_cols_; ++c) {
      if (row_ptr[c]) {
        index_type slot = ptr_[r] + row_nnz[r];
        indices_[slot] = c;
        ++row_nnz[r];
      }
    }
  }
}

sparse_binary_matrix sparse_binary_matrix::to_csc() const {
  if (layout_ == sparse_binary_matrix_layout::csc) {
    return sparse_binary_matrix(sparse_binary_matrix_layout::csc, num_rows_,
                                num_cols_, ptr_, indices_);
  }
  // CSR -> CSC: for each column j, gather row indices i where (i,j) is stored
  // In CSR, row i has col indices in indices_[row_ptrs[i] .. row_ptrs[i+1]-1]
  std::vector<index_type> col_nnz(num_cols_, 0);
  for (index_type i = 0; i < num_rows_; ++i) {
    for (index_type p = ptr_[i]; p < ptr_[i + 1]; ++p) {
      index_type j = indices_[p];
      ++col_nnz[j];
    }
  }
  std::vector<index_type> col_ptrs(num_cols_ + 1);
  col_ptrs[0] = 0;
  for (index_type j = 0; j < num_cols_; ++j) {
    col_ptrs[j + 1] = col_ptrs[j] + col_nnz[j];
  }
  std::fill(col_nnz.begin(), col_nnz.end(), 0);
  std::vector<index_type> row_indices(indices_.size());
  for (index_type i = 0; i < num_rows_; ++i) {
    for (index_type p = ptr_[i]; p < ptr_[i + 1]; ++p) {
      index_type j = indices_[p];
      index_type q = col_ptrs[j] + col_nnz[j];
      row_indices[q] = i;
      ++col_nnz[j];
    }
  }
  return sparse_binary_matrix(sparse_binary_matrix_layout::csc, num_rows_,
                              num_cols_, std::move(col_ptrs),
                              std::move(row_indices));
}

sparse_binary_matrix sparse_binary_matrix::to_csr() const {
  if (layout_ == sparse_binary_matrix_layout::csr) {
    return sparse_binary_matrix(sparse_binary_matrix_layout::csr, num_rows_,
                                num_cols_, ptr_, indices_);
  }
  // CSC -> CSR: for each row i, gather column indices j where (i,j) is stored
  // In CSC, col j has row indices in indices_[col_ptrs[j] .. col_ptrs[j+1]-1]
  std::vector<index_type> row_nnz(num_rows_, 0);
  for (index_type j = 0; j < num_cols_; ++j) {
    for (index_type p = ptr_[j]; p < ptr_[j + 1]; ++p) {
      index_type i = indices_[p];
      ++row_nnz[i];
    }
  }
  std::vector<index_type> row_ptrs(num_rows_ + 1);
  row_ptrs[0] = 0;
  for (index_type i = 0; i < num_rows_; ++i) {
    row_ptrs[i + 1] = row_ptrs[i] + row_nnz[i];
  }
  std::fill(row_nnz.begin(), row_nnz.end(), 0);
  std::vector<index_type> col_indices(indices_.size());
  for (index_type j = 0; j < num_cols_; ++j) {
    for (index_type p = ptr_[j]; p < ptr_[j + 1]; ++p) {
      index_type i = indices_[p];
      index_type q = row_ptrs[i] + row_nnz[i];
      col_indices[q] = j;
      ++row_nnz[i];
    }
  }
  return sparse_binary_matrix(sparse_binary_matrix_layout::csr, num_rows_,
                              num_cols_, std::move(row_ptrs),
                              std::move(col_indices));
}

cudaqx::tensor<std::uint8_t> sparse_binary_matrix::to_dense() const {
  cudaqx::tensor<std::uint8_t> dense(
      std::vector<std::size_t>{num_rows_, num_cols_});
  for (std::size_t r = 0; r < num_rows_; ++r) {
    std::memset(&dense.at({r, 0}), 0, num_cols_ * sizeof(std::uint8_t));
  }
  if (layout_ == sparse_binary_matrix_layout::csc) {
    for (index_type j = 0; j < num_cols_; ++j) {
      for (index_type p = ptr_[j]; p < ptr_[j + 1]; ++p) {
        index_type i = indices_[p];
        dense.at({i, j}) = 1;
      }
    }
  } else {
    for (index_type i = 0; i < num_rows_; ++i) {
      for (index_type p = ptr_[i]; p < ptr_[i + 1]; ++p) {
        index_type j = indices_[p];
        dense.at({i, j}) = 1;
      }
    }
  }
  return dense;
}

std::vector<std::vector<sparse_binary_matrix::index_type>>
sparse_binary_matrix::to_nested_csc() const {
  std::vector<std::vector<index_type>> out(num_cols_);
  if (layout_ == sparse_binary_matrix_layout::csc) {
    for (index_type j = 0; j < num_cols_; ++j) {
      out[j].assign(indices_.begin() + ptr_[j], indices_.begin() + ptr_[j + 1]);
    }
  } else {
    for (index_type i = 0; i < num_rows_; ++i) {
      for (index_type p = ptr_[i]; p < ptr_[i + 1]; ++p) {
        index_type j = indices_[p];
        out[j].push_back(i);
      }
    }
  }
  return out;
}

std::vector<std::vector<sparse_binary_matrix::index_type>>
sparse_binary_matrix::to_nested_csr() const {
  std::vector<std::vector<index_type>> out(num_rows_);
  if (layout_ == sparse_binary_matrix_layout::csr) {
    for (index_type i = 0; i < num_rows_; ++i) {
      out[i].assign(indices_.begin() + ptr_[i], indices_.begin() + ptr_[i + 1]);
    }
  } else {
    for (index_type j = 0; j < num_cols_; ++j) {
      for (index_type p = ptr_[j]; p < ptr_[j + 1]; ++p) {
        index_type i = indices_[p];
        out[i].push_back(j);
      }
    }
  }
  return out;
}

} // namespace cudaq::qec
