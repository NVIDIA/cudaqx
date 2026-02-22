/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/qec/sparse_binary_matrix.h"
#include <cassert>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace cudaq::qec {

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
  assert(col_ptrs.size() == static_cast<std::size_t>(num_cols) + 1);
  assert(col_ptrs.back() == row_indices.size());
  return sparse_binary_matrix(sparse_binary_matrix_layout::csc, num_rows,
                              num_cols, std::move(col_ptrs),
                              std::move(row_indices));
}

sparse_binary_matrix
sparse_binary_matrix::from_csr(index_type num_rows, index_type num_cols,
                               std::vector<index_type> row_ptrs,
                               std::vector<index_type> col_indices) {
  assert(row_ptrs.size() == static_cast<std::size_t>(num_rows) + 1);
  assert(row_ptrs.back() == col_indices.size());
  return sparse_binary_matrix(sparse_binary_matrix_layout::csr, num_rows,
                              num_cols, std::move(row_ptrs),
                              std::move(col_indices));
}

sparse_binary_matrix sparse_binary_matrix::from_nested_csc(
    index_type num_rows, index_type num_cols,
    const std::vector<std::vector<index_type>> &nested) {
  if (nested.size() != static_cast<std::size_t>(num_cols)) {
    throw std::invalid_argument(
        "sparse_pcm::from_nested_csc: nested.size() must equal num_cols");
  }
  std::vector<index_type> col_ptrs(num_cols + 1);
  col_ptrs[0] = 0;
  std::vector<index_type> row_indices;
  row_indices.reserve(nested.size() * 2);
  for (index_type j = 0; j < num_cols; ++j) {
    for (index_type r : nested[j]) {
      if (r >= num_rows) {
        throw std::invalid_argument(
            "sparse_pcm::from_nested_csc: row index out of range");
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
        "sparse_pcm::from_nested_csr: nested.size() must equal num_rows");
  }
  std::vector<index_type> row_ptrs(num_rows + 1);
  row_ptrs[0] = 0;
  std::vector<index_type> col_indices;
  col_indices.reserve(nested.size() * 2);
  for (index_type i = 0; i < num_rows; ++i) {
    for (index_type c : nested[i]) {
      if (c >= num_cols) {
        throw std::invalid_argument(
            "sparse_pcm::from_nested_csr: column index out of range");
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
        "sparse_pcm: dense PCM tensor must have rank 2");
  }
  const std::size_t nrows = dense.shape()[0];
  const std::size_t ncols = dense.shape()[1];
  if (nrows >
          static_cast<std::size_t>(std::numeric_limits<index_type>::max()) ||
      ncols >
          static_cast<std::size_t>(std::numeric_limits<index_type>::max())) {
    throw std::invalid_argument(
        "sparse_pcm: dense PCM dimensions exceed index_type range");
  }
  num_rows_ = static_cast<index_type>(nrows);
  num_cols_ = static_cast<index_type>(ncols);
  layout_ = layout;

  if (layout_ == sparse_binary_matrix_layout::csc) {
    std::vector<index_type> row_indices;
    // row_indices.reserve(nrows * ncols / 2);
    ptr_.resize(num_cols_ + 1);
    ptr_[0] = 0;
    for (index_type c = 0; c < num_cols_; ++c) {
      for (index_type r = 0; r < num_rows_; ++r) {
        if (dense.at(
                {static_cast<std::size_t>(r), static_cast<std::size_t>(c)}))
          row_indices.push_back(r);
      }
      ptr_[c + 1] = static_cast<index_type>(row_indices.size());
    }
    indices_ = std::move(row_indices);
  } else {
    std::vector<index_type> col_indices;
    // col_indices.reserve(nrows * ncols / 2);
    ptr_.resize(num_rows_ + 1);
    ptr_[0] = 0;
    for (index_type r = 0; r < num_rows_; ++r) {
      for (index_type c = 0; c < num_cols_; ++c) {
        if (dense.at(
                {static_cast<std::size_t>(r), static_cast<std::size_t>(c)}))
          col_indices.push_back(c);
      }
      ptr_[r + 1] = static_cast<index_type>(col_indices.size());
    }
    indices_ = std::move(col_indices);
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
