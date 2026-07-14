/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cuda-qx/core/tensor.h"
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace cudaq::qec::details {

struct inlined_feedback_layout {
  std::vector<std::size_t> detector_indices;
  std::vector<std::size_t> detector_offsets;
  std::vector<std::size_t> observable_indices;
  std::vector<std::size_t> observable_offsets;
};

inline void validate_inlined_feedback_tensor(
    const cudaqx::tensor<uint8_t> &tensor, std::size_t expected_rows,
    std::size_t expected_cols, const std::string &name) {
  if (tensor.size() == 0)
    return;

  if (tensor.rank() != 2 || tensor.shape()[0] != expected_rows ||
      tensor.shape()[1] != expected_cols) {
    std::string actual;
    for (std::size_t d = 0; d < tensor.rank(); ++d)
      actual += (d ? ", " : "") + std::to_string(tensor.shape()[d]);
    throw std::runtime_error(name + " has invalid shape [" + actual +
                             "] - expected [" + std::to_string(expected_rows) +
                             ", " + std::to_string(expected_cols) +
                             "] or an empty tensor.");
  }

  for (std::size_t i = 0; i < tensor.size(); ++i) {
    const auto value = tensor.data()[i];
    if (value > 1)
      throw std::runtime_error(
          name + " has non-binary value " +
          std::to_string(static_cast<unsigned int>(value)) + " at flat index " +
          std::to_string(i) + "; entries must be exactly 0 or 1.");
  }
}

inline void feedback_rows_to_csr(const cudaqx::tensor<uint8_t> &matrix,
                                 std::size_t num_rows, std::size_t num_cols,
                                 const std::string &name,
                                 std::vector<std::size_t> &indices,
                                 std::vector<std::size_t> &offsets) {
  validate_inlined_feedback_tensor(matrix, num_rows, num_cols, name);
  if (matrix.size() == 0)
    return;

  offsets.resize(num_rows + 1);
  offsets[0] = 0;
  for (std::size_t row = 0; row < num_rows; ++row) {
    for (std::size_t col = 0; col < num_cols; ++col)
      if (matrix.at({row, col}) != 0)
        indices.push_back(col);
    offsets[row + 1] = indices.size();
  }
}

inline inlined_feedback_layout build_inlined_feedback_layout(
    const cudaqx::tensor<uint8_t> &feedback,
    const cudaqx::tensor<uint8_t> &observable_feedback,
    std::size_t num_syndromes_per_round, std::size_t num_observables,
    const std::string &feedback_name = "inlined feedback",
    const std::string &observable_feedback_name =
        "observable inlined feedback") {
  inlined_feedback_layout layout;
  feedback_rows_to_csr(feedback, num_syndromes_per_round,
                       num_syndromes_per_round, feedback_name,
                       layout.detector_indices, layout.detector_offsets);
  feedback_rows_to_csr(observable_feedback, num_observables,
                       num_syndromes_per_round, observable_feedback_name,
                       layout.observable_indices, layout.observable_offsets);
  return layout;
}

} // namespace cudaq::qec::details
