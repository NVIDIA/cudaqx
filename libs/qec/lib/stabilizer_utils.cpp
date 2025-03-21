/*******************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "cudaq/qec/stabilizer_utils.h"

namespace cudaq::qec {
// Sort the stabilizers by the occurrence of 'Z' first, then by 'X'.
// An earlier occurrence is considered "less".
// Example: a = "ZZX", b = "XXZ" -> a < b
static bool spinOpComparator(const cudaq::spin_op &a, const cudaq::spin_op &b) {
  // TODO - update this logic once the stabilizers are converted to
  // spin_op_terms.
  auto astr = a.to_string(false);
  auto bstr = b.to_string(false);
  auto zIdxA = astr.find_first_of("Z");
  auto zIdxB = bstr.find_first_of("Z");
  if (zIdxA == std::string::npos) {
    if (zIdxB != std::string::npos)
      return false;

    // No Z in either, must both contain a X
    auto xIdxA = astr.find_first_of("X");
    return xIdxA < bstr.find_first_of("X");
  }

  // First contains a Z
  if (zIdxB == std::string::npos)
    return true;

  return zIdxA < zIdxB;
}

static bool isStabilizerSorted(const std::vector<cudaq::spin_op> &ops) {
  return std::is_sorted(ops.begin(), ops.end(), spinOpComparator);
}

void sortStabilizerOps(std::vector<cudaq::spin_op> &ops) {
  std::sort(ops.begin(), ops.end(), spinOpComparator);
}

// Need to push into the form
// H = [ H_Z | 0   ]
//     [ 0   | H_X ]
cudaqx::tensor<uint8_t>
to_parity_matrix(const std::vector<cudaq::spin_op> &stabilizers,
                 stabilizer_type type) {
  if (stabilizers.empty())
    return cudaqx::tensor<uint8_t>();

  // Stabilizers must be sorted prior to use, but they are "const" into this
  // function. First check to see if they are already sorted, and if so, proceed
  // without making any copies. Otherwise make a copy, sort the copy, and make
  // the downstream code use the sorted copy.
  const std::vector<cudaq::spin_op> *p_stabilizers = &stabilizers;
  std::vector<cudaq::spin_op> stabilizers_copy; // only use if necessary
  if (!isStabilizerSorted(stabilizers)) {
    // Make a copy and sort them
    stabilizers_copy = stabilizers;
    sortStabilizerOps(stabilizers_copy);
    p_stabilizers = &stabilizers_copy;
  }

  // Get the number of qubits.
  std::size_t numQubits = [&]() {
    std::size_t max_id = 0;
    // Loop over all the stabilizers to find the maximum degree.
    for (const auto &s : stabilizers)
      max_id = std::max(max_id, s.max_degree());
    return max_id + 1; // add 1 to get the number of qubits
  }();

  if (type == stabilizer_type::XZ) {
    cudaqx::tensor<uint8_t> t({p_stabilizers->size(), 2 * numQubits});
    // Start by counting the number of Z spin_ops
    std::size_t numZRows = 0;
    for (auto &op : *p_stabilizers)
      if (op.to_string(false).find("Z") != std::string::npos)
        numZRows++;
      else
        break;

    // Need to shift Z bits left
    for (std::size_t row = 0; row < numZRows; row++) {
      for (std::size_t i = numQubits; i < 2 * numQubits; i++) {
        if ((*p_stabilizers)[row].get_raw_data().first[0][i])
          t.at({row, i - numQubits}) = 1;
      }
    }

    auto numXRows = p_stabilizers->size() - numZRows;

    for (std::size_t row = 0; row < numXRows; row++) {
      for (std::size_t i = 0; i < numQubits; i++) {
        if ((*p_stabilizers)[numZRows + row].get_raw_data().first[0][i])
          t.at({numZRows + row, i + numQubits}) = 1;
      }
    }

    return t;
  }

  if (type == stabilizer_type::Z) {
    // Start by counting the number of Z spin_ops
    std::size_t numZRows = 0;
    for (auto &op : *p_stabilizers)
      if (op.to_string(false).find("Z") != std::string::npos)
        numZRows++;
      else
        break;

    if (numZRows == 0)
      return cudaqx::tensor<uint8_t>();

    cudaqx::tensor<uint8_t> ret({numZRows, numQubits});
    for (std::size_t row = 0; row < numZRows; row++) {
      for (std::size_t i = numQubits; i < 2 * numQubits; i++) {
        if ((*p_stabilizers)[row].get_raw_data().first[0][i])
          ret.at({row, i - numQubits}) = 1;
      }
    }

    return ret;
  }

  // Start by counting the number of Z spin_ops
  std::size_t numZRows = 0;
  for (auto &op : *p_stabilizers)
    if (op.to_string(false).find("Z") != std::string::npos)
      numZRows++;
    else
      break;

  auto numXRows = p_stabilizers->size() - numZRows;

  if (numXRows == 0)
    return cudaqx::tensor<uint8_t>();

  cudaqx::tensor<uint8_t> ret({numXRows, numQubits});
  for (std::size_t row = 0; row < numXRows; row++) {
    for (std::size_t i = 0; i < numQubits; i++) {
      if ((*p_stabilizers)[numZRows + row].get_raw_data().first[0][i])
        ret.at({row, i}) = 1;
    }
  }

  return ret;
}

cudaqx::tensor<uint8_t> to_parity_matrix(const std::vector<std::string> &words,
                                         stabilizer_type type) {

  std::vector<cudaq::spin_op> ops;
  for (auto &os : words)
    ops.emplace_back(cudaq::spin_op::from_word(os));
  sortStabilizerOps(ops);
  return to_parity_matrix(ops, type);
}
} // namespace cudaq::qec
