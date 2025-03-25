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
static bool spinOpComparatorStr(const std::string &astr,
                                const std::string &bstr) {
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

static bool spinOpComparator(const cudaq::spin_op &a, const cudaq::spin_op &b) {
  auto astr = a.begin()->get_pauli_word();
  auto bstr = b.begin()->get_pauli_word();
  return spinOpComparatorStr(astr, bstr);
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

  std::vector<std::string> stab_strings;
  for (auto &s : stabilizers)
    stab_strings.emplace_back(s.begin()->get_pauli_word());
  std::sort(stab_strings.begin(), stab_strings.end(), spinOpComparatorStr);

  auto numQubits = stab_strings[0].size();
  auto numStabilizers = stab_strings.size();
  if (type == stabilizer_type::XZ) {
    cudaqx::tensor<uint8_t> t({numStabilizers, 2 * numQubits});
    // Start by counting the number of Z spin_ops
    std::size_t numZRows = 0;
    for (const auto &op_str : stab_strings)
      if (op_str.find("Z") != std::string::npos)
        numZRows++;
      else
        break;

    // Need to shift Z bits left
    for (std::size_t row = 0; row < numZRows; row++) {
      for (std::size_t i = numQubits; i < 2 * numQubits; i++) {
        if (stab_strings[row][i - numQubits] == 'Z')
          t.at({row, i - numQubits}) = 1;
      }
    }

    auto numXRows = numStabilizers - numZRows;

    for (std::size_t row = 0; row < numXRows; row++) {
      for (std::size_t i = 0; i < numQubits; i++) {
        if (stab_strings[numZRows + row][i] == 'X')
          t.at({numZRows + row, i + numQubits}) = 1;
      }
    }

    return t;
  }

  if (type == stabilizer_type::Z) {
    // Start by counting the number of Z spin_ops
    std::size_t numZRows = 0;
    for (const auto &op_str : stab_strings)
      if (op_str.find("Z") != std::string::npos)
        numZRows++;
      else
        break;

    if (numZRows == 0)
      return cudaqx::tensor<uint8_t>();

    cudaqx::tensor<uint8_t> ret({numZRows, numQubits});
    for (std::size_t row = 0; row < numZRows; row++) {
      for (std::size_t i = numQubits; i < 2 * numQubits; i++) {
        if (stab_strings[row][i - numQubits] == 'Z')
          ret.at({row, i - numQubits}) = 1;
      }
    }

    return ret;
  }

  // Start by counting the number of Z spin_ops
  std::size_t numZRows = 0;
  for (const auto &op_str : stab_strings)
    if (op_str.find("Z") != std::string::npos)
      numZRows++;
    else
      break;

  auto numXRows = numStabilizers - numZRows;

  if (numXRows == 0)
    return cudaqx::tensor<uint8_t>();

  cudaqx::tensor<uint8_t> ret({numXRows, numQubits});
  for (std::size_t row = 0; row < numXRows; row++) {
    for (std::size_t i = 0; i < numQubits; i++) {
      if (stab_strings[numZRows + row][i] == 'X')
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
