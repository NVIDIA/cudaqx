/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/solvers/operators/operator_pools/qaoa_operator_pool.h"

namespace cudaq::solvers {

std::vector<cudaq::spin_op>
qaoa_pool::generate(const heterogeneous_map &config) const {
  if (!config.contains({"num-qubits", "num_qubits", "n-qubits", "n_qubits"}))
    throw std::runtime_error(
        "must provide num-qubits when constructing the qaoa operator pool.");
  auto qubits_num = config.get<std::size_t>(
      {"num-qubits", "n-qubits", "num_qubits", "n_qubits"});
  if (qubits_num == 0)
    return {};

  std::vector<cudaq::spin_op> op;

  // Single qubit X terms
  for (std::size_t i = 0; i < qubits_num; ++i) {
    op.push_back(cudaq::spin_op::x(i));
  }

  // Single qubit Y terms
  for (std::size_t i = 0; i < qubits_num; ++i) {
    op.push_back(cudaq::spin_op::y(i));
  }

  // XX terms
  for (std::size_t i = 0; i < qubits_num - 1; ++i) {
    for (std::size_t j = i + 1; j < qubits_num; ++j) {
      op.push_back(cudaq::spin_op::x(i) * cudaq::spin_op::x(j));
    }
  }

  // YY terms
  for (std::size_t i = 0; i < qubits_num - 1; ++i) {
    for (std::size_t j = i + 1; j < qubits_num; ++j) {
      op.push_back(cudaq::spin_op::y(i) * cudaq::spin_op::y(j));
    }
  }

  // YZ terms
  for (std::size_t i = 0; i < qubits_num - 1; ++i) {
    for (std::size_t j = i + 1; j < qubits_num; ++j) {
      op.push_back(cudaq::spin_op::y(i) * cudaq::spin_op::z(j));
    }
  }

  // ZY terms
  for (std::size_t i = 0; i < qubits_num - 1; ++i) {
    for (std::size_t j = i + 1; j < qubits_num; ++j) {
      op.push_back(cudaq::spin_op::z(i) * cudaq::spin_op::y(j));
    }
  }

  // XY terms
  for (std::size_t i = 0; i < qubits_num - 1; ++i) {
    for (std::size_t j = i + 1; j < qubits_num; ++j) {
      op.push_back(cudaq::spin_op::x(i) * cudaq::spin_op::y(j));
    }
  }

  // YX terms
  for (std::size_t i = 0; i < qubits_num - 1; ++i) {
    for (std::size_t j = i + 1; j < qubits_num; ++j) {
      op.push_back(cudaq::spin_op::y(i) * cudaq::spin_op::x(j));
    }
  }

  // XZ terms
  for (std::size_t i = 0; i < qubits_num - 1; ++i) {
    for (std::size_t j = i + 1; j < qubits_num; ++j) {
      op.push_back(cudaq::spin_op::x(i) * cudaq::spin_op::z(j));
    }
  }

  // ZX terms
  for (std::size_t i = 0; i < qubits_num - 1; ++i) {
    for (std::size_t j = i + 1; j < qubits_num; ++j) {
      op.push_back(cudaq::spin_op::z(i) * cudaq::spin_op::x(j));
    }
  }

  return op;
}

} // namespace cudaq::solvers