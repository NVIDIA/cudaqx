/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include <memory>

#include "cuda-qx/core/tensor.h"

#include "cudaq/qis/pauli_word.h"
#include "cudaq/operators.h"

namespace cudaq::qec {
enum class stabilizer_type { XZ, X, Z };
void sortStabilizerOps(std::vector<cudaq::spin_op_term> &ops);

/// Convert stabilizers to a parity check matrix
/// @return Tensor representing the parity check matrix
cudaqx::tensor<uint8_t>
to_parity_matrix(const std::vector<cudaq::spin_op_term> &stabilizers,
                 stabilizer_type type = stabilizer_type::XZ);
cudaqx::tensor<uint8_t>
to_parity_matrix(const std::vector<std::string> &words,
                 stabilizer_type type = stabilizer_type::XZ);

} // namespace cudaq::qec