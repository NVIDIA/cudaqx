/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq.h"
#include <cstddef>

namespace cudaq {

__qpu__ void
adapt_kernel(std::size_t numQubits,
             const cudaq::qkernel<void(cudaq::qvector<> &)> &statePrep,
             const std::vector<double> &thetas,
             const std::vector<double> &coefficients,
             const std::vector<cudaq::pauli_word> &trotterOpList,
             const std::vector<std::size_t> poolIndices) {
  cudaq::qvector q(numQubits);
  statePrep(q);

  auto i = 0;
  for (std::size_t j = 0; j < trotterOpList.size();) {
    auto index = poolIndices[j];
    auto &term = trotterOpList[j];
    exp_pauli(thetas[i] * coefficients[j], q, term);
    j++;
    if (index != poolIndices[j]) {
      i++;
    }
  }
}
} // namespace cudaq
