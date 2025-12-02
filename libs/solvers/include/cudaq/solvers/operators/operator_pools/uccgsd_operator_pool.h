/****************************************************************-*- C++ -*-****
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "../operator_pool.h"

namespace cudaq::solvers {

/// @brief UCCGSD operator pool is a class which generates a pool
/// of UCCGSD operators for use in quantum algorithms, like Adapt-VQE.
/// @details This class extends the operator_pool interface
/// therefore inherits the extension_point template, allowing for
/// runtime extensibility.
class uccgsd : public operator_pool {

public:
  /// @brief Generate a vector of spin operators based on the provided
  /// configuration.
  /// @details The UCCGSD operator pool is generated with an imaginary factor
  /// 'i' in the coefficients of the operators, which simplifies the use for
  /// running Adapt-VQE.
  /// @param config A heterogeneous map containing configuration parameters for
  /// operator generation.
  /// @return A vector of cudaq::spin_op objects representing the generated
  /// operator pool.
  std::vector<cudaq::spin_op>
  generate(const heterogeneous_map &config) const override;

  /// @brief Call to macro for defining the creator function for an extension
  /// @details This function is used by the extension point mechanism to create
  /// instances of the uccgsd class.
  CUDAQ_EXTENSION_CREATOR_FUNCTION(operator_pool, uccgsd)
};
/// @brief Register the uccgsd extension type with the CUDA-Q framework
CUDAQ_REGISTER_TYPE(uccgsd)

} // namespace cudaq::solvers
