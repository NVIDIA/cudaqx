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

/// @brief UpCCGSD operator pool is a class which generates a pool
/// of UpCCGSD operators for use in quantum algorithms, like Adapt-VQE.
/// @details This class extends the operator_pool interface
/// therefore inherits the extension_point template, allowing for
/// runtime extensibility. The UpCCGSD pool restricts generalized
/// doubles to paired αβ→αβ excitations while retaining spin-preserving
/// generalized singles.
class upccgsd : public operator_pool {

public:
  /// @brief Generate a vector of spin operators based on the provided
  /// configuration.
  /// @details The UpCCGSD operator pool is generated with an imaginary factor
  /// 'i' in the coefficients of the operators, which simplifies the use for
  /// running Adapt-VQE.
  /// @param config A heterogeneous map containing configuration parameters for
  /// operator generation, including the number of spatial orbitals
  /// ("num-orbitals" or "num_orbitals").
  /// @return A vector of cudaq::spin_op objects representing the generated
  /// UpCCGSD operator pool.
  std::vector<cudaq::spin_op>
  generate(const heterogeneous_map &config) const override;

  /// @brief Call to macro for defining the creator function for an extension
  /// @details This function is used by the extension point mechanism to create
  /// instances of the upccgsd class.
  CUDAQ_EXTENSION_CREATOR_FUNCTION(operator_pool, upccgsd)
};

/// @brief Register the upccgsd extension type with the CUDA-Q framework
CUDAQ_REGISTER_TYPE(upccgsd)

} // namespace cudaq::solvers
