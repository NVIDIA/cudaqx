/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "../operator_pool.h"

namespace cudaq::solvers {

// adapted from
// https://github.com/mayhallgroup/adapt-vqe/blob/master/src/operator_pools.py

/// @brief spin_complement_gsd operator pool is a class which generates a pool
/// of operators with the spin complement ground state degeneracy method for use
/// in quantum algorithms, like Adapt-VQE.
/// @details This class extends the operator_pool interface
/// therefore inherits extension_point template, allowing for
/// runtime extensibility.
class spin_complement_gsd : public operator_pool {

public:
  /// @brief Generate a vector of spin operators based on the provided
  /// configuration.
  /// @param config A heterogeneous map containing configuration parameters for
  /// operator generation.
  /// @return A vector of cudaq::spin_op objects representing the generated
  /// operator pool.
  std::vector<cudaq::spin_op>
  generate(const heterogeneous_map &config) const override;

  /// @brief Call to macro for defining the creator function for the
  /// spin_complement_gsd extension
  /// @details This function is used by the extension point mechanism to create
  /// instances of the spin_complement_gsd class.
  CUDAQ_ADD_EXTENSION_CREATOR_FUNCTION(operator_pool, spin_complement_gsd)
};
/// @brief Register the spin_complement_gsd extension type with the CUDA-Q
/// framework
CUDAQ_REGISTER_EXTENSION(spin_complement_gsd)
} // namespace cudaq::solvers
