/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "../operator_pool.h"

namespace cudaq::solvers {

/// @brief QAOA operator pool is a class with generates a pool
/// of QAOA operators for use in quantum algorithms, like Adapt-QAOA.
/// @details This class extends the operator_pool interface
/// therefore inherits the extension_point template, allowing for
/// runtime extensibility.
class qaoa_pool : public operator_pool {
public:
  /// @brief Generate a vector of spin operators based on the provided
  /// configuration.
  /// @param config A heterogeneous map containing configuration parameters for
  /// operator generation.
  /// @return A vector of cudaq::spin_op objects representing the generated
  /// operator pool.
  std::vector<cudaq::spin_op>
  generate(const heterogeneous_map &config) const override;

  /// @brief Call to macro for defining the creator function for the qaoa_pool
  /// extension
  /// @details This function is used by the extension point mechanism to create
  /// instances of the qaoa_pool class. The extension is registered with a name
  CUDAQ_ADD_EXTENSION_CUSTOM_CREATOR_FUNCTION_WITH_NAME(
      qaoa_pool, "qaoa", static std::unique_ptr<operator_pool> create() {
        return std::make_unique<qaoa_pool>();
      })
};

/// @brief Register the qaoa_pool extension type with the CUDA-Q framework
CUDAQ_REGISTER_EXTENSION(qaoa_pool)

} // namespace cudaq::solvers
