/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <any>
#include <optional>
#include <unordered_map>

#include "cuda-qx/core/heterogeneous_map.h"
#include "cudaq/spin_op.h"
#include "cudaq/utils/extension_point.h"

using namespace cudaqx;

namespace cudaq::solvers {

/// @brief Interface for generating quantum operator pools used in quantum
/// algorithms.
/// @details This class extends the extension_point template, allowing for
/// runtime extensibility.
class operator_pool : public cudaq::extension_point<operator_pool> {
public:
  /// @brief Default constructor.
  operator_pool() = default;

  /// @brief Virtual destructor to ensure proper cleanup of derived classes.
  virtual ~operator_pool() {}

  /// @brief Generate a vector of spin operators based on the provided
  /// configuration.
  /// @param config A heterogeneous map containing configuration parameters for
  /// operator generation.
  /// @return A vector of cudaq::spin_op objects representing the generated
  /// operator pool.
  virtual std::vector<cudaq::spin_op>
  generate(const heterogeneous_map &config) const = 0;
};

/// @brief Retrieves a quantum operator pool based on the specified name and
/// configuration options.
///
/// This function creates and returns a vector of quantum spin operators by
/// instantiating the appropriate operator_pool implementation specified by the
/// name parameter. The generated operators are configured according to the
/// provided options.
///
/// @param name The identifier string for the desired operator pool
/// implementation
/// @param options Configuration parameters for operator pool generation stored
/// in a heterogeneous map
/// @return std::vector<cudaq::spin_op> A vector containing the generated
/// quantum spin operators
std::vector<cudaq::spin_op> get_operator_pool(const std::string &name,
                                              const heterogeneous_map &options);

} // namespace cudaq::solvers
