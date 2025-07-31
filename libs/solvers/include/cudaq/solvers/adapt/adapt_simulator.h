/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "../adapt.h"

using namespace cudaqx;
namespace cudaq::solvers::adapt {

/// @brief Simulator implementation of the ADAPT-VQE algorithm
///
/// This class provides a simulator-specific implementation of the ADAPT-VQE
/// algorithm. It is designed to run the algorithm on quantum simulators rather
/// than actual quantum hardware. It attempts to distribute the work with MPI
/// if possible.
class simulator : public adapt_impl {
public:
  /// @brief Run the ADAPT-VQE algorithm on a simulator
  /// @param initState Function to initialize the quantum state
  /// @param H Hamiltonian operator
  /// @param pool Pool of operators
  /// @param optimizer Optimization algorithm (unused in this implementation)
  /// @param gradient Gradient calculation method (optional)
  /// @param options Additional options for the algorithm
  /// @return Energy value obtained from the ADAPT-VQE algorithm
  /// @note This implementation is specific to quantum simulators
  result run(const cudaq::qkernel<void(cudaq::qvector<> &)> &initState,
             const spin_op &H, const std::vector<spin_op> &pool,
             const optim::optimizer &optimizer, const std::string &gradient,
             const heterogeneous_map options) override;

  /// @brief Creator function for the simulator implementation
  /// @details This function is used by the extension point mechanism to create
  /// instances of the simulator class.
  CUDAQ_ADD_EXTENSION_CREATOR_FUNCTION(adapt_impl, simulator);

  virtual ~simulator() {}
};

/// @brief Register the simulator type with the CUDA-Q framework
CUDAQ_REGISTER_EXTENSION(simulator)

} // namespace cudaq::solvers::adapt
