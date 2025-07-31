/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/qis/qubit_qis.h"
#include "cudaq/solvers/operators/operator_pool.h"
#include "cudaq/solvers/vqe.h"

#include <functional>

using namespace cudaqx;

/**
 * @file
 * @brief Implementation of the ADAPT-VQE algorithm
 *
 * This file contains the implementation of the Adaptive Derivative-Assembled
 * Pseudo-Trotter Variational Quantum Eigensolver (ADAPT-VQE) algorithm.
 *
 * @details
 * ADAPT-VQE is an advanced quantum algorithm designed to improve upon the
 * standard Variational Quantum Eigensolver (VQE) approach for solving quantum
 * chemistry problems. It addresses key challenges faced by traditional VQE
 * methods by dynamically constructing a problem-specific ansatz, offering
 * several advantages:
 *
 * - Faster convergence: Adaptively selects the most impactful operators,
 *   potentially achieving convergence more quickly than fixed-ansatz VQE
 * methods.
 * - Enhanced efficiency: Builds a compact ansatz tailored to the specific
 *   problem, potentially reducing overall circuit depth.
 * - Increased accuracy: Has demonstrated the ability to outperform standard
 *   VQE approaches in terms of accuracy for certain molecular systems.
 * - Adaptability: Automatically adjusts to different molecular systems without
 *   requiring significant user intervention or prior knowledge of the system's
 *   electronic structure.
 *
 * The ADAPT-VQE algorithm works by iteratively growing the quantum circuit
 * ansatz, selecting operators from a predefined pool based on their gradient
 * magnitudes. This adaptive approach allows the algorithm to focus
 * computational resources on the most relevant aspects of the problem,
 * potentially leading to more efficient and accurate simulations of molecular
 * systems on quantum computers.
 */

namespace cudaq::solvers {

namespace adapt {

/// Result type for ADAPT-VQE algorithm
/// @return Tuple containing:
///   - Final energy (double)
///   - Optimized parameters (vector of doubles)
///   - Selected operators (vector of cudaq::spin_op)
using result =
    std::tuple<double, std::vector<double>, std::vector<cudaq::spin_op>>;

/// Abstract base class for ADAPT-VQE implementation
class adapt_impl : public cudaq::extension_point<adapt_impl> {
public:
  /// Run the ADAPT-VQE algorithm
  /// @param initState Initial state preparation quantum kernel
  /// @param H Hamiltonian operator
  /// @param pool Pool of operators
  /// @param optimizer Optimization algorithm
  /// @param gradient Gradient calculation method
  /// @param options Additional options for the algorithm. Supported Keys:
  ///  - "max_iter" (int): Maximum number of iterations [default: 30]
  ///  - "grad_norm_tolerance" (double): Convergence tolerance for gradient norm
  ///  [default: 1e-5]
  ///  - "grad_norm_diff_tolerance" (double): Tolerance for difference between
  ///  gradient norms [default: 1e-5]
  ///  - "threshold_energy" (double): Energy convergence threshold [default:
  ///  1e-6]
  ///  - "initial_theta" (double): Initial value for theta parameter [default:
  ///  0.0]
  ///  - "verbose" (bool): Enable detailed output logging [default: false]
  ///  - "shots" (int): Number of measurement shots (-1 for exact simulation)
  ///  [default: -1]
  ///  - "dynamic_start" (string): Optimization mode for the theta parameters at
  ///  each iteration.
  ///      It can be either "warm", or "cold". [default: "cold"]
  ///  - "tol" (double): Tolerance for optimization [default: 1e-12]
  /// @return Result of the ADAPT-VQE algorithm
  virtual result run(const cudaq::qkernel<void(cudaq::qvector<> &)> &initState,
                     const spin_op &H, const std::vector<spin_op> &pool,
                     const optim::optimizer &optimizer,
                     const std::string &gradient,
                     const heterogeneous_map options) = 0;

  /// Virtual destructor
  virtual ~adapt_impl() {}
};

} // namespace adapt

/// @brief Run ADAPT-VQE algorithm with default optimizer
/// @param initialState Initial state preparation quantum kernel
/// @param H Hamiltonian operator
/// @param poolList Pool of operators
/// @param options Additional options for the algorithm. Supported Keys:
///  - "max_iter" (int): Maximum number of iterations [default: 30]
///  - "grad_norm_tolerance" (double): Convergence tolerance for gradient norm
///  [default: 1e-5]
///  - "grad_norm_diff_tolerance" (double): Tolerance for difference between
///  gradient norms [default: 1e-5]
///  - "threshold_energy" (double): Energy convergence threshold [default: 1e-6]
///  - "initial_theta" (double): Initial value for theta parameter [default:
///  0.0]
///  - "verbose" (bool): Enable detailed output logging [default: false]
///  - "shots" (int): Number of measurement shots (-1 for exact simulation)
///  [default: -1]
///  - "dynamic_start" (string): Optimization mode for the theta parameters at
///  each iteration.
///      It can be either "warm", or "cold". [default: "cold"]
///  - "tol" (double): Tolerance for optimization [default: 1e-12]
/// @return Result of the ADAPT-VQE algorithm
static inline adapt::result
adapt_vqe(const cudaq::qkernel<void(cudaq::qvector<> &)> &initialState,
          const spin_op &H, const std::vector<spin_op> &poolList,
          const heterogeneous_map options = heterogeneous_map()) {
  auto &platform = cudaq::get_platform();
  auto impl =
      adapt::adapt_impl::get(platform.is_simulator() ? "simulator" : "remote");
  auto opt = optim::optimizer::get("cobyla");
  return impl->run(initialState, H, poolList, *opt, "", options);
}

/// @brief Run ADAPT-VQE algorithm with custom optimizer
/// @param initialState Initial state preparation quantum kernel
/// @param H Hamiltonian operator
/// @param poolList Pool of operators
/// @param optimizer Custom optimization algorithm
/// @param options Additional options for the algorithm. Supported Keys:
///  - "max_iter" (int): Maximum number of iterations [default: 30]
///  - "grad_norm_tolerance" (double): Convergence tolerance for gradient norm
///  [default: 1e-5]
///  - "grad_norm_diff_tolerance" (double): Tolerance for difference between
///  gradient norms [default: 1e-5]
///  - "threshold_energy" (double): Energy convergence threshold [default: 1e-6]
///  - "initial_theta" (double): Initial value for theta parameter [default:
///  0.0]
///  - "verbose" (bool): Enable detailed output logging [default: false]
///  - "shots" (int): Number of measurement shots (-1 for exact simulation)
///  [default: -1]
///  - "dynamic_start" (string): Optimization mode for the theta parameters at
///  each iteration.
///      It can be either "warm", or "cold". [default: "cold"]
///  - "tol" (double): Tolerance for optimization [default: 1e-12]
/// @return Result of the ADAPT-VQE algorithm
static inline adapt::result
adapt_vqe(const cudaq::qkernel<void(cudaq::qvector<> &)> &initialState,
          const spin_op &H, const std::vector<spin_op> &poolList,
          const optim::optimizer &optimizer,
          const heterogeneous_map options = heterogeneous_map()) {
  auto &platform = cudaq::get_platform();
  auto impl =
      adapt::adapt_impl::get(platform.is_simulator() ? "simulator" : "remote");
  return impl->run(initialState, H, poolList, optimizer, "", options);
}

/// @brief Run ADAPT-VQE algorithm with custom optimizer and gradient method
/// @param initialState Initial state preparation quantum kernel
/// @param H Hamiltonian operator
/// @param poolList Pool of operators
/// @param optimizer Custom optimization algorithm
/// @param gradient Gradient calculation method
/// @param options Additional options for the algorithm. Supported Keys:
///  - "max_iter" (int): Maximum number of iterations [default: 30]
///  - "grad_norm_tolerance" (double): Convergence tolerance for gradient norm
///  [default: 1e-5]
///  - "grad_norm_diff_tolerance" (double): Tolerance for difference between
///  gradient norms [default: 1e-5]
///  - "threshold_energy" (double): Energy convergence threshold [default: 1e-6]
///  - "initial_theta" (double): Initial value for theta parameter [default:
///  0.0]
///  - "verbose" (bool): Enable detailed output logging [default: false]
///  - "shots" (int): Number of measurement shots (-1 for exact simulation)
///  [default: -1]
///  - "dynamic_start" (string): Optimization mode for the theta parameters at
///  each iteration.
///      It can be either "warm", or "cold". [default: "cold"]
///  - "tol" (double): Tolerance for optimization [default: 1e-12]
/// @return Result of the ADAPT-VQE algorithm
static inline adapt::result
adapt_vqe(const cudaq::qkernel<void(cudaq::qvector<> &)> &initialState,
          const spin_op &H, const std::vector<spin_op> &poolList,
          const optim::optimizer &optimizer, const std::string &gradient,
          const heterogeneous_map options = heterogeneous_map()) {
  auto &platform = cudaq::get_platform();
  auto impl =
      adapt::adapt_impl::get(platform.is_simulator() ? "simulator" : "remote");
  return impl->run(initialState, H, poolList, optimizer, gradient, options);
}

} // namespace cudaq::solvers
