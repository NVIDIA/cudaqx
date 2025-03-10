/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "common/MeasureCounts.h"
#include "cudaq/spin_op.h"

#include "cuda-qx/core/graph.h"
#include "optimizer.h"

namespace cudaq::solvers {

/// @brief Result structure for QAOA optimization
struct qaoa_result {
  /// @brief The optimal value found by the QAOA algorithm
  double optimal_value = 0.0;

  /// @brief The optimal variational parameters that achieved the optimal value
  std::vector<double> optimal_parameters;

  /// @brief The measurement results for the optimal circuit configuration
  cudaq::sample_result optimal_config;
};

/// @brief Execute the Quantum Approximate Optimization Algorithm (QAOA) with
/// custom mixing Hamiltonian
/// @param problemHamiltonian The cost Hamiltonian encoding the optimization
/// problem
/// @param referenceHamiltonian The mixing Hamiltonian for the QAOA evolution
/// (typically X-rotation terms)
/// @param optimizer The classical optimizer to use for parameter optimization
/// @param numLayers The number of QAOA layers (p-value)
/// @param initialParameters Initial guess for the variational parameters
/// @param options Additional algorithm options passed as key-value pairs
///
/// @note User can provide the following options - {"counterdiabatic",
/// true/false} to run Digitized-Counterdiabatic QAOA (adds Ry rotations after
/// QAOA single layer)
///
/// @return qaoa_result containing the optimization results
qaoa_result qaoa(const cudaq::spin_op &problemHamiltonian,
                 const cudaq::spin_op &referenceHamiltonian,
                 const optim::optimizer &optimizer, std::size_t numLayers,
                 const std::vector<double> &initialParameters,
                 const heterogeneous_map options = {});

/// @brief Execute QAOA with default transverse field mixing Hamiltonian
/// @param problemHamiltonian The cost Hamiltonian encoding the optimization
/// problem
/// @param optimizer The classical optimizer to use for parameter optimization
/// @param numLayers The number of QAOA layers (p-value)
/// @param initialParameters Initial guess for the variational parameters
/// @param options Additional algorithm options passed as key-value pairs
///
/// @note User can provide the following options - {"counterdiabatic",
/// true/false} to run Digitized-Counterdiabatic QAOA (adds Ry rotations after
/// QAOA single layer)
///
/// @return qaoa_result containing the optimization results
qaoa_result qaoa(const cudaq::spin_op &problemHamiltonian,
                 const optim::optimizer &optimizer, std::size_t numLayers,
                 const std::vector<double> &initialParameters,
                 const heterogeneous_map options = {});

/// @brief Execute QAOA with default optimizer and mixing Hamiltonian
/// @param problemHamiltonian The cost Hamiltonian encoding the optimization
/// problem
/// @param numLayers The number of QAOA layers (p-value)
/// @param initialParameters Initial guess for the variational parameters
/// be size 2*numLayers)
/// @param options Additional algorithm options passed as key-value pairs
///
/// @note User can provide the following options - {"counterdiabatic",
/// true/false} to run Digitized-Counterdiabatic QAOA (adds Ry rotations after
/// QAOA single layer)
///
/// @return qaoa_result containing the optimization results
qaoa_result qaoa(const cudaq::spin_op &problemHamiltonian,
                 std::size_t numLayers,
                 const std::vector<double> &initialParameters,
                 const heterogeneous_map options = {});

/// @brief Execute the Quantum Approximate Optimization Algorithm (QAOA) with
/// custom mixing Hamiltonian
/// @param problemHamiltonian The cost Hamiltonian encoding the optimization
/// problem
/// @param referenceHamiltonian The mixing Hamiltonian for the QAOA evolution
/// (typically X-rotation terms)
/// @param numLayers The number of QAOA layers (p-value)
/// @param initialParameters Initial guess for the variational parameters
/// @param options Additional algorithm options passed as key-value pairs
///
/// @note User can provide the following options - {"counterdiabatic",
/// true/false} to run Digitized-Counterdiabatic QAOA (adds Ry rotations after
/// QAOA single layer)
///
/// @return qaoa_result containing the optimization results
qaoa_result qaoa(const cudaq::spin_op &problemHamiltonian,
                 const cudaq::spin_op &referenceHamiltonian,
                 std::size_t numLayers,
                 const std::vector<double> &initialParameters,
                 const heterogeneous_map options = {});

/// @brief Calculate the number of variational parameters needed for QAOA with
/// custom mixing Hamiltonian
///
/// @details This function determines the total number of variational parameters
/// required for QAOA execution based on the problem setup and options. When
/// full_parameterization is true, an angle will be used for every term in both
/// the problem and reference Hamiltonians. Otherwise, one angle per layer is
/// used for each Hamiltonian.
///
/// @param problemHamiltonian The cost Hamiltonian encoding the optimization
/// problem
/// @param referenceHamiltonian The mixing Hamiltonian for the QAOA evolution
/// @param numLayers The number of QAOA layers (p-value)
/// @param options Additional algorithm options:
///               - "full_parameterization": bool - Use individual angles for
///               each Hamiltonian term
///               - "counterdiabatic": bool - Enable counterdiabatic QAOA
///               variant, adds an Ry to every qubit in the system with its own
///               angle to optimize.
///
/// @return The total number of variational parameters needed
////
std::size_t get_num_qaoa_parameters(const cudaq::spin_op &problemHamiltonian,
                                    const cudaq::spin_op &referenceHamiltonian,
                                    std::size_t numLayers,
                                    const heterogeneous_map options = {});

/// @brief Calculate the number of variational parameters needed for QAOA with
/// default mixing Hamiltonian
///
/// @details This function determines the total number of variational parameters
/// required for QAOA execution using the default transverse field mixing
/// Hamiltonian. When full_parameterization is true, an angle will be used for
/// every term in both the problem and reference Hamiltonians. Otherwise, one
/// angle per layer is used for each Hamiltonian.
///
/// @param problemHamiltonian The cost Hamiltonian encoding the optimization
/// problem
/// @param numLayers The number of QAOA layers (p-value)
/// @param options Additional algorithm options:
///               - "full_parameterization": bool - Use individual angles for
///               each Hamiltonian term
///               - "counterdiabatic": bool - Enable counterdiabatic QAOA
///               variant, adds an Ry to every qubit in the system with its own
///               angle to optimize.
///
/// @return The total number of variational parameters needed
///
std::size_t get_num_qaoa_parameters(const cudaq::spin_op &problemHamiltonian,
                                    std::size_t numLayers,
                                    const heterogeneous_map options = {});
} // namespace cudaq::solvers
