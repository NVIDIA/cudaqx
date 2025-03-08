/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/solvers/qaoa/qaoa_device.h"
#include "cudaq/solvers/vqe.h"

#include "cudaq/qis/pauli_word.h"
#include "cudaq/solvers/qaoa.h"

namespace cudaq::solvers {
cudaq::spin_op
getDefaultReferenceHamiltonian(const cudaq::spin_op &problemHamiltonian) {
  cudaq::spin_op referenceHamiltonian;
  auto numQubits = problemHamiltonian.num_qubits();

  // Add X terms for each qubit as the default mixer
  for (std::size_t i = 0; i < numQubits; i++) {
    referenceHamiltonian += cudaq::spin::i(numQubits - 1) * cudaq::spin::x(i);
  }

  referenceHamiltonian -= cudaq::spin::i(numQubits - 1);
  return referenceHamiltonian;
}

std::size_t get_num_qaoa_parameters(const cudaq::spin_op &problemHamiltonian,
                                    const cudaq::spin_op &referenceHamiltonian,
                                    std::size_t numLayers,
                                    const heterogeneous_map options) {
  auto counterdiabatic = options.get<bool>("counterdiabatic", false);
  auto full_parameterization =
      options.get<bool>(std::vector<std::string>{"full-parameterization",
                                                 "full_parameterization"},
                        false);

  // Compute the expected number of parameters
  std::size_t expectedNumParams = 0;
  if (full_parameterization) {
    auto nonIdTerms = 0;
    for (const auto &term : referenceHamiltonian)
      if (!term.is_identity())
        nonIdTerms++;

    expectedNumParams =
        numLayers * (problemHamiltonian.num_terms() + nonIdTerms);
  } else {
    expectedNumParams = 2 * numLayers;
  }

  if (counterdiabatic)
    expectedNumParams += numLayers * problemHamiltonian.num_qubits();

  return expectedNumParams;
}

std::size_t get_num_qaoa_parameters(const cudaq::spin_op &problemHamiltonian,
                                    std::size_t numLayers,
                                    const heterogeneous_map options) {
  return get_num_qaoa_parameters(
      problemHamiltonian, getDefaultReferenceHamiltonian(problemHamiltonian),
      numLayers, options);
}

qaoa_result qaoa(const cudaq::spin_op &problemHamiltonian,
                 const cudaq::spin_op &referenceHamiltonian,
                 const optim::optimizer &optimizer, std::size_t numLayers,
                 const std::vector<double> &initialParameters,
                 const heterogeneous_map options) {
  auto expectedNumParams = get_num_qaoa_parameters(
      problemHamiltonian, referenceHamiltonian, numLayers, options);

  if (initialParameters.size() != expectedNumParams)
    throw std::runtime_error(
        "qaoa error - invalid number of initial parameters. " +
        std::to_string(expectedNumParams) + " parameters are required, but " +
        std::to_string(initialParameters.size()) + " provided.");

  auto counterdiabatic = options.get<bool>("counterdiabatic", false);
  auto full_parameterization =
      options.get<bool>(std::vector<std::string>{"full-parameterization",
                                                 "full_parameterization"},
                        false);
  std::vector<double> probHCoeffs, refHCoeffs;
  std::vector<cudaq::pauli_word> probHWords, refHWords;
  for (const auto &o : problemHamiltonian) {
    probHWords.emplace_back(o.to_string(false));
    probHCoeffs.push_back(o.get_coefficient().real());
  }

  for (const auto &o : referenceHamiltonian) {
    refHWords.emplace_back(o.to_string(false));
    refHCoeffs.push_back(o.get_coefficient().real());
  }

  auto numQubits = problemHamiltonian.num_qubits();
  auto argsTranslator = [&](std::vector<double> x) {
    return std::make_tuple(numQubits, numLayers, x, probHCoeffs, probHWords,
                           refHCoeffs, refHWords, full_parameterization,
                           counterdiabatic);
  };

  auto [optVal, optParams, data] =
      vqe(qaoa_kernel, problemHamiltonian, initialParameters, argsTranslator,
          options);
  auto counts = cudaq::sample(qaoa_kernel, numQubits, numLayers, optParams,
                              probHCoeffs, probHWords, refHCoeffs, refHWords,
                              full_parameterization, counterdiabatic);
  return qaoa_result{optVal, optParams, counts};
}

qaoa_result qaoa(const cudaq::spin_op &problemHamiltonian,
                 const optim::optimizer &optimizer, std::size_t numLayers,
                 const std::vector<double> &initialParameters,
                 const heterogeneous_map options) {
  // Create default transverse field mixing Hamiltonian
  cudaq::spin_op referenceHamiltonian =
      getDefaultReferenceHamiltonian(problemHamiltonian);

  // Delegate to the full implementation
  return qaoa(problemHamiltonian, referenceHamiltonian, optimizer, numLayers,
              initialParameters, options);
}

qaoa_result qaoa(const cudaq::spin_op &problemHamiltonian,
                 std::size_t numLayers,
                 const std::vector<double> &initialParameters,
                 const heterogeneous_map options) {
  // Validate inputs
  if (initialParameters.empty())
    throw std::invalid_argument("Initial parameters cannot be empty");

  // Create default COBYLA optimizer
  auto defaultOptimizer = optim::optimizer::get("cobyla");

  // Delegate to the version with explicit optimizer
  return qaoa(problemHamiltonian, *defaultOptimizer, numLayers,
              initialParameters, options);
}

qaoa_result qaoa(const cudaq::spin_op &problemHamiltonian,
                 const cudaq::spin_op &referenceHamiltonian,
                 std::size_t numLayers,
                 const std::vector<double> &initialParameters,
                 const heterogeneous_map options) {
  // Create default transverse field mixing Hamiltonian
  auto numQubits = problemHamiltonian.num_qubits();
  auto defaultOptimizer = optim::optimizer::get("cobyla");

  // Delegate to the full implementation
  return qaoa(problemHamiltonian, referenceHamiltonian, *defaultOptimizer,
              numLayers, initialParameters, options);
}

} // namespace cudaq::solvers
