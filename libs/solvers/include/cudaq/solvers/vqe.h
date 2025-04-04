/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "observe_gradient.h"
#include "optimizer.h"

using namespace cudaqx;

namespace cudaq::solvers {

/// @brief A vqe_result encapsulates all the data produced
/// by a standard variational quantum eigensolver execution. It
/// provides the programmer with the optimal energy and parameters
/// as well as a list of all execution data at each iteration.
struct vqe_result {
  double energy;
  std::vector<double> optimal_parameters;
  std::vector<observe_iteration> iteration_data;
  operator double() { return energy; }

  // FIXME add to/from file functionality
};

/// @brief Compute the minimal eigenvalue of the given Hamiltonian with VQE.
/// @details Given a quantum kernel of signature `void(std::vector<double>)`,
/// run the variational quantum eigensolver routine to compute
/// the minimum eigenvalue of the specified hermitian `spin_op`.
/// @tparam QuantumKernel Type of the quantum kernel
/// @param kernel Quantum kernel to be optimized
/// @param hamiltonian Spin operator representing the Hamiltonian
/// @param optimizer Optimization algorithm to use
/// @param gradient Gradient computation method
/// @param initial_parameters Initial parameters for the optimization
/// @param options Additional options for the VQE algorithm. Available options
/// - "tol" (double): Tolerance for optimization [default: 1e-12]
/// @return VQE result containing optimal energy, parameters, and iteration data
template <typename QuantumKernel>
  requires std::invocable<QuantumKernel, std::vector<double>>
static inline vqe_result vqe(QuantumKernel &&kernel, const spin_op &hamiltonian,
                             optim::optimizer &optimizer,
                             observe_gradient &gradient,
                             const std::vector<double> &initial_parameters,
                             heterogeneous_map options = heterogeneous_map()) {
  if (!optimizer.requiresGradients())
    throw std::runtime_error("[vqe] provided optimizer does not require "
                             "gradients, yet gradient instance provided.");

  options.insert("initial_parameters", initial_parameters);
  std::vector<observe_iteration> data;

  /// Run the optimization
  auto [groundEnergy, optParams] = optimizer.optimize(
      initial_parameters.size(),
      [&](const std::vector<double> &x, std::vector<double> &dx) {
        auto res =
            cudaq::observe(options.get("shots", -1), kernel, hamiltonian, x);
        if (options.get("verbose", false))
          printf("<H> = %.12lf\n", res.expectation());
        data.emplace_back(x, res, observe_execution_type::function);
        gradient.compute(x, dx, res.expectation(), options.get("shots", -1));
        for (auto datum : gradient.data)
          data.push_back(datum);

        return res.expectation();
      },
      options);

  return {groundEnergy, optParams, data};
}

/// @brief Overloaded VQE function using string-based optimizer and gradient
/// selection
template <typename QuantumKernel>
  requires std::invocable<QuantumKernel, std::vector<double>>
static inline vqe_result vqe(QuantumKernel &&kernel, const spin_op &hamiltonian,
                             const std::string &optName,
                             const std::string &gradName,
                             const std::vector<double> &initial_parameters,
                             heterogeneous_map options = heterogeneous_map()) {

  if (!cudaq::optim::optimizer::is_registered(optName))
    throw std::runtime_error("provided optimizer is not valid.");

  if (!cudaq::observe_gradient::is_registered(gradName))
    throw std::runtime_error("provided optimizer is not valid.");

  auto optimizer = cudaq::optim::optimizer::get(optName);
  auto gradient = cudaq::observe_gradient::get(gradName, kernel, hamiltonian);

  if (!optimizer->requiresGradients())
    throw std::runtime_error("[vqe] provided optimizer does not require "
                             "gradients, yet gradient instance provided.");

  options.insert("initial_parameters", initial_parameters);
  std::vector<observe_iteration> data;

  /// Run the optimization
  auto [groundEnergy, optParams] = optimizer->optimize(
      initial_parameters.size(),
      [&](const std::vector<double> &x, std::vector<double> &dx) {
        auto res =
            cudaq::observe(options.get("shots", -1), kernel, hamiltonian, x);
        if (options.get("verbose", false))
          printf("<H> = %.12lf\n", res.expectation());
        data.emplace_back(x, res, observe_execution_type::function);
        gradient->compute(x, dx, res.expectation(), options.get("shots", -1));
        for (auto datum : gradient->data)
          data.push_back(datum);

        return res.expectation();
      },
      options);

  return {groundEnergy, optParams, data};
}

/// @brief Overloaded VQE function using string-based optimizer selection
/// without gradient
template <typename QuantumKernel>
  requires std::invocable<QuantumKernel, std::vector<double>>
static inline vqe_result vqe(QuantumKernel &&kernel, const spin_op &hamiltonian,
                             const std::string &optName,
                             const std::vector<double> &initial_parameters,
                             heterogeneous_map options = heterogeneous_map()) {

  if (!cudaq::optim::optimizer::is_registered(optName))
    throw std::runtime_error("provided optimizer is not valid.");

  auto optimizer = cudaq::optim::optimizer::get(optName);

  if (optimizer->requiresGradients())
    throw std::runtime_error("[vqe] provided optimizer requires "
                             "gradients, yet no gradient instance provided.");

  options.insert("initial_parameters", initial_parameters);
  std::vector<observe_iteration> data;

  /// Run the optimization
  auto [groundEnergy, optParams] = optimizer->optimize(
      initial_parameters.size(),
      [&](const std::vector<double> &x, std::vector<double> &dx) {
        auto res =
            cudaq::observe(options.get("shots", -1), kernel, hamiltonian, x);
        if (options.get("verbose", false))
          printf("<H> = %.12lf\n", res.expectation());
        data.emplace_back(x, res, observe_execution_type::function);
        return res.expectation();
      },
      options);

  return {groundEnergy, optParams, data};
}

/// @brief Overloaded VQE function using string-based optimizer and provided
/// gradient object
template <typename QuantumKernel>
  requires std::invocable<QuantumKernel, std::vector<double>>
static inline vqe_result vqe(QuantumKernel &&kernel, const spin_op &hamiltonian,
                             const std::string &optName,
                             observe_gradient &gradient,
                             const std::vector<double> &initial_parameters,
                             heterogeneous_map options = heterogeneous_map()) {

  if (!cudaq::optim::optimizer::is_registered(optName))
    throw std::runtime_error("provided optimizer is not valid.");

  auto optimizer = cudaq::optim::optimizer::get(optName);
  if (!optimizer->requiresGradients())
    throw std::runtime_error("[vqe] provided optimizer does not require "
                             "gradients, yet gradient instance provided.");

  options.insert("initial_parameters", initial_parameters);
  std::vector<observe_iteration> data;

  /// Run the optimization
  auto [groundEnergy, optParams] = optimizer->optimize(
      initial_parameters.size(),
      [&](const std::vector<double> &x, std::vector<double> &dx) {
        auto res =
            cudaq::observe(options.get("shots", -1), kernel, hamiltonian, x);
        if (options.get("verbose", false))
          printf("<H> = %.12lf\n", res.expectation());
        data.emplace_back(x, res, observe_execution_type::function);
        gradient.compute(x, dx, res.expectation(), options.get("shots", -1));
        for (auto datum : gradient.data)
          data.push_back(datum);

        return res.expectation();
      },
      options);

  return {groundEnergy, optParams, data};
}

/// @brief Overloaded VQE function using provided optimizer and string-based
/// gradient selection
template <typename QuantumKernel>
  requires std::invocable<QuantumKernel, std::vector<double>>
static inline vqe_result vqe(QuantumKernel &&kernel, const spin_op &hamiltonian,
                             optim::optimizer &optimizer,
                             const std::string &gradName,
                             const std::vector<double> &initial_parameters,
                             heterogeneous_map options = heterogeneous_map()) {

  if (!cudaq::observe_gradient::is_registered(gradName))
    throw std::runtime_error("provided optimizer is not valid.");

  auto gradient = cudaq::observe_gradient::get(gradName, kernel, hamiltonian);

  if (!optimizer.requiresGradients())
    throw std::runtime_error("[vqe] provided optimizer does not require "
                             "gradients, yet gradient instance provided.");

  options.insert("initial_parameters", initial_parameters);
  std::vector<observe_iteration> data;

  /// Run the optimization
  auto [groundEnergy, optParams] = optimizer.optimize(
      initial_parameters.size(),
      [&](const std::vector<double> &x, std::vector<double> &dx) {
        auto res =
            cudaq::observe(options.get("shots", -1), kernel, hamiltonian, x);
        if (options.get("verbose", false))
          printf("<H> = %.12lf\n", res.expectation());
        data.emplace_back(x, res, observe_execution_type::function);
        gradient->compute(x, dx, res.expectation(), options.get("shots", -1));
        for (auto datum : gradient->data)
          data.push_back(datum);

        return res.expectation();
      },
      options);

  return {groundEnergy, optParams, data};
}

/// @brief Overloaded VQE function using provided optimizer without gradient
template <typename QuantumKernel>
  requires std::invocable<QuantumKernel, std::vector<double>>
static inline vqe_result vqe(QuantumKernel &&kernel, const spin_op &hamiltonian,
                             optim::optimizer &optimizer,
                             const std::vector<double> &initial_parameters,
                             heterogeneous_map options = heterogeneous_map()) {

  if (optimizer.requiresGradients())
    throw std::runtime_error("[vqe] provided optimizer does not require "
                             "gradients, yet gradient instance provided.");

  options.insert("initial_parameters", initial_parameters);

  std::vector<observe_iteration> data;

  /// Run the optimization
  auto [groundEnergy, optParams] = optimizer.optimize(
      initial_parameters.size(),
      [&](const std::vector<double> &x, std::vector<double> &dx) {
        auto res =
            cudaq::observe(options.get("shots", -1), kernel, hamiltonian, x);
        if (options.get("verbose", false))
          printf("<H> = %.12lf\n", res.expectation());
        data.emplace_back(x, res, observe_execution_type::function);

        return res.expectation();
      },
      options);

  return {groundEnergy, optParams, data};
}

template <typename QuantumKernel>
  requires std::invocable<QuantumKernel, std::vector<double>>
static inline vqe_result vqe(QuantumKernel &&kernel, const spin_op &hamiltonian,
                             const std::vector<double> &initial_parameters,
                             heterogeneous_map options = heterogeneous_map()) {

  auto optimizer = optim::optimizer::get("cobyla");
  options.insert("initial_parameters", initial_parameters);

  std::vector<observe_iteration> data;

  /// Run the optimization
  auto [groundEnergy, optParams] = optimizer->optimize(
      initial_parameters.size(),
      [&](const std::vector<double> &x, std::vector<double> &dx) {
        auto res =
            cudaq::observe(options.get("shots", -1), kernel, hamiltonian, x);
        if (options.get("verbose", false))
          printf("<H> = %.12lf\n", res.expectation());
        data.emplace_back(x, res, observe_execution_type::function);

        return res.expectation();
      },
      options);

  return {groundEnergy, optParams, data};
}

template <typename QuantumKernel, typename ArgTranslator>
static inline vqe_result vqe(QuantumKernel &&kernel, const spin_op &hamiltonian,
                             optim::optimizer &optimizer,
                             const std::vector<double> &initial_parameters,
                             ArgTranslator &&translator,
                             heterogeneous_map options = heterogeneous_map()) {

  if (optimizer.requiresGradients())
    throw std::runtime_error("[vqe] provided optimizer requires "
                             "gradients, yet gradient instance not provided.");

  options.insert("initial_parameters", initial_parameters);

  std::vector<observe_iteration> data;

  /// Run the optimization
  auto [groundEnergy, optParams] = optimizer.optimize(
      initial_parameters.size(),
      [&](const std::vector<double> &x, std::vector<double> &dx) {
        auto res = std::apply(
            [&](auto &&...arg) {
              return cudaq::observe(kernel, hamiltonian, arg...);
            },
            translator(x));
        if (options.get("verbose", false))
          printf("<H> = %.12lf\n", res.expectation());
        data.emplace_back(x, res, observe_execution_type::function);

        return res.expectation();
      },
      options);

  return {groundEnergy, optParams, data};
}

template <typename QuantumKernel, typename ArgTranslator>
static inline vqe_result vqe(QuantumKernel &&kernel, const spin_op &hamiltonian,
                             const std::vector<double> &initial_parameters,
                             ArgTranslator &&translator,
                             heterogeneous_map options = heterogeneous_map()) {

  auto optimizer = optim::optimizer::get("cobyla");
  options.insert("initial_parameters", initial_parameters);

  std::vector<observe_iteration> data;

  /// Run the optimization
  auto [groundEnergy, optParams] = optimizer->optimize(
      initial_parameters.size(),
      [&](const std::vector<double> &x, std::vector<double> &dx) {
        auto res = std::apply(
            [&](auto &&...arg) {
              return cudaq::observe(kernel, hamiltonian, arg...);
            },
            translator(x));
        if (options.get("verbose", false))
          printf("<H> = %.12lf\n", res.expectation());
        data.emplace_back(x, res, observe_execution_type::function);

        return res.expectation();
      },
      options);

  return {groundEnergy, optParams, data};
}

template <typename QuantumKernel, typename ArgTranslator>
static inline vqe_result
vqe(QuantumKernel &&kernel, const spin_op &hamiltonian,
    optim::optimizer &optimizer, observe_gradient &gradient,
    const std::vector<double> &initial_parameters, ArgTranslator &&translator,
    heterogeneous_map options = heterogeneous_map()) {

  if (!optimizer.requiresGradients())
    throw std::runtime_error("[vqe] provided optimizer does not require "
                             "gradients, yet gradient instance provided.");

  options.insert("initial_parameters", initial_parameters);

  std::vector<observe_iteration> data;

  /// Run the optimization
  auto [groundEnergy, optParams] = optimizer.optimize(
      initial_parameters.size(),
      [&](const std::vector<double> &x, std::vector<double> &dx) {
        auto res = std::apply(
            [&](auto &&...arg) {
              return cudaq::observe(kernel, hamiltonian, arg...);
            },
            translator(x));
        if (options.get("verbose", false))
          printf("<H> = %.12lf\n", res.expectation());
        data.emplace_back(x, res, observe_execution_type::function);
        gradient.compute(x, dx, res.expectation(), options.get("shots", -1));
        return res.expectation();
      },
      options);

  return {groundEnergy, optParams, data};
}

} // namespace cudaq::solvers
