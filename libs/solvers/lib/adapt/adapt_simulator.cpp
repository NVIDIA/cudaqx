/*******************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <iostream>

#include "common/Logger.h"
#include "cudaq.h"

#include "device/adapt.h"
#include "device/prepare_state.h"
#include "cudaq/solvers/adapt/adapt_simulator.h"
#include "cudaq/solvers/vqe.h"
#include "cudaq/solvers/adapt/variant.h"

#include <nlohmann/json.hpp>

namespace cudaq::solvers::adapt {

result
simulator::run(const cudaq::qkernel<void(cudaq::qvector<> &)> &initialState,
               const spin_op &H, const std::vector<spin_op> &pool,
               const optim::optimizer &optimizer, const std::string &gradient,
               const heterogeneous_map options) {

  if (pool.empty())
    throw std::runtime_error("Invalid adapt input, operator pool is empty.");

  std::vector<cudaq::pauli_word> pauliWords;
  std::vector<double> thetas, coefficients;
  std::vector<std::size_t> poolIndices;
  std::vector<cudaq::spin_op> chosenOps;
  double latestEnergy = std::numeric_limits<double>::max();
  double ediff = std::numeric_limits<double>::max();

  auto variant = adapt::AdaptVariant::VQE;
  if (options.contains("adapt_variant")) {
    // String first
    try {
      auto s = options.get<std::string>("adapt_variant");
      adapt::AdaptVariant tmp;
      if (adapt::parse_variant(s, tmp)) variant = tmp;
    } catch (...) {
      // Int fallback
      try {
        int v = options.get<int>("adapt_variant");
        variant = static_cast<adapt::AdaptVariant>(v);
      } catch (...) {}
    }
  }

  if (variant == adapt::AdaptVariant::QAOA) {
    throw std::runtime_error("ADAPT-QAOA not yet implemented.");
  }

  int maxIter = options.get<int>("max_iter", 30);
  auto grad_norm_tolerance = options.get<double>("grad_norm_tolerance", 1e-5);
  auto tolNormDiff = options.get<double>("grad_norm_diff_tolerance", 1e-5);
  auto thresholdE = options.get<double>("threshold_energy", 1e-6);
  auto initTheta = options.get<double>("initial_theta", 0.0);
  auto mutable_options = options;

  auto numQubits = H.num_qubits();

  // Assumes each rank can see numQpus, models a distributed
  // architecture where each rank is a compute node, and each node
  // has numQpus GPUs available. Each GPU is indexed 0, 1, 2, ..
  std::size_t numQpus = cudaq::get_platform().num_qpus();
  std::size_t numRanks =
      cudaq::mpi::is_initialized() ? cudaq::mpi::num_ranks() : 1;
  std::size_t rank = cudaq::mpi::is_initialized() ? cudaq::mpi::rank() : 0;
  
  double energy = 0.0, lastNorm = std::numeric_limits<double>::max();

  std::vector<spin_op> commutators;
  std::size_t total_elements = pool.size();

  // Check if operator has only imaginary coefficients
  // checking the first one is enough, we assume the pool is homogeneous
  const auto &c = pool[0].begin()->evaluate_coefficient();
  bool isImaginary =
      (std::abs(c.real()) <= 1e-9) && (std::abs(c.imag()) > 1e-9);
  auto coeff = (!isImaginary) ? std::complex<double>{0.0, 1.0}
                              : std::complex<double>{1.0, 0.0};
  
  for (auto &op : pool) {
    auto commutator = H * op - op * H;
    commutator.canonicalize().trim();
    if (commutator.num_terms() > 0)
      commutators.push_back(coeff * commutator);
  }

  // Start of with the initial |psi_n>
  cudaq::state state = get_state(adapt_kernel, numQubits, initialState, thetas,
                                 coefficients, pauliWords, poolIndices);


  int step = 0;
  while (true) {
    if (options.get("verbose", false))
      printf("Step %d\n", step);
    if (step >= maxIter) {
      std::cerr
          << "Warning: Timed out, number of iteration steps exceeds maxIter!"
          << std::endl;
      break;
    }
    step++;
    // Step 1 - compute <psi|[H,Oi]|psi> vector
    std::vector<double> gradients;
    std::vector<observe_result> results;
    double gradNorm = 0.0;

    if (numQpus == 1) {
      for (std::size_t i = 0; i < commutators.size(); i++) {
        cudaq::info("Compute commutator {}", i);
        results.emplace_back(observe(prepare_state, commutators[i], state));
      }
    }

    // Get the gradient results
    std::transform(results.begin(), results.end(),
                   std::back_inserter(gradients),
                   [](auto &&el) { return std::fabs(el.expectation()); });

    // Compute the local gradient norm
    double norm = 0.0;
    for (auto &g : gradients)
      norm += g * g;
    norm = std::sqrt(norm);

    auto iter = std::max_element(gradients.begin(), gradients.end());
    double maxGrad = *iter;
    auto maxOpIdx = std::distance(gradients.begin(), iter);

    // Convergence is reached if gradient values are small
    if (norm < grad_norm_tolerance ||
        std::fabs(lastNorm - norm) < tolNormDiff || ediff < thresholdE)
      break;

    // Use the operator from the pool
    auto op = pool[maxOpIdx];
    if (!isImaginary)
      op = std::complex<double>{0.0, 1.0} * pool[maxOpIdx];

    chosenOps.push_back(op);
    thetas.push_back(initTheta);

    for (auto o : op) {
      pauliWords.emplace_back(o.get_pauli_word(numQubits));
      coefficients.push_back(o.evaluate_coefficient().imag());
      poolIndices.push_back(maxOpIdx);
    }

    optim::optimizable_function objective;
    std::unique_ptr<observe_gradient> defaultGradient;
    // If we don't need gradients, objective is simple
    if (!optimizer.requiresGradients()) {
      objective = [&, thetas, coefficients](const std::vector<double> &x,
                                            std::vector<double> &dx) mutable {
        auto res = cudaq::observe(adapt_kernel, H, numQubits, initialState, x,
                                  coefficients, pauliWords, poolIndices);
        if (options.get("verbose", false))
          printf("<H> = %.12lf\n", res.expectation());
        return res.expectation();
      };
    } else {
      auto localGradientName = gradient;
      if (gradient.empty())
        localGradientName = "parameter_shift";

      defaultGradient = observe_gradient::get(
          localGradientName,
          [&, thetas, coefficients, pauliWords](const std::vector<double> xx) {
            std::apply([&](auto &&...new_args) { adapt_kernel(new_args...); },
                       std::forward_as_tuple(numQubits, initialState, xx,
                                             coefficients, pauliWords,
                                             poolIndices));
          },
          H);
      objective = [&, thetas, coefficients](const std::vector<double> &x,
                                            std::vector<double> &dx) mutable {
        // FIXME get shots in here...
        auto res = cudaq::observe(adapt_kernel, H, numQubits, initialState, x,
                                  coefficients, pauliWords, poolIndices);
        if (options.get("verbose", false))
          printf("<H> = %.12lf\n", res.expectation());
        defaultGradient->compute(x, dx, res.expectation(),
                                 options.get("shots", -1));
        return res.expectation();
      };
    }

    if (options.contains("dynamic_start")) {
      if (options.get<std::string>("dynamic_start") == "warm")
        mutable_options.insert("initial_parameters", thetas);
    }

    auto [groundEnergy, optParams] =
        const_cast<optim::optimizer &>(optimizer).optimize(
            thetas.size(), objective, mutable_options);

    // Set the new optimzal parameters
    thetas = optParams;
    energy = groundEnergy;

    // Set the norm for the next iteration's check
    lastNorm = norm;
    state = get_state(adapt_kernel, numQubits, initialState, thetas,
                      coefficients, pauliWords, poolIndices);

    ediff = std::fabs(latestEnergy - groundEnergy);
    latestEnergy = groundEnergy;
  }

  return std::make_tuple(energy, thetas, chosenOps);

  }

}