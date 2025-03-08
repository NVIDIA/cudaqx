/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/Logger.h"
#include "cudaq.h"

#include "device/adapt.h"
#include "device/prepare_state.h"
#include "cudaq/solvers/adapt/adapt_simulator.h"
#include "cudaq/solvers/vqe.h"

#include <iostream>
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
  std::vector<cudaq::spin_op> chosenOps;
  auto tol = options.get<double>("grad_norm_tolerance", 1e-5);
  auto numQubits = H.num_qubits();
  // Assumes each rank can see numQpus, models a distributed
  // architecture where each rank is a compute node, and each node
  // has numQpus GPUs available. Each GPU is indexed 0, 1, 2, ..
  std::size_t numQpus = cudaq::get_platform().num_qpus();
  std::size_t numRanks =
      cudaq::mpi::is_initialized() ? cudaq::mpi::num_ranks() : 1;
  std::size_t rank = cudaq::mpi::is_initialized() ? cudaq::mpi::rank() : 0;
  double energy = 0.0, lastNorm = std::numeric_limits<double>::max();

  // poolList is split into numRanks chunks, and each chunk can be
  // further parallelized across numQpus.
  // Compute the [H,Oi]
  std::vector<spin_op> commutators;
  std::size_t total_elements = pool.size();
  std::size_t elements_per_rank = total_elements / numRanks;
  std::size_t remainder = total_elements % numRanks;
  std::size_t start = rank * elements_per_rank + std::min(rank, remainder);
  std::size_t end = start + elements_per_rank + (rank < remainder ? 1 : 0);
  for (int i = start; i < end; i++) {
    auto op = pool[i];
    commutators.emplace_back(H * op - op * H);
  }

  nlohmann::json initInfo = {{"num-qpus", numQpus},
                             {"numRanks", numRanks},
                             {"num-pool-elements", pool.size()},
                             {"num-elements-per-rank", end - start}};
  if (rank == 0)
    cudaq::info("[adapt] init info: {}", initInfo.dump(4));

  // We'll need to know the local to global index map
  std::vector<std::size_t> localToGlobalMap(end - start);
  for (int i = 0; i < end - start; i++)
    localToGlobalMap[i] = start + i;

  // Start of with the initial |psi_n>
  cudaq::state state = get_state(adapt_kernel, numQubits, initialState, thetas,
                                 coefficients, pauliWords);
  std::size_t count = 0;
  while (true) {

    // Step 1 - compute <psi|[H,Oi]|psi> vector
    std::vector<double> gradients;
    double gradNorm = 0.0;
    std::vector<async_observe_result> resultHandles;
    for (std::size_t i = 0, qpuCounter = 0; i < commutators.size(); i++) {
      if (rank == 0)
        cudaq::info("Compute commutator {}", i);
      if (qpuCounter % numQpus == 0)
        qpuCounter = 0;

      resultHandles.emplace_back(
          observe_async(qpuCounter++, prepare_state, commutators[i], state));
    }

    std::vector<observe_result> results;
    for (auto &handle : resultHandles)
      results.emplace_back(handle.get());

    // Get the gradient results
    std::transform(results.begin(), results.end(),
                   std::back_inserter(gradients),
                   [](auto &&el) { return std::fabs(el.expectation()); });

    // Compute the local gradient norm
    double norm = 0.0;
    for (auto &g : gradients)
      norm += g * g;

    // All ranks have a norm, need to reduce that across all
    if (mpi::is_initialized())
      norm = cudaq::mpi::all_reduce(norm, std::plus<double>());

    // All ranks have a max gradient and index
    auto iter = std::max_element(gradients.begin(), gradients.end());
    double maxGrad = *iter;
    auto maxOpIdx = std::distance(gradients.begin(), iter);
    if (mpi::is_initialized()) {
      std::vector<int> allMaxOpIndices(numRanks);
      std::vector<double> allMaxGrads(numRanks);
      // Distribute the max gradient from this rank to others
      cudaq::mpi::all_gather(allMaxGrads, {*iter});
      // Distribute the corresponding idx from this rank to others,
      // make sure we map back to global indices
      cudaq::mpi::all_gather(allMaxOpIndices,
                             {static_cast<int>(localToGlobalMap[maxOpIdx])});

      // Everyone has the indices, loop over and pick out the
      // max from all calculations
      std::size_t cachedIdx = 0;
      double cachedGrad = 0.0;
      for (std::size_t i = 0; i < allMaxGrads.size(); i++)
        if (allMaxGrads[i] > cachedGrad) {
          cachedGrad = allMaxGrads[i];
          cachedIdx = allMaxOpIndices[i];
        }

      maxOpIdx = cachedIdx;
    }

    if (rank == 0) {
      cudaq::info("[adapt] index of element with max gradient is {}", maxOpIdx);
      cudaq::info("current norm is {} and last iteration norm is {}", norm,
                  lastNorm);
    }

    // Convergence is reached if gradient values are small
    if (std::sqrt(std::fabs(norm)) < tol || std::fabs(lastNorm - norm) < tol)
      break;

    // Use the operator from the pool
    auto op = pool[maxOpIdx];
    chosenOps.push_back(op);
    thetas.push_back(0.0);
    for (auto o : op) {
      pauliWords.emplace_back(o.to_string(false));
      coefficients.push_back(o.get_coefficient().evaluate().imag());
    }

    optim::optimizable_function objective;
    std::unique_ptr<observe_gradient> defaultGradient;
    // If we don't need gradients, objective is simple
    if (!optimizer.requiresGradients()) {
      objective = [&, thetas, coefficients](const std::vector<double> &x,
                                            std::vector<double> &dx) mutable {
        auto res = cudaq::observe(adapt_kernel, H, numQubits, initialState, x,
                                  coefficients, pauliWords);
        if (options.get("verbose", false))
          printf("<H> = %.12lf\n", res.expectation());
        // data.emplace_back(x, res, observe_execution_type::function);
        // for (auto datum : gradient->data)
        //   data.push_back(datum);

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
                                             coefficients, pauliWords));
          },
          H);
      objective = [&, thetas, coefficients](const std::vector<double> &x,
                                            std::vector<double> &dx) mutable {
        // FIXME get shots in here...
        auto res = cudaq::observe(adapt_kernel, H, numQubits, initialState, x,
                                  coefficients, pauliWords);
        if (options.get("verbose", false))
          printf("<H> = %.12lf\n", res.expectation());
        defaultGradient->compute(x, dx, res.expectation(),
                                 options.get("shots", -1));

        // data.emplace_back(x, res, observe_execution_type::function);
        // for (auto datum : gradient->data)
        //   data.push_back(datum);

        return res.expectation();
      };
    }

    // FIXME fix the const_cast
    auto [groundEnergy, optParams] =
        const_cast<optim::optimizer &>(optimizer).optimize(thetas.size(),
                                                           objective, options);
    // Set the new optimzal parameters
    thetas = optParams;
    energy = groundEnergy;

    // Set the norm for the next iteration's check
    lastNorm = norm;
    state = get_state(adapt_kernel, numQubits, initialState, thetas,
                      coefficients, pauliWords);
  }

  return std::make_tuple(energy, thetas, chosenOps);
}

} // namespace cudaq::solvers::adapt