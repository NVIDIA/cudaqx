/*******************************************************************************
 * Copyright (c) 2024 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <cmath>
#include <gtest/gtest.h>

#include "nvqpp/test_kernels.h"

#include "cudaq.h"
#include "cudaq/solvers/vqe.h"

TEST(SolversVQETester, checkAPI) {

  using namespace cudaq::spin;

  cudaq::spin_op h = 5.907 - 2.1433 * x(0) * x(1) - 2.1433 * y(0) * y(1) +
                     .21829 * z(0) - 6.125 * z(1);
  {
    auto cobyla = cudaq::optim::optimizer::get("cobyla");
    auto result =
        cudaq::solvers::vqe(ansatz, h, *cobyla, {0.0}, {{"verbose", true}});
    EXPECT_NEAR(result.energy, -1.748, 1e-3);
  }

  {
    auto lbfgs = cudaq::optim::optimizer::get("lbfgs");
    auto gradient =
        cudaq::observe_gradient::get("central_difference", ansatz, h);
    auto [energy, params, data] =
        cudaq::solvers::vqe(ansatz, h, *lbfgs, *gradient, {0.0});
    EXPECT_NEAR(energy, -1.748, 1e-3);
  }

  {
    // Test how one can handle non-standard kernel signature
    auto optimizer = cudaq::optim::optimizer::get("cobyla");
    constexpr int N = 2;
    auto result = cudaq::solvers::vqe(
        ansatzNonStdSignature, h, *optimizer, {0.0},
        [&](const std::vector<double> &x) { return std::make_tuple(x[0], N); });

    EXPECT_NEAR(result.energy, -1.748, 1e-3);
    EXPECT_TRUE(result.iteration_data.size() > 1);
  }

  {
    // Test how one can handle non-standard kernel signature
    constexpr int N = 2;
    auto translator = [&](const std::vector<double> &x) {
      return std::make_tuple(x[0], N);
    };
    auto optimizer = cudaq::optim::optimizer::get("lbfgs");
    auto gradient = cudaq::observe_gradient::get(
        "central_difference", ansatzNonStdSignature, h, translator);
    // Wrap the kernel in another kernel with the standard signature
    auto result = cudaq::solvers::vqe(ansatzNonStdSignature, h, *optimizer,
                                      *gradient, {0.0}, translator);

    EXPECT_NEAR(result.energy, -1.748, 1e-3);
    EXPECT_TRUE(result.iteration_data.size() > 1);
  }
  // Handle shots-based simulation
  {
    cudaq::set_random_seed(13);
    auto optimizer = cudaq::optim::optimizer::get("cobyla");
    auto result = cudaq::solvers::vqe(
        ansatz, h, *optimizer, std::vector<double>{0.0}, {{"shots", 10000}});
    printf("TEST %lf\n", result.energy);
    result.iteration_data[0].result.dump();
    EXPECT_TRUE(result.energy > -2.0 && result.energy < -1.5);
  }

  // Handle shots-based simulation with gradient
  {
    cudaq::set_random_seed(13);
    auto optimizer = cudaq::optim::optimizer::get("lbfgs");
    auto gradient = cudaq::observe_gradient::get("parameter_shift", ansatz, h);
    auto result = cudaq::solvers::vqe(ansatz, h, *optimizer, *gradient, {0.0},
                                      {{"shots", 10000}});
    printf("TEST %lf\n", result.energy);
    result.iteration_data[0].result.dump();
    for (auto &o : result.iteration_data) {
      printf("Type: %s\n", static_cast<int>(o.type) ? "gradient" : "function");
      o.result.dump();
    }
    EXPECT_TRUE(result.energy > -2.0 && result.energy < -1.5);
  }
}

TEST(SolversVQETester, checkCentralDifference2Params) {
  // VQE with central_difference gradient and 2 parameters.
  using namespace cudaq::spin;
  cudaq::spin_op h = 5.907 - 2.1433 * x(0) * x(1) - 2.1433 * y(0) * y(1) +
                     .21829 * z(0) - 6.125 * z(1);

  auto lbfgs = cudaq::optim::optimizer::get("lbfgs");
  auto gradient =
      cudaq::observe_gradient::get("central_difference", ansatz2Params, h);
  auto [energy, params, data] =
      cudaq::solvers::vqe(ansatz2Params, h, *lbfgs, *gradient, {0.0, 0.0});
  // The H2 ground state energy is -1.748. With correct gradients and a
  // 2-parameter ansatz, L-BFGS should converge to this value.
  EXPECT_NEAR(energy, -1.748, 1e-3);
}

TEST(SolversVQETester, checkParameterShift2Params) {
  // VQE with parameter_shift gradient and 2 parameters.
  using namespace cudaq::spin;
  cudaq::spin_op h = 5.907 - 2.1433 * x(0) * x(1) - 2.1433 * y(0) * y(1) +
                     .21829 * z(0) - 6.125 * z(1);

  auto lbfgs = cudaq::optim::optimizer::get("lbfgs");
  auto gradient =
      cudaq::observe_gradient::get("parameter_shift", ansatz2Params, h);
  auto [energy, params, data] =
      cudaq::solvers::vqe(ansatz2Params, h, *lbfgs, *gradient, {0.0, 0.0});
  // The H2 ground state energy is -1.748. With correct gradients and a
  // 2-parameter ansatz, L-BFGS should converge to this value.
  EXPECT_NEAR(energy, -1.748, 1e-3);
}

TEST(SolversVQETester, checkForwardDifference2Params) {
  // VQE with forward_difference gradient and 2 parameters.
  using namespace cudaq::spin;
  cudaq::spin_op h = 5.907 - 2.1433 * x(0) * x(1) - 2.1433 * y(0) * y(1) +
                     .21829 * z(0) - 6.125 * z(1);

  auto lbfgs = cudaq::optim::optimizer::get("lbfgs");
  auto gradient =
      cudaq::observe_gradient::get("forward_difference", ansatz2Params, h);
  auto [energy, params, data] =
      cudaq::solvers::vqe(ansatz2Params, h, *lbfgs, *gradient, {0.0, 0.0});
  // The H2 ground state energy is -1.748. With correct gradients and a
  // 2-parameter ansatz, L-BFGS should converge to this value.
  EXPECT_NEAR(energy, -1.748, 1e-3);
}

TEST(SolversVQETester, checkCentralDifference3Params) {
  using namespace cudaq::spin;
  cudaq::spin_op h = 5.907 - 2.1433 * x(0) * x(1) - 2.1433 * y(0) * y(1) +
                     .21829 * z(0) - 6.125 * z(1);

  auto lbfgs = cudaq::optim::optimizer::get("lbfgs");
  auto gradient =
      cudaq::observe_gradient::get("central_difference", ansatz3Params, h);
  auto [energy, params, data] = cudaq::solvers::vqe(
      ansatz3Params, h, *lbfgs, *gradient, {0.0, 0.0, 0.0});
  // Same H2 ground state. 3 parameters give the ansatz more freedom, but
  // the optimizer still must find E = -1.748 with correct gradients.
  EXPECT_NEAR(energy, -1.748, 1e-3);
}

TEST(SolversVQETester, checkParameterShift3Params) {
  using namespace cudaq::spin;
  cudaq::spin_op h = 5.907 - 2.1433 * x(0) * x(1) - 2.1433 * y(0) * y(1) +
                     .21829 * z(0) - 6.125 * z(1);

  auto lbfgs = cudaq::optim::optimizer::get("lbfgs");
  auto gradient =
      cudaq::observe_gradient::get("parameter_shift", ansatz3Params, h);
  auto [energy, params, data] = cudaq::solvers::vqe(
      ansatz3Params, h, *lbfgs, *gradient, {0.0, 0.0, 0.0});
  // Same H2 ground state. 3 parameters give the ansatz more freedom, but
  // the optimizer still must find E = -1.748 with correct gradients.
  EXPECT_NEAR(energy, -1.748, 1e-3);
}

TEST(SolversVQETester, checkForwardDifference3Params) {
  using namespace cudaq::spin;
  cudaq::spin_op h = 5.907 - 2.1433 * x(0) * x(1) - 2.1433 * y(0) * y(1) +
                     .21829 * z(0) - 6.125 * z(1);

  auto lbfgs = cudaq::optim::optimizer::get("lbfgs");
  auto gradient =
      cudaq::observe_gradient::get("forward_difference", ansatz3Params, h);
  auto [energy, params, data] = cudaq::solvers::vqe(
      ansatz3Params, h, *lbfgs, *gradient, {0.0, 0.0, 0.0});
  // Same H2 ground state. 3 parameters give the ansatz more freedom, but
  // the optimizer still must find E = -1.748 with correct gradients.
  EXPECT_NEAR(energy, -1.748, 1e-3);
}
