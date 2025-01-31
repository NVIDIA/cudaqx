/*******************************************************************************
 * Copyright (c) 2024 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <cmath>
#include <gtest/gtest.h>

#include "cudaq.h"
#include "nvqpp/test_kernels.h"
#include "cudaq/solvers/adapt.h"

std::vector<double> h2_data{
    3, 1, 1, 3, 0.0454063,  0,  2, 0, 0, 0, 0.17028,    0,
    0, 0, 2, 0, -0.220041,  -0, 1, 3, 3, 1, 0.0454063,  0,
    0, 0, 0, 0, -0.106477,  0,  0, 2, 0, 0, 0.17028,    0,
    0, 0, 0, 2, -0.220041,  -0, 3, 3, 1, 1, -0.0454063, -0,
    2, 2, 0, 0, 0.168336,   0,  2, 0, 2, 0, 0.1202,     0,
    0, 2, 0, 2, 0.1202,     0,  2, 0, 0, 2, 0.165607,   0,
    0, 2, 2, 0, 0.165607,   0,  0, 0, 2, 2, 0.174073,   0,
    1, 1, 3, 3, -0.0454063, -0, 15};

TEST(SolversTester, checkSimpleAdapt) {
  cudaq::spin_op h(h2_data, 4);
  auto pool = cudaq::solvers::operator_pool::get("spin_complement_gsd");
  auto poolList = pool->generate({{"num-orbitals", h.num_qubits() / 2}});
  auto [energy, thetas, ops] = cudaq::solvers::adapt_vqe(
      hartreeFock2Electrons, h, poolList,
      {{"grad_norm_tolerance", 1e-3}, {"verbose", true}});
  EXPECT_NEAR(energy, -1.13, 1e-2);
}

TEST(SolversTester, checkSimpleAdaptGradient) {
  cudaq::spin_op h(h2_data, 4);
  auto pool = cudaq::solvers::operator_pool::get("spin_complement_gsd");
  auto poolList = pool->generate({{"num-orbitals", h.num_qubits() / 2}});
  auto opt = cudaq::optim::optimizer::get("lbfgs");
  auto [energy, thetas, ops] = cudaq::solvers::adapt_vqe(
      hartreeFock2Electrons, h, poolList, *opt, "central_difference",
      {{"grad_norm_tolerance", 1e-3}, {"verbose", true}});
  EXPECT_NEAR(energy, -1.13, 1e-2);

  for (std::size_t i = 0; i < thetas.size(); i++)
    printf("%lf -> %s\n", thetas[i], ops[i].to_string().c_str());
}

TEST(SolversTester, checkSimpleAdaptUCCSD) {
  cudaq::spin_op h(h2_data, 4);
  auto pool = cudaq::solvers::operator_pool::get("uccsd");
  heterogeneous_map config;
  config.insert("num-qubits", h.num_qubits());
  config.insert("num-electrons", 2);
  auto poolList = pool->generate(config);

  auto [energy, thetas, ops] = cudaq::solvers::adapt_vqe(
      hartreeFock2Electrons, h, poolList,
      {{"grad_norm_tolerance", 1e-3}, {"verbose", true}});
  EXPECT_NEAR(energy, -1.13, 1e-2);
}

TEST(SolversTester, checkSimpleAdaptGradientUCCSD) {
  cudaq::spin_op h(h2_data, 4);
  auto pool = cudaq::solvers::operator_pool::get("uccsd");
  heterogeneous_map config;
  config.insert("num-qubits", h.num_qubits());
  config.insert("num-electrons", 2);
  auto poolList = pool->generate(config);

  auto opt = cudaq::optim::optimizer::get("lbfgs");
  auto [energy, thetas, ops] = cudaq::solvers::adapt_vqe(
      hartreeFock2Electrons, h, poolList, *opt, "central_difference",
      {{"grad_norm_tolerance", 1e-3}, {"verbose", true}});
  EXPECT_NEAR(energy, -1.13, 1e-2);

  for (std::size_t i = 0; i < thetas.size(); i++)
    printf("%lf -> %s\n", thetas[i], ops[i].to_string().c_str());
}
