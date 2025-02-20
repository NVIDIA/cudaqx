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
#include "cudaq/solvers/operators.h"

class SolversTester : public ::testing::Test {
protected:
  cudaq::spin_op h;
  cudaq::spin_op hamli;
  cudaq::spin_op hamhh;
  void SetUp() override {
    std::vector<double> h2_data{
        3, 1, 1, 3, 0.0454063,  0,  2, 0, 0, 0, 0.17028,    0,
        0, 0, 2, 0, -0.220041,  -0, 1, 3, 3, 1, 0.0454063,  0,
        0, 0, 0, 0, -0.106477,  0,  0, 2, 0, 0, 0.17028,    0,
        0, 0, 0, 2, -0.220041,  -0, 3, 3, 1, 1, -0.0454063, -0,
        2, 2, 0, 0, 0.168336,   0,  2, 0, 2, 0, 0.1202,     0,
        0, 2, 0, 2, 0.1202,     0,  2, 0, 0, 2, 0.165607,   0,
        0, 2, 2, 0, 0.165607,   0,  0, 0, 2, 2, 0.174073,   0,
        1, 1, 3, 3, -0.0454063, -0, 15};
    this->h = cudaq::spin_op(h2_data, 4);

    cudaq::solvers::molecular_geometry geometryLiH = {{"Li", {0.3925, 0., 0.}},
                                                      {"H", {-1.1774, 0., 0.}}};
    cudaq::solvers::molecular_geometry geometryHH = {{"H", {0., 0., 0.}},
                                                     {"H", {0., 0., .7474}}};
    auto hh = cudaq::solvers::create_molecule(
        geometryHH, "sto-3g", 0, 0,
        {.casci = true, .ccsd = true, .verbose = true});
    auto lih = cudaq::solvers::create_molecule(
        geometryLiH, "sto-3g", 0, 0,
        {.casci = true, .ccsd = true, .verbose = true});

    hamli = lih.hamiltonian;
    hamhh = hh.hamiltonian;
  }
};

TEST_F(SolversTester, checkSimpleAdapt) {
  auto pool = cudaq::solvers::operator_pool::get("spin_complement_gsd");

  auto poolList = pool->generate({{"num-orbitals", h.num_qubits() / 2}});
  auto [energy1, thetash1, ops1] = cudaq::solvers::adapt_vqe(
      hartreeFock2Electrons, h, poolList,
      {{"grad_norm_tolerance", 1e-2}, {"verbose", true}});
  EXPECT_NEAR(energy1, -1.13, 1e-2);

  poolList = pool->generate({{"num-orbitals", hamhh.num_qubits() / 2}});
  auto [energy3, thetas3, ops3] = cudaq::solvers::adapt_vqe(
      hartreeFock2Electrons, hamhh, poolList,
      {{"grad_norm_tolerance", 1e-2}, {"verbose", true}});
  EXPECT_NEAR(energy3, -1.13, 1e-2);

  poolList = pool->generate({{"num-orbitals", hamli.num_qubits() / 2}});
  auto [energy2, thetas2, ops2] = cudaq::solvers::adapt_vqe(
      statePrep4Electrons, hamli, poolList,
      {{"grad_norm_tolerance", 0.5}, {"verbose", true}});
  EXPECT_NEAR(energy2, -7.88, 1e-2);
}

TEST_F(SolversTester, checkSimpleAdaptGradient) {
  auto pool = cudaq::solvers::operator_pool::get("spin_complement_gsd");
  auto opt = cudaq::optim::optimizer::get("lbfgs");

  auto poolList = pool->generate({{"num-orbitals", h.num_qubits() / 2}});
  auto [energy1, thetas1, ops1] = cudaq::solvers::adapt_vqe(
      hartreeFock2Electrons, h, poolList, *opt, "central_difference",
      {{"grad_norm_tolerance", 1e-2}, {"verbose", true}});
  EXPECT_NEAR(energy1, -1.13, 1e-2);
  for (std::size_t i = 0; i < thetas1.size(); i++)
    printf("%lf -> %s\n", thetas1[i], ops1[i].to_string().c_str());

  poolList = pool->generate({{"num-orbitals", hamhh.num_qubits() / 2}});
  auto [energy3, thetas3, ops3] = cudaq::solvers::adapt_vqe(
      hartreeFock2Electrons, hamhh, poolList, *opt, "central_difference",
      {{"grad_norm_tolerance", 1e-2}, {"verbose", true}});
  EXPECT_NEAR(energy3, -1.13, 1e-2);
  for (std::size_t i = 0; i < thetas3.size(); i++)
    printf("%lf -> %s\n", thetas3[i], ops3[i].to_string().c_str());

  poolList = pool->generate({{"num-orbitals", hamli.num_qubits() / 2}});
  auto [energy2, thetas2, ops2] = cudaq::solvers::adapt_vqe(
      statePrep4Electrons, hamli, poolList, *opt, "central_difference",
      {{"grad_norm_tolerance", 0.5}, {"verbose", true}});
  EXPECT_NEAR(energy2, -7.88, 1e-2);
  for (std::size_t i = 0; i < thetas2.size(); i++)
    printf("%lf -> %s\n", thetas2[i], ops2[i].to_string().c_str());
}

TEST_F(SolversTester, checkSimpleAdaptUCCSD) {
  auto pool = cudaq::solvers::operator_pool::get("uccsd");

  heterogeneous_map config1;
  config1.insert("num-qubits", h.num_qubits());
  config1.insert("num-electrons", 2);
  auto poolList = pool->generate(config1);
  auto [energy1, thetas1, ops1] = cudaq::solvers::adapt_vqe(
      hartreeFock2Electrons, h, poolList,
      {{"grad_norm_tolerance", 1e-2}, {"verbose", true}});
  EXPECT_NEAR(energy1, -1.13, 1e-2);

  heterogeneous_map config3;
  config3.insert("num-qubits", hamhh.num_qubits());
  config3.insert("num-electrons", 2);
  poolList = pool->generate(config3);
  auto [energy3, thetas3, ops3] = cudaq::solvers::adapt_vqe(
      hartreeFock2Electrons, hamhh, poolList,
      {{"grad_norm_tolerance", 1e-2}, {"verbose", true}});
  EXPECT_NEAR(energy3, -1.13, 1e-2);

  heterogeneous_map config2;
  config2.insert("num-qubits", hamli.num_qubits());
  config2.insert("num-electrons", 4);
  poolList = pool->generate(config2);
  auto [energy2, thetas2, ops2] = cudaq::solvers::adapt_vqe(
      statePrep4Electrons, hamli, poolList,
      {{"grad_norm_tolerance", 0.5}, {"verbose", true}});
  EXPECT_NEAR(energy2, -7.88, 1e-2);
}

TEST_F(SolversTester, checkSimpleAdaptGradientUCCSD) {
  auto pool = cudaq::solvers::operator_pool::get("uccsd");
  auto opt = cudaq::optim::optimizer::get("lbfgs");

  heterogeneous_map config1;
  config1.insert("num-qubits", h.num_qubits());
  config1.insert("num-electrons", 2);
  auto poolList = pool->generate(config1);
  auto [energy, thetas, ops] = cudaq::solvers::adapt_vqe(
      hartreeFock2Electrons, h, poolList, *opt, "central_difference",
      {{"grad_norm_tolerance", 1e-2}, {"verbose", true}});
  EXPECT_NEAR(energy, -1.13, 1e-2);

  for (std::size_t i = 0; i < thetas.size(); i++)
    printf("%lf -> %s\n", thetas[i], ops[i].to_string().c_str());

  heterogeneous_map config3;
  config3.insert("num-qubits", hamhh.num_qubits());
  config3.insert("num-electrons", 2);
  poolList = pool->generate(config3);
  auto [energy3, thetas3, ops3] = cudaq::solvers::adapt_vqe(
      hartreeFock2Electrons, hamhh, poolList, *opt, "central_difference",
      {{"grad_norm_tolerance", 1e-2}, {"verbose", true}});
  EXPECT_NEAR(energy3, -1.13, 1e-2);

  for (std::size_t i = 0; i < thetas3.size(); i++)
    printf("%lf -> %s\n", thetas3[i], ops3[i].to_string().c_str());

  heterogeneous_map config2;
  config2.insert("num-qubits", hamli.num_qubits());
  config2.insert("num-electrons", 4);
  poolList = pool->generate(config2);
  auto [energy2, thetas2, ops2] = cudaq::solvers::adapt_vqe(
      statePrep4Electrons, hamli, poolList, *opt, "central_difference",
      {{"grad_norm_tolerance", 0.5}, {"verbose", true}});
  EXPECT_NEAR(energy2, -7.88, 1e-2);

  for (std::size_t i = 0; i < thetas2.size(); i++)
    printf("%lf -> %s\n", thetas2[i], ops2[i].to_string().c_str());
}
