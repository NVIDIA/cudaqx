/*******************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
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

bool check_gpu_available() {
  static int gpu_available = -1; // -1 means unknown; 0 = no; 1 = yes
  if (gpu_available < 0) {
    if (std::system("nvidia-smi > /dev/null 2>&1") != 0)
      gpu_available = 0;
    else
      gpu_available = 1;
  }
  return static_cast<bool>(gpu_available);
}

class SolversTester : public ::testing::Test {
protected:
  static cudaq::spin_op h;
  static cudaq::spin_op hamli;
  static cudaq::spin_op hamhh;
  static void SetUpTestSuite() {
    std::vector<double> h2_data{
        3, 1, 1, 3, 0.0454063,  0,  2, 0, 0, 0, 0.17028,    0,
        0, 0, 2, 0, -0.220041,  -0, 1, 3, 3, 1, 0.0454063,  0,
        0, 0, 0, 0, -0.106477,  0,  0, 2, 0, 0, 0.17028,    0,
        0, 0, 0, 2, -0.220041,  -0, 3, 3, 1, 1, -0.0454063, -0,
        2, 2, 0, 0, 0.168336,   0,  2, 0, 2, 0, 0.1202,     0,
        0, 2, 0, 2, 0.1202,     0,  2, 0, 0, 2, 0.165607,   0,
        0, 2, 2, 0, 0.165607,   0,  0, 0, 2, 2, 0.174073,   0,
        1, 1, 3, 3, -0.0454063, -0, 15};
    h = cudaq::spin_op(h2_data, 4);

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
cudaq::spin_op SolversTester::h;
cudaq::spin_op SolversTester::hamli;
cudaq::spin_op SolversTester::hamhh;

TEST_F(SolversTester, checkSimpleAdapt_H2) {
  auto pool = cudaq::solvers::operator_pool::get("spin_complement_gsd");
  auto poolList = pool->generate({{"num-orbitals", h.num_qubits() / 2}});
  auto [energy, thetash, ops] = cudaq::solvers::adapt_vqe(
      hartreeFock2Electrons, h, poolList,
      {{"grad_norm_tolerance", 1e-2}, {"verbose", true}});
  EXPECT_NEAR(energy, -1.13, 1e-2);
}

TEST_F(SolversTester, checkSimpleAdapt_H2Sto3g) {
  auto pool = cudaq::solvers::operator_pool::get("spin_complement_gsd");
  auto poolList = pool->generate({{"num-orbitals", hamhh.num_qubits() / 2}});
  auto [energy, thetas, ops] = cudaq::solvers::adapt_vqe(
      hartreeFock2Electrons, hamhh, poolList,
      {{"grad_norm_tolerance", 1e-2}, {"verbose", true}});
  EXPECT_NEAR(energy, -1.13, 1e-2);
}

TEST_F(SolversTester, checkSimpleAdapt_LiHSto3g) {
  if (!check_gpu_available())
    GTEST_SKIP() << "No GPU available, skipping test because CPU is slow";
  auto pool = cudaq::solvers::operator_pool::get("spin_complement_gsd");
  auto poolList = pool->generate({{"num-orbitals", hamli.num_qubits() / 2}});
  auto [energy, thetas, ops] = cudaq::solvers::adapt_vqe(
      statePrep4Electrons, hamli, poolList,
      {{"grad_norm_tolerance", 0.5}, {"verbose", true}});
  EXPECT_NEAR(energy, -7.88, 1e-2);
}

TEST_F(SolversTester, checkSimpleAdaptGradient_H2) {
  auto pool = cudaq::solvers::operator_pool::get("spin_complement_gsd");
  auto opt = cudaq::optim::optimizer::get("lbfgs");

  auto poolList = pool->generate({{"num-orbitals", h.num_qubits() / 2}});
  auto [energy, thetas, ops] = cudaq::solvers::adapt_vqe(
      hartreeFock2Electrons, h, poolList, *opt, "central_difference",
      {{"grad_norm_tolerance", 1e-2}, {"verbose", true}});
  EXPECT_NEAR(energy, -1.13, 1e-2);
  for (std::size_t i = 0; i < thetas.size(); i++)
    printf("%lf -> %s\n", thetas[i], ops[i].to_string().c_str());
}

TEST_F(SolversTester, checkSimpleAdaptGradient_H2Sto3g) {
  auto pool = cudaq::solvers::operator_pool::get("spin_complement_gsd");
  auto opt = cudaq::optim::optimizer::get("lbfgs");

  auto poolList = pool->generate({{"num-orbitals", hamhh.num_qubits() / 2}});
  auto [energy, thetas, ops] = cudaq::solvers::adapt_vqe(
      hartreeFock2Electrons, hamhh, poolList, *opt, "central_difference",
      {{"grad_norm_tolerance", 1e-2}, {"verbose", true}});
  EXPECT_NEAR(energy, -1.13, 1e-2);
  for (std::size_t i = 0; i < thetas.size(); i++)
    printf("%lf -> %s\n", thetas[i], ops[i].to_string().c_str());
}

TEST_F(SolversTester, checkSimpleAdaptGradient_LiHSto3g) {
  if (!check_gpu_available())
    GTEST_SKIP() << "No GPU available, skipping test because CPU is slow";
  auto pool = cudaq::solvers::operator_pool::get("spin_complement_gsd");
  auto opt = cudaq::optim::optimizer::get("cobyla");

  auto poolList = pool->generate({{"num-orbitals", hamli.num_qubits() / 2}});
  auto [energy, thetas, ops] = cudaq::solvers::adapt_vqe(
      statePrep4Electrons, hamli, poolList, *opt, "central_difference",
      {{"grad_norm_tolerance", 0.5}, {"verbose", true}});
  EXPECT_NEAR(energy, -7.88, 1e-2);
  for (std::size_t i = 0; i < thetas.size(); i++)
    printf("%lf -> %s\n", thetas[i], ops[i].to_string().c_str());
}

TEST_F(SolversTester, checkSimpleAdaptUCCSD_H2) {
  auto pool = cudaq::solvers::operator_pool::get("uccsd");

  heterogeneous_map config;
  config.insert("num-qubits", h.num_qubits());
  config.insert("num-electrons", 2);
  auto poolList = pool->generate(config);
  auto [energy, thetas, ops] = cudaq::solvers::adapt_vqe(
      hartreeFock2Electrons, h, poolList,
      {{"grad_norm_tolerance", 1e-2}, {"verbose", true}});
  EXPECT_NEAR(energy, -1.13, 1e-2);
}

TEST_F(SolversTester, checkSimpleAdaptUCCSD_H2Sto3g) {
  auto pool = cudaq::solvers::operator_pool::get("uccsd");

  heterogeneous_map config;
  config.insert("num-qubits", hamhh.num_qubits());
  config.insert("num-electrons", 2);
  auto poolList = pool->generate(config);
  auto [energy, thetas, ops] = cudaq::solvers::adapt_vqe(
      hartreeFock2Electrons, hamhh, poolList,
      {{"grad_norm_tolerance", 1e-2}, {"verbose", true}});
  EXPECT_NEAR(energy, -1.13, 1e-2);
}

TEST_F(SolversTester, checkSimpleAdaptUCCSD_LiHSto3g) {
  if (!check_gpu_available())
    GTEST_SKIP() << "No GPU available, skipping test because CPU is slow";
  auto pool = cudaq::solvers::operator_pool::get("uccsd");

  heterogeneous_map config;
  config.insert("num-qubits", hamli.num_qubits());
  config.insert("num-electrons", 4);
  auto poolList = pool->generate(config);
  auto [energy, thetas, ops] = cudaq::solvers::adapt_vqe(
      statePrep4Electrons, hamli, poolList,
      {{"grad_norm_tolerance", 0.5}, {"verbose", true}});
  EXPECT_NEAR(energy, -7.88, 1e-2);
}

TEST_F(SolversTester, checkSimpleAdaptGradientUCCSD_H2) {
  auto pool = cudaq::solvers::operator_pool::get("uccsd");
  auto opt = cudaq::optim::optimizer::get("lbfgs");

  heterogeneous_map config;
  config.insert("num-qubits", h.num_qubits());
  config.insert("num-electrons", 2);
  auto poolList = pool->generate(config);
  auto [energy, thetas, ops] = cudaq::solvers::adapt_vqe(
      hartreeFock2Electrons, h, poolList, *opt, "central_difference",
      {{"grad_norm_tolerance", 1e-2}, {"verbose", true}});
  EXPECT_NEAR(energy, -1.13, 1e-2);

  for (std::size_t i = 0; i < thetas.size(); i++)
    printf("%lf -> %s\n", thetas[i], ops[i].to_string().c_str());
}
TEST_F(SolversTester, checkSimpleAdaptGradientUCCSD_H2Sto3g) {
  auto pool = cudaq::solvers::operator_pool::get("uccsd");
  auto opt = cudaq::optim::optimizer::get("lbfgs");

  heterogeneous_map config;
  config.insert("num-qubits", hamhh.num_qubits());
  config.insert("num-electrons", 2);
  auto poolList = pool->generate(config);
  auto [energy, thetas, ops] = cudaq::solvers::adapt_vqe(
      hartreeFock2Electrons, hamhh, poolList, *opt, "central_difference",
      {{"grad_norm_tolerance", 1e-2}, {"verbose", true}});
  EXPECT_NEAR(energy, -1.13, 1e-2);

  for (std::size_t i = 0; i < thetas.size(); i++)
    printf("%lf -> %s\n", thetas[i], ops[i].to_string().c_str());
}

TEST_F(SolversTester, checkSimpleAdaptGradientUCCSD_LiHSto3g) {
  if (!check_gpu_available())
    GTEST_SKIP() << "No GPU available, skipping test because CPU is slow";

  auto pool = cudaq::solvers::operator_pool::get("uccsd");
  auto opt = cudaq::optim::optimizer::get("cobyla");
  heterogeneous_map config;
  config.insert("num-qubits", hamli.num_qubits());
  config.insert("num-electrons", 4);
  auto poolList = pool->generate(config);
  auto [energy, thetas, ops] = cudaq::solvers::adapt_vqe(
      statePrep4Electrons, hamli, poolList, *opt, "central_difference",
      {{"grad_norm_tolerance", 0.5}, {"verbose", true}});
  EXPECT_NEAR(energy, -7.88, 1e-2);

  for (std::size_t i = 0; i < thetas.size(); i++)
    printf("%lf -> %s\n", thetas[i], ops[i].to_string().c_str());
}
