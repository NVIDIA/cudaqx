/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include <algorithm>
#include <complex>
#include <iostream>
#include <iterator>
#include <set>

#include <gtest/gtest.h>

#include "cudaq/solvers/operators/molecule/fermion_compilers/bravyi_kitaev.h"

// One- and Two-body integrals were copied from test_molecule.cpp.
// They were further validated using the script ./support/h2_pyscf_hf.py.
//
TEST(BravyiKitaev, testH2Hamiltonian) {
  using double_complex = std::complex<double>;

  cudaqx::tensor<> hpq({4, 4});
  cudaqx::tensor<> hpqrs({4, 4, 4, 4});

  double h_constant = 0.7080240981000804;
  hpq.at({0, 0}) = -1.2488;
  hpq.at({1, 1}) = -1.2488;
  hpq.at({2, 2}) = -.47967;
  hpq.at({3, 3}) = -.47967;
  hpqrs.at({0, 0, 0, 0}) = 0.3366719725032414;
  hpqrs.at({0, 0, 2, 2}) = 0.0908126657382825;
  hpqrs.at({0, 1, 1, 0}) = 0.3366719725032414;
  hpqrs.at({0, 1, 3, 2}) = 0.0908126657382825;
  hpqrs.at({0, 2, 0, 2}) = 0.09081266573828267;
  hpqrs.at({0, 2, 2, 0}) = 0.33121364716348484;
  hpqrs.at({0, 3, 1, 2}) = 0.09081266573828267;
  hpqrs.at({0, 3, 3, 0}) = 0.33121364716348484;
  hpqrs.at({1, 0, 0, 1}) = 0.3366719725032414;
  hpqrs.at({1, 0, 2, 3}) = 0.0908126657382825;
  hpqrs.at({1, 1, 1, 1}) = 0.3366719725032414;
  hpqrs.at({1, 1, 3, 3}) = 0.0908126657382825;
  hpqrs.at({1, 2, 0, 3}) = 0.09081266573828267;
  hpqrs.at({1, 2, 2, 1}) = 0.33121364716348484;
  hpqrs.at({1, 3, 1, 3}) = 0.09081266573828267;
  hpqrs.at({1, 3, 3, 1}) = 0.33121364716348484;
  hpqrs.at({2, 0, 0, 2}) = 0.3312136471634851;
  hpqrs.at({2, 0, 2, 0}) = 0.09081266573828246;
  hpqrs.at({2, 1, 1, 2}) = 0.3312136471634851;
  hpqrs.at({2, 1, 3, 0}) = 0.09081266573828246;
  hpqrs.at({2, 2, 0, 0}) = 0.09081266573828264;
  hpqrs.at({2, 2, 2, 2}) = 0.34814578499360427;
  hpqrs.at({2, 3, 1, 0}) = 0.09081266573828264;
  hpqrs.at({2, 3, 3, 2}) = 0.34814578499360427;
  hpqrs.at({3, 0, 0, 3}) = 0.3312136471634851;
  hpqrs.at({3, 0, 2, 1}) = 0.09081266573828246;
  hpqrs.at({3, 1, 1, 3}) = 0.3312136471634851;
  hpqrs.at({3, 1, 3, 1}) = 0.09081266573828246;
  hpqrs.at({3, 2, 0, 1}) = 0.09081266573828264;
  hpqrs.at({3, 2, 2, 3}) = 0.34814578499360427;
  hpqrs.at({3, 3, 1, 1}) = 0.09081266573828264;
  hpqrs.at({3, 3, 3, 3}) = 0.34814578499360427;

  cudaq::solvers::bravyi_kitaev transform{};
  cudaq::spin_op result = transform.generate(h_constant, hpq, hpqrs, {});
  cudaq::spin_op gold =
      -0.1064770114930045 * cudaq::spin_op::i(0) + 0.04540633286914125 * cudaq::spin_op::x(0) * cudaq::spin_op::z(1) * cudaq::spin_op::x(2) +
      0.04540633286914125 * cudaq::spin_op::x(0) * cudaq::spin_op::z(1) * cudaq::spin_op::x(2) * cudaq::spin_op::z(3) +
      0.04540633286914125 * cudaq::spin_op::y(0) * cudaq::spin_op::z(1) * cudaq::spin_op::y(2) +
      0.04540633286914125 * cudaq::spin_op::y(0) * cudaq::spin_op::z(1) * cudaq::spin_op::y(2) * cudaq::spin_op::z(3) +
      0.17028010135220506 * cudaq::spin_op::z(0) + 0.1702801013522051 * cudaq::spin_op::z(0) * cudaq::spin_op::z(1) +
      0.16560682358174256 * cudaq::spin_op::z(0) * cudaq::spin_op::z(1) * cudaq::spin_op::z(2) +
      0.16560682358174256 * cudaq::spin_op::z(0) * cudaq::spin_op::z(1) * cudaq::spin_op::z(2) * cudaq::spin_op::z(3) +
      0.12020049071260128 * cudaq::spin_op::z(0) * cudaq::spin_op::z(2) +
      0.12020049071260128 * cudaq::spin_op::z(0) * cudaq::spin_op::z(2) * cudaq::spin_op::z(3) + 0.1683359862516207 * cudaq::spin_op::z(1) -
      0.22004130022421792 * cudaq::spin_op::z(1) * cudaq::spin_op::z(2) * cudaq::spin_op::z(3) +
      0.17407289249680227 * cudaq::spin_op::z(1) * cudaq::spin_op::z(3) - 0.22004130022421792 * cudaq::spin_op::z(2);
  auto [terms, residuals] = (result - gold).get_raw_data();
  for (auto r : residuals)
    EXPECT_NEAR(std::abs(r), 0.0, 1e-4);
}

TEST(BravyiKitaev, testSRLCase0) {
  using double_complex = std::complex<double>;

  auto result = cudaq::solvers::seeley_richard_love(2, 2, 4.0, 20);

  cudaq::spin_op gold = double_complex(-2.0, 0.0) * cudaq::spin_op::i(0) * cudaq::spin_op::i(1) * cudaq::spin_op::z(2) +
                        double_complex(2.0, 0.0) * cudaq::spin_op::i(0) * cudaq::spin_op::i(1) * cudaq::spin_op::i(2);

  auto [terms, residuals] = (result - gold).get_raw_data();
  for (auto r : residuals)
    EXPECT_NEAR(std::abs(r), 0.0, 1e-4);
}

TEST(BravyiKitaev, testSRLCase1) {
  using double_complex = std::complex<double>;

  auto result = cudaq::solvers::seeley_richard_love(2, 6, 4.0, 20);
  cudaq::spin_op gold = double_complex(1.0, 0.0) * cudaq::spin_op::i(0) * cudaq::spin_op::z(1) * cudaq::spin_op::x(2) * cudaq::spin_op::y(3) *
                            cudaq::spin_op::i(4) * cudaq::spin_op::z(5) * cudaq::spin_op::y(6) +
                        double_complex(-1.0, 0.0) * cudaq::spin_op::i(0) * cudaq::spin_op::z(1) * cudaq::spin_op::y(2) * cudaq::spin_op::y(3) *
                            cudaq::spin_op::i(4) * cudaq::spin_op::z(5) * cudaq::spin_op::x(6) +
                        double_complex(0.0, -1.0) * cudaq::spin_op::i(0) * cudaq::spin_op::z(1) * cudaq::spin_op::x(2) * cudaq::spin_op::y(3) *
                            cudaq::spin_op::i(4) * cudaq::spin_op::z(5) * cudaq::spin_op::x(6) +
                        double_complex(0.0, -1.0) * cudaq::spin_op::i(0) * cudaq::spin_op::z(1) * cudaq::spin_op::y(2) * cudaq::spin_op::y(3) *
                            cudaq::spin_op::i(4) * cudaq::spin_op::z(5) * cudaq::spin_op::y(6);
  auto [terms, residuals] = (result - gold).get_raw_data();
  for (auto r : residuals)
    EXPECT_NEAR(std::abs(r), 0.0, 1e-4);
}

TEST(BravyiKitaev, testSRLCase2) {
  using double_complex = std::complex<double>;

  auto result = cudaq::solvers::seeley_richard_love(5, 2, 4.0, 20);
  cudaq::spin_op gold =
      double_complex(-1.0, 0.0) * cudaq::spin_op::z(1) * cudaq::spin_op::y(2) * cudaq::spin_op::y(3) * cudaq::spin_op::z(4) * cudaq::spin_op::x(5) +
      double_complex(0.0, 1.0) * cudaq::spin_op::z(1) * cudaq::spin_op::x(2) * cudaq::spin_op::y(3) * cudaq::spin_op::z(4) * cudaq::spin_op::x(5) +
      double_complex(1.0, 0.0) * cudaq::spin_op::z(1) * cudaq::spin_op::x(2) * cudaq::spin_op::y(3) * cudaq::spin_op::y(5) +
      double_complex(0.0, 1.0) * cudaq::spin_op::z(1) * cudaq::spin_op::y(2) * cudaq::spin_op::y(3) * cudaq::spin_op::y(5);

  auto [terms, residuals] = (result - gold).get_raw_data();
  for (auto r : residuals)
    EXPECT_NEAR(std::abs(r), 0.0, 1e-4);
}

TEST(BravyiKitaev, testSRLCase3) {
  using double_complex = std::complex<double>;

  auto result = cudaq::solvers::seeley_richard_love(1, 2, 4.0, 20);
  cudaq::spin_op gold = double_complex(1.0, 0.0) * cudaq::spin_op::z(0) * cudaq::spin_op::y(1) * cudaq::spin_op::y(2) +
                        double_complex(0.0, -1.0) * cudaq::spin_op::z(0) * cudaq::spin_op::y(1) * cudaq::spin_op::x(2) +
                        double_complex(1.0, 0.0) * cudaq::spin_op::i(0) * cudaq::spin_op::x(1) * cudaq::spin_op::x(2) +
                        double_complex(0.0, 1.0) * cudaq::spin_op::i(0) * cudaq::spin_op::x(1) * cudaq::spin_op::y(2);

  auto [terms, residuals] = (result - gold).get_raw_data();
  for (auto r : residuals)
    EXPECT_NEAR(std::abs(r), 0.0, 1e-4);
}

TEST(BravyiKitaev, testSRLCase4) {
  using double_complex = std::complex<double>;

  auto result = cudaq::solvers::seeley_richard_love(0, 5, 4.0, 20);
  cudaq::spin_op gold =
      double_complex(-1.0, 0.0) * cudaq::spin_op::y(0) * cudaq::spin_op::x(1) * cudaq::spin_op::i(2) * cudaq::spin_op::y(3) * cudaq::spin_op::z(4) * cudaq::spin_op::x(5) +
      double_complex(0.0, -1.0) * cudaq::spin_op::x(0) * cudaq::spin_op::x(1) * cudaq::spin_op::i(2) * cudaq::spin_op::y(3) * cudaq::spin_op::z(4) * cudaq::spin_op::x(5) +
      double_complex(1.0, 0.0) * cudaq::spin_op::x(0) * cudaq::spin_op::x(1) * cudaq::spin_op::i(2) * cudaq::spin_op::y(3) * cudaq::spin_op::i(4) * cudaq::spin_op::y(5) +
      double_complex(0.0, -1.0) * cudaq::spin_op::y(0) * cudaq::spin_op::x(1) * cudaq::spin_op::i(2) * cudaq::spin_op::y(3) * cudaq::spin_op::i(4) * cudaq::spin_op::y(5);

  auto [terms, residuals] = (result - gold).get_raw_data();
  for (auto r : residuals)
    EXPECT_NEAR(std::abs(r), 0.0, 1e-4);
}

TEST(BravyiKitaev, testSRLCase6) {
  using double_complex = std::complex<double>;

  auto result = cudaq::solvers::seeley_richard_love(18, 19, 4.0, 20);
  cudaq::spin_op gold = double_complex(1.0, 0.0) * cudaq::spin_op::x(18) * cudaq::spin_op::i(19) +
                        double_complex(0.0, -1.0) * cudaq::spin_op::y(18) * cudaq::spin_op::i(19) +
                        double_complex(0.0, 1.0) * cudaq::spin_op::z(17) * cudaq::spin_op::y(18) * cudaq::spin_op::z(19) +
                        double_complex(-1.0, 0.0) * cudaq::spin_op::z(17) * cudaq::spin_op::x(18) * cudaq::spin_op::z(19);

  auto [terms, residuals] = (result - gold).get_raw_data();
  for (auto r : residuals)
    EXPECT_NEAR(std::abs(r), 0.0, 1e-4);
}

TEST(BravyiKitaev, testSRLCase7) {
  using double_complex = std::complex<double>;

  auto result = cudaq::solvers::seeley_richard_love(11, 5, 4.0, 20);
  cudaq::spin_op gold =
      double_complex(0.0, 1.0) * cudaq::spin_op::z(3) * cudaq::spin_op::z(4) * cudaq::spin_op::x(5) * cudaq::spin_op::y(7) * cudaq::spin_op::z(9) * cudaq::spin_op::z(10) *
          cudaq::spin_op::x(11) +
      double_complex(-1.0, 0.0) * cudaq::spin_op::z(3) * cudaq::spin_op::y(5) * cudaq::spin_op::y(7) * cudaq::spin_op::z(9) * cudaq::spin_op::z(10) * cudaq::spin_op::x(11) +
      double_complex(1.0, 0.0) * cudaq::spin_op::z(3) * cudaq::spin_op::z(4) * cudaq::spin_op::x(5) * cudaq::spin_op::y(7) * cudaq::spin_op::y(11) +
      double_complex(0.0, 1.0) * cudaq::spin_op::z(3) * cudaq::spin_op::y(5) * cudaq::spin_op::y(7) * cudaq::spin_op::y(11);

  auto [terms, residuals] = (result - gold).get_raw_data();
  for (auto r : residuals)
    EXPECT_NEAR(std::abs(r), 0.0, 1e-4);
}

TEST(BravyiKitaev, testSRLCase8) {
  using double_complex = std::complex<double>;

  auto result = cudaq::solvers::seeley_richard_love(7, 9, 4.0, 20);
  cudaq::spin_op gold =
      double_complex(0.0, -1.0) * cudaq::spin_op::z(3) * cudaq::spin_op::z(5) * cudaq::spin_op::z(6) * cudaq::spin_op::y(7) * cudaq::spin_op::z(8) * cudaq::spin_op::x(9) *
          cudaq::spin_op::x(11) +
      double_complex(1.0, 0.0) * cudaq::spin_op::z(3) * cudaq::spin_op::z(5) * cudaq::spin_op::z(6) * cudaq::spin_op::y(7) * cudaq::spin_op::y(9) * cudaq::spin_op::x(11) +
      double_complex(1.0, 0.0) * cudaq::spin_op::x(7) * cudaq::spin_op::z(8) * cudaq::spin_op::x(9) * cudaq::spin_op::x(11) +
      double_complex(0.0, 1.0) * cudaq::spin_op::x(7) * cudaq::spin_op::y(9) * cudaq::spin_op::x(11);

  auto [terms, residuals] = (result - gold).get_raw_data();
  for (auto r : residuals)
    EXPECT_NEAR(std::abs(r), 0.0, 1e-4);
}

TEST(BravyiKitaev, testSRLCase9) {
  using double_complex = std::complex<double>;

  auto result = cudaq::solvers::seeley_richard_love(9, 15, 4.0, 20);
  cudaq::spin_op gold =
      double_complex(-1.0, 0.0) * cudaq::spin_op::y(9) * cudaq::spin_op::y(11) * cudaq::spin_op::z(13) * cudaq::spin_op::z(14) +
      double_complex(0.0, -1.0) * cudaq::spin_op::z(8) * cudaq::spin_op::x(9) * cudaq::spin_op::y(11) * cudaq::spin_op::z(13) * cudaq::spin_op::z(14) +
      double_complex(-1.0, 0.0) * cudaq::spin_op::z(7) * cudaq::spin_op::z(8) * cudaq::spin_op::x(9) * cudaq::spin_op::x(11) * cudaq::spin_op::z(15) +
      double_complex(0.0, 1.0) * cudaq::spin_op::z(7) * cudaq::spin_op::y(9) * cudaq::spin_op::x(11) * cudaq::spin_op::z(15);

  auto [terms, residuals] = (result - gold).get_raw_data();
  for (auto r : residuals)
    EXPECT_NEAR(std::abs(r), 0.0, 1e-4);
}

TEST(BravyiKitaev, testSRLCase10) {
  using double_complex = std::complex<double>;

  auto result = cudaq::solvers::seeley_richard_love(3, 7, 4.0, 20);
  cudaq::spin_op gold =
      double_complex(0.0, -1.0) * cudaq::spin_op::z(1) * cudaq::spin_op::z(2) * cudaq::spin_op::y(3) * cudaq::spin_op::z(5) * cudaq::spin_op::z(6) +
      double_complex(1.0, 0.0) * cudaq::spin_op::x(3) * cudaq::spin_op::z(5) * cudaq::spin_op::z(6) +
      double_complex(-1.0, 0.0) * cudaq::spin_op::z(1) * cudaq::spin_op::z(2) * cudaq::spin_op::x(3) * cudaq::spin_op::z(7) +
      double_complex(0.0, 1.0) * cudaq::spin_op::y(3) * cudaq::spin_op::z(7);

  auto [terms, residuals] = (result - gold).get_raw_data();
  for (auto r : residuals)
    EXPECT_NEAR(std::abs(r), 0.0, 1e-4);
}
