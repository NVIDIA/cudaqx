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

void checkEqual(cudaq::complex_matrix a, cudaq::complex_matrix b, double tol = 1e-8) {
  // Uncomment the following for debugging
  // std::cout << "matrix a:" << std::endl;
  // a.dump();
  // std::cout << "matrix b:" << std::endl;
  // b.dump();

  ASSERT_EQ(a.get_rank(), b.get_rank());
  ASSERT_EQ(a.rows(), b.rows());
  ASSERT_EQ(a.cols(), b.cols());
  ASSERT_EQ(a.size(), b.size());
  for (std::size_t i = 0; i < a.rows(); i++) {
    for (std::size_t j = 0; j < a.cols(); j++) {
      auto a_val = a[{i, j}];
      auto b_val = b[{i, j}];
      EXPECT_NEAR(a_val.real(), b_val.real(), tol);
      EXPECT_NEAR(a_val.imag(), b_val.imag(), tol);
    }
  }
}

// One- and Two-body integrals were copied from test_molecule.cpp.
// They were further validated using the script ./support/h2_pyscf_hf.py.
//
TEST(BravyiKitaev, testH2Hamiltonian) {
  using double_complex = std::complex<double>;
  using namespace cudaq::spin;

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
      -0.1064770114930045 + 0.04540633286914125 * x(0) * z(1) * x(2) +
      0.04540633286914125 * x(0) * z(1) * x(2) * z(3) +
      0.04540633286914125 * y(0) * z(1) * y(2) +
      0.04540633286914125 * y(0) * z(1) * y(2) * z(3) +
      0.17028010135220506 * z(0) + 0.1702801013522051 * z(0) * z(1) +
      0.16560682358174256 * z(0) * z(1) * z(2) +
      0.16560682358174256 * z(0) * z(1) * z(2) * z(3) +
      0.12020049071260128 * z(0) * z(2) +
      0.12020049071260128 * z(0) * z(2) * z(3) + 0.1683359862516207 * z(1) -
      0.22004130022421792 * z(1) * z(2) * z(3) +
      0.17407289249680227 * z(1) * z(3) - 0.22004130022421792 * z(2);
  checkEqual(gold.to_matrix(), result.to_matrix(), 1.5e-4);
}

TEST(BravyiKitaev, testSRLCase0) {
  using double_complex = std::complex<double>;
  using namespace cudaq::spin;

  auto result = cudaq::solvers::seeley_richard_love(2, 2, 4.0, 20);

  cudaq::spin_op gold = double_complex(-2.0, 0.0) * z(2) +
                        double_complex(2.0, 0.0);

  checkEqual(gold.to_matrix(), result.to_matrix(), 1.5e-4);
}

TEST(BravyiKitaev, testSRLCase1) {
  using double_complex = std::complex<double>;
  using namespace cudaq::spin;

  auto result = cudaq::solvers::seeley_richard_love(2, 6, 4.0, 20);
  cudaq::spin_op gold = double_complex(1.0, 0.0) * z(1) * x(2) * y(3) *
                            z(5) * y(6) +
                        double_complex(-1.0, 0.0) * z(1) * y(2) * y(3) *
                            z(5) * x(6) +
                        double_complex(0.0, -1.0) * z(1) * x(2) * y(3) *
                            z(5) * x(6) +
                        double_complex(0.0, -1.0) * z(1) * y(2) * y(3) *
                            z(5) * y(6);
  auto t1 = gold.to_matrix();
  auto t2 = result.to_matrix();
  checkEqual(gold.to_matrix(), result.to_matrix(), 1.5e-4);
}

TEST(BravyiKitaev, testSRLCase2) {
  using double_complex = std::complex<double>;
  using namespace cudaq::spin;

  auto result = cudaq::solvers::seeley_richard_love(5, 2, 4.0, 20);
  cudaq::spin_op gold =
      double_complex(-1.0, 0.0) * z(1) * y(2) * y(3) * z(4) * x(5) +
      double_complex(0.0, 1.0) * z(1) * x(2) * y(3) * z(4) * x(5) +
      double_complex(1.0, 0.0) * z(1) * x(2) * y(3) * y(5) +
      double_complex(0.0, 1.0) * z(1) * y(2) * y(3) * y(5);

  checkEqual(gold.to_matrix(), result.to_matrix(), 1.5e-4);
}

TEST(BravyiKitaev, testSRLCase3) {
  using double_complex = std::complex<double>;
  using namespace cudaq::spin;

  auto result = cudaq::solvers::seeley_richard_love(1, 2, 4.0, 20);
  cudaq::spin_op gold = double_complex(1.0, 0.0) * z(0) * y(1) * y(2) +
                        double_complex(0.0, -1.0) * z(0) * y(1) * x(2) +
                        double_complex(1.0, 0.0) * i(0) * x(1) * x(2) +
                        double_complex(0.0, 1.0) * i(0) * x(1) * y(2);

  checkEqual(gold.to_matrix(), result.to_matrix(), 1.5e-4);
}

TEST(BravyiKitaev, testSRLCase4) {
  using double_complex = std::complex<double>;
  using namespace cudaq::spin;

  auto result = cudaq::solvers::seeley_richard_love(0, 5, 4.0, 20);
  cudaq::spin_op gold =
      double_complex(-1.0, 0.0) * y(0) * x(1) * y(3) * z(4) * x(5) +
      double_complex(0.0, -1.0) * x(0) * x(1) * y(3) * z(4) * x(5) +
      double_complex(1.0, 0.0) * x(0) * x(1) * y(3) * y(5) +
      double_complex(0.0, -1.0) * y(0) * x(1) * y(3) * y(5);

  checkEqual(gold.to_matrix(), result.to_matrix(), 1.5e-4);
}

TEST(BravyiKitaev, testSRLCase6) {
  using double_complex = std::complex<double>;
  using namespace cudaq::spin;

  auto result = cudaq::solvers::seeley_richard_love(18, 19, 4.0, 20);
  cudaq::spin_op gold = double_complex(1.0, 0.0) * x(18) +
                        double_complex(0.0, -1.0) * y(18) +
                        double_complex(0.0, 1.0) * z(17) * y(18) * z(19) +
                        double_complex(-1.0, 0.0) * z(17) * x(18) * z(19);

  checkEqual(gold.to_matrix(), result.to_matrix(), 1.5e-4);
}

TEST(BravyiKitaev, testSRLCase7) {
  using double_complex = std::complex<double>;
  using namespace cudaq::spin;

  auto result = cudaq::solvers::seeley_richard_love(11, 5, 4.0, 20);
  cudaq::spin_op gold =
      double_complex(0.0, 1.0) * z(3) * z(4) * x(5) * y(7) * z(9) * z(10) *
          x(11) +
      double_complex(-1.0, 0.0) * z(3) * y(5) * y(7) * z(9) * z(10) * x(11) +
      double_complex(1.0, 0.0) * z(3) * z(4) * x(5) * y(7) * y(11) +
      double_complex(0.0, 1.0) * z(3) * y(5) * y(7) * y(11);

  checkEqual(gold.to_matrix(), result.to_matrix(), 1.5e-4);
}

TEST(BravyiKitaev, testSRLCase8) {
  using double_complex = std::complex<double>;
  using namespace cudaq::spin;

  auto result = cudaq::solvers::seeley_richard_love(7, 9, 4.0, 20);
  cudaq::spin_op gold =
      double_complex(0.0, -1.0) * z(3) * z(5) * z(6) * y(7) * z(8) * x(9) *
          x(11) +
      double_complex(1.0, 0.0) * z(3) * z(5) * z(6) * y(7) * y(9) * x(11) +
      double_complex(1.0, 0.0) * x(7) * z(8) * x(9) * x(11) +
      double_complex(0.0, 1.0) * x(7) * y(9) * x(11);

  checkEqual(gold.to_matrix(), result.to_matrix(), 1.5e-4);
}

TEST(BravyiKitaev, testSRLCase9) {
  using double_complex = std::complex<double>;
  using namespace cudaq::spin;

  auto result = cudaq::solvers::seeley_richard_love(9, 15, 4.0, 20);
  cudaq::spin_op gold =
      double_complex(-1.0, 0.0) * y(9) * y(11) * z(13) * z(14) +
      double_complex(0.0, -1.0) * z(8) * x(9) * y(11) * z(13) * z(14) +
      double_complex(-1.0, 0.0) * z(7) * z(8) * x(9) * x(11) * z(15) +
      double_complex(0.0, 1.0) * z(7) * y(9) * x(11) * z(15);

  checkEqual(gold.to_matrix(), result.to_matrix(), 1.5e-4);
}

TEST(BravyiKitaev, testSRLCase10) {
  using double_complex = std::complex<double>;
  using namespace cudaq::spin;

  auto result = cudaq::solvers::seeley_richard_love(3, 7, 4.0, 20);
  cudaq::spin_op gold =
      double_complex(0.0, -1.0) * z(1) * z(2) * y(3) * z(5) * z(6) +
      double_complex(1.0, 0.0) * x(3) * z(5) * z(6) +
      double_complex(-1.0, 0.0) * z(1) * z(2) * x(3) * z(7) +
      double_complex(0.0, 1.0) * y(3) * z(7);

  checkEqual(gold.to_matrix(), result.to_matrix(), 1.5e-4);
}
