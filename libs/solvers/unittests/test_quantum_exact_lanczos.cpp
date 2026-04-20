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
#include "cudaq/solvers/quantum_exact_lanczos.h"

using namespace cudaqx;

TEST(QuantumExactLanczos, checkH2_Molecule) {
  using namespace cudaq::spin;
  using namespace cudaq::solvers;

  // H2 Hamiltonian (STO-3G, simplified)
  cudaq::spin_op h2 = -1.0523732 + 0.39793742 * z(0) - 0.39793742 * z(1) -
                      0.01128010 * z(2) + 0.01128010 * z(3) +
                      0.18093120 * x(0) * x(1) * y(2) * y(3);

  heterogeneous_map options;
  options.insert("krylov_dim", 5);
  options.insert("verbose", false);

  // Run QEL
  auto result = quantum_exact_lanczos(h2, 4, 2, options);

  // Check that matrices were built
  EXPECT_EQ(result.hamiltonian_matrix.size(), 25); // 5x5
  EXPECT_EQ(result.overlap_matrix.size(), 25);     // 5x5

  // Check metadata
  EXPECT_EQ(result.krylov_dimension, 5);
  EXPECT_EQ(result.num_system, 4);
  EXPECT_EQ(result.num_ancilla, 3);     // log2(6 terms) = 3
  EXPECT_EQ(result.moments.size(), 10); // 2 * krylov_dim

  // Check normalization is positive
  EXPECT_GT(result.normalization, 0.0);
}

TEST(QuantumExactLanczos, checkSimpleXYZ) {
  using namespace cudaq::spin;
  using namespace cudaq::solvers;

  // Simple 2-qubit Hamiltonian: H = 0.5*X0 + 0.3*Y1 - 0.2*Z0Z1
  cudaq::spin_op h = 0.5 * x(0) + 0.3 * y(1) - 0.2 * z(0) * z(1);

  heterogeneous_map options;
  options.insert("krylov_dim", 3);
  options.insert("verbose", false);

  // Run QEL
  auto result = quantum_exact_lanczos(h, 2, 0, options);

  // Check matrices were built
  EXPECT_EQ(result.hamiltonian_matrix.size(), 9); // 3x3
  EXPECT_EQ(result.overlap_matrix.size(), 9);     // 3x3

  // Check normalization (should be |0.5| + |0.3| + |-0.2| = 1.0)
  EXPECT_NEAR(result.normalization, 1.0, 1e-10);
}

TEST(QuantumExactLanczos, checkMetadata) {
  using namespace cudaq::spin;
  using namespace cudaq::solvers;

  cudaq::spin_op h = 0.7 * z(0) + 0.3 * x(0);

  heterogeneous_map options;
  options.insert("krylov_dim", 4);

  auto result = quantum_exact_lanczos(h, 1, 0, options);

  // Check metadata
  EXPECT_EQ(result.krylov_dimension, 4);
  EXPECT_EQ(result.num_system, 1);
  EXPECT_EQ(result.num_ancilla, 1); // 2 terms -> 1 ancilla
  EXPECT_GT(result.normalization, 0.0);
  EXPECT_EQ(result.moments.size(), 8);             // 2 * krylov_dim
  EXPECT_EQ(result.hamiltonian_matrix.size(), 16); // 4x4
  EXPECT_EQ(result.overlap_matrix.size(), 16);     // 4x4
}
