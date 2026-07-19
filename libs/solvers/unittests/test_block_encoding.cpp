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
#include "cudaq/solvers/operators/block_encoding.h"

TEST(BlockEncodingTester, checkPauliLCU_H2) {
  using namespace cudaq::spin;
  using namespace cudaq::solvers;

  // H2 Hamiltonian (simplified, 4 qubits)
  cudaq::spin_op h2 = -1.0523732 + 0.39793742 * z(0) - 0.39793742 * z(1) -
                      0.01128010 * z(2) + 0.01128010 * z(3) +
                      0.18093120 * x(0) * x(1) * y(2) * y(3);

  // Create block encoding
  pauli_lcu encoding(h2, 4);

  // Check basic properties
  EXPECT_GT(encoding.num_ancilla(), 0);
  EXPECT_EQ(encoding.num_system(), 4);
  EXPECT_GT(encoding.normalization(), 0.0);

  // Check that we have the right number of ancilla qubits
  // log2(6 terms) = 3 ancilla qubits needed
  EXPECT_EQ(encoding.num_ancilla(), 3);

  // Check that normalization is approximately the 1-norm
  // The actual terms included may vary based on how spin_op handles constants
  // and identity terms. The normalization should be positive and reasonable.
  EXPECT_GT(encoding.normalization(), 2.0);
  EXPECT_LT(encoding.normalization(), 2.1);

  // Check that angles are computed (2^3 - 1 = 7 angles for 3-qubit tree)
  EXPECT_EQ(encoding.get_angles().size(), 7);

  // Check term data structures have correct sizes
  EXPECT_GT(encoding.get_term_controls().size(), 0);
  EXPECT_GT(encoding.get_term_lengths().size(), 0);
  EXPECT_EQ(encoding.get_term_signs().size(),
            encoding.get_term_lengths().size());
}

TEST(BlockEncodingTester, checkPauliLCU_SimpleXYZ) {
  using namespace cudaq::spin;
  using namespace cudaq::solvers;

  // Simple 2-qubit Hamiltonian: H = 0.5*X0 + 0.3*Y1 - 0.2*Z0Z1
  cudaq::spin_op h = 0.5 * x(0) + 0.3 * y(1) - 0.2 * z(0) * z(1);

  // Create block encoding
  pauli_lcu encoding(h, 2);

  // Check properties
  EXPECT_EQ(encoding.num_system(), 2);
  EXPECT_EQ(encoding.num_ancilla(), 2); // log2(3) = 2 ancilla

  // Normalization should be |0.5| + |0.3| + |-0.2| = 1.0
  EXPECT_NEAR(encoding.normalization(), 1.0, 1e-10);

  // Should have 3 terms with correct signs
  EXPECT_EQ(encoding.get_term_signs().size(), 3);
  EXPECT_EQ(encoding.get_term_lengths().size(), 3);

  // Each term should have the right number of Paulis
  auto lengths = encoding.get_term_lengths();
  EXPECT_EQ(lengths[0], 1); // X0 has 1 Pauli
  EXPECT_EQ(lengths[1], 1); // Y1 has 1 Pauli
  EXPECT_EQ(lengths[2], 2); // Z0Z1 has 2 Paulis
}

TEST(BlockEncodingTester, checkKernelExecution) {
  using namespace cudaq::spin;
  using namespace cudaq::solvers;

  // Simple Hamiltonian
  cudaq::spin_op h = 0.5 * x(0) + 0.3 * z(0);

  // Create block encoding
  pauli_lcu encoding(h, 1);

  // Test kernel: Apply PREPARE on ancilla qubits
  auto prepare_test = [&]() __qpu__ {
    cudaq::qvector<> anc(encoding.num_ancilla());
    encoding.prepare(anc);
    // The state should be prepared now
    // In a real test, we'd measure and check probabilities
  };

  // This should compile and run without error
  EXPECT_NO_THROW(prepare_test());

  // Test SELECT kernel
  auto select_test = [&]() __qpu__ {
    cudaq::qvector<> anc(encoding.num_ancilla());
    cudaq::qvector<> sys(encoding.num_system());
    encoding.select(anc, sys);
  };

  EXPECT_NO_THROW(select_test());

  // Test full block encoding
  auto full_test = [&]() __qpu__ {
    cudaq::qvector<> anc(encoding.num_ancilla());
    cudaq::qvector<> sys(encoding.num_system());
    encoding.apply(anc, sys);
  };

  EXPECT_NO_THROW(full_test());
}

TEST(BlockEncodingTester, checkIdentityTerm) {
  using namespace cudaq::spin;
  using namespace cudaq::solvers;

  // Hamiltonian with constant term
  cudaq::spin_op h = 2.0 * i(0) + 0.5 * x(0);

  // Create block encoding - should handle identity term
  pauli_lcu encoding(h, 1);

  EXPECT_EQ(encoding.num_system(), 1);
  EXPECT_NEAR(encoding.normalization(), 2.5, 1e-10);
}

TEST(BlockEncodingTester, checkLargeHamiltonian) {
  using namespace cudaq::spin;
  using namespace cudaq::solvers;

  // Build a larger Hamiltonian (8 qubits, multiple terms)
  cudaq::spin_op h;
  for (int i = 0; i < 7; ++i) {
    h += 0.1 * z(i) * z(i + 1);
  }
  for (int i = 0; i < 8; ++i) {
    h += 0.05 * x(i);
  }

  // Create block encoding
  pauli_lcu encoding(h, 8);

  // Should need ceil(log2(15)) = 4 ancilla qubits
  EXPECT_EQ(encoding.num_ancilla(), 4);
  EXPECT_EQ(encoding.num_system(), 8);

  // Normalization should be sum of absolute coefficients
  // 7 * 0.1 + 8 * 0.05 = 0.7 + 0.4 = 1.1
  EXPECT_NEAR(encoding.normalization(), 1.1, 1e-10);
}
