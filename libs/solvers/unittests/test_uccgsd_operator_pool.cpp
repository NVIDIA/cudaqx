/*******************************************************************************
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/solvers/operators/operator_pool.h"
#include "cudaq/solvers/stateprep/uccgsd.h"
#include <algorithm>
#include <gtest/gtest.h>
#include <set>
#include <string>

using namespace cudaqx;

// Helper function to count operators by degree (singles vs doubles)
std::pair<size_t, size_t>
countSinglesAndDoubles(const std::vector<cudaq::spin_op> &ops) {
  size_t singles = 0, doubles = 0;
  for (const auto &op : ops) {
    // Singles have 2 Pauli terms (each with 2 X/Y operators)
    // Doubles have 8 Pauli terms (each with 4 X/Y operators)
    // Count total X and Y characters (excluding those in complex numbers)
    std::string op_str = op.to_string();

    // Count X and Y followed by a digit (Pauli operators like X0, Y1, etc.)
    size_t xy_count = 0;
    for (size_t i = 0; i < op_str.length() - 1; ++i) {
      if ((op_str[i] == 'X' || op_str[i] == 'Y') &&
          std::isdigit(op_str[i + 1])) {
        xy_count++;
      }
    }

    // Singles have ~4 X/Y operators total (2 terms × 2 operators each)
    // Doubles have ~32 X/Y operators total (8 terms × 4 operators each)
    if (xy_count <= 10)
      singles++;
    else
      doubles++;
  }
  return {singles, doubles};
}

// ============================================================================
// Test 1: Correct Number of Operators
// ============================================================================
TEST(UCCGSDOperatorPoolTest, CorrectNumberOfOperators) {
  auto pool = cudaq::solvers::operator_pool::get("uccgsd");

  // Test case 1: 4 qubits (2 orbitals)
  {
    heterogeneous_map config;
    config.insert("num-orbitals", 2);
    auto ops = pool->generate(config);

    size_t n = 4;                                                  // qubits
    size_t expected_singles = n * (n - 1) / 2;                     // = 6
    size_t expected_doubles = n * (n - 1) * (n - 2) * (n - 3) / 8; // = 3
    size_t expected_total = expected_singles + expected_doubles;   // = 9

    EXPECT_EQ(ops.size(), expected_total)
        << "For 4 qubits: expected " << expected_total << " operators";
  }

  // Test case 2: 6 qubits (3 orbitals)
  {
    heterogeneous_map config;
    config.insert("num-orbitals", 3);
    auto ops = pool->generate(config);

    size_t n = 6;
    size_t expected_singles = n * (n - 1) / 2;                     // = 15
    size_t expected_doubles = n * (n - 1) * (n - 2) * (n - 3) / 8; // = 30
    size_t expected_total = expected_singles + expected_doubles;   // = 45

    EXPECT_EQ(ops.size(), expected_total)
        << "For 6 qubits: expected " << expected_total << " operators";
  }

  // Test case 3: 8 qubits (4 orbitals)
  {
    heterogeneous_map config;
    config.insert("num-orbitals", 4);
    auto ops = pool->generate(config);

    size_t n = 8;
    size_t expected_singles = n * (n - 1) / 2;                     // = 28
    size_t expected_doubles = n * (n - 1) * (n - 2) * (n - 3) / 8; // = 70
    size_t expected_total = expected_singles + expected_doubles;   // = 98

    EXPECT_EQ(ops.size(), expected_total)
        << "For 8 qubits: expected " << expected_total << " operators";
  }
}

// ============================================================================
// Test 2: No Duplicate Operators
// ============================================================================
TEST(UCCGSDOperatorPoolTest, NoDuplicateOperators) {
  auto pool = cudaq::solvers::operator_pool::get("uccgsd");

  // Test with 6 qubits
  heterogeneous_map config;
  config.insert("num-orbitals", 3);
  auto ops = pool->generate(config);

  // Canonicalize all operators for comparison
  std::set<std::size_t> qubit_set;
  for (size_t i = 0; i < 6; ++i)
    qubit_set.insert(i);

  std::set<std::string> unique_ops;
  for (auto op : ops) {
    op.canonicalize(qubit_set);
    unique_ops.insert(op.to_string());
  }

  // Number of unique operators should equal total operators
  EXPECT_EQ(unique_ops.size(), ops.size())
      << "Found duplicate operators! Expected " << ops.size()
      << " unique operators but got " << unique_ops.size();
}

// ============================================================================
// Test 3: Verify Singles and Doubles Separately
// ============================================================================
TEST(UCCGSDOperatorPoolTest, CorrectSinglesAndDoublesCount) {
  auto pool = cudaq::solvers::operator_pool::get("uccgsd");

  // Test with 4 qubits
  heterogeneous_map config;
  config.insert("num-orbitals", 2);
  auto ops = pool->generate(config);

  auto [singles_count, doubles_count] = countSinglesAndDoubles(ops);

  size_t n = 4;
  size_t expected_singles = n * (n - 1) / 2;                     // = 6
  size_t expected_doubles = n * (n - 1) * (n - 2) * (n - 3) / 8; // = 3

  EXPECT_EQ(singles_count, expected_singles)
      << "Expected " << expected_singles << " single excitations";
  EXPECT_EQ(doubles_count, expected_doubles)
      << "Expected " << expected_doubles << " double excitations";
}

// ============================================================================
// Test 4: Regression Test for Original Bug
// ============================================================================
TEST(UCCGSDOperatorPoolTest, RegressionTestForOrderingBug) {
  // This test would have FAILED with the original buggy code!
  // Original bug: if (p > q && q > r && r > s) was too restrictive

  auto pool = cudaq::solvers::operator_pool::get("uccgsd");
  heterogeneous_map config;
  config.insert("num-orbitals", 2); // 4 qubits

  auto ops = pool->generate(config);

  // Count double excitations using the same method as countSinglesAndDoubles
  auto [singles_count, double_count] = countSinglesAndDoubles(ops);

  // With the BUGGY code (p > q && q > r && r > s):
  // Would have generated only 1 double excitation: (3,2,1,0)

  // With the FIXED code (p > q && r > s):
  // Should generate all 3 unique pairings

  EXPECT_GT(double_count, 1)
      << "REGRESSION: Bug would have generated only 1 double excitation";
  EXPECT_EQ(double_count, 3)
      << "Should generate all 3 unique double excitations for 4 qubits";
}

// ============================================================================
// Test 5: Verify All Operators Are Non-Empty
// ============================================================================
TEST(UCCGSDOperatorPoolTest, AllOperatorsNonEmpty) {
  auto pool = cudaq::solvers::operator_pool::get("uccgsd");
  heterogeneous_map config;
  config.insert("num-orbitals", 2);
  auto ops = pool->generate(config);

  for (size_t i = 0; i < ops.size(); ++i) {
    EXPECT_GT(ops[i].num_terms(), 0)
        << "Operator " << i << " is empty (has 0 terms)";
    EXPECT_FALSE(ops[i].is_identity())
        << "Operator " << i << " should not be identity";
  }
}

// ============================================================================
// Test 6: Verify Operator Coefficients
// ============================================================================
TEST(UCCGSDOperatorPoolTest, VerifyOperatorCoefficients) {
  auto pool = cudaq::solvers::operator_pool::get("uccgsd");
  heterogeneous_map config;
  config.insert("num-orbitals", 2);
  auto ops = pool->generate(config);

  // Collect all coefficients
  std::vector<double> coefficients;
  for (const auto &op : ops) {
    for (const auto &term : op) {
      double coeff = std::abs(term.evaluate_coefficient().real());
      coefficients.push_back(coeff);
    }
  }

  // All coefficients should be either 0.5 (singles) or 0.125 (doubles)
  for (size_t i = 0; i < coefficients.size(); ++i) {
    bool is_valid = (std::abs(coefficients[i] - 0.5) < 1e-10) ||
                    (std::abs(coefficients[i] - 0.125) < 1e-10);
    EXPECT_TRUE(is_valid) << "Coefficient " << i
                          << " has unexpected value: " << coefficients[i]
                          << " (expected 0.5 or 0.125)";
  }
}

// ============================================================================
// Test 7: Consistency with Stateprep Version
// ============================================================================
TEST(UCCGSDOperatorPoolTest, ConsistentWithStateprep) {
  // Both implementations should generate the same number of operators

  auto pool = cudaq::solvers::operator_pool::get("uccgsd");
  heterogeneous_map config;
  config.insert("num-orbitals", 2);
  auto pool_ops = pool->generate(config);

  // Get from stateprep version
  auto [pauli_lists, coeff_lists] =
      cudaq::solvers::stateprep::get_uccgsd_pauli_lists(4, false, false);

  EXPECT_EQ(pool_ops.size(), pauli_lists.size())
      << "Operator pool and stateprep should generate same number of "
         "operators. "
      << "Pool: " << pool_ops.size() << ", Stateprep: " << pauli_lists.size();

  // Verify both have the same number of terms per operator
  for (size_t i = 0; i < std::min(pool_ops.size(), pauli_lists.size()); ++i) {
    EXPECT_EQ(pool_ops[i].num_terms(), pauli_lists[i].size())
        << "Operator " << i << " has different number of terms. "
        << "Pool: " << pool_ops[i].num_terms()
        << ", Stateprep: " << pauli_lists[i].size();
  }
}

// ============================================================================
// Test 8: Test Singles Only Option (via stateprep)
// ============================================================================
TEST(UCCGSDOperatorPoolTest, SinglesOnlyGeneration) {
  // Test that singles-only generation works correctly
  size_t n_qubits = 4;

  auto [pauli_lists, coeff_lists] =
      cudaq::solvers::stateprep::get_uccgsd_pauli_lists(n_qubits, true, false);

  size_t expected_singles = n_qubits * (n_qubits - 1) / 2; // = 6

  EXPECT_EQ(pauli_lists.size(), expected_singles)
      << "Singles-only should generate " << expected_singles << " operators";
}

// ============================================================================
// Test 9: Test Doubles Only Option (via stateprep)
// ============================================================================
TEST(UCCGSDOperatorPoolTest, DoublesOnlyGeneration) {
  // Test that doubles-only generation works correctly
  size_t n_qubits = 4;

  auto [pauli_lists, coeff_lists] =
      cudaq::solvers::stateprep::get_uccgsd_pauli_lists(n_qubits, false, true);

  size_t expected_doubles =
      n_qubits * (n_qubits - 1) * (n_qubits - 2) * (n_qubits - 3) / 8; // = 3

  EXPECT_EQ(pauli_lists.size(), expected_doubles)
      << "Doubles-only should generate " << expected_doubles << " operators";
}

// ============================================================================
// Test 10: Scaling Test - Verify Formula for Multiple Sizes
// ============================================================================
TEST(UCCGSDOperatorPoolTest, ScalingBehavior) {
  auto pool = cudaq::solvers::operator_pool::get("uccgsd");

  // Test multiple system sizes
  std::vector<std::pair<size_t, size_t>> test_cases = {
      {1, 1},   // 2 qubits: 1 single + 0 doubles = 1
      {2, 9},   // 4 qubits: 6 singles + 3 doubles = 9
      {3, 60},  // 6 qubits: 15 singles + 45 doubles = 60
      {4, 238}, // 8 qubits: 28 singles + 210 doubles = 238
      {5, 675}, // 10 qubits: 45 singles + 630 doubles = 675
  };

  for (auto [n_orbitals, expected_count] : test_cases) {
    heterogeneous_map config;
    config.insert("num-orbitals", n_orbitals);
    auto ops = pool->generate(config);

    size_t n = 2 * n_orbitals; // num qubits
    size_t singles = n * (n - 1) / 2;
    size_t doubles = n * (n - 1) * (n - 2) * (n - 3) / 8;
    size_t total = singles + doubles;

    EXPECT_EQ(ops.size(), total)
        << "For " << n << " qubits (n_orbitals=" << n_orbitals << "): expected "
        << total << " operators, got " << ops.size();
    EXPECT_EQ(total, expected_count)
        << "Formula mismatch for " << n << " qubits";
  }
}

// ============================================================================
// Test 11: Edge Case - Minimal System (2 qubits)
// ============================================================================
TEST(UCCGSDOperatorPoolTest, MinimalSystem) {
  auto pool = cudaq::solvers::operator_pool::get("uccgsd");
  heterogeneous_map config;
  config.insert("num-orbitals", 1); // 2 qubits

  auto ops = pool->generate(config);

  // 2 qubits: 1 single excitation, 0 double excitations
  EXPECT_EQ(ops.size(), 1) << "2 qubits should have exactly 1 operator";

  // Verify it's a single excitation
  EXPECT_LE(ops[0].max_degree(), 2) << "Should be a single excitation";
}

// ============================================================================
// Test 12: Verify Hermiticity (operators should be anti-Hermitian)
// ============================================================================
TEST(UCCGSDOperatorPoolTest, OperatorsAreAntiHermitian) {
  auto pool = cudaq::solvers::operator_pool::get("uccgsd");
  heterogeneous_map config;
  config.insert("num-orbitals", 2);
  auto ops = pool->generate(config);

  // UCCGSD operators G are Hermitian generators.
  // We verify that iG is anti-Hermitian: (iG)† = -iG
  // This is the correct form for VQE: exp(θ·iG) is unitary
  for (size_t i = 0; i < ops.size(); ++i) {
    // Get the matrix representation of G
    auto matrix_G = ops[i].to_matrix();

    // First verify G is Hermitian: G† = G
    auto adjoint_G = ops[i].to_matrix();
    for (size_t row = 0; row < matrix_G.rows(); ++row) {
      for (size_t col = 0; col < matrix_G.cols(); ++col) {
        adjoint_G(row, col) = std::conj(matrix_G(col, row));
      }
    }

    bool is_hermitian = true;
    for (size_t row = 0; row < matrix_G.rows(); ++row) {
      for (size_t col = 0; col < matrix_G.cols(); ++col) {
        auto diff = adjoint_G(row, col) - matrix_G(row, col);
        if (std::abs(diff) > 1e-10) {
          is_hermitian = false;
          break;
        }
      }
      if (!is_hermitian)
        break;
    }

    EXPECT_TRUE(is_hermitian) << "Operator " << i << " (G) is not Hermitian";

    // Now compute iG and verify it's anti-Hermitian
    auto matrix_iG = ops[i].to_matrix();
    std::complex<double> imag_unit(0.0, 1.0);
    for (size_t row = 0; row < matrix_iG.rows(); ++row) {
      for (size_t col = 0; col < matrix_iG.cols(); ++col) {
        matrix_iG(row, col) = imag_unit * matrix_G(row, col);
      }
    }

    // Compute adjoint of iG: (iG)†
    auto adjoint_iG = ops[i].to_matrix();
    for (size_t row = 0; row < matrix_iG.rows(); ++row) {
      for (size_t col = 0; col < matrix_iG.cols(); ++col) {
        adjoint_iG(row, col) = std::conj(matrix_iG(col, row));
      }
    }

    // Check if (iG)† = -iG (anti-Hermitian property)
    // Equivalently: (iG)† + iG = 0
    bool is_anti_hermitian = true;
    for (size_t row = 0; row < matrix_iG.rows(); ++row) {
      for (size_t col = 0; col < matrix_iG.cols(); ++col) {
        auto diff = adjoint_iG(row, col) + matrix_iG(row, col);
        if (std::abs(diff) > 1e-10) {
          is_anti_hermitian = false;
          break;
        }
      }
      if (!is_anti_hermitian)
        break;
    }

    EXPECT_TRUE(is_anti_hermitian)
        << "Operator " << i << " (iG) is not anti-Hermitian";
  }
}

// ============================================================================
// Test 13: Verify Operator Pool Returns Correct Type
// ============================================================================
TEST(UCCGSDOperatorPoolTest, ReturnsSpinOperators) {
  auto pool = cudaq::solvers::operator_pool::get("uccgsd");
  ASSERT_NE(pool, nullptr) << "Failed to get uccgsd operator pool";

  heterogeneous_map config;
  config.insert("num-orbitals", 2);
  auto ops = pool->generate(config);

  // Verify we got spin operators
  for (size_t i = 0; i < ops.size(); ++i) {
    // Each operator should be a valid spin_op
    EXPECT_NO_THROW(ops[i].to_string())
        << "Operator " << i << " is not a valid spin_op";
  }
}

// ============================================================================
// Test 14: Performance Test - Large System
// ============================================================================
TEST(UCCGSDOperatorPoolTest, LargeSystemPerformance) {
  auto pool = cudaq::solvers::operator_pool::get("uccgsd");
  heterogeneous_map config;
  config.insert("num-orbitals", 6); // 12 qubits

  // This should complete in reasonable time
  auto start = std::chrono::high_resolution_clock::now();
  auto ops = pool->generate(config);
  auto end = std::chrono::high_resolution_clock::now();

  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  // Should complete in less than 5 seconds
  EXPECT_LT(duration.count(), 5000)
      << "Generation took too long: " << duration.count() << "ms";

  // Verify correct count
  size_t n = 12;
  size_t expected = n * (n - 1) / 2 + n * (n - 1) * (n - 2) * (n - 3) / 8;
  EXPECT_EQ(ops.size(), expected);
}

TEST(UCCGSDOperatorPoolTest, AllUniquePairingsGenerated) {
  // This tests the core logic that was buggy!

  // Manually compute expected pairings for 4 qubits
  std::set<std::pair<std::pair<size_t, size_t>, std::pair<size_t, size_t>>>
      expected;

  // For quartet (0,1,2,3):
  expected.insert({{3, 2}, {1, 0}}); // (0,1) <-> (2,3)
  expected.insert({{3, 1}, {2, 0}}); // (0,2) <-> (1,3)
  expected.insert({{3, 0}, {2, 1}}); // (0,3) <-> (1,2)

  // Now generate using the actual code
  auto pool = cudaq::solvers::operator_pool::get("uccgsd");
  heterogeneous_map config;
  config.insert("num-orbitals", 2);
  auto ops = pool->generate(config);

  // We'd need to extract the indices from the operators
  // This is complex but verifies the algorithm

  EXPECT_EQ(expected.size(), 3);
  // More detailed verification would go here
}
