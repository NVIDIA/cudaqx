/*******************************************************************************
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/solvers/operators/operator_pool.h"
#include "cudaq/solvers/stateprep/upccgsd.h"
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
    // Heuristic: count X/Y in the string representation.
    std::string op_str = op.to_string();

    size_t xy_count = 0;
    for (size_t i = 0; i + 1 < op_str.length(); ++i) {
      if ((op_str[i] == 'X' || op_str[i] == 'Y') &&
          std::isdigit(op_str[i + 1])) {
        xy_count++;
      }
    }

    // Same threshold logic as before: singles are "small", doubles "large".
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
TEST(UPCCGSDOperatorPoolTest, CorrectNumberOfOperators) {
  auto pool = cudaq::solvers::operator_pool::get("upccgsd");

  // Recall: M = numOrbitals, n_qubits = 2M
  // singles = M (M - 1)
  // doubles = M (M - 1) / 2
  // total = 3/2 M (M - 1)

  // Test case 1: 2 orbitals (4 qubits)
  {
    heterogeneous_map config;
    config.insert("num-orbitals", 2);
    auto ops = pool->generate(config);

    std::size_t M = 2;
    std::size_t expected_singles = M * (M - 1);        // 2
    std::size_t expected_doubles = M * (M - 1) / 2;    // 1
    std::size_t expected_total   = expected_singles + expected_doubles; // 3

    EXPECT_EQ(ops.size(), expected_total)
        << "For 2 orbitals: expected " << expected_total << " operators";
  }

  // Test case 2: 3 orbitals (6 qubits)
  {
    heterogeneous_map config;
    config.insert("num-orbitals", 3);
    auto ops = pool->generate(config);

    std::size_t M = 3;
    std::size_t expected_singles = M * (M - 1);        // 6
    std::size_t expected_doubles = M * (M - 1) / 2;    // 3
    std::size_t expected_total   = expected_singles + expected_doubles; // 9

    EXPECT_EQ(ops.size(), expected_total)
        << "For 3 orbitals: expected " << expected_total << " operators";
  }

  // Test case 3: 4 orbitals (8 qubits)
  {
    heterogeneous_map config;
    config.insert("num-orbitals", 4);
    auto ops = pool->generate(config);

    std::size_t M = 4;
    std::size_t expected_singles = M * (M - 1);        // 12
    std::size_t expected_doubles = M * (M - 1) / 2;    // 6
    std::size_t expected_total   = expected_singles + expected_doubles; // 18

    EXPECT_EQ(ops.size(), expected_total)
        << "For 4 orbitals: expected " << expected_total << " operators";
  }
}

// ============================================================================
// Test 2: No Duplicate Operators
// ============================================================================
TEST(UPCCGSDOperatorPoolTest, NoDuplicateOperators) {
  auto pool = cudaq::solvers::operator_pool::get("upccgsd");

  heterogeneous_map config;
  config.insert("num-orbitals", 3); // 6 qubits
  auto ops = pool->generate(config);

  std::set<std::size_t> qubit_set;
  for (size_t i = 0; i < 6; ++i)
    qubit_set.insert(i);

  std::set<std::string> unique_ops;
  for (auto op : ops) {
    op.canonicalize(qubit_set);
    unique_ops.insert(op.to_string());
  }

  EXPECT_EQ(unique_ops.size(), ops.size())
      << "Found duplicate operators! Expected " << ops.size()
      << " unique operators but got " << unique_ops.size();
}

// ============================================================================
// Test 3: Verify Singles and Doubles Separately
// ============================================================================
TEST(UPCCGSDOperatorPoolTest, CorrectSinglesAndDoublesCount) {
  auto pool = cudaq::solvers::operator_pool::get("upccgsd");

  heterogeneous_map config;
  config.insert("num-orbitals", 3); // 6 qubits
  auto ops = pool->generate(config);

  auto [singles_count, doubles_count] = countSinglesAndDoubles(ops);

  std::size_t M = 3;
  std::size_t expected_singles = M * (M - 1);        // 6
  std::size_t expected_doubles = M * (M - 1) / 2;    // 3

  EXPECT_EQ(singles_count, expected_singles)
      << "Expected " << expected_singles << " single excitations";
  EXPECT_EQ(doubles_count, expected_doubles)
      << "Expected " << expected_doubles << " double excitations";
}

// ============================================================================
// Test 4: Verify All Operators Are Non-Empty
// ============================================================================
TEST(UPCCGSDOperatorPoolTest, AllOperatorsNonEmpty) {
  auto pool = cudaq::solvers::operator_pool::get("upccgsd");
  heterogeneous_map config;
  config.insert("num-orbitals", 2);
  auto ops = pool->generate(config);

  for (size_t i = 0; i < ops.size(); ++i) {
    EXPECT_GT(ops[i].num_terms(), 0)
        << "Operator " << i << " is empty (has 0 terms)";
    for (const auto &term : ops[i])
      EXPECT_FALSE(term.is_identity())
          << "Term " << term.to_string() << " is identity";
  }
}

// ============================================================================
// Test 5: Verify Operator Coefficients
// (same coefficients as UCCGSD, just fewer operators)
// ============================================================================
TEST(UPCCGSDOperatorPoolTest, VerifyOperatorCoefficients) {
  auto pool = cudaq::solvers::operator_pool::get("upccgsd");
  heterogeneous_map config;
  config.insert("num-orbitals", 2);
  auto ops = pool->generate(config);

  std::vector<double> coefficients;
  for (const auto &op : ops) {
    for (const auto &term : op) {
      double coeff = std::abs(term.evaluate_coefficient().real());
      coefficients.push_back(coeff);
    }
  }

  for (size_t i = 0; i < coefficients.size(); ++i) {
    bool is_valid = (std::abs(coefficients[i] - 0.5) < 1e-10) ||
                    (std::abs(coefficients[i] - 0.125) < 1e-10);
    EXPECT_TRUE(is_valid) << "Coefficient " << i
                          << " has unexpected value: " << coefficients[i]
                          << " (expected 0.5 or 0.125)";
  }
}

// ============================================================================
// Test 6: Consistency with Stateprep Version
// ============================================================================
TEST(UPCCGSDOperatorPoolTest, ConsistentWithStateprep) {
  auto pool = cudaq::solvers::operator_pool::get("upccgsd");
  heterogeneous_map config;
  config.insert("num-orbitals", 2); // 4 qubits
  auto pool_ops = pool->generate(config);

  // stateprep uses "norbitals" = #spin orbitals = #qubits
  std::size_t n_qubits = 4;
  auto [pauli_lists, coeff_lists] =
      cudaq::solvers::stateprep::get_upccgsd_pauli_lists(n_qubits, false, false);

  EXPECT_EQ(pool_ops.size(), pauli_lists.size())
      << "Operator pool and stateprep should generate same number of "
         "operators. "
      << "Pool: " << pool_ops.size() << ", Stateprep: " << pauli_lists.size();

  for (size_t i = 0; i < std::min(pool_ops.size(), pauli_lists.size()); ++i) {
    EXPECT_EQ(pool_ops[i].num_terms(), pauli_lists[i].size())
        << "Operator " << i << " has different number of terms. "
        << "Pool: " << pool_ops[i].num_terms()
        << ", Stateprep: " << pauli_lists[i].size();
  }
}

// ============================================================================
// Test 7: Singles Only Option (via stateprep)
// ============================================================================
TEST(UPCCGSDOperatorPoolTest, SinglesOnlyGeneration) {
  std::size_t M = 3;           // spatial orbitals
  std::size_t n_qubits = 2*M;  // spin orbitals

  auto [pauli_lists, coeff_lists] =
      cudaq::solvers::stateprep::get_upccgsd_pauli_lists(
          n_qubits, /*only_singles=*/true, /*only_doubles=*/false);

  std::size_t expected_singles = M * (M - 1); // generalized spin-preserving
  EXPECT_EQ(pauli_lists.size(), expected_singles)
      << "Singles-only should generate " << expected_singles << " operators";
}

// ============================================================================
// Test 8: Doubles Only Option (via stateprep)
// ============================================================================
TEST(UPCCGSDOperatorPoolTest, DoublesOnlyGeneration) {
  std::size_t M = 3;           // spatial orbitals
  std::size_t n_qubits = 2*M;  // spin orbitals

  auto [pauli_lists, coeff_lists] =
      cudaq::solvers::stateprep::get_upccgsd_pauli_lists(
          n_qubits, /*only_singles=*/false, /*only_doubles=*/true);

  std::size_t expected_doubles = M * (M - 1) / 2; // paired αβ→αβ
  EXPECT_EQ(pauli_lists.size(), expected_doubles)
      << "Doubles-only should generate " << expected_doubles << " operators";
}

// ============================================================================
// Test 9: Scaling Test - Verify Formula for Multiple Sizes
// ============================================================================
TEST(UPCCGSDOperatorPoolTest, ScalingBehavior) {
  auto pool = cudaq::solvers::operator_pool::get("upccgsd");

  // (M = numOrbitals, expected total count)
  std::vector<std::pair<std::size_t, std::size_t>> test_cases = {
      {2, 3},   // M=2: singles=2, doubles=1
      {3, 9},   // M=3: singles=6, doubles=3
      {4, 18},  // M=4: singles=12, doubles=6
      {5, 30},  // M=5: singles=20, doubles=10
  };

  for (auto [M, expected_count] : test_cases) {
    heterogeneous_map config;
    config.insert("num-orbitals", M);
    auto ops = pool->generate(config);

    std::size_t singles = M * (M - 1);
    std::size_t doubles = M * (M - 1) / 2;
    std::size_t total   = singles + doubles;

    EXPECT_EQ(ops.size(), total)
        << "For M=" << M << " orbitals: expected "
        << total << " operators, got " << ops.size();
    EXPECT_EQ(total, expected_count)
        << "Formula mismatch for M=" << M << " orbitals";
  }
}

// ============================================================================
// Test 10: Edge Case - Minimal System
// ============================================================================
TEST(UPCCGSDOperatorPoolTest, MinimalSystem) {
  auto pool = cudaq::solvers::operator_pool::get("upccgsd");

  // With 1 orbital (2 qubits), there are no spin-preserving generalized
  // singles and no paired doubles, so the pool should be empty.
  {
    heterogeneous_map config;
    config.insert("num-orbitals", 1);
    auto ops = pool->generate(config);
    EXPECT_EQ(ops.size(), 0)
        << "With 1 orbital, UpCCGSD should generate 0 operators";
  }

  // With 2 orbitals, we should have 2 singles + 1 double = 3 operators.
  {
    heterogeneous_map config;
    config.insert("num-orbitals", 2);
    auto ops = pool->generate(config);
    EXPECT_EQ(ops.size(), 3)
        << "With 2 orbitals, UpCCGSD should generate 3 operators";
  }
}

// ============================================================================
// Test 11: Verify Hermiticity / Anti-Hermiticity Structure
// ============================================================================
TEST(UPCCGSDOperatorPoolTest, OperatorsAreHermitianGenerators) {
  auto pool = cudaq::solvers::operator_pool::get("upccgsd");
  heterogeneous_map config;
  config.insert("num-orbitals", 2);
  auto ops = pool->generate(config);

  for (size_t i = 0; i < ops.size(); ++i) {
    auto matrix_G = ops[i].to_matrix();

    // Check G is Hermitian
    auto adjoint_G = ops[i].to_matrix();
    for (size_t row = 0; row < matrix_G.rows(); ++row)
      for (size_t col = 0; col < matrix_G.cols(); ++col)
        adjoint_G(row, col) = std::conj(matrix_G(col, row));

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
  }
}

// ============================================================================
// Test 12: Verify Operator Pool Returns Correct Type
// ============================================================================
TEST(UPCCGSDOperatorPoolTest, ReturnsSpinOperators) {
  auto pool = cudaq::solvers::operator_pool::get("upccgsd");
  ASSERT_NE(pool, nullptr) << "Failed to get upccgsd operator pool";

  heterogeneous_map config;
  config.insert("num-orbitals", 2);
  auto ops = pool->generate(config);

  for (size_t i = 0; i < ops.size(); ++i) {
    EXPECT_NO_THROW(ops[i].to_string())
        << "Operator " << i << " is not a valid spin_op";
  }
}

// ============================================================================
// Test 13: Performance Test - Larger System
// ============================================================================
TEST(UPCCGSDOperatorPoolTest, LargeSystemPerformance) {
  auto pool = cudaq::solvers::operator_pool::get("upccgsd");
  heterogeneous_map config;
  config.insert("num-orbitals", 8); // 16 qubits

  auto start = std::chrono::high_resolution_clock::now();
  auto ops = pool->generate(config);
  auto end = std::chrono::high_resolution_clock::now();

  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  EXPECT_LT(duration.count(), 5000)
      << "Generation took too long: " << duration.count() << "ms";

  std::size_t M = 8;
  std::size_t expected =
      M * (M - 1) + M * (M - 1) / 2; // singles + doubles
  EXPECT_EQ(ops.size(), expected);
}
