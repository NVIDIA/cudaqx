/*******************************************************************************
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/solvers/operators/operator_pool.h"
#include "cudaq/solvers/stateprep/ceo.h"
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
    // Singles have 2 Pauli terms (Y_q X_p - X_q Y_p)
    // Doubles have multiple Pauli terms (4 terms each for operators A and B)
    // Heuristic: count X/Y in the string representation.
    std::string op_str = op.to_string();

    size_t xy_count = 0;
    for (size_t i = 0; i + 1 < op_str.length(); ++i) {
      if ((op_str[i] == 'X' || op_str[i] == 'Y') &&
          std::isdigit(op_str[i + 1])) {
        xy_count++;
      }
    }

    // Singles should have 4 X/Y operators
    if (xy_count == 4)
      singles++;
    // Doubles should have 16 X/Y operators
    else if (xy_count == 16)
      doubles++;
    else
      throw std::runtime_error("Invalid operator: " + op_str);
  }

  return {singles, doubles};
}

// ============================================================================
// Test 1: Correct Number of Operators
// ============================================================================
TEST(CEOOperatorPoolTest, CorrectNumberOfOperators) {

  // Test with num_orbitals = 2
  {
    heterogeneous_map config{{"num_orbitals", 2}};
    auto ops = cudaq::solvers::operator_pool::get("ceo")->generate(config);

    // Expected: 1 single for each spin
    // 1 mixed double set of indices, resulting in 2 doubles operators for CEO

    // Total: 2 singles + 8 mixed doubles = 10 operators
    EXPECT_EQ(ops.size(), 4);

    auto [singles, doubles] = countSinglesAndDoubles(ops);
    EXPECT_EQ(singles, 2);
    EXPECT_EQ(doubles, 2);
  }

  // Test with num_orbitals = 3
  {
    heterogeneous_map config{{"num_orbitals", 3}};
    auto ops = cudaq::solvers::operator_pool::get("ceo")->generate(config);

    // Alpha singles: 3*2/2 = 3
    // Beta singles: 3*2/2 = 3
    // Total singles: 6

    // Mixed doubles: 3 * 2 / 2 alpha pairs, same for beta pairs.
    // 9 alpha-beta combinations, each creates 2 operators = 18 operators

    // Same-spin doubles: 0 (need 4 same-spin orbitals)

    // Total: 6 + 18 = 24
    EXPECT_EQ(ops.size(), 24);

    auto [singles, doubles] = countSinglesAndDoubles(ops);
    EXPECT_EQ(singles, 6);
    EXPECT_EQ(doubles, 18);
  }

  // Test with num_orbitals = 4 to check same spin doubles
  {
    heterogeneous_map config{{"num_orbitals", 4}};
    auto ops = cudaq::solvers::operator_pool::get("ceo")->generate(config);

    // Alpha singles: 4*3/2 = 6
    // Beta singles: 4*3/2 = 6
    // Total singles: 12

    // Mixed doubles: 4 * 3 / 2 alpha pairs, same for beta pairs.
    // 36 alpha-beta combinations, each creates 2 operators = 72 operators

    // Same-spin doubles: single set of indices for each spin, each creates 6
    // operators = 12 operators

    // Total: 12 + 72 + 12
    EXPECT_EQ(ops.size(), 96);

    auto [singles, doubles] = countSinglesAndDoubles(ops);
    EXPECT_EQ(singles, 12);
    EXPECT_EQ(doubles, 84);
  }
}

// ============================================================================
// Test 2: Configuration Parameter Validation
// ============================================================================
TEST(CEOOperatorPoolTest, ConfigurationValidation) {
  // Test missing num_orbitals
  {
    heterogeneous_map config{};
    EXPECT_THROW(cudaq::solvers::operator_pool::get("ceo")->generate(config),
                 std::exception);
  }

  // Test num_orbitals = 0
  {
    heterogeneous_map config{{"num_orbitals", 0}};
    EXPECT_THROW(cudaq::solvers::operator_pool::get("ceo")->generate(config),
                 std::invalid_argument);
  }

  // Test valid configuration
  {
    heterogeneous_map config{{"num_orbitals", 2}};
    EXPECT_NO_THROW(
        cudaq::solvers::operator_pool::get("ceo")->generate(config));
  }
}

// ============================================================================
// Test 3: Operator Uniqueness - No duplicate operators
// ============================================================================
TEST(CEOOperatorPoolTest, NoDuplicateOperators) {
  size_t num_orbitals = 5;
  size_t num_qubits = 2 * num_orbitals;
  heterogeneous_map config{{"num_orbitals", num_orbitals}};
  auto ops = cudaq::solvers::operator_pool::get("ceo")->generate(config);

  // Canonicalize all operators for comparison
  std::set<std::size_t> qubit_set;
  for (size_t i = 0; i < num_qubits; ++i)
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
// Test 4: Verify All Operators Are Non-Empty
// ============================================================================
TEST(CEOOperatorPoolTest, AllOperatorsNonEmpty) {
  heterogeneous_map config{{"num_orbitals", 2}};
  auto ops = cudaq::solvers::operator_pool::get("ceo")->generate(config);

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
// ============================================================================
TEST(CEOOperatorPoolTest, VerifyOperatorCoefficients) {
  heterogeneous_map config{{"num_orbitals", 4}};
  auto ops = cudaq::solvers::operator_pool::get("ceo")->generate(config);

  std::vector<double> coefficients;
  for (const auto &op : ops) {
    for (const auto &term : op) {
      double coeff = std::abs(term.evaluate_coefficient().real());
      coefficients.push_back(coeff);
    }
  }

  for (size_t i = 0; i < coefficients.size(); ++i) {
    bool is_valid = (std::abs(coefficients[i] - 0.5) < 1e-10) ||
                    (std::abs(coefficients[i] - 0.25) < 1e-10);
    EXPECT_TRUE(is_valid) << "Coefficient " << i
                          << " has unexpected value: " << coefficients[i]
                          << " (expected 0.5 or 0.25)";
  }
}

// ============================================================================
// Test 6: CEO State Preparation Helper Function
// ============================================================================
TEST(CEOOperatorPoolTest, StatePrepPauliLists) {
  std::size_t norbitals = 5;
  auto [pauliWords, coeffs] =
      cudaq::solvers::stateprep::get_ceo_pauli_lists(norbitals);

  // Should match the operator pool size
  heterogeneous_map config{{"num_orbitals", norbitals}};
  auto ops = cudaq::solvers::operator_pool::get("ceo")->generate(config);
  EXPECT_EQ(pauliWords.size(), ops.size());
  EXPECT_EQ(coeffs.size(), ops.size());

  for (size_t i = 0; i < std::min(ops.size(), pauliWords.size()); ++i) {
    EXPECT_EQ(ops[i].num_terms(), pauliWords[i].size())
        << "Operator " << i << " has different number of terms. "
        << "Pool: " << ops[i].num_terms()
        << ", Stateprep: " << pauliWords[i].size();
  }
}

// ============================================================================
// Test 7: Verify Hermiticity / Anti-Hermiticity Structure
// ============================================================================
TEST(CEOOperatorPoolTest, OperatorsAreHermitianGenerators) {
  heterogeneous_map config{{"num_orbitals", 4}};
  auto ops = cudaq::solvers::operator_pool::get("ceo")->generate(config);

  for (size_t i = 0; i < ops.size(); ++i) {
    auto matrix_G = ops[i].to_matrix();

    // Check G is Hermitian
    bool is_hermitian = true;
    auto adjoint_G = ops[i].to_matrix();
    for (size_t row = 0; row < matrix_G.rows(); ++row) {
      for (size_t col = 0; col < matrix_G.cols(); ++col) {
        auto diff = std::conj(matrix_G(col, row)) - matrix_G(row, col);
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
// Test 8: Performance Test - Large System
// ============================================================================
TEST(CEOOperatorPoolTest, LargeSystemPerformance) {
  auto pool = cudaq::solvers::operator_pool::get("ceo");
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
  size_t n = 6;
  size_t expected = n * (n - 1); // Singles alpha and beta
  expected += (n * (n - 1) / 2) * (n * (n - 1) / 2) * 2; // Mixed doubles
  expected +=
      (n * (n - 1) * (n - 2) * (n - 3) / 24) * 12; // Doubles alpha and beta
  EXPECT_EQ(ops.size(), expected);
}
