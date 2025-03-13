/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/solvers/operators/operator_pool.h"
#include <complex>
#include <cstddef>
#include <gtest/gtest.h>
#include <string>
#include <vector>

using namespace cudaqx;

TEST(UCCSDTest, GenerateWithDefaultConfig) {
  auto pool = cudaq::solvers::operator_pool::get("uccsd");
  heterogeneous_map config;
  config.insert("num-qubits", 4);
  config.insert("num-electrons", 2);

  auto operators = pool->generate(config);
  ASSERT_FALSE(operators.empty());
  EXPECT_EQ(operators.size(), 2 + 1);

  for (const auto &op : operators) {
    EXPECT_EQ(op.num_qubits(), 4);
  }
}

TEST(UCCSDTest, GenerateFromAPIFunction) {
  auto operators = cudaq::solvers::get_operator_pool(
      "uccsd", {{"num-qubits", 4}, {"num-electrons", 2}});
  ASSERT_FALSE(operators.empty());
  EXPECT_EQ(operators.size(), 2 + 1);

  for (const auto &op : operators) {
    EXPECT_EQ(op.num_qubits(), 4);
  }
}

TEST(UCCSDTest, GenerateWithCustomCoefficients) {
  auto pool = cudaq::solvers::operator_pool::get("uccsd");
  heterogeneous_map config;
  config.insert("num-qubits", 4);
  config.insert("num-electrons", 2);

  auto operators = pool->generate(config);

  ASSERT_FALSE(operators.empty());
  EXPECT_EQ(operators.size(), 2 + 1);

  std::vector<std::complex<double>> temp_coeffs;
  for (size_t i = 0; i < operators.size(); ++i) {
    EXPECT_EQ(operators[i].num_qubits(), 4);

    operators[i].for_each_term([&](const auto &term) {
      temp_coeffs.push_back(term.get_coefficient());
    });
  }

  for (size_t j = 0; j < temp_coeffs.size(); ++j) {
    double real_part = temp_coeffs[j].real();
    EXPECT_TRUE(real_part == 0.5 || real_part == -0.5 || real_part == 0.125 ||
                real_part == -0.125)
        << "Coefficient at index " << j
        << " has unexpected value: " << real_part;
  }
}

TEST(UCCSDTest, GenerateWithOddElectrons) {
  auto pool = cudaq::solvers::operator_pool::get("uccsd");
  heterogeneous_map config;
  config.insert("num-qubits", 6);
  config.insert("num-electrons", 3);
  config.insert("spin", 1);

  auto operators = pool->generate(config);

  ASSERT_FALSE(operators.empty());
  EXPECT_EQ(operators.size(), 2 * 2 + 4);

  for (const auto &op : operators)
    EXPECT_EQ(op.num_qubits(), 6);
}

TEST(UCCSDTest, GenerateWithLargeSystem) {
  auto pool = cudaq::solvers::operator_pool::get("uccsd");
  heterogeneous_map config;
  config.insert("num-qubits", 20);
  config.insert("num-electrons", 10);

  auto operators = pool->generate(config);

  ASSERT_FALSE(operators.empty());
  EXPECT_EQ(operators.size(), 875);

  for (const auto &op : operators) {
    EXPECT_EQ(op.num_qubits(), 20);
  }
}

TEST(UccsdOperatorPoolTest, GeneratesCorrectOperators) {
  // Arrange
  auto pool = cudaq::solvers::operator_pool::get("uccsd");
  heterogeneous_map config;
  config.insert("num-qubits", 4);
  config.insert("num-electrons", 2);

  // Act
  auto operators = pool->generate(config);

  // Convert SpinOperators to strings
  std::vector<std::vector<std::string>> terms_strings;
  std::vector<std::vector<std::complex<double>>> coefficients;
  for (const auto &op : operators) {
    std::vector<std::complex<double>> temp_coeffs;
    std::vector<std::string> string_rep;
    op.for_each_term([&](const auto &term) {
      string_rep.push_back(term.to_string(false));
      temp_coeffs.push_back(term.get_coefficient());
    });
    terms_strings.push_back(string_rep);
    coefficients.push_back(temp_coeffs);
  }

  // Assert
  std::vector<std::vector<std::string>> expected_operators = {
      {"XZYI", "YZXI"},
      {"IXZY", "IYZX"},
      {"YYYX", "YXXX", "XXYX", "YYXY", "XYYY", "XXXY", "YXYY", "XYXX"}};

  std::vector<std::vector<std::complex<double>>> expected_coefficients = {
      {std::complex<double>(-0.5, 0), std::complex<double>(0.5, 0)},
      {std::complex<double>(-0.5, 0), std::complex<double>(0.5, 0)},
      {std::complex<double>(-0.125, 0), std::complex<double>(-0.125, 0),
       std::complex<double>(0.125, 0), std::complex<double>(-0.125, 0),
       std::complex<double>(0.125, 0), std::complex<double>(0.125, 0),
       std::complex<double>(0.125, 0), std::complex<double>(-0.125, 0)}};
  ASSERT_EQ(terms_strings.size(), expected_operators.size())
      << "Number of generated operators does not match expected count";

  for (size_t i = 0; i < expected_operators.size(); ++i) {
    for (size_t j = 0; j < expected_operators[i].size(); ++j) {

      EXPECT_EQ(expected_operators[i][j].length(), 4)
          << "Operator " << expected_operators[i][j]
          << " does not have the expected length of 4";

      EXPECT_EQ(terms_strings[i][j], expected_operators[i][j])
          << "Mismatch at index " << i << ", " << j;
      std::cout << coefficients[i][j] << std::endl;
      std::cout << expected_coefficients[i][j] << std::endl;
      EXPECT_EQ(coefficients[i][j], expected_coefficients[i][j])
          << "Mismatch at index " << i << ", " << j;
    }
  }

  // Additional checks
  for (size_t k = 0; k < terms_strings.size(); ++k) {
    for (size_t l = 0; l < terms_strings[k].size(); ++l) {
      std::string term_string = terms_strings[k][l];
      EXPECT_EQ(term_string.size(), 4)
          << "Operator  " << term_string
          << " does not have the expected length of 4";
    }
  }
}

TEST(UCCSDTest, GenerateWithInvalidConfig) {
  auto pool = cudaq::solvers::operator_pool::get("uccsd");
  heterogeneous_map config;
  // Missing required parameters

  EXPECT_THROW(pool->generate(config), std::runtime_error);
}

// Test for single qubit terms
TEST(MixerPoolTest, SingleQubitTerms) {
  auto opPool = cudaq::solvers::operator_pool::get("qaoa");
  std::vector<cudaq::spin_op> ops = opPool->generate({{"n-qubits", 2}});

  // First 2 operators should be X(0) and X(1)
  EXPECT_EQ(ops[0], cudaq::spin::x(0));
  EXPECT_EQ(ops[1], cudaq::spin::x(1));

  // Next 2 operators should be Y(0) and Y(1)
  EXPECT_EQ(ops[2], cudaq::spin::y(0));
  EXPECT_EQ(ops[3], cudaq::spin::y(1));
}

// Test for two-qubit XX terms
TEST(MixerPoolTest, TwoQubitXXTerms) {
  auto opPool = cudaq::solvers::operator_pool::get("qaoa");
  std::vector<cudaq::spin_op> ops = opPool->generate({{"n-qubits", 3}});

  // Find XX terms (they start after single qubit terms)
  int xx_start_idx = 6; // After 3 X terms and 3 Y terms
  EXPECT_EQ(ops[xx_start_idx], cudaq::spin::x(0) * cudaq::spin::x(1));
  EXPECT_EQ(ops[xx_start_idx + 1], cudaq::spin::x(0) * cudaq::spin::x(2));
  EXPECT_EQ(ops[xx_start_idx + 2], cudaq::spin::x(1) * cudaq::spin::x(2));
}

// Test vector size for different qubit numbers
TEST(MixerPoolTest, VectorSizes) {
  // For n qubits, we expect:
  // - n single X terms
  // - n single Y terms
  // - (n*(n-1))/2 terms for each two-qubit combination (XX, YY, YZ, ZY, XY, YX,
  // XZ, ZX)

  // Test for 2 qubits
  auto opPool = cudaq::solvers::operator_pool::get("qaoa");
  std::vector<cudaq::spin_op> ops_2q = opPool->generate({{"n-qubits", 2}});
  int expected_size_2q = 4 + 8; // 4 single-qubit + 8 two-qubit terms
  EXPECT_EQ(ops_2q.size(), expected_size_2q);

  // Test for 3 qubits
  std::vector<cudaq::spin_op> ops_3q = opPool->generate({{"n-qubits", 3}});

  int expected_size_3q = 6 + 24; // 6 single-qubit + 24 two-qubit terms
  EXPECT_EQ(ops_3q.size(), expected_size_3q);
}

// Test for empty and single qubit cases
TEST(MixerPoolTest, EdgeCases) {
  // Test with 0 qubits
  auto opPool = cudaq::solvers::operator_pool::get("qaoa");
  std::vector<cudaq::spin_op> ops_0q = opPool->generate({{"n-qubits", 0}});

  EXPECT_EQ(ops_0q.size(), 0);

  // Test with 1 qubit
  // auto ops_1q = mixer_pool(1);
  std::vector<cudaq::spin_op> ops_1q = opPool->generate({{"n-qubits", 1}});

  EXPECT_EQ(ops_1q.size(), 2); // Only X(0) and Y(0)
  EXPECT_EQ(ops_1q[0], cudaq::spin::x(0));
  EXPECT_EQ(ops_1q[1], cudaq::spin::y(0));
}