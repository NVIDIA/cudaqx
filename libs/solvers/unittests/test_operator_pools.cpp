/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/solvers/operators/operator_pool.h"
#include <gtest/gtest.h>

using namespace cudaqx;

TEST(UCCSDTest, GenerateWithDefaultConfig) {
  auto pool = cudaq::solvers::operator_pool::get("uccsd");
  heterogeneous_map config;
  config.insert("num-qubits", 4);
  config.insert("num-electrons", 2);

  auto operators = pool->generate(config);
  ASSERT_FALSE(operators.empty());
  EXPECT_EQ(operators.size(), 2 * 2 + 1 * 8);

  for (const auto &op : operators) {
    EXPECT_EQ(op.num_qubits(), 4);
  }
}

TEST(UCCSDTest, GenerateFromAPIFunction) {
  auto operators = cudaq::solvers::get_operator_pool(
      "uccsd", {{"num-qubits", 4}, {"num-electrons", 2}});
  ASSERT_FALSE(operators.empty());
  EXPECT_EQ(operators.size(), 2 * 2 + 1 * 8);

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
  EXPECT_EQ(operators.size(), (2 * 2 + 1 * 8));

  for (size_t i = 0; i < operators.size(); ++i) {
    EXPECT_EQ(operators[i].num_qubits(), 4);
    EXPECT_DOUBLE_EQ(1.0, operators[i].get_coefficient().real());
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
  EXPECT_EQ(operators.size(), 2 * 4 + 4 * 8);

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
  EXPECT_GT(operators.size(), 875);

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
  std::vector<std::string> operator_strings;
  for (const auto &op : operators) {
    operator_strings.push_back(op.to_string(false));
  }

  // Assert
  std::vector<std::string> expected_operators = {
      "YZXI", "XZYI", "IYZX", "IXZY", "XXXY", "XXYX",
      "XYYY", "YXYY", "XYXX", "YXXX", "YYXY", "YYYX"};

  ASSERT_EQ(operator_strings.size(), expected_operators.size())
      << "Number of generated operators does not match expected count";

  for (size_t i = 0; i < expected_operators.size(); ++i) {
    EXPECT_EQ(operator_strings[i], expected_operators[i])
        << "Mismatch at index " << i;
  }

  // Additional checks
  for (const auto &op_string : operator_strings) {
    EXPECT_EQ(op_string.length(), 4)
        << "Operator " << op_string
        << " does not have the expected length of 4";

    EXPECT_TRUE(op_string.find_first_not_of("IXYZ") == std::string::npos)
        << "Operator " << op_string << " contains invalid characters";
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