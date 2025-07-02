/*******************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <cmath>
#include <gtest/gtest.h>

#include "cudaq/qec/codes/surface_code.h"
#include "cudaq/qec/experiments.h"
#include "cudaq/qec/pcm_utils.h"
#include "cudaq/qec/version.h"

TEST(StabilizerTester, checkConstructFromSpinOps) {
  {
    // Constructor will always auto sort
    std::vector<cudaq::spin_op_term> stab{cudaq::spin_op::from_word("ZZZZIII"),
                                          cudaq::spin_op::from_word("XXXXIII"),
                                          cudaq::spin_op::from_word("IXXIXXI"),
                                          cudaq::spin_op::from_word("IIXXIXX"),
                                          cudaq::spin_op::from_word("IZZIZZI"),
                                          cudaq::spin_op::from_word("IIZZIZZ")};
    EXPECT_EQ(stab.size(), 6);
    auto parity = cudaq::qec::to_parity_matrix(stab);
    parity.dump();
    EXPECT_EQ(parity.rank(), 2);
    std::vector<std::size_t> expected_shape{6, 14};
    EXPECT_EQ(parity.shape(), expected_shape);

    {
      std::vector<int> data = {
          1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  /* row 0 */
          0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,  /* row 1 */
          0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0,  /* row 2 */
          0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0,  /* row 3 */
          0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0,  /* row 4 */
          0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1}; /* row 5 */

      cudaqx::tensor<int> t(expected_shape);
      t.borrow(data.data(), expected_shape);
      for (std::size_t i = 0; i < 6; i++)
        for (std::size_t j = 0; j < 14; j++)
          EXPECT_EQ(t.at({i, j}), parity.at({i, j}));
    }
    {
      auto parity_x =
          cudaq::qec::to_parity_matrix(stab, cudaq::qec::stabilizer_type::X);
      printf("Hx:\n");
      parity_x.dump();
      EXPECT_EQ(parity_x.rank(), 2);
      std::vector<std::size_t> expected_shape{3, 7};
      EXPECT_EQ(parity_x.shape(), expected_shape);
      std::vector<int> data = {1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0,
                               1, 1, 0, 0, 0, 1, 1, 0, 1, 1};
      cudaqx::tensor<int> t(expected_shape);
      t.borrow(data.data(), expected_shape);
      for (std::size_t i = 0; i < 3; i++)
        for (std::size_t j = 0; j < 7; j++)
          EXPECT_EQ(t.at({i, j}), parity_x.at({i, j}));
    }
  }
  {

    // Note testing here also that constructor sorts them
    std::vector<std::string> stab{"ZZZZIII", "XXXXIII", "IXXIXXI",
                                  "IIXXIXX", "IZZIZZI", "IIZZIZZ"};
    EXPECT_EQ(stab.size(), 6);
    auto parity = cudaq::qec::to_parity_matrix(stab);
    parity.dump();
    EXPECT_EQ(parity.rank(), 2);
    std::vector<std::size_t> expected_shape{6, 14};
    EXPECT_EQ(parity.shape(), expected_shape);
    {
      std::vector<int> data = {
          1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  /* row 0 */
          0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,  /* row 1 */
          0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0,  /* row 2 */
          0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0,  /* row 3 */
          0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0,  /* row 4 */
          0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1}; /* row 5 */

      cudaqx::tensor<int> t(expected_shape);
      t.borrow(data.data(), expected_shape);
      for (std::size_t i = 0; i < 6; i++)
        for (std::size_t j = 0; j < 14; j++)
          EXPECT_EQ(t.at({i, j}), parity.at({i, j}));
    }
    {
      auto parity_z =
          cudaq::qec::to_parity_matrix(stab, cudaq::qec::stabilizer_type::Z);
      parity_z.dump();
      EXPECT_EQ(parity_z.rank(), 2);
      std::vector<std::size_t> expected_shape{3, 7};
      EXPECT_EQ(parity_z.shape(), expected_shape);
      std::vector<int> data = {1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0,
                               1, 1, 0, 0, 0, 1, 1, 0, 1, 1};
      cudaqx::tensor<int> t(expected_shape);
      t.borrow(data.data(), expected_shape);
      for (std::size_t i = 0; i < 3; i++)
        for (std::size_t j = 0; j < 7; j++)
          EXPECT_EQ(t.at({i, j}), parity_z.at({i, j}));
    }
  }
}

TEST(QECCodeTester, checkSampleMemoryCircuit) {
  {
    // Steane tests
    auto steane = cudaq::qec::get_code("steane");
    cudaqx::tensor<uint8_t> parity = steane->get_parity();
    cudaqx::tensor<uint8_t> observables =
        steane->get_pauli_observables_matrix();
    cudaqx::tensor<uint8_t> Lx = steane->get_observables_x();
    cudaqx::tensor<uint8_t> Lz = steane->get_observables_z();

    int nShots = 10;
    int nRounds = 4;
    {
      // Prep0 experiment. Prep all data qubits in Z basis.
      // Measure all data qubits in the Z basis.
      // To correct it, find out how many times it flipped.
      // X errors flip the Z observable.
      // So when we get the predicted error data string E = E_X | E_Z
      // from the decoder, we apply E_X to our L_mz to correct it.
      auto [syndromes, d] = cudaq::qec::sample_memory_circuit(
          *steane, cudaq::qec::operation::prep0, nShots, nRounds);
      syndromes.dump();
      EXPECT_EQ(syndromes.shape()[0], nShots * nRounds);
      EXPECT_EQ(syndromes.shape()[1], 6);

      // No noise here, should be all zeros
      int sum = 0;
      for (std::size_t i = 0; i < syndromes.shape()[0]; i++)
        for (std::size_t j = 0; j < syndromes.shape()[1]; j++)
          sum += syndromes.at({i, j});
      EXPECT_TRUE(sum == 0);

      // Prep0, should measure out logical |0> each shot
      printf("data:\n");
      d.dump();
      printf("Lz:\n");
      Lz.dump();
      cudaqx::tensor<uint8_t> logical_mz = Lz.dot(d.transpose()) % 2;
      printf("logical_mz:\n");
      logical_mz.dump();
      EXPECT_FALSE(logical_mz.any());
    }
    {
      // Prep1, should measure out logical |1> each shot
      auto [syndromes, d] = cudaq::qec::sample_memory_circuit(
          *steane, cudaq::qec::operation::prep1, nShots, nRounds);
      printf("data:\n");
      d.dump();
      printf("Lz:\n");
      Lz.dump();
      cudaqx::tensor<uint8_t> logical_mz = Lz.dot(d.transpose()) % 2;
      printf("logical_mz:\n");
      logical_mz.dump();
      EXPECT_EQ(nShots, logical_mz.sum_all());
    }
    {
      // Prepp, should measure out logical |+> each shot
      auto [syndromes, d] = cudaq::qec::sample_memory_circuit(
          *steane, cudaq::qec::operation::prepp, nShots, nRounds);
      printf("data:\n");
      d.dump();
      printf("Lx:\n");
      Lx.dump();
      cudaqx::tensor<uint8_t> logical_mx = Lx.dot(d.transpose()) % 2;
      printf("logical_mx:\n");
      logical_mx.dump();
      EXPECT_FALSE(logical_mx.any());
    }
    {
      // Prepm, should measure out logical |-> each shot
      auto [syndromes, d] = cudaq::qec::sample_memory_circuit(
          *steane, cudaq::qec::operation::prepm, nShots, nRounds);
      printf("data:\n");
      d.dump();
      printf("Lx:\n");
      Lx.dump();
      cudaqx::tensor<uint8_t> logical_mx = Lx.dot(d.transpose()) % 2;
      printf("logical_mx:\n");
      logical_mx.dump();
      EXPECT_EQ(nShots, logical_mx.sum_all());
    }
  }
}

TEST(QECCodeTester, checkSteane) {
  {
    // with default stabilizers
    auto steane = cudaq::qec::get_code("steane");
    cudaqx::tensor<uint8_t> parity = steane->get_parity();
    cudaqx::tensor<uint8_t> observables =
        steane->get_pauli_observables_matrix();
    cudaqx::tensor<uint8_t> Lx = steane->get_observables_x();
    cudaqx::tensor<uint8_t> Lz = steane->get_observables_z();
    {
      std::vector<uint8_t> data = {
          1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  /* row 0 */
          0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,  /* row 1 */
          0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0,  /* row 2 */
          0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0,  /* row 3 */
          0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0,  /* row 4 */
          0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1}; /* row 5 */

      std::vector<std::size_t> expected_shape{6, 14};
      cudaqx::tensor<uint8_t> t(expected_shape);
      t.borrow(data.data());
      for (std::size_t i = 0; i < 6; i++)
        for (std::size_t j = 0; j < 14; j++)
          EXPECT_EQ(t.at({i, j}), parity.at({i, j}));
    }
    EXPECT_EQ(2, observables.rank());
    EXPECT_EQ(2, observables.shape()[0]);
    EXPECT_EQ(14, observables.shape()[1]);
    EXPECT_EQ(2, Lx.rank());
    EXPECT_EQ(1, Lx.shape()[0]);
    EXPECT_EQ(7, Lx.shape()[1]);
    EXPECT_EQ(2, Lz.rank());
    EXPECT_EQ(1, Lz.shape()[0]);
    EXPECT_EQ(7, Lz.shape()[1]);
    {
      std::vector<std::vector<uint8_t>> true_observables = {
          {0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0},
          {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1}};
      std::vector<std::vector<uint8_t>> true_Lx = {{0, 0, 0, 0, 1, 1, 1}};
      std::vector<std::vector<uint8_t>> true_Lz = {{0, 0, 0, 0, 1, 1, 1}};
      for (std::size_t i = 0; i < observables.shape()[0]; ++i)
        for (std::size_t j = 0; j < observables.shape()[1]; ++j)
          EXPECT_EQ(true_observables[i][j], observables.at({i, j}));

      for (std::size_t i = 0; i < Lx.shape()[0]; ++i)
        for (std::size_t j = 0; j < Lx.shape()[1]; ++j)
          EXPECT_EQ(true_Lx[i][j], Lx.at({i, j}));

      for (std::size_t i = 0; i < Lz.shape()[0]; ++i)
        for (std::size_t j = 0; j < Lz.shape()[1]; ++j)
          EXPECT_EQ(true_Lz[i][j], Lz.at({i, j}));
    }

    auto [syndromes, d] = cudaq::qec::sample_memory_circuit(*steane, 10, 4);
    syndromes.dump();

    // No noise here, should be all zeros
    int sum = 0;
    for (std::size_t i = 0; i < syndromes.shape()[0]; i++)
      for (std::size_t j = 0; j < syndromes.shape()[1]; j++)
        sum += syndromes.at({i, j});

    EXPECT_TRUE(sum == 0);
  }
  {
    // From Stabilizers
    std::vector<std::string> words{"ZZZZIII", "XXXXIII", "IXXIXXI",
                                   "IIXXIXX", "IZZIZZI", "IIZZIZZ"};
    std::vector<cudaq::spin_op_term> ops;
    for (auto &os : words)
      ops.emplace_back(cudaq::spin_op::from_word(os));
    cudaq::qec::sortStabilizerOps(ops);
    auto steane = cudaq::qec::get_code("steane", ops);
    auto parity = steane->get_parity();
    {
      std::vector<uint8_t> data = {
          1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  /* row 0 */
          0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,  /* row 1 */
          0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0,  /* row 2 */
          0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0,  /* row 3 */
          0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0,  /* row 4 */
          0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1}; /* row 5 */
      std::vector<std::size_t> expected_shape{6, 14};
      cudaqx::tensor<uint8_t> t(expected_shape);
      t.borrow(data.data());
      for (std::size_t i = 0; i < 6; i++)
        for (std::size_t j = 0; j < 14; j++)
          EXPECT_EQ(t.at({i, j}), parity.at({i, j}));
    }

    auto [syndromes, d] = cudaq::qec::sample_memory_circuit(*steane, 10, 4);
    syndromes.dump();
    // No noise here, should be all zeros
    int sum = 0;
    for (std::size_t i = 0; i < syndromes.shape()[0]; i++)
      for (std::size_t j = 0; j < syndromes.shape()[1]; j++)
        sum += syndromes.at({i, j});

    EXPECT_TRUE(sum == 0);
  }
}

TEST(QECCodeTester, checkCodeCapacity) {
  {
    auto steane = cudaq::qec::get_code("steane");
    auto Hz = steane->get_parity_z();
    int nShots = 10;
    double error_prob = 0;

    auto [syndromes, data] =
        cudaq::qec::sample_code_capacity(Hz, nShots, error_prob);
    EXPECT_EQ(2, Hz.rank());
    EXPECT_EQ(3, Hz.shape()[0]);
    EXPECT_EQ(7, Hz.shape()[1]);
    EXPECT_EQ(nShots, syndromes.shape()[0]);
    EXPECT_EQ(Hz.shape()[0], syndromes.shape()[1]);
    EXPECT_EQ(nShots, data.shape()[0]);
    EXPECT_EQ(Hz.shape()[1], data.shape()[1]);

    // Error prob = 0 should be all zeros
    for (size_t i = 0; i < nShots; ++i) {
      for (size_t j = 0; j < Hz.shape()[1]; ++j) {
        EXPECT_EQ(0, data.at({i, j}));
      }
    }

    for (size_t i = 0; i < nShots; ++i) {
      for (size_t j = 0; j < Hz.shape()[0]; ++j) {
        EXPECT_EQ(0, syndromes.at({i, j}));
      }
    }
  }
  {
    auto steane = cudaq::qec::get_code("steane");
    auto Hz = steane->get_parity_z();
    int nShots = 10;
    double error_prob = 0.15;
    unsigned seed = 1337;

    auto [syndromes, data] =
        cudaq::qec::sample_code_capacity(Hz, nShots, error_prob, seed);
    EXPECT_EQ(2, Hz.rank());
    EXPECT_EQ(3, Hz.shape()[0]);
    EXPECT_EQ(7, Hz.shape()[1]);
    EXPECT_EQ(nShots, syndromes.shape()[0]);
    EXPECT_EQ(Hz.shape()[0], syndromes.shape()[1]);
    EXPECT_EQ(nShots, data.shape()[0]);
    EXPECT_EQ(Hz.shape()[1], data.shape()[1]);
    // seed = 1337, error_prob = 0.15, nShots = 10
    // produces this data set:
    // This seed happens to only have weight 0 or 1 errors,
    // which are easy to check by hand.
    std::vector<std::vector<uint8_t>> seeded_data = {
        {0, 0, 0, 1, 0, 0, 0}, {0, 0, 0, 1, 0, 0, 0}, {0, 0, 1, 0, 0, 0, 0},
        {0, 0, 0, 0, 1, 0, 0}, {0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 1, 0, 0}, {0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 1, 0},
        {0, 0, 0, 0, 0, 0, 0}};
    EXPECT_EQ(nShots, seeded_data.size());
    EXPECT_EQ(Hz.shape()[1], seeded_data[0].size());
    for (size_t i = 0; i < nShots; ++i) {
      for (size_t j = 0; j < Hz.shape()[1]; ++j) {
        EXPECT_EQ(seeded_data[i][j], data.at({i, j}));
      }
    }

    // Hand-checked syndromes
    std::vector<std::vector<uint8_t>> checked_syndromes = {
        {1, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 0}, {0, 0, 0},
        {0, 0, 0}, {0, 1, 0}, {0, 0, 0}, {0, 1, 1}, {0, 0, 0}};
    EXPECT_EQ(nShots, checked_syndromes.size());
    EXPECT_EQ(Hz.shape()[0], checked_syndromes[0].size());
    for (size_t i = 0; i < nShots; ++i) {
      for (size_t j = 0; j < Hz.shape()[0]; ++j) {
        EXPECT_EQ(checked_syndromes[i][j], syndromes.at({i, j}));
      }
    }
  }
  {
    auto steane = cudaq::qec::get_code("steane");
    auto Hz = steane->get_parity_z();
    int nShots = 10;
    double error_prob = 0.25;
    unsigned seed = 1337;

    auto [syndromes, data] =
        cudaq::qec::sample_code_capacity(Hz, nShots, error_prob, seed);
    EXPECT_EQ(2, Hz.rank());
    EXPECT_EQ(3, Hz.shape()[0]);
    EXPECT_EQ(7, Hz.shape()[1]);
    EXPECT_EQ(nShots, syndromes.shape()[0]);
    EXPECT_EQ(Hz.shape()[0], syndromes.shape()[1]);
    EXPECT_EQ(nShots, data.shape()[0]);
    EXPECT_EQ(Hz.shape()[1], data.shape()[1]);
    // seed = 1337, error_prob = 0.25, nShots = 10
    // produces this data set:
    // This seed has some higher weight errors which
    // where checked by hand
    std::vector<std::vector<uint8_t>> seeded_data = {
        {0, 1, 0, 1, 1, 0, 0}, {0, 0, 0, 1, 0, 0, 1}, {0, 0, 1, 0, 0, 0, 0},
        {0, 0, 0, 0, 1, 0, 0}, {0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 1, 0, 0}, {0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 1, 0},
        {0, 0, 0, 0, 0, 0, 0}};
    EXPECT_EQ(nShots, seeded_data.size());
    EXPECT_EQ(Hz.shape()[1], seeded_data[0].size());
    for (size_t i = 0; i < nShots; ++i) {
      for (size_t j = 0; j < Hz.shape()[1]; ++j) {
        EXPECT_EQ(seeded_data[i][j], data.at({i, j}));
      }
    }

    // Hand-checked syndromes
    std::vector<std::vector<uint8_t>> checked_syndromes = {
        {0, 0, 1}, {1, 0, 0}, {1, 1, 1}, {0, 1, 0}, {0, 0, 0},
        {0, 0, 0}, {0, 1, 0}, {0, 0, 0}, {0, 1, 1}, {0, 0, 0}};

    EXPECT_EQ(nShots, checked_syndromes.size());
    EXPECT_EQ(Hz.shape()[0], checked_syndromes[0].size());
    for (size_t i = 0; i < nShots; ++i) {
      for (size_t j = 0; j < Hz.shape()[0]; ++j) {
        EXPECT_EQ(checked_syndromes[i][j], syndromes.at({i, j}));
      }
    }
  }
}

TEST(QECCodeTester, checkRepetition) {
  {
    // must provide distance
    EXPECT_THROW(cudaq::qec::get_code("repetition"), std::runtime_error);
  }
  auto repetition = cudaq::qec::get_code(
      "repetition", cudaqx::heterogeneous_map{{"distance", 9}});

  {
    auto stabilizers = repetition->get_stabilizers();

    std::vector<std::string> actual_stabs;
    for (auto &s : stabilizers)
      actual_stabs.push_back(s.get_pauli_word());

    std::vector<std::string> expected_strings = {
        "ZZIIIIIII", "IZZIIIIII", "IIZZIIIII", "IIIZZIIII",
        "IIIIZZIII", "IIIIIZZII", "IIIIIIZZI", "IIIIIIIZZ"};

    EXPECT_EQ(actual_stabs, expected_strings);
    auto parity = repetition->get_parity();
    auto Hx = repetition->get_parity_x();
    auto Hz = repetition->get_parity_z();
    EXPECT_EQ(0, Hx.rank());
    EXPECT_EQ(2, Hz.rank());
    parity.dump();
    std::vector<uint8_t> data = {
        1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  /* row 0 */
        0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  /* row 1 */
        0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  /* row 2 */
        0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  /* row 3 */
        0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  /* row 4 */
        0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  /* row 5 */
        0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  /* row 6 */
        0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0}; /* row 7 */
    std::vector<std::size_t> expected_shape{8, 18};
    cudaqx::tensor<uint8_t> t(expected_shape);
    t.borrow(data.data());
    for (std::size_t i = 0; i < t.shape()[0]; i++)
      for (std::size_t j = 0; j < t.shape()[1]; j++)
        EXPECT_EQ(t.at({i, j}), parity.at({i, j}));
  }
  {
    cudaqx::tensor<uint8_t> observables =
        repetition->get_pauli_observables_matrix();
    cudaqx::tensor<uint8_t> Lx = repetition->get_observables_x();
    cudaqx::tensor<uint8_t> Lz = repetition->get_observables_z();

    EXPECT_EQ(2, observables.rank());
    EXPECT_EQ(1, observables.shape()[0]);
    EXPECT_EQ(18, observables.shape()[1]);
    EXPECT_EQ(0, Lx.rank());
    EXPECT_EQ(2, Lz.rank());
    EXPECT_EQ(1, Lz.shape()[0]);
    EXPECT_EQ(9, Lz.shape()[1]);
    {
      std::vector<std::vector<uint8_t>> true_observables = {
          {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};
      std::vector<std::vector<uint8_t>> true_Lz = {{1, 0, 0, 0, 0, 0, 0, 0, 0}};
      for (std::size_t i = 0; i < observables.shape()[0]; ++i)
        for (std::size_t j = 0; j < observables.shape()[1]; ++j)
          EXPECT_EQ(true_observables[i][j], observables.at({i, j}));

      for (std::size_t i = 0; i < Lz.shape()[0]; ++i)
        for (std::size_t j = 0; j < Lz.shape()[1]; ++j)
          EXPECT_EQ(true_Lz[i][j], Lz.at({i, j}));
    }
  }
  {
    auto parity = repetition->get_parity();
    auto parity_z = repetition->get_parity_z();
    int nShots = 10;
    int nRounds = 4;
    auto [syndromes, data_mz] =
        cudaq::qec::sample_memory_circuit(*repetition, nShots, nRounds);
    syndromes.dump();
    data_mz.dump();
    EXPECT_EQ(nShots * nRounds, syndromes.shape()[0]);
    EXPECT_EQ(parity.shape()[0], syndromes.shape()[1]);
    EXPECT_EQ(nShots, data_mz.shape()[0]);
    EXPECT_EQ(parity_z.shape()[1], data_mz.shape()[1]);
    // No noise here, should be all zeros
    int sum = 0;
    for (std::size_t i = 0; i < nShots - 1; i++)
      for (std::size_t j = 0; j < parity.shape()[0]; j++)
        sum += syndromes.at({i, j});

    EXPECT_TRUE(sum == 0);
  }
}

TEST(QECCodeTester, checkSurfaceCode) {
  {
    // must provide distance
    EXPECT_THROW(cudaq::qec::get_code("surface_code"), std::runtime_error);
  }
  {
    // with default stabilizers
    auto surf_code = cudaq::qec::get_code(
        "surface_code", cudaqx::heterogeneous_map{{"distance", 3}});
    cudaqx::tensor<uint8_t> parity = surf_code->get_parity();
    cudaqx::tensor<uint8_t> parity_x = surf_code->get_parity_x();
    cudaqx::tensor<uint8_t> parity_z = surf_code->get_parity_z();
    cudaqx::tensor<uint8_t> observables =
        surf_code->get_pauli_observables_matrix();
    cudaqx::tensor<uint8_t> Lx = surf_code->get_observables_x();
    cudaqx::tensor<uint8_t> Lz = surf_code->get_observables_z();

    {
      // This is just a regression check, this has not been hand tested
      std::vector<uint8_t> data = {
          1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  /* row 0 */
          0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  /* row 1 */
          0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,  /* row 2 */
          0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  /* row 3 */
          0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0,  /* row 4 */
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0,  /* row 5 */
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0,  /* row 6 */
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1}; /* row 7 */

      std::vector<std::size_t> expected_shape{8, 18};
      cudaqx::tensor<uint8_t> t(expected_shape);
      t.borrow(data.data());
      for (std::size_t i = 0; i < 8; i++)
        for (std::size_t j = 0; j < 18; j++)
          EXPECT_EQ(t.at({i, j}), parity.at({i, j}));
    }
    {
      // This is just a regression check, this has not been hand tested
      std::vector<uint8_t> data = {1, 0, 0, 1, 0, 0, 0, 0, 0,  /* row 0 */
                                   0, 1, 1, 0, 1, 1, 0, 0, 0,  /* row 1 */
                                   0, 0, 0, 1, 1, 0, 1, 1, 0,  /* row 2 */
                                   0, 0, 0, 0, 0, 1, 0, 0, 1}; /* row 3 */

      std::vector<std::size_t> expected_shape{4, 9};
      cudaqx::tensor<uint8_t> t(expected_shape);
      t.borrow(data.data());
      for (std::size_t i = 0; i < 4; i++)
        for (std::size_t j = 0; j < 9; j++)
          EXPECT_EQ(t.at({i, j}), parity_x.at({i, j}));
    }
    {
      // This is just a regression check, this has not been hand tested
      std::vector<uint8_t> data = {1, 1, 0, 1, 1, 0, 0, 0, 0,  /* row 0 */
                                   0, 1, 1, 0, 0, 0, 0, 0, 0,  /* row 1 */
                                   0, 0, 0, 0, 1, 1, 0, 1, 1,  /* row 2 */
                                   0, 0, 0, 0, 0, 0, 1, 1, 0}; /* row 3 */

      std::vector<std::size_t> expected_shape{4, 9};
      cudaqx::tensor<uint8_t> t(expected_shape);
      t.borrow(data.data());
      for (std::size_t i = 0; i < 4; i++)
        for (std::size_t j = 0; j < 9; j++)
          EXPECT_EQ(t.at({i, j}), parity_z.at({i, j}));
    }
    EXPECT_EQ(2, observables.rank());
    EXPECT_EQ(2, observables.shape()[0]);
    EXPECT_EQ(18, observables.shape()[1]);
    EXPECT_EQ(2, Lx.rank());
    EXPECT_EQ(1, Lx.shape()[0]);
    EXPECT_EQ(9, Lx.shape()[1]);
    EXPECT_EQ(2, Lz.rank());
    EXPECT_EQ(1, Lz.shape()[0]);
    EXPECT_EQ(9, Lz.shape()[1]);
    {
      std::vector<std::vector<uint8_t>> true_observables = {
          {1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},  // Z first
          {0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0}}; // Then X
      std::vector<std::vector<uint8_t>> true_Lx = {{1, 1, 1, 0, 0, 0, 0, 0, 0}};
      std::vector<std::vector<uint8_t>> true_Lz = {{1, 0, 0, 1, 0, 0, 1, 0, 0}};
      for (std::size_t i = 0; i < observables.shape()[0]; ++i)
        for (std::size_t j = 0; j < observables.shape()[1]; ++j)
          EXPECT_EQ(true_observables[i][j], observables.at({i, j}));

      for (std::size_t i = 0; i < Lx.shape()[0]; ++i)
        for (std::size_t j = 0; j < Lx.shape()[1]; ++j)
          EXPECT_EQ(true_Lx[i][j], Lx.at({i, j}));

      for (std::size_t i = 0; i < Lz.shape()[0]; ++i)
        for (std::size_t j = 0; j < Lz.shape()[1]; ++j)
          EXPECT_EQ(true_Lz[i][j], Lz.at({i, j}));
    }
  }
}

// expect |0>, |+> to measure out 0 in respective bases
// expect |1>, |-> to measure out 1 in respective bases
bool noiseless_logical_SPAM_test(const cudaq::qec::code &code,
                                 cudaq::qec::operation statePrep,
                                 uint8_t expected_result) {
  cudaqx::tensor<uint8_t> Lx = code.get_observables_x();
  cudaqx::tensor<uint8_t> Lz = code.get_observables_z();

  // We measure Z observable in prep0, prep1 experiments
  cudaqx::tensor<uint8_t> measured_observable(Lz.shape());
  measured_observable.borrow(Lz.data());
  // We measure X observable in prepp, prepm experiments
  if (statePrep == cudaq::qec::operation::prepp ||
      statePrep == cudaq::qec::operation::prepm) {
    measured_observable = cudaqx::tensor<uint8_t>(Lx.shape());
    measured_observable.borrow(Lx.data());
  }

  int nShots = 10;
  // Number of rounds does not matter for noiseless, all should be zero.
  int nRounds = 4;
  auto [syndromes, d] =
      cudaq::qec::sample_memory_circuit(code, statePrep, nShots, nRounds);
  syndromes.dump();

  printf("data:\n");
  d.dump();
  printf("Obs:\n");
  measured_observable.dump();
  cudaqx::tensor<uint8_t> logical_measurement =
      measured_observable.dot(d.transpose()) % 2;
  printf("logical_measurement:\n");
  logical_measurement.dump();

  // With no noise, each shot should measure out the expected value
  for (size_t shot = 0; shot < nShots; ++shot) {
    // All codes have only 1 logical qubit for now
    for (size_t lQ = 0; lQ < 1; ++lQ) {
      if (logical_measurement.at({lQ, shot}) != expected_result) {
        printf("shot: %zu, lQ: %zu\n", shot, lQ);
        std::cout << +logical_measurement.at({lQ, shot}) << "\n";
        return false;
      }
    }
  }
  return true;
}

TEST(QECCodeTester, checkSteaneSPAM) {
  auto steane = cudaq::qec::get_code("steane");
  EXPECT_TRUE(
      noiseless_logical_SPAM_test(*steane, cudaq::qec::operation::prep0, 0));
  EXPECT_TRUE(
      noiseless_logical_SPAM_test(*steane, cudaq::qec::operation::prep1, 1));
  EXPECT_TRUE(
      noiseless_logical_SPAM_test(*steane, cudaq::qec::operation::prepp, 0));
  EXPECT_TRUE(
      noiseless_logical_SPAM_test(*steane, cudaq::qec::operation::prepm, 1));
  EXPECT_FALSE(
      noiseless_logical_SPAM_test(*steane, cudaq::qec::operation::prep0, 1));
  EXPECT_FALSE(
      noiseless_logical_SPAM_test(*steane, cudaq::qec::operation::prep1, 0));
  EXPECT_FALSE(
      noiseless_logical_SPAM_test(*steane, cudaq::qec::operation::prepp, 1));
  EXPECT_FALSE(
      noiseless_logical_SPAM_test(*steane, cudaq::qec::operation::prepm, 0));
}

TEST(QECCodeTester, checkRepetitionSPAM) {
  // only Z basis for repetition
  auto repetition = cudaq::qec::get_code(
      "repetition", cudaqx::heterogeneous_map{{"distance", 9}});
  EXPECT_TRUE(noiseless_logical_SPAM_test(*repetition,
                                          cudaq::qec::operation::prep0, 0));
  EXPECT_TRUE(noiseless_logical_SPAM_test(*repetition,
                                          cudaq::qec::operation::prep1, 1));
  EXPECT_FALSE(noiseless_logical_SPAM_test(*repetition,
                                           cudaq::qec::operation::prep0, 1));
  EXPECT_FALSE(noiseless_logical_SPAM_test(*repetition,
                                           cudaq::qec::operation::prep1, 0));
}

TEST(QECCodeTester, checkSurfaceCodeSPAM) {
  // Must compile with stim for larger distances
  auto surf_code = cudaq::qec::get_code(
      "surface_code", cudaqx::heterogeneous_map{{"distance", 3}});
  EXPECT_TRUE(
      noiseless_logical_SPAM_test(*surf_code, cudaq::qec::operation::prep0, 0));
  EXPECT_TRUE(
      noiseless_logical_SPAM_test(*surf_code, cudaq::qec::operation::prep1, 1));
  EXPECT_TRUE(
      noiseless_logical_SPAM_test(*surf_code, cudaq::qec::operation::prepp, 0));
  EXPECT_TRUE(
      noiseless_logical_SPAM_test(*surf_code, cudaq::qec::operation::prepm, 1));
  EXPECT_FALSE(
      noiseless_logical_SPAM_test(*surf_code, cudaq::qec::operation::prep0, 1));
  EXPECT_FALSE(
      noiseless_logical_SPAM_test(*surf_code, cudaq::qec::operation::prep1, 0));
  EXPECT_FALSE(
      noiseless_logical_SPAM_test(*surf_code, cudaq::qec::operation::prepp, 1));
  EXPECT_FALSE(
      noiseless_logical_SPAM_test(*surf_code, cudaq::qec::operation::prepm, 0));
}

TEST(QECCodeTester, checkStabilizerGrid) {
  {
    int distance = 3;

    cudaq::qec::surface_code::stabilizer_grid grid(distance);

    grid.print_stabilizer_grid();
    grid.print_stabilizer_coords();
    grid.print_stabilizer_indices();
    grid.print_stabilizer_maps();
    grid.print_data_grid();
    grid.print_stabilizers();

    EXPECT_EQ(3, grid.distance);
    EXPECT_EQ(4, grid.grid_length);
    EXPECT_EQ(16, grid.roles.size());
    EXPECT_EQ(4, grid.x_stab_coords.size());
    EXPECT_EQ(4, grid.z_stab_coords.size());
    EXPECT_EQ(4, grid.x_stab_indices.size());
    EXPECT_EQ(4, grid.z_stab_indices.size());
    EXPECT_EQ(9, grid.data_coords.size());
    EXPECT_EQ(9, grid.data_indices.size());
    EXPECT_EQ(4, grid.x_stabilizers.size());
    EXPECT_EQ(4, grid.z_stabilizers.size());
  }
  {
    int distance = 5;

    cudaq::qec::surface_code::stabilizer_grid grid(distance);

    grid.print_stabilizer_grid();
    grid.print_stabilizer_coords();
    grid.print_stabilizer_indices();
    grid.print_stabilizer_maps();
    grid.print_data_grid();
    grid.print_stabilizers();

    EXPECT_EQ(5, grid.distance);
    EXPECT_EQ(6, grid.grid_length);
    EXPECT_EQ(36, grid.roles.size());
    EXPECT_EQ(12, grid.x_stab_coords.size());
    EXPECT_EQ(12, grid.z_stab_coords.size());
    EXPECT_EQ(12, grid.x_stab_indices.size());
    EXPECT_EQ(12, grid.z_stab_indices.size());
    EXPECT_EQ(25, grid.data_coords.size());
    EXPECT_EQ(25, grid.data_indices.size());
    EXPECT_EQ(12, grid.x_stabilizers.size());
    EXPECT_EQ(12, grid.z_stabilizers.size());
  }
  {
    int distance = 17;

    cudaq::qec::surface_code::stabilizer_grid grid(distance);

    EXPECT_EQ(17, grid.distance);
    EXPECT_EQ(18, grid.grid_length);
    EXPECT_EQ(324, grid.roles.size());
    EXPECT_EQ(144, grid.x_stab_coords.size());
    EXPECT_EQ(144, grid.z_stab_coords.size());
    EXPECT_EQ(144, grid.x_stab_indices.size());
    EXPECT_EQ(144, grid.z_stab_indices.size());
    EXPECT_EQ(289, grid.data_coords.size());
    EXPECT_EQ(289, grid.data_indices.size());
    EXPECT_EQ(144, grid.x_stabilizers.size());
    EXPECT_EQ(144, grid.z_stabilizers.size());
  }
}

TEST(PCMUtilsTester, checkReorderPCMColumns) {
  std::vector<uint8_t> data = {
      0, 1, 0, 0, 1, 0, 0, 0, 1, /* row 0 */
      1, 0, 0, 1, 1, 0, 0, 0, 0, /* row 1 */
      0, 0, 1, 0, 1, 0, 1, 0, 0, /* row 2 */
      0, 0, 0, 1, 1, 0, 0, 1, 0, /* row 3 */
      0, 0, 0, 0, 1, 1, 1, 1, 1, /* row 4 */
  };
  cudaqx::tensor<uint8_t> pcm(std::vector<std::size_t>{5, 9});
  pcm.borrow(data.data());
  auto column_order = cudaq::qec::get_sorted_pcm_column_indices(pcm);
  const std::vector<std::uint32_t> expected_order = {1, 8, 4, 0, 3, 2, 6, 7, 5};
  EXPECT_EQ(column_order, expected_order);
  auto pcm_reordered = cudaq::qec::reorder_pcm_columns(pcm, column_order);

  const std::vector<std::vector<uint8_t>> expected_data = {
      {1, 1, 1, 0, 0, 0, 0, 0, 0}, /* row 0 */
      {0, 0, 1, 1, 1, 0, 0, 0, 0}, /* row 1 */
      {0, 0, 1, 0, 0, 1, 1, 0, 0}, /* row 2 */
      {0, 0, 1, 0, 1, 0, 0, 1, 0}, /* row 3 */
      {0, 1, 1, 0, 0, 0, 1, 1, 1}  /* row 4 */
  };

  // Compare expected data with reordered data
  for (std::size_t i = 0; i < pcm.shape()[0]; ++i)
    for (std::size_t j = 0; j < pcm_reordered.shape()[1]; ++j)
      EXPECT_EQ(expected_data[i][j], pcm_reordered.at({i, j}));

  // Now try the whole flow with sort_pcm_columns
  auto pcm_reordered2 = cudaq::qec::sort_pcm_columns(pcm);
  for (std::size_t i = 0; i < pcm_reordered2.shape()[0]; ++i)
    for (std::size_t j = 0; j < pcm_reordered2.shape()[1]; ++j)
      EXPECT_EQ(expected_data[i][j], pcm_reordered2.at({i, j}));
}

TEST(PCMUtilsTester, checkSimplifyPCM1) {
  // No simplification occurs here, but reordering occurs.
  std::vector<uint8_t> data = {
      0, 1, /* row 0 */
      1, 0  /* row 1 */
  };
  std::vector<double> weights = {0.5, 0.5};
  cudaqx::tensor<uint8_t> pcm(std::vector<std::size_t>{2, 2});
  pcm.borrow(data.data());
  auto column_order = cudaq::qec::get_sorted_pcm_column_indices(pcm);
  const std::vector<std::uint32_t> expected_order = {1, 0};
  EXPECT_EQ(column_order, expected_order);
  auto pcm_reordered = cudaq::qec::reorder_pcm_columns(pcm, column_order);
  auto [H_new, weights_new] = cudaq::qec::simplify_pcm(pcm_reordered, weights);
  std::vector<double> expected_weights = {0.5, 0.5};
  EXPECT_EQ(weights_new, expected_weights);
}

TEST(PCMUtilsTester, checkSimplifyPCM2) {
  // Simplification (combining columns) occurs here.
  std::vector<uint8_t> data = {
      0, 1, 0, /* row 0 */
      1, 0, 1  /* row 1 */
  };
  std::vector<double> weights = {0.1, 0.2, 0.3};
  cudaqx::tensor<uint8_t> pcm(std::vector<std::size_t>{2, 3});
  pcm.borrow(data.data());

  auto column_order = cudaq::qec::get_sorted_pcm_column_indices(pcm);
  const std::vector<std::uint32_t> expected_order = {1, 0, 2};
  EXPECT_EQ(column_order, expected_order);

  auto [H_new, weights_new] = cudaq::qec::simplify_pcm(pcm, weights);
  std::vector<double> expected_weights = {0.2, 0.1 + 0.3 - 2 * 0.1 * 0.3};
  for (std::size_t i = 0; i < weights_new.size(); ++i)
    EXPECT_NEAR(weights_new[i], expected_weights[i], 1e-6);
}

bool are_pcms_equal(const cudaqx::tensor<uint8_t> &a,
                    const cudaqx::tensor<uint8_t> &b) {
  if (a.rank() != 2 || b.rank() != 2) {
    throw std::runtime_error("PCM must be a 2D tensor");
  }
  if (a.shape() != b.shape())
    return false;
  for (std::size_t r = 0; r < a.shape()[0]; ++r)
    for (std::size_t c = 0; c < a.shape()[1]; ++c)
      if (a.at({r, c}) != b.at({r, c}))
        return false;
  return true;
}

void check_pcm_equality(const cudaqx::tensor<uint8_t> &a,
                        const cudaqx::tensor<uint8_t> &b,
                        bool use_assert = true) {
  if (a.rank() != 2 || b.rank() != 2) {
    throw std::runtime_error("PCM must be a 2D tensor");
  }
  ASSERT_EQ(a.shape(), b.shape());
  auto num_rows = a.shape()[0];
  auto num_cols = a.shape()[1];
  for (std::size_t r = 0; r < num_rows; ++r) {
    for (std::size_t c = 0; c < num_cols; ++c) {
      if (a.at({r, c}) != b.at({r, c})) {
        if (use_assert)
          ASSERT_EQ(a.at({r, c}), b.at({r, c}))
              << "a.at({" << r << ", " << c << "}) = " << a.at({r, c})
              << ", b.at({" << r << ", " << c << "}) = " << b.at({r, c})
              << "\n";
        else
          EXPECT_EQ(a.at({r, c}), b.at({r, c}))
              << "a.at({" << r << ", " << c << "}) = " << a.at({r, c})
              << ", b.at({" << r << ", " << c << "}) = " << b.at({r, c})
              << "\n";
      }
    }
  }
}

TEST(PCMUtilsTester, checkSparsePCM) {
  std::size_t n_rounds = 4;
  std::size_t n_errs_per_round = 30;
  std::size_t n_syndromes_per_round = 10;
  std::size_t n_cols = n_rounds * n_errs_per_round;
  std::size_t n_rows = n_rounds * n_syndromes_per_round;
  std::size_t weight = 3;
  cudaqx::tensor<uint8_t> pcm = cudaq::qec::generate_random_pcm(
      n_rounds, n_errs_per_round, n_syndromes_per_round, weight,
      std::mt19937_64(13));
  printf("--------------------------------\n");
  printf("Original PCM:\n");
  pcm.dump_bits();
  auto pcm2 = cudaq::qec::sort_pcm_columns(pcm);
  printf("--------------------------------\n");
  printf("Sorted PCM (without specifying num syndromes per round):\n");
  pcm2.dump_bits();
  pcm2 = cudaq::qec::sort_pcm_columns(pcm, n_syndromes_per_round);
  printf("--------------------------------\n");
  printf("Sorted PCM (with num syndromes per round specified):\n");
  pcm2.dump_bits();
  printf("--------------------------------\n");

  // Make sure that the first round where an error occurs is non-decreasing
  // and the last round where an error occurs is non-increasing.
  std::size_t prev_col_first_round = 0;
  std::size_t prev_col_last_round = 0;
  for (std::size_t c = 0; c < n_cols; ++c) {
    // Find the first and last row where an error occurs in this column.
    auto first_row = std::numeric_limits<std::size_t>::max();
    auto last_row = std::numeric_limits<std::size_t>::min();
    for (std::size_t r = 0; r < n_rows; ++r) {
      if (pcm2.at({r, c}) == 1) {
        first_row = std::min(first_row, r);
        last_row = std::max(last_row, r);
      }
    }
    // Convert rows to rounds.
    auto first_round = first_row / n_syndromes_per_round;
    auto last_round = last_row / n_syndromes_per_round;

    // Make sure we are not going backwards.
    ASSERT_GE(first_round, prev_col_first_round);
    ASSERT_GE(last_round, prev_col_last_round);

    // Save for the next iteration.
    prev_col_first_round = first_round;
    prev_col_last_round = last_round;
  }

  // Make sure that the sort is stable, regardless of the input order of the
  // columns.
  int num_tests = 10;
  std::mt19937_64 rng(/*seed=*/13);
  for (int iter = 0; iter < num_tests; ++iter) {
    std::vector<std::uint32_t> shuffle_vector(n_cols);
    std::iota(shuffle_vector.begin(), shuffle_vector.end(), 0);
    std::shuffle(shuffle_vector.begin(), shuffle_vector.end(), rng);
    auto pcm_shuffled = cudaq::qec::reorder_pcm_columns(pcm, shuffle_vector);

    for (std::size_t c = 0; c < n_cols; ++c)
      for (std::size_t r = 0; r < n_rows; ++r)
        pcm_shuffled.at({r, c}) = pcm.at({r, shuffle_vector[c]});
    auto pcm_sorted =
        cudaq::qec::sort_pcm_columns(pcm_shuffled, n_syndromes_per_round);
    check_pcm_equality(pcm2, pcm_sorted);
  }
}

TEST(PCMUtilsTester, checkGetPCMForRounds) {
  std::size_t n_rounds = 4;
  std::size_t n_errs_per_round = 30;
  std::size_t n_syndromes_per_round = 10;
  std::size_t n_cols = n_rounds * n_errs_per_round;
  std::size_t n_rows = n_rounds * n_syndromes_per_round;
  std::size_t weight = 3;

  cudaqx::tensor<uint8_t> pcm = cudaq::qec::generate_random_pcm(
      n_rounds, n_errs_per_round, n_syndromes_per_round, weight,
      std::mt19937_64(13));

  pcm = cudaq::qec::sort_pcm_columns(pcm, n_syndromes_per_round);
  auto [pcm_for_rounds, first_column, last_column] =
      cudaq::qec::get_pcm_for_rounds(pcm, n_syndromes_per_round, 0,
                                     n_rounds - 1);
  check_pcm_equality(pcm_for_rounds, pcm);

  // Try all possible combinations of start and end rounds.
  for (int start_round = 0; start_round < n_rounds; ++start_round) {
    for (int end_round = start_round; end_round < n_rounds; ++end_round) {
      auto [pcm_test, first_column_test, last_column_test] =
          cudaq::qec::get_pcm_for_rounds(pcm, n_syndromes_per_round,
                                         start_round, end_round);
      // I don't have a good test criteria for this yet. It mainly just runs to
      // see if it runs without errors.
      printf("pcm_test for start_round = %u, end_round = %u:\n", start_round,
             end_round);
      pcm_test.dump_bits();
    }
  }
}

TEST(PCMUtilsTester, checkShufflePCMColumns) {
  std::size_t n_rounds = 4;
  std::size_t n_errs_per_round = 30;
  std::size_t n_syndromes_per_round = 10;
  std::size_t weight = 3;
  std::mt19937_64 rng(13);
  cudaqx::tensor<uint8_t> pcm = cudaq::qec::generate_random_pcm(
      n_rounds, n_errs_per_round, n_syndromes_per_round, weight,
      std::move(rng));
  pcm = cudaq::qec::sort_pcm_columns(pcm, n_syndromes_per_round);
  auto pcm_permuted = cudaq::qec::shuffle_pcm_columns(pcm, std::move(rng));
  // Verify that the new PCM is different from the original.
  EXPECT_FALSE(are_pcms_equal(pcm, pcm_permuted));
  // Verify that the resorted permutedPCM is the same as the original.
  auto pcm_permuted_and_sorted =
      cudaq::qec::sort_pcm_columns(pcm_permuted, n_syndromes_per_round);
  check_pcm_equality(pcm_permuted_and_sorted, pcm);
}

TEST(QECCodeTester, checkVersion) {
  std::string version = cudaq::qec::getVersion();
  EXPECT_FALSE(version.empty());
  EXPECT_TRUE(version.find("CUDAQX_QEC_VERSION") == std::string::npos);

  std::string fullVersion = cudaq::qec::getFullRepositoryVersion();
  EXPECT_TRUE(fullVersion.find("NVIDIA/cudaqx") != std::string::npos);
  EXPECT_TRUE(fullVersion.find("CUDAQX_SOLVERS_COMMIT_SHA") ==
              std::string::npos);
}
