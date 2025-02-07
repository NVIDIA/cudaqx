/*******************************************************************************
 * Copyright (c) 2024 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <cmath>
#include <gtest/gtest.h>

#include "cudaq.h"

#include "cudaq/qec/codes/surface_code.h"
#include "cudaq/qec/experiments.h"

TEST(StabilizerTester, checkConstructFromSpinOps) {
  {
    // Constructor will always auto sort
    std::vector<cudaq::spin_op> stab{cudaq::spin_op::from_word("ZZZZIII"),
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
    std::vector<cudaq::spin_op> ops;
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
  auto repetition = cudaq::qec::get_code("repetition", {{"distance", 9}});

  {
    auto stabilizers = repetition->get_stabilizers();

    std::vector<std::string> actual_stabs;
    for (auto &s : stabilizers)
      actual_stabs.push_back(s.to_string(false));

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
    auto surf_code = cudaq::qec::get_code("surface_code", {{"distance", 3}});
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
  auto repetition = cudaq::qec::get_code("repetition", {{"distance", 9}});
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
  auto surf_code = cudaq::qec::get_code("surface_code", {{"distance", 3}});
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
