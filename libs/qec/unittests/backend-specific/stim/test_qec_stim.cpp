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

#include "cudaq/qec/decoder.h"
#include "cudaq/qec/experiments.h"

TEST(QECCodeTester, checkRepetitionNoiseStim) {

  auto repetition = cudaq::qec::get_code(
      "repetition", cudaqx::heterogeneous_map{{"distance", 9}});
  {
    cudaq::set_random_seed(13);
    cudaq::noise_model noise;
    noise.add_all_qubit_channel("x", cudaq::qec::two_qubit_bitflip(0.1),
                                /*num_controls=*/1);

    auto [syndromes, d] =
        cudaq::qec::sample_memory_circuit(*repetition, 2, 2, noise);
    printf("syndrome\n");
    syndromes.dump();
    printf("data\n");
    d.dump();
    EXPECT_EQ(syndromes.shape()[0], 4);
    EXPECT_EQ(syndromes.shape()[1], 8);

    // Should have some 1s since it's noisy
    int sum = 0;
    for (std::size_t i = 0; i < 2; i++)
      for (std::size_t j = 0; j < 8; j++)
        sum += syndromes.at({i, j});

    EXPECT_TRUE(sum > 0);
  }
  {
    cudaq::set_random_seed(13);
    cudaq::noise_model noise;
    noise.add_all_qubit_channel("x", cudaq::qec::two_qubit_depolarization(0.1),
                                /*num_controls=*/1);

    auto [syndromes, d] =
        cudaq::qec::sample_memory_circuit(*repetition, 2, 2, noise);
    printf("syndrome\n");
    syndromes.dump();
    printf("data\n");
    d.dump();

    // Should have some 1s since it's noisy
    int sum = 0;
    for (std::size_t i = 0; i < 2; i++)
      for (std::size_t j = 0; j < 8; j++)
        sum += syndromes.at({i, j});

    EXPECT_TRUE(sum > 0);
  }
}

TEST(QECCodeTester, checkSteaneNoiseStim) {

  auto steane = cudaq::qec::get_code("steane");
  int nShots = 10;
  int nRounds = 3;
  {
    // two qubit bitflip
    cudaq::set_random_seed(13);
    cudaq::noise_model noise;
    noise.add_all_qubit_channel("x", cudaq::qec::two_qubit_bitflip(0.1), 1);

    auto [syndromes, d] =
        cudaq::qec::sample_memory_circuit(*steane, nShots, nRounds, noise);
    printf("syndrome\n");
    syndromes.dump();
    printf("data\n");
    d.dump();
    EXPECT_EQ(syndromes.shape()[0], nShots * nRounds);
    EXPECT_EQ(syndromes.shape()[1], 6);

    // Should have some 1s since it's noisy
    // bitflip should only trigger x syndromes
    int x_sum = 0;
    int z_sum = 0;

    for (std::size_t i = 0; i < syndromes.shape()[0]; i++) {
      for (std::size_t j_x = 0; j_x < syndromes.shape()[1] / 2; j_x++) {
        x_sum += syndromes.at({i, j_x});
      }
      for (std::size_t j_z = syndromes.shape()[1] / 2;
           j_z < syndromes.shape()[1]; j_z++) {
        z_sum += syndromes.at({i, j_z});
      }
    }
    EXPECT_TRUE(x_sum > 0);
    EXPECT_TRUE(z_sum == 0);
  }
  {
    // two qubit depol
    cudaq::set_random_seed(13);
    cudaq::noise_model noise;
    noise.add_all_qubit_channel("x", cudaq::qec::two_qubit_depolarization(0.1),
                                1);

    auto [syndromes, d] =
        cudaq::qec::sample_memory_circuit(*steane, nShots, nRounds, noise);
    printf("syndrome\n");
    syndromes.dump();
    printf("data\n");
    d.dump();

    // Should have some 1s since it's noisy
    // depolarizing triggers x and z syndromes
    int x_sum = 0;
    int z_sum = 0;
    for (std::size_t i = 0; i < syndromes.shape()[0]; i++) {
      for (std::size_t j_x = 0; j_x < syndromes.shape()[1] / 2; j_x++) {
        x_sum += syndromes.at({i, j_x});
      }
      for (std::size_t j_z = syndromes.shape()[1] / 2;
           j_z < syndromes.shape()[1]; j_z++) {
        z_sum += syndromes.at({i, j_z});
      }
    }
    EXPECT_TRUE(x_sum > 0);
    EXPECT_TRUE(z_sum > 0);
  }
  {
    // one qubit bitflip
    cudaq::set_random_seed(13);
    cudaq::noise_model noise;
    noise.add_all_qubit_channel("h", cudaq::bit_flip_channel(0.1));

    auto [syndromes, d] = cudaq::qec::sample_memory_circuit(
        *steane, cudaq::qec::operation::prepp, nShots, nRounds, noise);
    printf("syndrome\n");
    syndromes.dump();
    printf("data\n");
    d.dump();

    // Should have some 1s since it's noisy
    // only getting detectible errors on s_z ancillas
    int x_sum = 0;
    int z_sum = 0;
    for (std::size_t i = 0; i < syndromes.shape()[0]; i++) {
      for (std::size_t j_x = 0; j_x < syndromes.shape()[1] / 2; j_x++) {
        x_sum += syndromes.at({i, j_x});
      }
      for (std::size_t j_z = syndromes.shape()[1] / 2;
           j_z < syndromes.shape()[1]; j_z++) {
        z_sum += syndromes.at({i, j_z});
      }
    }
    EXPECT_TRUE(x_sum > 0);
    EXPECT_TRUE(z_sum > 0);
  }
  {
    // one qubit phase
    cudaq::set_random_seed(13);
    cudaq::noise_model noise;
    noise.add_all_qubit_channel("h", cudaq::phase_flip_channel(0.1));

    auto [syndromes, d] = cudaq::qec::sample_memory_circuit(
        *steane, cudaq::qec::operation::prepp, nShots, nRounds, noise);
    printf("syndrome\n");
    syndromes.dump();
    printf("data\n");
    d.dump();

    // Should have some 1s since it's noisy
    // only getting detectible errors on s_z ancillas
    int x_sum = 0;
    int z_sum = 0;
    for (std::size_t i = 0; i < syndromes.shape()[0]; i++) {
      for (std::size_t j_x = 0; j_x < syndromes.shape()[1] / 2; j_x++) {
        x_sum += syndromes.at({i, j_x});
      }
      for (std::size_t j_z = syndromes.shape()[1] / 2;
           j_z < syndromes.shape()[1]; j_z++) {
        z_sum += syndromes.at({i, j_z});
      }
    }
    // Even though phase flip is a z error,
    // additional hadamards in prepp can convert to x error (first round only)
    EXPECT_TRUE(x_sum > 0);
    EXPECT_TRUE(z_sum > 0);
  }
  {
    // one qubit depol
    cudaq::set_random_seed(13);
    cudaq::noise_model noise;
    noise.add_all_qubit_channel("h", cudaq::depolarization_channel(0.1));

    auto [syndromes, d] = cudaq::qec::sample_memory_circuit(
        *steane, cudaq::qec::operation::prepp, nShots, nRounds, noise);
    printf("syndrome\n");
    syndromes.dump();
    printf("data\n");
    d.dump();

    // Should have some 1s since it's noisy
    // only getting detectible errors on s_z ancillas
    int x_sum = 0;
    int z_sum = 0;
    for (std::size_t i = 0; i < syndromes.shape()[0]; i++) {
      for (std::size_t j_x = 0; j_x < syndromes.shape()[1] / 2; j_x++) {
        x_sum += syndromes.at({i, j_x});
      }
      for (std::size_t j_z = syndromes.shape()[1] / 2;
           j_z < syndromes.shape()[1]; j_z++) {
        z_sum += syndromes.at({i, j_z});
      }
    }
    EXPECT_TRUE(x_sum > 0);
    EXPECT_TRUE(z_sum > 0);
  }
}

TEST(QECCodeTester, checkSampleMemoryCircuitStim) {
  {
    // Steane tests
    auto steane = cudaq::qec::get_code("steane");
    cudaqx::tensor<uint8_t> observables =
        steane->get_pauli_observables_matrix();
    cudaqx::tensor<uint8_t> Lx = steane->get_observables_x();
    cudaqx::tensor<uint8_t> Lz = steane->get_observables_z();

    int nShots = 10;
    int nRounds = 4;
    {
      auto [syndromes, d] = cudaq::qec::sample_memory_circuit(
          *steane, cudaq::qec::operation::prep0, nShots, nRounds);
      syndromes.dump();

      // No noise here, should be all zeros
      int sum = 0;
      for (std::size_t i = 0; i < syndromes.shape()[0]; i++)
        for (std::size_t j = 0; j < syndromes.shape()[1]; j++)
          sum += syndromes.at({i, j});
      EXPECT_TRUE(sum == 0);

      // Prep0, should measure out logical 0 each shot
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
      // Prep1, should measure out logical 1 each shot
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
      // Prepp, should measure out logical + each shot
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
      // Prepm, should measure out logical - each shot
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
    {
      cudaq::set_random_seed(13);
      cudaq::noise_model noise;
      noise.add_all_qubit_channel("x", cudaq::qec::two_qubit_bitflip(0.1), 1);

      nShots = 10;
      nRounds = 4;

      auto [syndromes, d] = cudaq::qec::sample_memory_circuit(
          *steane, cudaq::qec::operation::prep0, nShots, nRounds, noise);
      printf("syndromes:\n");
      syndromes.dump();

      // Noise here, expect a nonzero
      int sum = 0;
      for (std::size_t i = 0; i < syndromes.shape()[0]; i++)
        for (std::size_t j = 0; j < syndromes.shape()[1]; j++)
          sum += syndromes.at({i, j});
      EXPECT_TRUE(sum > 0);

      // With noise, Lz will sometimes be flipped
      printf("data:\n");
      d.dump();
      printf("Lz:\n");
      Lz.dump();
      cudaqx::tensor<uint8_t> logical_mz = Lz.dot(d.transpose()) % 2;
      printf("logical_mz:\n");
      logical_mz.dump();
      EXPECT_TRUE(logical_mz.any());
    }
  }
}

TEST(QECCodeTester, checkTwoQubitBitflipStim) {
  // This circuit should read out |00> with and without bitflip noise
  struct null1 {
    void operator()() __qpu__ {
      cudaq::qvector q(2);
      h(q);
      x<cudaq::ctrl>(q[0], q[1]);
      h(q);
    }
  };

  // This circuit should read out |00> without bitflip noise, and random values
  // with
  struct null2 {
    void operator()() __qpu__ {
      cudaq::qvector q(2);
      x<cudaq::ctrl>(q[0], q[1]);
    }
  };
  cudaq::set_random_seed(13);
  cudaq::noise_model noise;
  noise.add_all_qubit_channel("x", cudaq::qec::two_qubit_bitflip(0.1), 1);
  cudaq::set_noise(noise);

  auto counts1 = cudaq::sample(100, null1{});
  EXPECT_FLOAT_EQ(1.0, counts1.probability("00"));

  auto counts2 = cudaq::sample(100, null2{});
  EXPECT_TRUE(counts2.probability("00") < 0.9);
  cudaq::unset_noise();
}

TEST(QECCodeTester, checkBitflip) {
  // This circuit should read out |0> when noiseless
  struct null1 {
    void operator()() __qpu__ {
      cudaq::qubit q;
      h(q);
      h(q);
    }
  };

  auto counts1 = cudaq::sample(100, null1{});
  EXPECT_FLOAT_EQ(1.0, counts1.probability("0"));

  cudaq::set_random_seed(13);
  cudaq::noise_model noise;
  noise.add_all_qubit_channel("h", cudaq::bit_flip_channel(0.5));
  cudaq::set_noise(noise);
  auto counts2 = cudaq::sample(100, null1{});
  cudaq::unset_noise();
  EXPECT_TRUE(counts2.probability("0") < 0.9);
}

TEST(QECCodeTester, checkNoisySampleMemoryCircuitAndDecodeStim) {
  {
    // Steane tests
    auto steane = cudaq::qec::get_code("steane");
    cudaqx::tensor<uint8_t> H = steane->get_parity();
    cudaqx::tensor<uint8_t> observables =
        steane->get_pauli_observables_matrix();
    cudaqx::tensor<uint8_t> Lx = steane->get_observables_x();
    cudaqx::tensor<uint8_t> Lz = steane->get_observables_z();

    int nShots = 1;
    int nRounds = 10;
    cudaq::set_random_seed(13);
    cudaq::noise_model noise;
    noise.add_all_qubit_channel("x", cudaq::qec::two_qubit_depolarization(0.01),
                                1);

    auto [syndromes, d] = cudaq::qec::sample_memory_circuit(
        *steane, cudaq::qec::operation::prep0, nShots, nRounds, noise);
    printf("syndromes:\n");
    syndromes.dump();

    // Noise here, expect a nonzero
    int sum = 0;
    for (std::size_t i = 0; i < syndromes.shape()[0]; i++)
      for (std::size_t j = 0; j < syndromes.shape()[1]; j++)
        sum += syndromes.at({i, j});
    EXPECT_TRUE(sum > 0);

    // With noise, Lz will sometimes be flipped
    printf("data:\n");
    d.dump();
    printf("Lz:\n");
    Lz.dump();
    cudaqx::tensor<uint8_t> logical_mz = Lz.dot(d.transpose()) % 2;
    printf("logical_mz:\n");
    logical_mz.dump();

    // s = (sx | sz)
    // sx = Hz . ex
    // sz = Hx . ez

    printf("Obs:\n");
    observables.dump();
    auto decoder = cudaq::qec::get_decoder("single_error_lut", H);
    printf("Hz:\n");
    H.dump();
    printf("end\n");
    size_t numLerrors = 0;
    size_t stride = syndromes.shape()[1];
    cudaqx::tensor<uint8_t> pauli_frame({observables.shape()[0]});
    for (size_t i = 0; i < nRounds - 1; ++i) {
      cudaqx::tensor<uint8_t> syndrome({stride});
      syndrome.borrow(syndromes.data() + i * stride);
      printf("syndrome:\n");
      syndrome.dump();
      auto [converged, v_result] = decoder->decode(syndrome);
      cudaqx::tensor<uint8_t> result_tensor;
      cudaq::qec::convert_vec_soft_to_tensor_hard(v_result, result_tensor);
      printf("decode result:\n");
      result_tensor.dump();
      cudaqx::tensor<uint8_t> decoded_observables =
          observables.dot(result_tensor);
      printf("decoded observable:\n");
      decoded_observables.dump();
      pauli_frame = (pauli_frame + decoded_observables) % 2;
      printf("pauli frame:\n");
      pauli_frame.dump();
    }
    // prep0 means this is a z-basis experiment
    // Check if Lz + pauli_frame[0] = 0?
    printf("Lz: %d, xFlips: %d\n", Lz.at({0, 0}), pauli_frame.at({0}));
    if (Lz.at({0, 0}) != pauli_frame.at({0}))
      numLerrors++;
#ifdef __x86_64__
    // No logicals errors for this seed
    // TODO - find a comparable seed for ARM or modify test.
    EXPECT_EQ(0, numLerrors);
#endif
  }
  {
    // Test x-basis and x-flips
    auto steane = cudaq::qec::get_code("steane");
    cudaqx::tensor<uint8_t> H = steane->get_parity();
    cudaqx::tensor<uint8_t> Hx = steane->get_parity_x();
    cudaqx::tensor<uint8_t> Hz = steane->get_parity_z();
    cudaqx::tensor<uint8_t> observables =
        steane->get_pauli_observables_matrix();
    cudaqx::tensor<uint8_t> Lx = steane->get_observables_x();
    cudaqx::tensor<uint8_t> Lz = steane->get_observables_z();

    int nShots = 10;
    int nRounds = 4;
    cudaq::set_random_seed(13);
    cudaq::noise_model noise;
    noise.add_all_qubit_channel("x", cudaq::qec::two_qubit_bitflip(0.05), 1);

    // Bitflip is X-type error, detected by Z stabilizers (Hz)
    auto [syndromes, d] = cudaq::qec::sample_memory_circuit(
        *steane, cudaq::qec::operation::prepp, nShots, nRounds, noise);
    printf("syndromes:\n");
    syndromes.dump();
    EXPECT_EQ(syndromes.shape()[0], nShots * nRounds);
    EXPECT_EQ(syndromes.shape()[1], 6);

    // With noise, Lx will sometimes be flipped
    printf("data:\n");
    d.dump();
    printf("Lx:\n");
    Lx.dump();
    cudaqx::tensor<uint8_t> logical_mx = Lx.dot(d.transpose()) % 2;
    // Can make a column vector
    printf("logical_mx:\n");
    logical_mx.dump();
    // bit flip errors trigger Z-type stabilizers (ZZIII)
    // these will be extracted into the ancx syndrome registers
    // (s_x | s_z ) = ( X flip syndromes, Z Flip syndromes)

    cudaqx::tensor<uint8_t> final_sx = Hz.dot(d.transpose()) % 2;
    // If x basis experiment, this would be final sx
    printf("final sx:\n");
    final_sx.dump();

    printf("Obs:\n");
    observables.dump();
    auto decoder = cudaq::qec::get_decoder("single_error_lut", H);
    printf("end\n");
    size_t numLerrors = 0;
    size_t stride = syndromes.shape()[1];
    for (size_t shot = 0; shot < nShots; ++shot) {
      cudaqx::tensor<uint8_t> pauli_frame({observables.shape()[0]});
      for (size_t i = 0; i < nRounds; ++i) {
        size_t count = shot * nRounds + i;
        printf("shot: %zu, round: %zu, count: %zu\n", shot, i, count);
        cudaqx::tensor<uint8_t> syndrome({stride});
        syndrome.borrow(syndromes.data() + stride * count);
        printf("syndrome:\n");
        syndrome.dump();
        auto [converged, v_result] = decoder->decode(syndrome);
        cudaqx::tensor<uint8_t> result_tensor;
        cudaq::qec::convert_vec_soft_to_tensor_hard(v_result, result_tensor);

        printf("decode result:\n");
        result_tensor.dump();
        cudaqx::tensor<uint8_t> decoded_observables =
            observables.dot(result_tensor);
        printf("decoded observable:\n");
        decoded_observables.dump();
        pauli_frame = (pauli_frame + decoded_observables) % 2;
        printf("pauli frame:\n");
        pauli_frame.dump();
      }
      // prepp means this is a x-basis experiment
      // does LMx + pauli_frame[1] = |+>? (+ is read out as 0 after rotation)

      printf("Obs_x: %d, pfZ: %d\n", logical_mx.at({0, shot}),
             pauli_frame.at({1}));
      uint8_t corrected_obs =
          (logical_mx.at({0, shot}) + pauli_frame.at({1})) % 2;
      std::cout << "corrected_obs: " << +corrected_obs << "\n";
      if (corrected_obs != 0)
        numLerrors++;
    }
    printf("numLerrors: %zu\n", numLerrors);
    EXPECT_TRUE(numLerrors > 0);
  }
}
