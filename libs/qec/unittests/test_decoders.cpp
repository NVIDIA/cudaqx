/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/qec/decoder.h"
#include <cmath>
#include <future>
#include <gtest/gtest.h>
#include <random>

TEST(DecoderUtils, CovertHardToSoft) {
  std::vector<int> in = {1, 0, 1, 1};
  std::vector<float> out;
  std::vector<float> expected_out = {1.0, 0.0, 1.0, 1.0};

  cudaq::qec::convert_vec_hard_to_soft(in, out);
  ASSERT_EQ(out.size(), expected_out.size());
  for (int i = 0; i < out.size(); i++)
    ASSERT_EQ(out[i], expected_out[i]);

  expected_out = {0.9, 0.1, 0.9, 0.9};
  cudaq::qec::convert_vec_hard_to_soft(in, out, 0.9f, 0.1f);
  ASSERT_EQ(out.size(), expected_out.size());
  for (int i = 0; i < out.size(); i++)
    ASSERT_EQ(out[i], expected_out[i]);

  std::vector<std::vector<int>> in2 = {{1, 0}, {0, 1}};
  std::vector<std::vector<double>> out2;
  std::vector<std::vector<double>> expected_out2 = {{0.9, 0.1}, {0.1, 0.9}};
  cudaq::qec::convert_vec_hard_to_soft(in2, out2, 0.9, 0.1);
  for (int r = 0; r < out2.size(); r++) {
    ASSERT_EQ(out2.size(), expected_out2.size());
    for (int c = 0; c < out2.size(); c++)
      ASSERT_EQ(out2[r][c], expected_out2[r][c]);
  }
}

TEST(DecoderUtils, CovertSoftToHard) {
  std::vector<float> in = {0.6, 0.4, 0.7, 0.8};
  std::vector<bool> out;
  std::vector<bool> expected_out = {true, false, true, true};

  cudaq::qec::convert_vec_soft_to_hard(in, out);
  ASSERT_EQ(out.size(), expected_out.size());
  for (int i = 0; i < out.size(); i++)
    ASSERT_EQ(out[i], expected_out[i]);

  expected_out = {true, true, true, true};
  cudaq::qec::convert_vec_soft_to_hard(in, out, 0.4f);
  ASSERT_EQ(out.size(), expected_out.size());
  for (int i = 0; i < out.size(); i++)
    ASSERT_EQ(out[i], expected_out[i]);

  std::vector<std::vector<double>> in2 = {{0.6, 0.4}, {0.7, 0.8}};
  std::vector<std::vector<int>> out2;
  std::vector<std::vector<int>> expected_out2 = {{1, 0}, {1, 1}};
  cudaq::qec::convert_vec_soft_to_hard(in2, out2);
  for (int r = 0; r < out2.size(); r++) {
    ASSERT_EQ(out2.size(), expected_out2.size());
    for (int c = 0; c < out2.size(); c++)
      ASSERT_EQ(out2[r][c], expected_out2[r][c]);
  }
}

TEST(DecoderUtils, ConvertVecSoftToTensorHard) {
  // Generate a million random floats between 0 and 1 using mt19937
  std::mt19937_64 gen(13);
  std::uniform_real_distribution<> dis(0.0, 1.0);
  std::vector<double> in(1000000);
  for (int i = 0; i < in.size(); i++)
    in[i] = dis(gen);

  // Test the conversion to a tensor
  cudaqx::tensor<uint8_t> out_tensor;
  auto t0 = std::chrono::high_resolution_clock::now();
  cudaq::qec::convert_vec_soft_to_tensor_hard(in, out_tensor);
  auto t1 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = t1 - t0;
  std::cout << "Time taken for cudaqx::tensor: " << diff.count() * 1000.0
            << "ms" << std::endl;

  // Use the conversion to a vector as a baseline
  std::vector<uint8_t> out_vec(in.size());
  t0 = std::chrono::high_resolution_clock::now();
  cudaq::qec::convert_vec_soft_to_hard(in, out_vec);
  t1 = std::chrono::high_resolution_clock::now();
  diff = t1 - t0;
  std::cout << "Time taken for std::vector: " << diff.count() * 1000.0 << "ms"
            << std::endl;

  // Check the results are the same
  for (std::size_t i = 0; i < in.size(); i++)
    ASSERT_EQ(out_tensor.at({i}), out_vec[i]);
}

TEST(SampleDecoder, checkAPI) {
  using cudaq::qec::float_t;

  std::size_t block_size = 10;
  std::size_t syndrome_size = 4;
  cudaqx::tensor<uint8_t> H({syndrome_size, block_size});
  auto d = cudaq::qec::decoder::get("sample_decoder", H);
  std::vector<float_t> syndromes(syndrome_size);
  auto dec_result = d->decode(syndromes);
  ASSERT_EQ(dec_result.result.size(), block_size);
  for (auto x : dec_result.result)
    ASSERT_EQ(x, 0.0f);

  // Async test
  dec_result = d->decode_async(syndromes).get();
  ASSERT_EQ(dec_result.result.size(), block_size);
  for (auto x : dec_result.result)
    ASSERT_EQ(x, 0.0f);

  // Test the move constructor and move assignment operator

  // Multi test
  auto dec_results = d->decode_batch({syndromes, syndromes});
  ASSERT_EQ(dec_results.size(), 2);
  for (auto &m : dec_results)
    for (auto x : m.result)
      ASSERT_EQ(x, 0.0f);
}

TEST(SteaneLutDecoder, checkAPI) {
  using cudaq::qec::float_t;

  // Use Hx from the [7,1,3] Steane code from
  // https://en.wikipedia.org/wiki/Steane_code.
  std::size_t block_size = 7;
  std::size_t syndrome_size = 3;
  cudaqx::heterogeneous_map custom_args;

  std::vector<uint8_t> H_vec = {0, 0, 0, 1, 1, 1, 1,  // IIIXXXX
                                0, 1, 1, 0, 0, 1, 1,  // IXXIIXX
                                1, 0, 1, 0, 1, 0, 1}; // XIXIXIX
  cudaqx::tensor<uint8_t> H;
  H.copy(H_vec.data(), {syndrome_size, block_size});
  auto d = cudaq::qec::decoder::get("single_error_lut", H, custom_args);

  // Run decoding on all possible syndromes.
  const std::size_t num_syndromes_to_check = 1 << syndrome_size;
  bool convergeTrueFound = false;
  bool convergeFalseFound = false;
  assert(syndrome_size <= 64); // Assert due to "1 << bit" below.
  for (std::size_t syn_idx = 0; syn_idx < num_syndromes_to_check; syn_idx++) {
    // Construct a syndrome.
    std::vector<float_t> syndrome(syndrome_size, 0.0);
    for (int bit = 0; bit < syndrome_size; bit++)
      if (syn_idx & (1 << bit))
        syndrome[bit] = 1.0;

    // Perform decoding.
    auto dec_result = d->decode(syndrome);

    // Check results.
    ASSERT_EQ(dec_result.result.size(), block_size);
    const auto printResults = true;
    if (printResults) {
      std::string syndrome_str(syndrome_size, '0');
      for (std::size_t j = 0; j < syndrome_size; j++)
        if (syndrome[j] >= 0.5)
          syndrome_str[j] = '1';
      std::cout << "Syndrome " << syndrome_str
                << " returned: {converged: " << dec_result.converged
                << ", result: {";
      for (std::size_t j = 0; j < block_size; j++) {
        std::cout << dec_result.result[j];
        if (j < block_size - 1)
          std::cout << ",";
        else
          std::cout << "}}\n";
      }
    }
    convergeTrueFound |= dec_result.converged;
    convergeFalseFound |= !dec_result.converged;
  }
  ASSERT_TRUE(convergeTrueFound);
  ASSERT_FALSE(convergeFalseFound);
}

TEST(AsyncDecoderResultTest, MoveConstructorTransfersFuture) {
  std::promise<cudaq::qec::decoder_result> promise;
  std::future<cudaq::qec::decoder_result> future = promise.get_future();

  cudaq::qec::async_decoder_result original(std::move(future));
  EXPECT_TRUE(original.fut.valid());

  cudaq::qec::async_decoder_result moved(std::move(original));
  EXPECT_TRUE(moved.fut.valid());
  EXPECT_FALSE(original.fut.valid());
}

TEST(AsyncDecoderResultTest, MoveAssignmentTransfersFuture) {
  std::promise<cudaq::qec::decoder_result> promise;
  std::future<cudaq::qec::decoder_result> future = promise.get_future();

  cudaq::qec::async_decoder_result first(std::move(future));
  cudaq::qec::async_decoder_result second = std::move(first);

  EXPECT_TRUE(second.fut.valid());
  EXPECT_FALSE(first.fut.valid());
}

TEST(DecoderResultTest, DefaultConstructor) {
  cudaq::qec::decoder_result result;
  EXPECT_FALSE(result.converged);
  EXPECT_TRUE(result.result.empty());
  EXPECT_FALSE(result.opt_results.has_value());
}

TEST(DecoderResultTest, OptResultsAssignment) {
  cudaq::qec::decoder_result result;
  cudaqx::heterogeneous_map opt_map;
  opt_map.insert("test_key", 42);
  result.opt_results = opt_map;

  EXPECT_TRUE(result.opt_results.has_value());
  EXPECT_EQ(result.opt_results->get<int>("test_key"), 42);
}

TEST(DecoderResultTest, EqualityOperator) {
  cudaq::qec::decoder_result result1;
  cudaq::qec::decoder_result result2;

  // Test equality with no opt_results
  EXPECT_TRUE(result1 == result2);

  // Test inequality when one has opt_results
  cudaqx::heterogeneous_map opt_map;
  opt_map.insert("test_key", 42);
  result1.opt_results = opt_map;
  EXPECT_FALSE(result1 == result2);

  // Test inequality when both have opt_results (even if same)
  result2.opt_results = opt_map;
  EXPECT_FALSE(result1 == result2);
}
