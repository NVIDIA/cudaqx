/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/qec/decoder.h"
#include <cmath>
#include <future>
#include <gtest/gtest.h>

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
