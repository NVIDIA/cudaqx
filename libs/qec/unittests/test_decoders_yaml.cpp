/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/qec/decoder.h"
#include "cudaq/qec/pcm_utils.h"
#include "cudaq/qec/realtime/decoding_config.h"
#include <cmath>
#include <gtest/gtest.h>

/// Helper function to test that a decoder configuration can be serialized to
/// and from YAML.
void test_decoder_yaml_roundtrip(
    cudaq::qec::decoding::config::multi_decoder_config &multi_config) {
  // Serialize to YAML
  std::string config_str = multi_config.to_yaml_str(200);
  // Deserialize from YAML
  auto multi_config_from_yaml =
      cudaq::qec::decoding::config::multi_decoder_config::from_yaml_str(
          config_str);
  // And now serialize the deserialized configuration back to YAML, just for
  // good measure.
  std::string round_trip_config_str = multi_config_from_yaml.to_yaml_str(200);
  // Validate
  bool matchStrings = round_trip_config_str == config_str;
  bool matchConfigs = multi_config_from_yaml == multi_config;
  EXPECT_TRUE(matchStrings);
  EXPECT_TRUE(matchConfigs);

  // Retain for debug:
  // if (!matchStrings || !matchConfigs) {
  //   std::cout << "Orig config string: " << config_str << std::endl;
  //   std::cout << "Round trip config string: " <<
  //   multi_config_from_yaml.to_yaml_str(200) << std::endl;
  // }
}

/// Helper function to create and finalize a decoder configuration.
void test_decoder_creation(
    cudaq::qec::decoding::config::multi_decoder_config &multi_config) {
  int status = cudaq::qec::decoding::config::configure_decoders(multi_config);
  EXPECT_EQ(status, 0);
  cudaq::qec::decoding::config::finalize_decoders();
}

/// Helper function to create a sample, skeleton test decoder configuration for
/// a single error LUT decoder.
cudaq::qec::decoding::config::decoder_config
create_test_empty_decoder_config(int id) {
  cudaq::qec::decoding::config::decoder_config config;
  config.id = id;
  config.type = "single_error_lut";
  config.block_size = 20;
  config.syndrome_size = 10;
  cudaqx::tensor<uint8_t> H({config.syndrome_size, config.block_size});
  cudaqx::tensor<uint8_t> O({2, config.block_size});
  config.H_sparse = cudaq::qec::pcm_to_sparse_vec(H);
  config.O_sparse = cudaq::qec::pcm_to_sparse_vec(O);
  config.D_sparse = cudaq::qec::generate_timelike_sparse_detector_matrix(
      config.syndrome_size, 2, /*include_first_round=*/false);
  return config;
}

/// Helper function to create a sample, skeleton test decoder configuration for
/// the NV-QLDPC decoder.
cudaq::qec::decoding::config::decoder_config
create_test_decoder_config_nv_qldpc(int id) {
  cudaq::qec::decoding::config::decoder_config config =
      create_test_empty_decoder_config(id);
  config.type = "nv-qldpc-decoder";

  config.decoder_custom_args =
      cudaq::qec::decoding::config::nv_qldpc_decoder_config();
  auto &nv_config =
      std::get<cudaq::qec::decoding::config::nv_qldpc_decoder_config>(
          config.decoder_custom_args);
  nv_config.use_sparsity = true;
  nv_config.max_iterations = 50;
  nv_config.use_osd = true;
  nv_config.osd_order = 60;
  nv_config.osd_method = 3;
  nv_config.error_rate_vec =
      std::vector<cudaq::qec::float_t>(config.block_size, 0.1);

  nv_config.n_threads = 128;
  nv_config.bp_batch_size = 1;
  nv_config.osd_batch_size = 16;
  nv_config.iter_per_check = 2;
  nv_config.clip_value = 10.0;
  nv_config.bp_method = 3;
  nv_config.scale_factor = 1.0;
  nv_config.proc_float = "fp64";
  nv_config.gamma0 = 0.0;
  nv_config.gamma_dist = {0.1, 0.2};
  nv_config.srelay_config = cudaq::qec::decoding::config::srelay_bp_config();
  nv_config.srelay_config->pre_iter = 5;
  nv_config.srelay_config->num_sets = 10;
  nv_config.srelay_config->stopping_criterion = "NConv";
  nv_config.srelay_config->stop_nconv = 10;
  // explicit_gammas must have num_sets rows (10 in this case)
  nv_config.explicit_gammas = std::vector<std::vector<cudaq::qec::float_t>>(
      10, std::vector<cudaq::qec::float_t>(config.block_size, 0.1));
  nv_config.bp_seed = 42;
  nv_config.composition = 1;

  return config;
}

bool is_nv_qldpc_decoder_available() {
  try {
    std::size_t block_size = 7;
    std::size_t syndrome_size = 3;
    cudaqx::tensor<uint8_t> H;
    // clang-format off
    std::vector<uint8_t> H_vec = {1, 0, 0, 1, 0, 1, 1,
                                  0, 1, 0, 1, 1, 0, 1,
                                  0, 0, 1, 0, 1, 1, 1};
    // clang-format on
    H.copy(H_vec.data(), {syndrome_size, block_size});

    auto d = cudaq::qec::decoder::get("nv-qldpc-decoder", H);
    return true;
  } catch (const std::exception &e) {
    return false;
  }
}

TEST(DecoderYAMLTest, SingleDecoder) {
  if (!is_nv_qldpc_decoder_available()) {
    GTEST_SKIP() << "nv-qldpc-decoder is not available";
  }
  cudaq::qec::decoding::config::multi_decoder_config multi_config;
  cudaq::qec::decoding::config::decoder_config config =
      create_test_decoder_config_nv_qldpc(0);
  multi_config.decoders.push_back(config);

  test_decoder_yaml_roundtrip(multi_config);
  test_decoder_creation(multi_config);
}

TEST(DecoderYAMLTest, MultiDecoder) {
  if (!is_nv_qldpc_decoder_available()) {
    GTEST_SKIP() << "nv-qldpc-decoder is not available";
  }
  cudaq::qec::decoding::config::multi_decoder_config multi_config;
  cudaq::qec::decoding::config::decoder_config config1 =
      create_test_decoder_config_nv_qldpc(0);
  cudaq::qec::decoding::config::decoder_config config2 =
      create_test_decoder_config_nv_qldpc(1);
  multi_config.decoders.push_back(config1);
  multi_config.decoders.push_back(config2);

  test_decoder_yaml_roundtrip(multi_config);
  test_decoder_creation(multi_config);
}

TEST(DecoderYAMLTest, MultiLUTDecoder) {
  cudaq::qec::decoding::config::multi_decoder_config multi_config;
  cudaq::qec::decoding::config::decoder_config config =
      create_test_empty_decoder_config(0);
  config.type = "multi_error_lut";
  config.decoder_custom_args =
      cudaq::qec::decoding::config::multi_error_lut_config();
  auto &lut_config =
      std::get<cudaq::qec::decoding::config::multi_error_lut_config>(
          config.decoder_custom_args);
  lut_config.lut_error_depth = 2;
  multi_config.decoders.push_back(config);

  test_decoder_yaml_roundtrip(multi_config);
  test_decoder_creation(multi_config);
}

TEST(DecoderYAMLTest, SingleLUTDecoder) {
  cudaq::qec::decoding::config::multi_decoder_config multi_config;
  cudaq::qec::decoding::config::decoder_config config =
      create_test_empty_decoder_config(0);
  config.type = "single_error_lut";
  config.decoder_custom_args =
      cudaq::qec::decoding::config::single_error_lut_config();
  multi_config.decoders.push_back(config);

  test_decoder_yaml_roundtrip(multi_config);
  test_decoder_creation(multi_config);
}

TEST(DecoderYAMLTest, SlidingWindowDecoder) {
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

  // Top-level decoder config
  cudaq::qec::decoding::config::multi_decoder_config multi_config;
  cudaq::qec::decoding::config::decoder_config config =
      create_test_empty_decoder_config(0);
  config.type = "sliding_window";
  config.block_size = n_cols;
  config.syndrome_size = n_rows;

  // Sliding window config
  config.decoder_custom_args =
      cudaq::qec::decoding::config::sliding_window_config();
  auto &sw_config =
      std::get<cudaq::qec::decoding::config::sliding_window_config>(
          config.decoder_custom_args);
  config.H_sparse = cudaq::qec::pcm_to_sparse_vec(pcm);
  config.O_sparse =
      cudaq::qec::pcm_to_sparse_vec(cudaqx::tensor<uint8_t>({2, n_cols}));
  config.D_sparse = cudaq::qec::generate_timelike_sparse_detector_matrix(
      config.syndrome_size, 2, /*include_first_round=*/false);
  sw_config.window_size = 1;
  sw_config.step_size = 1;
  sw_config.num_syndromes_per_round = n_syndromes_per_round;
  sw_config.straddle_start_round = false;
  sw_config.straddle_end_round = true;
  sw_config.error_rate_vec =
      std::vector<cudaq::qec::float_t>(config.block_size, 0.1);

  // Inner decoder config
  sw_config.inner_decoder_name = "multi_error_lut";
  sw_config.multi_error_lut_params =
      cudaq::qec::decoding::config::multi_error_lut_config();
  sw_config.multi_error_lut_params->lut_error_depth = 2;

  multi_config.decoders.push_back(config);

  test_decoder_yaml_roundtrip(multi_config);
  test_decoder_creation(multi_config);
}
