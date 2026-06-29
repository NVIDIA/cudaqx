/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "../lib/realtime/realtime_decoding.h"
#include "cudaq/qec/decoder.h"
#include "cudaq/qec/pcm_utils.h"
#include "cudaq/qec/realtime/decoding_config.h"
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <optional>

namespace {
class ScopedEnv {
public:
  ScopedEnv(const char *name, const char *value) : name(name) {
    if (const char *old = std::getenv(name))
      oldValue = old;
    setenv(name, value, 1);
  }

  ~ScopedEnv() {
    if (oldValue.has_value())
      setenv(name.c_str(), oldValue->c_str(), 1);
    else
      unsetenv(name.c_str());
  }

private:
  std::string name;
  std::optional<std::string> oldValue;
};
} // namespace

namespace cudaq::qec::decoding::simulation {
void enqueue_syndromes(std::uint64_t decoder_id, uint8_t *syndromes,
                       std::uint64_t syndrome_length, std::uint64_t tag);
void get_corrections(std::uint64_t decoder_id, uint8_t *corrections,
                     std::uint64_t correction_length, bool reset);
} // namespace cudaq::qec::decoding::simulation

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

cudaq::qec::decoding::config::decoder_config
create_test_decoder_config_trt(int id) {
  cudaq::qec::decoding::config::decoder_config config =
      create_test_empty_decoder_config(id);
  config.type = "trt_decoder";

  cudaqx::tensor<uint8_t> O({2, config.block_size});
  O.at({0, 1}) = 1;
  O.at({1, 3}) = 1;
  config.O_sparse = cudaq::qec::pcm_to_sparse_vec(O);

  config.decoder_custom_args =
      cudaq::qec::decoding::config::trt_decoder_config();
  auto &trt_config = std::get<cudaq::qec::decoding::config::trt_decoder_config>(
      config.decoder_custom_args);
  trt_config.onnx_load_path = "/tmp/predecoder.onnx";
  trt_config.engine_save_path = "/tmp/predecoder.engine";
  trt_config.precision = "best";
  trt_config.memory_workspace = 1ULL << 20;
  trt_config.batch_size = 4;
  trt_config.use_cuda_graph = false;
  trt_config.global_decoder = "pymatching";
  auto pymatching_params = cudaq::qec::decoding::config::pymatching_config();
  pymatching_params.merge_strategy = "smallest_weight";
  pymatching_params.error_rate_vec =
      std::vector<double>(config.block_size, 0.1);
  trt_config.global_decoder_params = pymatching_params;

  return config;
}

TEST(DecoderYAMLTest, TrtDecoderConfigRoundTrip) {
  cudaq::qec::decoding::config::multi_decoder_config multi_config;
  multi_config.decoders.push_back(create_test_decoder_config_trt(0));

  test_decoder_yaml_roundtrip(multi_config);
  const auto &trt_config =
      std::get<cudaq::qec::decoding::config::trt_decoder_config>(
          multi_config.decoders[0].decoder_custom_args);
  EXPECT_TRUE(
      std::holds_alternative<cudaq::qec::decoding::config::pymatching_config>(
          trt_config.global_decoder_params));
}

TEST(DecoderYAMLTest, TrtDecoderConfigToHeterogeneousMap) {
  auto config = create_test_decoder_config_trt(0);
  auto params = config.decoder_custom_args_to_heterogeneous_map();

  EXPECT_EQ(params.get<std::string>("onnx_load_path"), "/tmp/predecoder.onnx");
  EXPECT_EQ(params.get<std::string>("engine_save_path"),
            "/tmp/predecoder.engine");
  EXPECT_EQ(params.get<std::string>("precision"), "best");
  EXPECT_EQ(params.get<std::size_t>("memory_workspace"), 1ULL << 20);
  EXPECT_EQ(params.get<std::size_t>("batch_size"), 4u);
  EXPECT_FALSE(params.get<bool>("use_cuda_graph"));
  EXPECT_EQ(params.get<std::string>("global_decoder"), "pymatching");

  auto global_params =
      params.get<cudaqx::heterogeneous_map>("global_decoder_params");
  EXPECT_EQ(global_params.get<std::string>("merge_strategy"),
            "smallest_weight");
  EXPECT_EQ(global_params.get<std::vector<double>>("error_rate_vec").size(),
            config.block_size);
}

TEST(DecoderYAMLTest, TrtDecoderRealtimeParamsIncludeObservableMatrix) {
  auto config = create_test_decoder_config_trt(0);
  auto params = cudaq::qec::decoding::host::prepare_decoder_params(config);

  auto O = params.get<cudaqx::tensor<uint8_t>>("O");
  EXPECT_EQ(O.shape()[0], 2u);
  EXPECT_EQ(O.shape()[1], config.block_size);
  EXPECT_EQ(O.at({0, 1}), 1);
  EXPECT_EQ(O.at({1, 3}), 1);

  auto global_params =
      params.get<cudaqx::heterogeneous_map>("global_decoder_params");
  auto global_O = global_params.get<cudaqx::tensor<uint8_t>>("O");
  EXPECT_EQ(global_O.shape()[0], 2u);
  EXPECT_EQ(global_O.shape()[1], config.block_size);
}

TEST(DecoderYAMLTest, TrtDecoderMonostateGlobalDecoderParams) {
  auto config = create_test_decoder_config_trt(0);
  auto &trt_config = std::get<cudaq::qec::decoding::config::trt_decoder_config>(
      config.decoder_custom_args);
  trt_config.global_decoder = "pymatching";
  trt_config.global_decoder_params = std::monostate{};

  auto params = config.decoder_custom_args_to_heterogeneous_map();
  EXPECT_TRUE(params.contains("global_decoder_params"));
  EXPECT_TRUE(
      params.get<cudaqx::heterogeneous_map>("global_decoder_params").empty());

  cudaq::qec::decoding::config::multi_decoder_config multi_config;
  multi_config.decoders.push_back(config);
  const auto yaml = multi_config.to_yaml_str(200);
  EXPECT_NE(yaml.find("global_decoder_params"), std::string::npos);
  auto round_tripped =
      cudaq::qec::decoding::config::multi_decoder_config::from_yaml_str(yaml);
  EXPECT_EQ(round_tripped.to_yaml_str(200), yaml);
  const auto &round_tripped_trt_config =
      std::get<cudaq::qec::decoding::config::trt_decoder_config>(
          round_tripped.decoders[0].decoder_custom_args);
  EXPECT_TRUE(
      std::holds_alternative<cudaq::qec::decoding::config::pymatching_config>(
          round_tripped_trt_config.global_decoder_params));

  params = cudaq::qec::decoding::host::prepare_decoder_params(config);
  EXPECT_TRUE(params.contains("global_decoder_params"));
  EXPECT_TRUE(params.contains("O"));

  config.O_sparse.clear();
  params = cudaq::qec::decoding::host::prepare_decoder_params(config);
  EXPECT_TRUE(params.contains("global_decoder_params"));
  EXPECT_FALSE(params.contains("O"));
}

TEST(DecoderYAMLTest, TrtDecoderDefaultGlobalDecoderParams) {
  cudaqx::heterogeneous_map map;
  map.insert("global_decoder", std::string("chromobius"));

  auto trt_config =
      cudaq::qec::decoding::config::trt_decoder_config::from_heterogeneous_map(
          map);
  EXPECT_TRUE(
      std::holds_alternative<cudaq::qec::decoding::config::chromobius_config>(
          trt_config.global_decoder_params));
  auto params = trt_config.to_heterogeneous_map();
  EXPECT_TRUE(params.contains("global_decoder_params"));
  EXPECT_TRUE(
      params.get<cudaqx::heterogeneous_map>("global_decoder_params").empty());

  const std::string yaml_without_params = R"(
decoders:
  - id: 0
    type: trt_decoder
    block_size: 1
    syndrome_size: 1
    H_sparse: [0, -1]
    O_sparse: []
    D_sparse: [0, -1]
    decoder_custom_args:
      global_decoder: chromobius
)";
  auto parsed_without_params =
      cudaq::qec::decoding::config::multi_decoder_config::from_yaml_str(
          yaml_without_params);
  const auto &parsed_trt_config =
      std::get<cudaq::qec::decoding::config::trt_decoder_config>(
          parsed_without_params.decoders[0].decoder_custom_args);
  EXPECT_TRUE(
      std::holds_alternative<cudaq::qec::decoding::config::chromobius_config>(
          parsed_trt_config.global_decoder_params));

  auto config = create_test_decoder_config_trt(0);
  auto &yaml_trt_config =
      std::get<cudaq::qec::decoding::config::trt_decoder_config>(
          config.decoder_custom_args);
  yaml_trt_config.global_decoder = "chromobius";
  yaml_trt_config.global_decoder_params = std::monostate{};
  cudaq::qec::decoding::config::multi_decoder_config multi_config;
  multi_config.decoders.push_back(config);
  const auto yaml = multi_config.to_yaml_str(200);
  EXPECT_NE(yaml.find("global_decoder_params"), std::string::npos);
  auto round_tripped =
      cudaq::qec::decoding::config::multi_decoder_config::from_yaml_str(yaml);
  const auto &round_tripped_trt_config =
      std::get<cudaq::qec::decoding::config::trt_decoder_config>(
          round_tripped.decoders[0].decoder_custom_args);
  EXPECT_TRUE(
      std::holds_alternative<cudaq::qec::decoding::config::chromobius_config>(
          round_tripped_trt_config.global_decoder_params));
}

TEST(DecoderYAMLTest, UnknownTrtGlobalDecoderParamsThrow) {
  cudaqx::heterogeneous_map map;
  map.insert("global_decoder", std::string("my_plugin"));
  map.insert("global_decoder_params", cudaqx::heterogeneous_map{});
  EXPECT_THROW(
      cudaq::qec::decoding::config::trt_decoder_config::from_heterogeneous_map(
          map),
      std::runtime_error);

  cudaq::qec::decoding::config::trt_decoder_config trt_config;
  trt_config.global_decoder = "my_plugin";
  auto params = trt_config.to_heterogeneous_map();
  EXPECT_EQ(params.get<std::string>("global_decoder"), "my_plugin");
  EXPECT_FALSE(params.contains("global_decoder_params"));

  map = cudaqx::heterogeneous_map();
  map.insert("global_decoder", std::string("my_plugin"));
  trt_config =
      cudaq::qec::decoding::config::trt_decoder_config::from_heterogeneous_map(
          map);
  EXPECT_TRUE(
      std::holds_alternative<std::monostate>(trt_config.global_decoder_params));

  const std::string yaml_with_unknown_params = R"(
decoders:
  - id: 0
    type: trt_decoder
    block_size: 1
    syndrome_size: 1
    H_sparse: [0, -1]
    O_sparse: []
    D_sparse: [0, -1]
    decoder_custom_args:
      global_decoder: my_plugin
      global_decoder_params: {}
)";
  EXPECT_THROW(
      cudaq::qec::decoding::config::multi_decoder_config::from_yaml_str(
          yaml_with_unknown_params),
      std::runtime_error);
}

TEST(DecoderYAMLTest, TrtDecoderParamsWithoutDecoderThrows) {
  cudaqx::heterogeneous_map map;
  map.insert("onnx_load_path", std::string("/tmp/predecoder.onnx"));
  cudaqx::heterogeneous_map gd_params;
  gd_params.insert("merge_strategy", std::string("smallest_weight"));
  map.insert("global_decoder_params", gd_params);
  EXPECT_THROW(
      cudaq::qec::decoding::config::trt_decoder_config::from_heterogeneous_map(
          map),
      std::runtime_error);

  cudaq::qec::decoding::config::trt_decoder_config trt_config;
  trt_config.onnx_load_path = "/tmp/predecoder.onnx";
  auto pymatching_params = cudaq::qec::decoding::config::pymatching_config();
  pymatching_params.merge_strategy = "smallest_weight";
  trt_config.global_decoder_params = pymatching_params;
  EXPECT_THROW(trt_config.to_heterogeneous_map(), std::runtime_error);
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

TEST(DecoderConfigMapTest, SRelayNvQldpcAndTrtRoundTrip) {
  using namespace cudaq::qec::decoding::config;

  srelay_bp_config relay;
  relay.pre_iter = 3;
  relay.num_sets = 5;
  relay.stopping_criterion = "NConv";
  relay.stop_nconv = 2;

  auto relay_map = relay.to_heterogeneous_map();
  auto relay_from_map = srelay_bp_config::from_heterogeneous_map(relay_map);
  EXPECT_EQ(relay_from_map, relay);

  nv_qldpc_decoder_config nv;
  nv.use_sparsity = true;
  nv.error_rate = 0.02;
  nv.error_rate_vec = std::vector<double>{0.1, 0.2, 0.3};
  nv.max_iterations = 8;
  nv.n_threads = 4;
  nv.use_osd = true;
  nv.osd_method = 1;
  nv.osd_order = 2;
  nv.bp_batch_size = 16;
  nv.osd_batch_size = 8;
  nv.iter_per_check = 3;
  nv.clip_value = 9.5;
  nv.bp_method = 2;
  nv.scale_factor = 0.75;
  nv.proc_float = "fp64";
  nv.gamma0 = 0.4;
  nv.gamma_dist = std::vector<double>{0.1, 0.2};
  nv.explicit_gammas = std::vector<std::vector<double>>{{0.1, 0.2}, {0.3, 0.4}};
  nv.srelay_config = relay;
  nv.bp_seed = 13;
  nv.composition = 1;

  auto nv_map = nv.to_heterogeneous_map();
  auto nv_from_map = nv_qldpc_decoder_config::from_heterogeneous_map(nv_map);
  EXPECT_EQ(nv_from_map, nv);

  cudaqx::heterogeneous_map nested_relay_map;
  nested_relay_map.insert("pre_iter", std::size_t{7});
  nested_relay_map.insert("num_sets", std::size_t{9});
  nested_relay_map.insert("stopping_criterion", std::string("RelErr"));
  nested_relay_map.insert("stop_nconv", std::size_t{4});
  cudaqx::heterogeneous_map nv_with_nested_relay;
  nv_with_nested_relay.insert("srelay_config", nested_relay_map);
  auto nv_from_nested =
      nv_qldpc_decoder_config::from_heterogeneous_map(nv_with_nested_relay);
  ASSERT_TRUE(nv_from_nested.srelay_config.has_value());
  EXPECT_EQ(nv_from_nested.srelay_config->pre_iter, std::size_t{7});
  EXPECT_EQ(nv_from_nested.srelay_config->num_sets, std::size_t{9});

  trt_decoder_config trt;
  trt.onnx_load_path = "/tmp/model.onnx";
  trt.engine_save_path = "/tmp/model.engine";
  trt.precision = "noTF32";
  trt.memory_workspace = std::size_t{4096};
  auto trt_map = trt.to_heterogeneous_map();
  auto trt_from_map = trt_decoder_config::from_heterogeneous_map(trt_map);
  EXPECT_EQ(trt_from_map, trt);
}

TEST(DecoderConfigMapTest, DecoderCustomArgsCoversNvQldpcAndTrtVariants) {
  using namespace cudaq::qec::decoding::config;

  decoder_config nv_decoder;
  nv_decoder.type = "nv-qldpc-decoder";
  nv_qldpc_decoder_config nv_args;
  nv_args.max_iterations = 11;
  nv_args.error_rate_vec = std::vector<double>{0.1, 0.1};
  nv_decoder.decoder_custom_args = nv_args;
  auto nv_map = nv_decoder.decoder_custom_args_to_heterogeneous_map();
  EXPECT_TRUE(nv_map.contains("max_iterations"));
  EXPECT_EQ(nv_map.get<int>("max_iterations"), 11);

  decoder_config trt_decoder;
  trt_decoder.type = "trt_decoder";
  trt_decoder_config trt_args;
  trt_args.engine_load_path = "/tmp/model.engine";
  trt_args.precision = "tf32";
  trt_decoder.decoder_custom_args = trt_args;
  auto trt_map = trt_decoder.decoder_custom_args_to_heterogeneous_map();
  EXPECT_TRUE(trt_map.contains("engine_load_path"));
  EXPECT_EQ(trt_map.get<std::string>("engine_load_path"), "/tmp/model.engine");
}

TEST(DecoderYAMLTest, TrtDecoderConfigRoundTripWithoutInstantiation) {
  using namespace cudaq::qec::decoding::config;

  multi_decoder_config multi_config;
  decoder_config config = create_test_empty_decoder_config(0);
  config.type = "trt_decoder";
  trt_decoder_config trt_config;
  trt_config.engine_load_path = "/tmp/prebuilt.engine";
  trt_config.engine_save_path = "/tmp/saved.engine";
  trt_config.precision = "best";
  trt_config.memory_workspace = std::size_t{1 << 20};
  config.decoder_custom_args = trt_config;
  multi_config.decoders.push_back(config);

  test_decoder_yaml_roundtrip(multi_config);
}

TEST(DecoderYAMLTest, SlidingWindowInnerDecoderVariantRoundTrips) {
  using namespace cudaq::qec::decoding::config;

  auto check_roundtrip = [](sliding_window_config sw_config) {
    multi_decoder_config multi_config;
    decoder_config config = create_test_empty_decoder_config(0);
    config.type = "sliding_window";
    config.block_size = 6;
    config.syndrome_size = 4;
    cudaqx::tensor<uint8_t> H({config.syndrome_size, config.block_size});
    cudaqx::tensor<uint8_t> O({1, config.block_size});
    config.H_sparse = cudaq::qec::pcm_to_sparse_vec(H);
    config.O_sparse = cudaq::qec::pcm_to_sparse_vec(O);
    config.D_sparse = cudaq::qec::generate_timelike_sparse_detector_matrix(
        config.syndrome_size, 2, /*include_first_round=*/false);
    config.decoder_custom_args = sw_config;
    multi_config.decoders.push_back(config);
    test_decoder_yaml_roundtrip(multi_config);
  };

  sliding_window_config single_lut_sw;
  single_lut_sw.window_size = std::size_t{1};
  single_lut_sw.step_size = std::size_t{1};
  single_lut_sw.num_syndromes_per_round = std::size_t{2};
  single_lut_sw.error_rate_vec = std::vector<double>(6, 0.1);
  single_lut_sw.inner_decoder_name = "single_error_lut";
  single_lut_sw.single_error_lut_params = single_error_lut_config();
  check_roundtrip(single_lut_sw);

  sliding_window_config nv_sw = single_lut_sw;
  nv_sw.inner_decoder_name = "nv-qldpc-decoder";
  nv_sw.single_error_lut_params.reset();
  nv_sw.nv_qldpc_decoder_params = nv_qldpc_decoder_config();
  nv_sw.nv_qldpc_decoder_params->max_iterations = 5;
  nv_sw.nv_qldpc_decoder_params->error_rate_vec = std::vector<double>(6, 0.1);
  check_roundtrip(nv_sw);
}

TEST(DecoderConfigTest, ConfigureRejectsDuplicateAndNegativeIds) {
  using namespace cudaq::qec::decoding::config;

  multi_decoder_config duplicate_ids;
  duplicate_ids.decoders.push_back(create_test_empty_decoder_config(0));
  duplicate_ids.decoders.push_back(create_test_empty_decoder_config(0));
  EXPECT_EQ(configure_decoders(duplicate_ids), 1);

  multi_decoder_config negative_id;
  negative_id.decoders.push_back(create_test_empty_decoder_config(-1));
  negative_id.decoders.push_back(create_test_empty_decoder_config(0));
  EXPECT_EQ(configure_decoders(negative_id), 3);
}

TEST(DecoderConfigTest, ConfigureFromFileWithDebugLogging) {
  using namespace cudaq::qec::decoding::config;

  ScopedEnv debugEnv("CUDAQ_QEC_DEBUG_DECODER", "1");

  multi_decoder_config multi_config;
  multi_config.decoders.push_back(create_test_empty_decoder_config(0));
  const auto path =
      std::filesystem::temp_directory_path() / "cudaq_qec_decoders.yaml";
  {
    std::ofstream out(path);
    out << multi_config.to_yaml_str(200);
  }

  EXPECT_EQ(configure_decoders_from_file(path.c_str()), 0);
  finalize_decoders();
  std::filesystem::remove(path);
}

TEST(DecoderConfigTest, ConfigureFromMissingFileReturnsError) {
  using namespace cudaq::qec::decoding::config;

  // Missing config files should return the documented nonzero status instead
  // of attempting to parse an empty or invalid YAML payload.
  const auto missing_path = std::filesystem::temp_directory_path() /
                            "cudaq_qec_missing_decoders.yaml";
  std::filesystem::remove(missing_path);
  EXPECT_EQ(configure_decoders_from_file(missing_path.c_str()), 1);
}

TEST(DecoderConfigTest, SimulationHostPointerWrappersForwardToHostRuntime) {
  using namespace cudaq::qec::decoding::config;

  // The simulation namespace pointer overloads are host trampolines; configure
  // a simple decoder and verify enqueue/get_corrections reaches the host state.
  multi_decoder_config multi_config;
  auto config = create_test_empty_decoder_config(0);
  cudaqx::tensor<uint8_t> O({1, config.block_size});
  O.at({0, 0}) = 1;
  config.O_sparse = cudaq::qec::pcm_to_sparse_vec(O);
  multi_config.decoders.push_back(config);
  ASSERT_EQ(configure_decoders(multi_config), 0);

  std::vector<uint8_t> syndromes(config.syndrome_size * 2, 0);
  syndromes[0] = 1;
  cudaq::qec::decoding::simulation::enqueue_syndromes(
      /*decoder_id=*/0, syndromes.data(), syndromes.size(), /*tag=*/17);

  std::vector<uint8_t> corrections(1, 0xff);
  cudaq::qec::decoding::simulation::get_corrections(
      /*decoder_id=*/0, corrections.data(), corrections.size(), /*reset=*/true);
  EXPECT_EQ(corrections, (std::vector<uint8_t>{0}));
  finalize_decoders();
}
