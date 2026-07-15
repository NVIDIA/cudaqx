/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "SessionRegistry.h"
#include "../lib/realtime/realtime_decoding.h"
#include "cudaq/qec/decoder.h"
#include "cudaq/qec/decoder_config_schema.h"
#include "cudaq/qec/pcm_utils.h"
#include "cudaq/qec/realtime/decoding_config.h"
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <limits>
#include <optional>
#include <stdexcept>

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

TEST(DecoderYAMLTest, RejectsParserErrors) {
  const std::string unknown_root_key = R"(
decoders:
  - id: 0
    type: pymatching
    block_size: 1
    syndrome_size: 1
    H_sparse: [0, -1]
    O_sparse: [0, -1]
    D_sparse: [0, -1]
unexpected: true
)";
  EXPECT_THROW(
      cudaq::qec::decoding::config::multi_decoder_config::from_yaml_str(
          unknown_root_key),
      std::runtime_error);

  const std::string misspelled_decoder_argument = R"(
decoders:
  - id: 0
    type: pymatching
    block_size: 1
    syndrome_size: 1
    H_sparse: [0, -1]
    O_sparse: [0, -1]
    D_sparse: [0, -1]
    decoder_custom_args:
      merge_stratgey: smallest_weight
)";
  EXPECT_THROW(
      cudaq::qec::decoding::config::multi_decoder_config::from_yaml_str(
          misspelled_decoder_argument),
      std::runtime_error);

  EXPECT_THROW(
      cudaq::qec::decoding::config::multi_decoder_config::from_yaml_str(
          "decoders: ["),
      std::runtime_error);
}

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

cudaq::qec::decoding::config::decoder_config
create_test_sample_realtime_decoder_config(int id) {
  auto config = create_test_empty_decoder_config(id);
  config.type = "sample_decoder";
  cudaqx::tensor<uint8_t> O({2, config.block_size});
  O.at({0, 0}) = 1;
  O.at({1, 1}) = 1;
  config.O_sparse = cudaq::qec::pcm_to_sparse_vec(O);
  return config;
}

/// Helper function to create a sample, skeleton test decoder configuration for
/// the NV-QLDPC decoder.
cudaq::qec::decoding::config::decoder_config
create_test_decoder_config_nv_qldpc(int id) {
  cudaq::qec::decoding::config::decoder_config config =
      create_test_empty_decoder_config(id);
  config.type = "nv-qldpc-decoder";

  cudaqx::heterogeneous_map nv_args;
  nv_args.insert("use_sparsity", true);
  nv_args.insert("max_iterations", 50);
  nv_args.insert("use_osd", true);
  nv_args.insert("osd_order", 60);
  nv_args.insert("osd_method", 3);
  nv_args.insert("error_rate_vec", std::vector<double>(config.block_size, 0.1));
  nv_args.insert("n_threads", 128);
  nv_args.insert("bp_batch_size", 1);
  nv_args.insert("osd_batch_size", 16);
  nv_args.insert("iter_per_check", 2);
  nv_args.insert("clip_value", 10.0);
  nv_args.insert("bp_method", 3);
  nv_args.insert("scale_factor", 1.0);
  nv_args.insert("proc_float", "fp64");
  nv_args.insert("gamma0", 0.0);
  nv_args.insert("gamma_dist", std::vector<double>{0.1, 0.2});
  cudaqx::heterogeneous_map srelay_args;
  srelay_args.insert("pre_iter", std::size_t{5});
  srelay_args.insert("num_sets", std::size_t{10});
  srelay_args.insert("stopping_criterion", "NConv");
  srelay_args.insert("stop_nconv", std::size_t{10});
  nv_args.insert("srelay_config", srelay_args);
  // explicit_gammas must have num_sets rows (10 in this case)
  nv_args.insert("explicit_gammas",
                 std::vector<std::vector<double>>(
                     10, std::vector<double>(config.block_size, 0.1)));
  nv_args.insert("bp_seed", 42);
  nv_args.insert("composition", 1);
  config.decoder_custom_args = nv_args;

  return config;
}

// The trt_decoder schema is registered by the trt_decoder plugin, which is
// only built when TensorRT is available. YAML paths for trt configs require
// it; typed-struct conversions do not.
bool is_trt_decoder_schema_available() {
  return cudaq::qec::decoding::config::find_decoder_schema("trt_decoder") !=
         nullptr;
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
  cudaqx::heterogeneous_map lut_args;
  lut_args.insert("lut_error_depth", 2);
  config.decoder_custom_args = lut_args;
  multi_config.decoders.push_back(config);

  test_decoder_yaml_roundtrip(multi_config);
  test_decoder_creation(multi_config);
}

TEST(DecoderYAMLTest, SingleLUTDecoder) {
  cudaq::qec::decoding::config::multi_decoder_config multi_config;
  cudaq::qec::decoding::config::decoder_config config =
      create_test_empty_decoder_config(0);
  config.type = "single_error_lut";
  config.decoder_custom_args = cudaqx::heterogeneous_map();
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

  cudaqx::heterogeneous_map trt_args;
  trt_args.insert("onnx_load_path", "/tmp/predecoder.onnx");
  trt_args.insert("engine_save_path", "/tmp/predecoder.engine");
  trt_args.insert("precision", "best");
  trt_args.insert("memory_workspace", std::size_t{1ULL << 20});
  trt_args.insert("batch_size", std::size_t{4});
  trt_args.insert("use_cuda_graph", false);
  trt_args.insert("global_decoder", "pymatching");
  cudaqx::heterogeneous_map pymatching_params;
  pymatching_params.insert("merge_strategy", "smallest_weight");
  pymatching_params.insert("error_rate_vec",
                           std::vector<double>(config.block_size, 0.1));
  trt_args.insert("global_decoder_params", pymatching_params);
  config.decoder_custom_args = trt_args;

  return config;
}

TEST(DecoderYAMLTest, TrtDecoderConfigRoundTrip) {
  if (!is_trt_decoder_schema_available())
    GTEST_SKIP() << "trt_decoder plugin (and its parameter schema) not built";
  cudaq::qec::decoding::config::multi_decoder_config multi_config;
  multi_config.decoders.push_back(create_test_decoder_config_trt(0));

  test_decoder_yaml_roundtrip(multi_config);
  const auto &args = multi_config.decoders[0].decoder_custom_args.map();
  ASSERT_TRUE(args.contains("global_decoder_params"));
  EXPECT_EQ(args.get<cudaqx::heterogeneous_map>("global_decoder_params")
                .get<std::string>("merge_strategy"),
            "smallest_weight");
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

TEST(DecoderYAMLTest, TrtDecoderEmptyGlobalDecoderParams) {
  if (!is_trt_decoder_schema_available())
    GTEST_SKIP() << "trt_decoder plugin (and its parameter schema) not built";
  // An explicitly empty global params section round-trips and reaches the
  // realtime decoder params.
  auto config = create_test_decoder_config_trt(0);
  auto args = config.decoder_custom_args.map();
  args.insert("global_decoder_params", cudaqx::heterogeneous_map());
  config.decoder_custom_args = args;

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
  const auto &round_tripped_args =
      round_tripped.decoders[0].decoder_custom_args.map();
  ASSERT_TRUE(round_tripped_args.contains("global_decoder_params"));
  EXPECT_TRUE(
      round_tripped_args.get<cudaqx::heterogeneous_map>("global_decoder_params")
          .empty());

  params = cudaq::qec::decoding::host::prepare_decoder_params(config);
  EXPECT_TRUE(params.contains("global_decoder_params"));
  EXPECT_TRUE(params.contains("O"));

  config.O_sparse.clear();
  params = cudaq::qec::decoding::host::prepare_decoder_params(config);
  EXPECT_TRUE(params.contains("global_decoder_params"));
  EXPECT_FALSE(params.contains("O"));
}

TEST(DecoderYAMLTest, TrtDecoderDefaultGlobalDecoderParams) {
  if (!is_trt_decoder_schema_available())
    GTEST_SKIP() << "trt_decoder plugin (and its parameter schema) not built";
  // When the YAML names a global decoder with a registered schema but gives
  // no params, an empty section is materialized on parse.
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
  auto parsed =
      cudaq::qec::decoding::config::multi_decoder_config::from_yaml_str(
          yaml_without_params);
  const auto &args = parsed.decoders[0].decoder_custom_args.map();
  ASSERT_TRUE(args.contains("global_decoder_params"));
  EXPECT_TRUE(
      args.get<cudaqx::heterogeneous_map>("global_decoder_params").empty());

  // Emission after materialization is stable.
  const auto emitted = parsed.to_yaml_str(200);
  EXPECT_NE(emitted.find("global_decoder_params"), std::string::npos);
  auto round_tripped =
      cudaq::qec::decoding::config::multi_decoder_config::from_yaml_str(
          emitted);
  EXPECT_EQ(round_tripped, parsed);
  EXPECT_EQ(round_tripped.to_yaml_str(200), emitted);
}

TEST(DecoderYAMLTest, UnknownTrtGlobalDecoderParamsThrow) {
  if (!is_trt_decoder_schema_available())
    GTEST_SKIP() << "trt_decoder plugin (and its parameter schema) not built";
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

  // A global decoder without a registered schema is allowed as long as no
  // params section is given (nothing is materialized for it).
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
      global_decoder: my_plugin
)";
  auto parsed =
      cudaq::qec::decoding::config::multi_decoder_config::from_yaml_str(
          yaml_without_params);
  const auto &args = parsed.decoders[0].decoder_custom_args.map();
  EXPECT_EQ(args.get<std::string>("global_decoder"), "my_plugin");
  EXPECT_FALSE(args.contains("global_decoder_params"));
}

TEST(DecoderYAMLTest, TrtDecoderParamsWithoutDecoderThrows) {
  if (!is_trt_decoder_schema_available())
    GTEST_SKIP() << "trt_decoder plugin (and its parameter schema) not built";
  const std::string yaml_params_without_decoder = R"(
decoders:
  - id: 0
    type: trt_decoder
    block_size: 1
    syndrome_size: 1
    H_sparse: [0, -1]
    O_sparse: []
    D_sparse: [0, -1]
    decoder_custom_args:
      onnx_load_path: /tmp/predecoder.onnx
      global_decoder_params:
        merge_strategy: smallest_weight
)";
  EXPECT_THROW(
      cudaq::qec::decoding::config::multi_decoder_config::from_yaml_str(
          yaml_params_without_decoder),
      std::runtime_error);
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
  config.H_sparse = cudaq::qec::pcm_to_sparse_vec(pcm);
  config.O_sparse =
      cudaq::qec::pcm_to_sparse_vec(cudaqx::tensor<uint8_t>({2, n_cols}));
  config.D_sparse = cudaq::qec::generate_timelike_sparse_detector_matrix(
      config.syndrome_size, 2, /*include_first_round=*/false);
  cudaqx::heterogeneous_map sw_args;
  sw_args.insert("window_size", std::size_t{1});
  sw_args.insert("step_size", std::size_t{1});
  sw_args.insert("num_syndromes_per_round", n_syndromes_per_round);
  sw_args.insert("straddle_start_round", false);
  sw_args.insert("straddle_end_round", true);
  sw_args.insert("error_rate_vec", std::vector<double>(config.block_size, 0.1));

  // Inner decoder config
  sw_args.insert("inner_decoder_name", "multi_error_lut");
  cudaqx::heterogeneous_map inner_lut_args;
  inner_lut_args.insert("lut_error_depth", 2);
  sw_args.insert("inner_decoder_params", inner_lut_args);
  config.decoder_custom_args = sw_args;

  multi_config.decoders.push_back(config);

  test_decoder_yaml_roundtrip(multi_config);
  test_decoder_creation(multi_config);
}

TEST(DecoderYAMLTest, TrtDecoderConfigRoundTripWithoutInstantiation) {
  if (!is_trt_decoder_schema_available())
    GTEST_SKIP() << "trt_decoder plugin (and its parameter schema) not built";
  using namespace cudaq::qec::decoding::config;

  multi_decoder_config multi_config;
  decoder_config config = create_test_empty_decoder_config(0);
  config.type = "trt_decoder";
  cudaqx::heterogeneous_map trt_args;
  trt_args.insert("engine_load_path", "/tmp/prebuilt.engine");
  trt_args.insert("engine_save_path", "/tmp/saved.engine");
  trt_args.insert("precision", "best");
  trt_args.insert("memory_workspace", std::size_t{1 << 20});
  config.decoder_custom_args = trt_args;
  multi_config.decoders.push_back(config);

  test_decoder_yaml_roundtrip(multi_config);
}

TEST(DecoderYAMLTest, SlidingWindowInnerDecoderVariantRoundTrips) {
  using namespace cudaq::qec::decoding::config;

  auto check_roundtrip = [](const cudaqx::heterogeneous_map &sw_args) {
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
    config.decoder_custom_args = sw_args;
    multi_config.decoders.push_back(config);
    test_decoder_yaml_roundtrip(multi_config);
  };

  cudaqx::heterogeneous_map single_lut_sw;
  single_lut_sw.insert("window_size", std::size_t{1});
  single_lut_sw.insert("step_size", std::size_t{1});
  single_lut_sw.insert("num_syndromes_per_round", std::size_t{2});
  single_lut_sw.insert("num_boundary_syndromes", std::size_t{1});
  single_lut_sw.insert("error_rate_vec", std::vector<double>(6, 0.1));
  single_lut_sw.insert("inner_decoder_name", "single_error_lut");
  check_roundtrip(single_lut_sw);

  auto nv_sw = single_lut_sw;
  nv_sw.insert("inner_decoder_name", "nv-qldpc-decoder");
  cudaqx::heterogeneous_map nv_inner;
  nv_inner.insert("max_iterations", 5);
  nv_inner.insert("error_rate_vec", std::vector<double>(6, 0.1));
  nv_sw.insert("inner_decoder_params", nv_inner);
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

TEST(DecoderConfigTest, CreateRealtimeDecoderConfiguresRuntimeState) {
  auto config = create_test_sample_realtime_decoder_config(7);

  auto decoder = cudaq::qec::decoding::host::create_realtime_decoder(config);

  ASSERT_NE(decoder, nullptr);
  EXPECT_EQ(decoder->get_decoder_id(), 7u);
  EXPECT_EQ(decoder->get_num_observables(), 2u);
  EXPECT_EQ(decoder->get_num_msyn_per_decode(), 20u);
}

TEST(DecoderConfigTest, CreateRealtimeDecoderRequiresDetectorMatrix) {
  auto config = create_test_sample_realtime_decoder_config(0);
  config.D_sparse.clear();

  EXPECT_THROW(cudaq::qec::decoding::host::create_realtime_decoder(config),
               std::runtime_error);
}

TEST(DecoderConfigTest, CreateRealtimeDecoderRejectsUnrepresentableId) {
  auto config = create_test_sample_realtime_decoder_config(0);
  config.id =
      static_cast<std::int64_t>(std::numeric_limits<std::uint32_t>::max()) + 1;

  EXPECT_THROW(cudaq::qec::decoding::host::create_realtime_decoder(config),
               std::invalid_argument);
}

TEST(DecoderConfigTest, SessionRegistryUsesConfiguredRealtimeDecoder) {
  cudaq::qec::decoding::config::multi_decoder_config config;
  auto decoder_config = create_test_sample_realtime_decoder_config(0);
  config.decoders.push_back(std::move(decoder_config));

  cudaq::qec::decoding_server::SessionRegistry registry;
  registry.load_from_config(config, "unit test");

  const auto &decoder = registry.get(0).dec;
  ASSERT_NE(decoder, nullptr);
  EXPECT_EQ(decoder->get_decoder_id(), 0u);
  EXPECT_EQ(decoder->get_num_observables(), 2u);
  EXPECT_EQ(decoder->get_num_msyn_per_decode(), 20u);
}

TEST(DecoderConfigTest, SessionRegistryRejectsMissingDetectorMatrix) {
  cudaq::qec::decoding::config::multi_decoder_config config;
  auto decoder_config = create_test_sample_realtime_decoder_config(0);
  decoder_config.D_sparse.clear();
  config.decoders.push_back(std::move(decoder_config));

  cudaq::qec::decoding_server::SessionRegistry registry;
  EXPECT_THROW(registry.load_from_config(config, "unit test"),
               std::runtime_error);
}

TEST(DecoderConfigTest, SessionRegistryRejectsNegativeDecoderId) {
  cudaq::qec::decoding::config::multi_decoder_config config;
  auto decoder_config = create_test_sample_realtime_decoder_config(-1);
  config.decoders.push_back(std::move(decoder_config));

  cudaq::qec::decoding_server::SessionRegistry registry;
  EXPECT_THROW(registry.load_from_config(config, "unit test"),
               std::runtime_error);
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

TEST(DecoderSchemaTest, ThirdPartySchemaRegistrationEnablesCustomArgs) {
  using namespace cudaq::qec::decoding::config;

  // A third-party decoder plugin registers a parameter schema (normally from
  // a static initializer in its own shared library); the YAML layer then
  // accepts and round-trips its decoder_custom_args with no framework
  // changes.
  register_decoder_schema({"third_party_demo_engine",
                           {
                               {"gain", param_kind::f64},
                           }});
  register_decoder_schema(
      {"third_party_demo_decoder",
       {
           {"strength", param_kind::f64},
           {"passes", param_kind::int32},
           {"mode", param_kind::string, /*required=*/true},
           {"weights", param_kind::f64_vec},
           {"engine", param_kind::string},
           {"engine_params", param_kind::discriminated, false, "", "engine",
            /*materialize_empty=*/true},
       }});

  const std::string yaml = R"(
decoders:
  - id: 0
    type: third_party_demo_decoder
    block_size: 2
    syndrome_size: 1
    H_sparse: [0, -1]
    O_sparse: [0, -1]
    D_sparse: [0, -1]
    decoder_custom_args:
      strength: 1.5
      passes: 3
      mode: fast
      weights: [0.25, 0.75]
      engine: third_party_demo_engine
)";
  auto config = multi_decoder_config::from_yaml_str(yaml);
  const auto &args = config.decoders[0].decoder_custom_args.map();
  EXPECT_EQ(args.get<double>("strength"), 1.5);
  EXPECT_EQ(args.get<int>("passes"), 3);
  EXPECT_EQ(args.get<std::string>("mode"), "fast");
  EXPECT_EQ(args.get<std::vector<double>>("weights"),
            (std::vector<double>{0.25, 0.75}));
  // The discriminated engine_params section is materialized (empty) because
  // "engine" names a registered schema and materialize_empty is set.
  ASSERT_TRUE(args.contains("engine_params"));
  EXPECT_TRUE(args.get<cudaqx::heterogeneous_map>("engine_params").empty());

  const auto emitted = config.to_yaml_str(200);
  auto round_tripped = multi_decoder_config::from_yaml_str(emitted);
  EXPECT_EQ(round_tripped, config);
  EXPECT_EQ(round_tripped.to_yaml_str(200), emitted);

  // Unknown keys are rejected against the schema.
  const std::string misspelled = R"(
decoders:
  - id: 0
    type: third_party_demo_decoder
    block_size: 2
    syndrome_size: 1
    H_sparse: [0, -1]
    O_sparse: [0, -1]
    D_sparse: [0, -1]
    decoder_custom_args:
      strenght: 1.5
      mode: fast
)";
  EXPECT_THROW(multi_decoder_config::from_yaml_str(misspelled),
               std::runtime_error);

  // Missing required keys are rejected when the section is present.
  const std::string missing_required = R"(
decoders:
  - id: 0
    type: third_party_demo_decoder
    block_size: 2
    syndrome_size: 1
    H_sparse: [0, -1]
    O_sparse: [0, -1]
    D_sparse: [0, -1]
    decoder_custom_args:
      strength: 1.5
)";
  EXPECT_THROW(multi_decoder_config::from_yaml_str(missing_required),
               std::runtime_error);

  // A populated discriminated section round-trips, and one that names an
  // unregistered schema is rejected.
  const std::string with_engine_params = R"(
decoders:
  - id: 0
    type: third_party_demo_decoder
    block_size: 2
    syndrome_size: 1
    H_sparse: [0, -1]
    O_sparse: [0, -1]
    D_sparse: [0, -1]
    decoder_custom_args:
      mode: fast
      engine: third_party_demo_engine
      engine_params:
        gain: 2.5
)";
  auto engine_config = multi_decoder_config::from_yaml_str(with_engine_params);
  const auto &engine_args = engine_config.decoders[0].decoder_custom_args.map();
  EXPECT_EQ(engine_args.get<cudaqx::heterogeneous_map>("engine_params")
                .get<double>("gain"),
            2.5);
  auto engine_round_tripped =
      multi_decoder_config::from_yaml_str(engine_config.to_yaml_str(200));
  EXPECT_EQ(engine_round_tripped, engine_config);

  const std::string unknown_engine = R"(
decoders:
  - id: 0
    type: third_party_demo_decoder
    block_size: 2
    syndrome_size: 1
    H_sparse: [0, -1]
    O_sparse: [0, -1]
    D_sparse: [0, -1]
    decoder_custom_args:
      mode: fast
      engine: engine_without_schema
      engine_params: {}
)";
  EXPECT_THROW(multi_decoder_config::from_yaml_str(unknown_engine),
               std::runtime_error);
}

TEST(DecoderSchemaTest, CustomArgsForUnregisteredTypeThrow) {
  const std::string yaml = R"(
decoders:
  - id: 0
    type: decoder_without_registered_schema
    block_size: 1
    syndrome_size: 1
    H_sparse: [0, -1]
    O_sparse: [0, -1]
    D_sparse: [0, -1]
    decoder_custom_args:
      anything: 1
)";
  EXPECT_THROW(
      cudaq::qec::decoding::config::multi_decoder_config::from_yaml_str(yaml),
      std::runtime_error);
}

TEST(DecoderSchemaTest, ExamplePluginRegistersSchema) {
  // The in-tree example decoder plugin registers a (parameter-less) schema
  // from its own shared library; its presence here proves the end-to-end
  // plugin registration path works.
  EXPECT_NE(cudaq::qec::decoding::config::find_decoder_schema(
                "single_error_lut_example"),
            nullptr);
}

TEST(DecoderSchemaTest, ValidateCustomArgsChecksProgrammaticMaps) {
  using namespace cudaq::qec::decoding::config;

  // Maps built programmatically (or from Python dicts) never pass through the
  // YAML parser, so validate_custom_args applies the same schema checks
  // explicitly.
  register_decoder_schema({"third_party_demo_engine",
                           {
                               {"gain", param_kind::f64},
                           }});
  register_decoder_schema(
      {"third_party_demo_decoder",
       {
           {"strength", param_kind::f64},
           {"passes", param_kind::int32},
           {"mode", param_kind::string, /*required=*/true},
           {"weights", param_kind::f64_vec},
           {"engine", param_kind::string},
           {"engine_params", param_kind::discriminated, false, "", "engine",
            /*materialize_empty=*/true},
       }});

  decoder_config config;
  config.type = "third_party_demo_decoder";
  cudaqx::heterogeneous_map args;
  args.insert("strength", 1.5);
  args.insert("mode", std::string("fast"));
  config.decoder_custom_args = args;
  EXPECT_NO_THROW(config.validate_custom_args());

  // Unknown key.
  args.insert("strenght", 1.5);
  config.decoder_custom_args = args;
  EXPECT_THROW(config.validate_custom_args(), std::runtime_error);

  // Missing required key.
  cudaqx::heterogeneous_map missing_mode;
  missing_mode.insert("strength", 1.5);
  config.decoder_custom_args = missing_mode;
  EXPECT_THROW(config.validate_custom_args(), std::runtime_error);

  // Nested discriminated sections are validated with the schema named by the
  // discriminator.
  cudaqx::heterogeneous_map engine_params;
  engine_params.insert("gain", 2.5);
  cudaqx::heterogeneous_map with_engine;
  with_engine.insert("mode", std::string("fast"));
  with_engine.insert("engine", std::string("third_party_demo_engine"));
  with_engine.insert("engine_params", engine_params);
  config.decoder_custom_args = with_engine;
  EXPECT_NO_THROW(config.validate_custom_args());

  engine_params.insert("gian", 2.5);
  with_engine.insert("engine_params", engine_params);
  config.decoder_custom_args = with_engine;
  EXPECT_THROW(config.validate_custom_args(), std::runtime_error);

  // Unregistered decoder types reject non-empty args (and accept empty ones).
  decoder_config unregistered;
  unregistered.type = "decoder_without_registered_schema";
  EXPECT_NO_THROW(unregistered.validate_custom_args());
  cudaqx::heterogeneous_map anything;
  anything.insert("anything", 1);
  unregistered.decoder_custom_args = anything;
  EXPECT_THROW(unregistered.validate_custom_args(), std::runtime_error);

  // multi_decoder_config validates every decoder.
  multi_decoder_config multi;
  multi.decoders.push_back(unregistered);
  EXPECT_THROW(multi.validate_custom_args(), std::runtime_error);
}

TEST(DecoderSchemaTest, ProgrammaticConfigsMaterializeSchemaDefaults) {
  using namespace cudaq::qec::decoding::config;

  // Schema-declared defaults (materialize_empty discriminated sections, e.g.
  // trt_decoder's global_decoder_params) must apply to programmatically
  // built configs at the decoder-construction seam, not only on the YAML
  // parse path.
  register_decoder_schema({"third_party_demo_engine",
                           {
                               {"gain", param_kind::f64},
                           }});
  register_decoder_schema(
      {"third_party_demo_decoder",
       {
           {"mode", param_kind::string, /*required=*/true},
           {"engine", param_kind::string},
           {"engine_params", param_kind::discriminated, false, "", "engine",
            /*materialize_empty=*/true},
       }});

  decoder_config config;
  config.type = "third_party_demo_decoder";
  cudaqx::heterogeneous_map args;
  args.insert("mode", std::string("fast"));
  args.insert("engine", std::string("third_party_demo_engine"));
  config.decoder_custom_args = args;

  auto materialized = config.decoder_custom_args_to_heterogeneous_map();
  ASSERT_TRUE(materialized.contains("engine_params"));
  EXPECT_TRUE(
      materialized.get<cudaqx::heterogeneous_map>("engine_params").empty());
  // The stored args are untouched; only the constructor-facing view defaults.
  EXPECT_FALSE(config.decoder_custom_args.map().contains("engine_params"));

  // A decoder type without a registered schema passes its args through.
  decoder_config unregistered;
  unregistered.type = "decoder_without_registered_schema";
  unregistered.decoder_custom_args = args;
  EXPECT_TRUE(custom_args_maps_equal(
      unregistered.decoder_custom_args_to_heterogeneous_map(), args));
}

TEST(DecoderSchemaTest, CustomArgsEqualityIsSignAware) {
  using namespace cudaq::qec::decoding::config;

  // size_t(2^64-1) must not compare equal to int(-1) via wraparound.
  cudaqx::heterogeneous_map a;
  a.insert("seed", std::numeric_limits<std::size_t>::max());
  cudaqx::heterogeneous_map b;
  b.insert("seed", int(-1));
  EXPECT_FALSE(custom_args_maps_equal(a, b));

  // Same-value cross-width comparisons still hold.
  cudaqx::heterogeneous_map c;
  c.insert("seed", std::size_t(7));
  cudaqx::heterogeneous_map d;
  d.insert("seed", int(7));
  EXPECT_TRUE(custom_args_maps_equal(c, d));
  cudaqx::heterogeneous_map e;
  e.insert("seed", int(-1));
  EXPECT_TRUE(custom_args_maps_equal(b, e));
}

TEST(DecoderSchemaTest, SlidingWindowValidateHookRejectsBadWindowing) {
  using namespace cudaq::qec::decoding::config;

  // The sliding_window schema registers a validate hook for the cross-field
  // constraints its per-key specs can't express; the hook runs both when YAML
  // is parsed and from validate_custom_args.
  const std::string yaml_template = R"(
decoders:
  - id: 0
    type: sliding_window
    block_size: 2
    syndrome_size: 2
    H_sparse: [0, -1, 1, -1]
    O_sparse: [0, -1, 1, -1]
    D_sparse: [0, -1, 1, -1]
    decoder_custom_args:
      window_size: WINDOW
      step_size: STEP
      error_rate_vec: [0.01, 0.01]
      inner_decoder_name: single_error_lut
)";
  auto make_yaml = [&](const std::string &window, const std::string &step) {
    std::string yaml = yaml_template;
    yaml.replace(yaml.find("WINDOW"), 6, window);
    yaml.replace(yaml.find("STEP"), 4, step);
    return yaml;
  };

  EXPECT_NO_THROW(multi_decoder_config::from_yaml_str(make_yaml("4", "2")));
  // step_size > window_size
  EXPECT_THROW(multi_decoder_config::from_yaml_str(make_yaml("2", "4")),
               std::runtime_error);
  // step_size == 0
  EXPECT_THROW(multi_decoder_config::from_yaml_str(make_yaml("2", "0")),
               std::runtime_error);

  decoder_config config;
  config.type = "sliding_window";
  cudaqx::heterogeneous_map args;
  args.insert("window_size", std::size_t(2));
  args.insert("step_size", std::size_t(4));
  args.insert("error_rate_vec", std::vector<double>{0.01, 0.01});
  args.insert("inner_decoder_name", std::string("single_error_lut"));
  config.decoder_custom_args = args;
  EXPECT_THROW(config.validate_custom_args(), std::runtime_error);

  args.insert("step_size", std::size_t(2));
  config.decoder_custom_args = args;
  EXPECT_NO_THROW(config.validate_custom_args());

  // num_boundary_syndromes must be <= num_syndromes_per_round (the boundary
  // layers can be narrower than the interior, never wider).
  args.insert("num_syndromes_per_round", std::size_t(2));
  args.insert("num_boundary_syndromes", std::size_t(3));
  config.decoder_custom_args = args;
  EXPECT_THROW(config.validate_custom_args(), std::runtime_error);

  args.insert("num_boundary_syndromes", std::size_t(2));
  config.decoder_custom_args = args;
  EXPECT_NO_THROW(config.validate_custom_args());

  args.insert("error_rate_vec", std::vector<double>{});
  config.decoder_custom_args = args;
  EXPECT_THROW(config.validate_custom_args(), std::runtime_error);
}

TEST(DecoderSchemaTest, JsonSchemaExportReflectsRegistry) {
  using namespace cudaq::qec::decoding::config;

  // Structural spot checks; the python test suite parses the document and
  // exercises it against real YAML configurations with the jsonschema
  // package.
  const std::string text = decoder_config_json_schema();
  EXPECT_NE(text.find("\"https://json-schema.org/draft/2020-12/schema\""),
            std::string::npos);
  EXPECT_NE(text.find("\"decoder_params\""), std::string::npos);
  EXPECT_NE(text.find("\"decoder_config\""), std::string::npos);
  EXPECT_NE(text.find("\"sparse_matrix\""), std::string::npos);

  // Every registered schema (built-in and plugin-registered alike) has a
  // $defs entry, referenced from the per-type dispatch.
  for (const auto &name : registered_decoder_schema_names()) {
    EXPECT_NE(text.find("\"" + name + "\""), std::string::npos) << name;
    EXPECT_NE(text.find("\"#/$defs/decoder_params/" + name + "\""),
              std::string::npos)
        << name;
  }

  // Required keys and unknown-key rejection are carried over.
  EXPECT_NE(text.find("\"error_rate_vec\""), std::string::npos);
  EXPECT_NE(text.find("\"additionalProperties\": false"), std::string::npos);
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

TEST(DecoderYAMLTest, CudaDeviceIdRoundTrip) {
  cudaq::qec::decoding::config::multi_decoder_config multi_config;
  auto config = create_test_empty_decoder_config(0);
  config.cuda_device_id = 2;
  multi_config.decoders.push_back(config);
  test_decoder_yaml_roundtrip(multi_config);
}

TEST(DecoderYAMLTest, PrepareDecoderParamsSurfacesCudaDeviceId) {
  // Non-trt type: the insert must happen before prepare_decoder_params()'s
  // trt-only early return, so the knob reaches every decoder type.
  auto config = create_test_empty_decoder_config(0);
  config.cuda_device_id = 3;
  auto params = cudaq::qec::decoding::host::prepare_decoder_params(config);
  ASSERT_TRUE(params.contains("cuda_device_id"));
  EXPECT_EQ(params.get<int>("cuda_device_id"), 3);

  // Absent -> key absent (decoder::get() treats absence as unpinned).
  auto config2 = create_test_empty_decoder_config(1);
  auto params2 = cudaq::qec::decoding::host::prepare_decoder_params(config2);
  EXPECT_FALSE(params2.contains("cuda_device_id"));

  // trt type: still surfaced on the trt branch. prepare_decoder_params only
  // manipulates the params map (no schema lookup, no filesystem), so empty
  // custom args exercise the trt path without needing the trt plugin.
  auto config3 = create_test_empty_decoder_config(2);
  config3.type = "trt_decoder";
  config3.cuda_device_id = 1;
  auto params3 = cudaq::qec::decoding::host::prepare_decoder_params(config3);
  ASSERT_TRUE(params3.contains("cuda_device_id"));
  EXPECT_EQ(params3.get<int>("cuda_device_id"), 1);
}

TEST(DecoderYAMLTest, ValidateCustomArgsChecksValueKinds) {
  // A validated map is guaranteed to serialize: every value must be readable
  // as its schema kind's canonical storage type, not just have a known key.
  using cudaq::qec::decoding::config::decoder_config;

  decoder_config config;
  config.type = "nv-qldpc-decoder";

  cudaqx::heterogeneous_map args;
  args.insert("clip_value", std::string("oops")); // f64 param
  config.decoder_custom_args = args;
  try {
    config.validate_custom_args();
    FAIL() << "expected kind mismatch to be rejected";
  } catch (const std::runtime_error &e) {
    EXPECT_NE(std::string(e.what()).find("clip_value"), std::string::npos);
    EXPECT_NE(std::string(e.what()).find("float"), std::string::npos);
  }

  // A std::size_t stored under an f64 param (the generic conversion used
  // for dicts assigned before `type` is set) is equally unreadable at
  // emission and must be rejected too.
  cudaqx::heterogeneous_map generic;
  generic.insert("clip_value", std::size_t{2});
  config.decoder_custom_args = generic;
  EXPECT_THROW(config.validate_custom_args(), std::runtime_error);

  // Canonically-typed values pass.
  cudaqx::heterogeneous_map good;
  good.insert("clip_value", 2.0);
  good.insert("max_iterations", 50);
  config.decoder_custom_args = good;
  EXPECT_NO_THROW(config.validate_custom_args());
}

TEST(DecoderYAMLTest, TrtFirstEmissionMaterializesGlobalDecoderParams) {
  if (!is_trt_decoder_schema_available())
    GTEST_SKIP() << "trt_decoder plugin (and its parameter schema) not built";
  // A programmatic config with only global_decoder set serializes with the
  // defaulted empty global_decoder_params on FIRST emission (as the old
  // typed path did), so emitted YAML is stable across round trips.
  auto config = create_test_empty_decoder_config(0);
  config.type = "trt_decoder";
  cudaqx::heterogeneous_map args;
  args.insert("global_decoder", std::string("pymatching"));
  config.decoder_custom_args = args;

  cudaq::qec::decoding::config::multi_decoder_config multi_config;
  multi_config.decoders.push_back(config);
  const auto first = multi_config.to_yaml_str(200);
  EXPECT_NE(first.find("global_decoder_params"), std::string::npos);

  auto round_tripped =
      cudaq::qec::decoding::config::multi_decoder_config::from_yaml_str(first);
  EXPECT_EQ(round_tripped.to_yaml_str(200), first);
}

TEST(DecoderYAMLTest, NonSchemaKeysDroppedFromDecoderParamsAndEmission) {
  // A key outside the registered schema can never round-trip through YAML,
  // so the constructor-facing map must not contain it either: local decoders
  // and remote targets see the same configuration.
  auto config = create_test_empty_decoder_config(0);
  config.type = "multi_error_lut";
  cudaqx::heterogeneous_map args;
  args.insert("lut_error_depth", 2);
  args.insert("not_a_real_param", 42);
  config.decoder_custom_args = args;

  auto params = config.decoder_custom_args_to_heterogeneous_map();
  EXPECT_TRUE(params.contains("lut_error_depth"));
  EXPECT_FALSE(params.contains("not_a_real_param"));

  cudaq::qec::decoding::config::multi_decoder_config multi_config;
  multi_config.decoders.push_back(config);
  const auto yaml = multi_config.to_yaml_str(200);
  EXPECT_NE(yaml.find("lut_error_depth"), std::string::npos);
  EXPECT_EQ(yaml.find("not_a_real_param"), std::string::npos);

  // The stored args are untouched -- only the derived views are filtered.
  EXPECT_TRUE(config.decoder_custom_args.map().contains("not_a_real_param"));
}
