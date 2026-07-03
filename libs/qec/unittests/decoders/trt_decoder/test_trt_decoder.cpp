/*******************************************************************************
 * Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "trt_test_data.h"
#include "cudaq/qec/decoder.h"
#include "cudaq/qec/trt_decoder_internal.h"
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <optional>
#include <random>
#include <thread>
#include <vector>

#include <cuda_runtime_api.h>

#if defined(__linux__)
#include <linux/mempolicy.h>
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

using namespace cudaq::qec;

// Resolved from the LD_PRELOAD affinity shim when preloaded; weak so the
// binary still links (and counter asserts are skipped) without it.
extern "C" __attribute__((weak)) void cudaqx_affinity_syscall_reset();
extern "C" __attribute__((weak)) long cudaqx_set_mempolicy_count();

static bool gpu_available() {
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

namespace {
class TestTrtLogger : public nvinfer1::ILogger {
public:
  void log(Severity, const char *) noexcept override {}
};

cudaqx::tensor<uint8_t> make_identity_h(std::size_t n) {
  cudaqx::tensor<uint8_t> H({n, n});
  for (std::size_t i = 0; i < n; ++i)
    H.at({i, i}) = 1;
  return H;
}

std::filesystem::path make_temp_engine_path(const std::string &name) {
  return std::filesystem::temp_directory_path() / name;
}

std::optional<std::string> get_dynamic_onnx_asset_path() {
#ifdef TRT_TEST_DYNAMIC_ONNX_PATH
  return std::string(TRT_TEST_DYNAMIC_ONNX_PATH);
#else
  return std::nullopt;
#endif
}

std::optional<std::string> get_uint8_onnx_asset_path() {
#ifdef TRT_TEST_UINT8_ONNX_PATH
  return std::string(TRT_TEST_UINT8_ONNX_PATH);
#else
  return std::nullopt;
#endif
}

std::optional<std::string> get_uint8_to_float_onnx_asset_path() {
#ifdef TRT_TEST_UINT8_TO_FLOAT_ONNX_PATH
  return std::string(TRT_TEST_UINT8_TO_FLOAT_ONNX_PATH);
#else
  return std::nullopt;
#endif
}
} // namespace

// Path to ONNX test asset. Set by CMake (absolute path) so the test finds it
// regardless of the executable's run directory.
static std::string get_onnx_asset_path() {
#ifdef TRT_TEST_ONNX_PATH
  return std::string(TRT_TEST_ONNX_PATH);
#else
  // Fallback when not built with CMake: relative to project root
  return "assets/tests/surface_code_decoder.onnx";
#endif
}

// Test fixture for TRT decoder tests
class TRTDecoderTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Set up test parameters
    block_size = 3;
    syndrome_size = 2;

    // Create a simple parity check matrix H
    H = cudaqx::tensor<uint8_t>({syndrome_size, block_size});
    H.at({0, 0}) = 1;
    H.at({0, 1}) = 0;
    H.at({0, 2}) = 1; // First syndrome bit
    H.at({1, 0}) = 0;
    H.at({1, 1}) = 1;
    H.at({1, 2}) = 1; // Second syndrome bit
  }

  void TearDown() override {
    // Clean up any test files
    std::filesystem::remove("test_load_file.txt");
  }

  std::size_t block_size;
  std::size_t syndrome_size;
  cudaqx::tensor<uint8_t> H;
};

// Test parameter validation function
TEST_F(TRTDecoderTest, ValidateParameters_ValidONNXPath) {
  cudaqx::heterogeneous_map params;
  params.insert("onnx_load_path", std::string("test_model.onnx"));

  // Should not throw
  EXPECT_NO_THROW(
      cudaq::qec::trt_decoder_internal::validate_trt_decoder_parameters(
          params));
}

TEST_F(TRTDecoderTest, ValidateParameters_ValidEnginePath) {
  cudaqx::heterogeneous_map params;
  params.insert("engine_load_path", std::string("test_engine.trt"));

  // Should not throw
  EXPECT_NO_THROW(
      cudaq::qec::trt_decoder_internal::validate_trt_decoder_parameters(
          params));
}

TEST_F(TRTDecoderTest, ValidateParameters_BothPathsProvided) {
  cudaqx::heterogeneous_map params;
  params.insert("onnx_load_path", std::string("test_model.onnx"));
  params.insert("engine_load_path", std::string("test_engine.trt"));

  // Should throw runtime_error
  EXPECT_THROW(
      cudaq::qec::trt_decoder_internal::validate_trt_decoder_parameters(params),
      std::runtime_error);
}

TEST_F(TRTDecoderTest, ValidateParameters_NoPathsProvided) {
  cudaqx::heterogeneous_map params;

  // Should throw runtime_error
  EXPECT_THROW(
      cudaq::qec::trt_decoder_internal::validate_trt_decoder_parameters(params),
      std::runtime_error);
}

TEST_F(TRTDecoderTest, ValidateParameters_EmptyStringPaths) {
  cudaqx::heterogeneous_map params;
  params.insert("onnx_load_path", std::string(""));
  params.insert("engine_load_path", std::string(""));

  // Should throw runtime_error (empty strings are still considered "provided")
  EXPECT_THROW(
      cudaq::qec::trt_decoder_internal::validate_trt_decoder_parameters(params),
      std::runtime_error);
}

// Test load_file function
TEST_F(TRTDecoderTest, LoadFile_ValidFile) {
  // Create a test file
  std::string test_filename = "test_load_file.txt";
  std::string test_content = "Hello, World!";

  std::ofstream file(test_filename);
  file << test_content;
  file.close();

  // Test loading the file
  auto loaded_content =
      cudaq::qec::trt_decoder_internal::load_file(test_filename);
  std::string loaded_string(loaded_content.begin(), loaded_content.end());

  EXPECT_EQ(loaded_string, test_content);
}

TEST_F(TRTDecoderTest, LoadFile_NonExistentFile) {
  // Test loading a non-existent file
  EXPECT_THROW(
      cudaq::qec::trt_decoder_internal::load_file("non_existent_file.txt"),
      std::runtime_error);
}

TEST_F(TRTDecoderTest, SaveEngineRejectsNullEngine) {
  EXPECT_THROW(cudaq::qec::trt_decoder_internal::save_engine_to_file(
                   nullptr, "unused.engine"),
               std::runtime_error);
}

TEST_F(TRTDecoderTest, ParsePrecisionAcceptsSupportedAndLegacyValues) {
  // Skip without a usable GPU: createInferBuilder initializes CUDA, and on
  // failure TRT 10.7 (CUDA 12.6) throws nvinfer1::APIUsageError while TRT 10.14
  // (CUDA 13.0) returns nullptr. Guarding like the other GPU tests yields the
  // same outcome (skip) on both toolkits instead of aborting on 12.6.
  if (!gpu_available())
    GTEST_SKIP() << "No CUDA GPU available";

  TestTrtLogger logger;

  // Defensive against the throwing variant: catch init failures rather than
  // relying solely on a nullptr return, so the test never terminates the
  // process.
  std::unique_ptr<nvinfer1::IBuilder> builder;
  std::unique_ptr<nvinfer1::IBuilderConfig> config;
  try {
    builder.reset(nvinfer1::createInferBuilder(logger));
    if (builder)
      config.reset(builder->createBuilderConfig());
  } catch (const std::exception &e) {
    GTEST_SKIP() << "TensorRT initialization failed: " << e.what();
  }
  if (!builder)
    GTEST_SKIP() << "TensorRT builder unavailable";
  if (!config)
    GTEST_SKIP() << "TensorRT builder config unavailable";

  // parse_precision must accept current ("tf32", "best"), legacy/ignored
  // ("noTF32", "fp16") and unknown values without throwing.
  EXPECT_NO_THROW(
      cudaq::qec::trt_decoder_internal::parse_precision("tf32", config.get()));
  EXPECT_NO_THROW(cudaq::qec::trt_decoder_internal::parse_precision(
      "noTF32", config.get()));
  EXPECT_NO_THROW(
      cudaq::qec::trt_decoder_internal::parse_precision("best", config.get()));
  EXPECT_NO_THROW(
      cudaq::qec::trt_decoder_internal::parse_precision("fp16", config.get()));
  EXPECT_NO_THROW(cudaq::qec::trt_decoder_internal::parse_precision(
      "unknown_precision", config.get()));
}

// Test parameter validation edge cases
TEST_F(TRTDecoderTest, ValidateParameters_EdgeCases) {
  // Test with whitespace-only strings
  cudaqx::heterogeneous_map params1;
  params1.insert("onnx_load_path", std::string("   "));
  params1.insert("engine_load_path", std::string("   "));

  EXPECT_THROW(
      cudaq::qec::trt_decoder_internal::validate_trt_decoder_parameters(
          params1),
      std::runtime_error);

  // Test with very long paths
  cudaqx::heterogeneous_map params2;
  std::string long_path(1000, 'a');
  params2.insert("onnx_load_path", long_path);

  EXPECT_NO_THROW(
      cudaq::qec::trt_decoder_internal::validate_trt_decoder_parameters(
          params2));
}

// Test TRT decoder with generated test data
// This test validates that the TRT decoder produces identical results to
// PyTorch
TEST_F(TRTDecoderTest, ValidateAgainstPyTorchModel) {
  if (!gpu_available())
    GTEST_SKIP() << "No CUDA GPU available";
  std::string onnx_path = get_onnx_asset_path();
  if (!std::filesystem::exists(onnx_path)) {
    GTEST_SKIP() << "ONNX model file not found: " << onnx_path;
  }

  // Create parity check matrix matching the test data
  // For distance-3 surface code: 24 detectors (syndromes), block_size matches
  // output
  std::size_t num_detectors = NUM_DETECTORS;
  std::size_t num_observables = NUM_OBSERVABLES;

  // Create a dummy H matrix (the TRT decoder doesn't actually use it for
  // inference, but the constructor requires it)
  cudaqx::tensor<uint8_t> H({num_detectors, num_detectors});
  for (std::size_t i = 0; i < num_detectors; ++i) {
    H.at({i, i}) = 1;
  }

  // Create the TRT decoder
  cudaqx::heterogeneous_map params;
  params.insert("onnx_load_path", onnx_path);

  std::unique_ptr<decoder> trt_decoder;
  try {
    trt_decoder = decoder::get("trt_decoder", H, params);
  } catch (const std::exception &e) {
    GTEST_SKIP() << "Failed to create TRT decoder: " << e.what();
  }

  // Tolerance for floating point comparison
  constexpr float TOLERANCE = 1e-4f;

  // Track statistics
  int num_passed = 0;
  int num_failed = 0;
  float max_error = 0.0f;
  float total_error = 0.0f;

  // Test each of the 100 test cases
  for (size_t i = 0; i < TEST_INPUTS.size(); ++i) {
    // Convert test input to the format expected by decoder
    std::vector<cudaq::qec::float_t> syndrome(TEST_INPUTS[i].begin(),
                                              TEST_INPUTS[i].end());

    // Run TRT decoder inference
    auto result = trt_decoder->decode(syndrome);

    // Get the PyTorch expected output
    float expected_output = TEST_OUTPUTS[i][0];

    // Get the TRT decoder output
    ASSERT_FALSE(result.result.empty())
        << "TRT decoder returned empty result for test case " << i;
    float trt_output = result.result[0];

    // Compute absolute error
    float error = std::abs(trt_output - expected_output);
    total_error += error;
    max_error = std::max(max_error, error);

    // Check if within tolerance
    if (error < TOLERANCE) {
      num_passed++;
    } else {
      num_failed++;
      // Print detailed error info for first few failures
      if (num_failed <= 5) {
        std::cout << "Test case " << i << " FAILED:" << std::endl;
        std::cout << "  Expected: " << expected_output << std::endl;
        std::cout << "  Got:      " << trt_output << std::endl;
        std::cout << "  Error:    " << error << std::endl;
      }
    }

    // Assert each individual test case
    EXPECT_LT(error, TOLERANCE)
        << "Test case " << i << " failed: "
        << "TRT output (" << trt_output << ") differs from PyTorch output ("
        << expected_output << ") by " << error;
  }

  // Print summary statistics
  std::cout << "\n=== TRT Decoder Validation Summary ===" << std::endl;
  std::cout << "Total test cases: " << TEST_INPUTS.size() << std::endl;
  std::cout << "Passed: " << num_passed << std::endl;
  std::cout << "Failed: " << num_failed << std::endl;
  std::cout << "Max error: " << max_error << std::endl;
  std::cout << "Average error: " << (total_error / TEST_INPUTS.size())
            << std::endl;
  std::cout << "====================================\n" << std::endl;

  // Overall test assertion: all cases must pass
  EXPECT_EQ(num_failed, 0) << num_failed << " test cases failed validation";
}

// Test a single specific case for detailed debugging
TEST_F(TRTDecoderTest, ValidateSingleTestCase) {
  if (!gpu_available())
    GTEST_SKIP() << "No CUDA GPU available";
  std::string onnx_path = get_onnx_asset_path();
  if (!std::filesystem::exists(onnx_path)) {
    GTEST_SKIP() << "ONNX model file not found: " << onnx_path;
  }

  // Create dummy H matrix
  std::size_t num_detectors = NUM_DETECTORS;
  cudaqx::tensor<uint8_t> H({num_detectors, num_detectors});
  for (std::size_t i = 0; i < num_detectors; ++i) {
    H.at({i, i}) = 1;
  }

  // Create the TRT decoder
  cudaqx::heterogeneous_map params;
  params.insert("onnx_load_path", onnx_path);

  std::unique_ptr<decoder> trt_decoder;
  try {
    trt_decoder = decoder::get("trt_decoder", H, params);
  } catch (const std::exception &e) {
    GTEST_SKIP() << "Failed to create TRT decoder: " << e.what();
  }

  // Test first case in detail
  std::vector<cudaq::qec::float_t> syndrome(TEST_INPUTS[0].begin(),
                                            TEST_INPUTS[0].end());

  std::cout << "Input syndrome (first 10 values): ";
  for (size_t i = 0; i < std::min(size_t(10), syndrome.size()); ++i) {
    std::cout << syndrome[i] << " ";
  }
  std::cout << std::endl;

  auto result = trt_decoder->decode(syndrome);

  float expected = TEST_OUTPUTS[0][0];
  float actual = result.result[0];
  float error = std::abs(actual - expected);

  std::cout << "Expected output: " << expected << std::endl;
  std::cout << "Actual output:   " << actual << std::endl;
  std::cout << "Absolute error:  " << error << std::endl;
  std::cout << "Converged:       " << (result.converged ? "yes" : "no")
            << std::endl;

  EXPECT_LT(error, 1e-4f) << "Single test case validation failed";
  EXPECT_TRUE(result.converged) << "Decoder did not converge";
}

// Test performance comparison: CUDA Graph vs Traditional execution
TEST_F(TRTDecoderTest, PerformanceComparisonCudaGraphVsTraditional) {
  if (!gpu_available())
    GTEST_SKIP() << "No CUDA GPU available";
  std::string onnx_path = get_onnx_asset_path();
  if (!std::filesystem::exists(onnx_path)) {
    GTEST_SKIP() << "ONNX model file not found: " << onnx_path;
  }

  // Create dummy H matrix
  std::size_t num_detectors = NUM_DETECTORS;
  cudaqx::tensor<uint8_t> H({num_detectors, num_detectors});
  for (std::size_t i = 0; i < num_detectors; ++i) {
    H.at({i, i}) = 1;
  }

  // Create test syndrome
  std::vector<cudaq::qec::float_t> syndrome(TEST_INPUTS[0].begin(),
                                            TEST_INPUTS[0].end());

  // =========================================================================
  // Create decoder WITH CUDA graphs (default)
  // =========================================================================
  cudaqx::heterogeneous_map params_cuda_graph;
  params_cuda_graph.insert("onnx_load_path", onnx_path);
  params_cuda_graph.insert("precision", "fp16");
  params_cuda_graph.insert("use_cuda_graph", true);

  std::unique_ptr<decoder> decoder_cuda_graph;
  try {
    decoder_cuda_graph = decoder::get("trt_decoder", H, params_cuda_graph);
  } catch (const std::exception &e) {
    GTEST_SKIP() << "Failed to create CUDA graph decoder: " << e.what();
  }

  // =========================================================================
  // Create decoder WITHOUT CUDA graphs (traditional)
  // =========================================================================
  cudaqx::heterogeneous_map params_traditional;
  params_traditional.insert("onnx_load_path", onnx_path);
  params_traditional.insert("precision", "fp16");
  params_traditional.insert("use_cuda_graph", false);

  std::unique_ptr<decoder> decoder_traditional;
  try {
    decoder_traditional = decoder::get("trt_decoder", H, params_traditional);
  } catch (const std::exception &e) {
    GTEST_SKIP() << "Failed to create traditional decoder: " << e.what();
  }

  // =========================================================================
  // Warm-up phase (for fair comparison)
  // =========================================================================
  const int warmup_iterations = 5;
  std::cout << "\n=== Warming up decoders ===" << std::endl;

  for (int i = 0; i < warmup_iterations; ++i) {
    decoder_cuda_graph->decode(syndrome);
    decoder_traditional->decode(syndrome);
  }

  // =========================================================================
  // Benchmark CUDA Graph Executor
  // =========================================================================
  const int benchmark_iterations = 200;
  std::cout << "Benchmarking CUDA Graph executor..." << std::endl;

  auto start_cuda_graph = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < benchmark_iterations; ++i) {
    auto result = decoder_cuda_graph->decode(syndrome);
    ASSERT_TRUE(result.converged)
        << "CUDA graph decoder failed at iteration " << i;
  }
  auto end_cuda_graph = std::chrono::high_resolution_clock::now();

  auto duration_cuda_graph =
      std::chrono::duration_cast<std::chrono::microseconds>(end_cuda_graph -
                                                            start_cuda_graph);
  double avg_time_cuda_graph =
      duration_cuda_graph.count() / static_cast<double>(benchmark_iterations);

  // =========================================================================
  // Benchmark Traditional Executor
  // =========================================================================
  std::cout << "Benchmarking Traditional executor..." << std::endl;

  auto start_traditional = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < benchmark_iterations; ++i) {
    auto result = decoder_traditional->decode(syndrome);
    ASSERT_TRUE(result.converged)
        << "Traditional decoder failed at iteration " << i;
  }
  auto end_traditional = std::chrono::high_resolution_clock::now();

  auto duration_traditional =
      std::chrono::duration_cast<std::chrono::microseconds>(end_traditional -
                                                            start_traditional);
  double avg_time_traditional =
      duration_traditional.count() / static_cast<double>(benchmark_iterations);

  // =========================================================================
  // Calculate and report performance improvement
  // =========================================================================
  double speedup = avg_time_traditional / avg_time_cuda_graph;
  double improvement_percent =
      ((avg_time_traditional - avg_time_cuda_graph) / avg_time_traditional) *
      100.0;

  std::cout << "\n=== Performance Comparison Results ===" << std::endl;
  std::cout << "Iterations: " << benchmark_iterations << std::endl;
  std::cout << "CUDA Graph avg time:   " << avg_time_cuda_graph << " μs"
            << std::endl;
  std::cout << "Traditional avg time:  " << avg_time_traditional << " μs"
            << std::endl;
  std::cout << "Speedup:               " << speedup << "x" << std::endl;
  std::cout << "Improvement:           " << improvement_percent << "%"
            << std::endl;
  std::cout << "======================================\n" << std::endl;

  // =========================================================================
  // Performance assertions
  // =========================================================================
  // CUDA graphs should provide at least 5% improvement
  // (Conservative threshold - typical improvement is 10-20%)
  EXPECT_GT(speedup, 1.05)
      << "CUDA graph execution should be at least 5% faster than traditional. "
      << "Speedup: " << speedup << "x, Improvement: " << improvement_percent
      << "%";

  // Sanity check: both should be reasonably fast (< 100ms per decode)
  EXPECT_LT(avg_time_cuda_graph, 100000.0)
      << "CUDA graph execution unexpectedly slow: " << avg_time_cuda_graph
      << " μs";
  EXPECT_LT(avg_time_traditional, 100000.0)
      << "Traditional execution unexpectedly slow: " << avg_time_traditional
      << " μs";
}

TEST_F(TRTDecoderTest, DecodeUninitializedDecoderReturnsUnconvergedBatch) {
  // A failed constructor leaves the decoder object present but not ready;
  // decode_batch should return one unconverged empty result per input syndrome.
  cudaqx::heterogeneous_map params;
  params.insert("engine_load_path",
                std::string("/no/such/cudaq-qec-test.engine"));
  auto trt_decoder = decoder::get("trt_decoder", make_identity_h(2), params);
  ASSERT_NE(trt_decoder, nullptr);

  auto results = trt_decoder->decode_batch({{}});
  ASSERT_EQ(results.size(), 1u);
  EXPECT_FALSE(results[0].converged);
  EXPECT_TRUE(results[0].result.empty());
}

TEST_F(TRTDecoderTest, EngineSavePathAndEngineLoadPathRoundTrip) {
  // Building from ONNX with engine_save_path must serialize a reusable engine;
  // loading the saved engine should preserve inference numerics.
  if (!gpu_available())
    GTEST_SKIP() << "No CUDA GPU available";
  const std::string onnx_path = get_onnx_asset_path();
  if (!std::filesystem::exists(onnx_path))
    GTEST_SKIP() << "ONNX model file not found: " << onnx_path;

  const auto engine_path =
      make_temp_engine_path("cudaq_qec_trt_decoder_roundtrip.engine");
  std::filesystem::remove(engine_path);
  auto H = make_identity_h(NUM_DETECTORS);

  cudaqx::heterogeneous_map build_params;
  build_params.insert("onnx_load_path", onnx_path);
  build_params.insert("engine_save_path", engine_path.string());
  build_params.insert("memory_workspace", std::size_t{1 << 20});
  build_params.insert("precision", std::string("fp16"));
  build_params.insert("use_cuda_graph", false);

  std::unique_ptr<decoder> built_decoder;
  try {
    built_decoder = decoder::get("trt_decoder", H, build_params);
  } catch (const std::exception &e) {
    GTEST_SKIP() << "Failed to build TRT decoder: " << e.what();
  }
  ASSERT_TRUE(std::filesystem::exists(engine_path));

  cudaqx::heterogeneous_map load_params;
  load_params.insert("engine_load_path", engine_path.string());
  load_params.insert("use_cuda_graph", false);
  std::unique_ptr<decoder> loaded_decoder;
  try {
    loaded_decoder = decoder::get("trt_decoder", H, load_params);
  } catch (const std::exception &e) {
    GTEST_SKIP() << "Failed to load TRT decoder: " << e.what();
  }

  std::vector<cudaq::qec::float_t> syndrome(TEST_INPUTS[0].begin(),
                                            TEST_INPUTS[0].end());
  auto built_result = built_decoder->decode(syndrome);
  auto loaded_result = loaded_decoder->decode(syndrome);
  ASSERT_EQ(built_result.result.size(), loaded_result.result.size());
  for (std::size_t i = 0; i < built_result.result.size(); ++i)
    EXPECT_NEAR(built_result.result[i], loaded_result.result[i], 1e-4);

  std::filesystem::remove(engine_path);
}

TEST_F(TRTDecoderTest, DynamicBatchIdentityModelUsesOptimizationProfile) {
  // A dynamic-batch ONNX model exercises TensorRT profile creation and the
  // runtime setInputShape path while preserving identity outputs.
  if (!gpu_available())
    GTEST_SKIP() << "No CUDA GPU available";
  auto onnx_path = get_dynamic_onnx_asset_path();
  if (!onnx_path || !std::filesystem::exists(*onnx_path))
    GTEST_SKIP() << "Generated dynamic ONNX fixture is unavailable";

  cudaqx::heterogeneous_map params;
  params.insert("onnx_load_path", *onnx_path);
  params.insert("batch_size", std::size_t{2});
  params.insert("use_cuda_graph", true);
  params.insert("memory_workspace", std::size_t{1 << 20});

  std::unique_ptr<decoder> trt_decoder;
  try {
    trt_decoder = decoder::get("trt_decoder", make_identity_h(3), params);
  } catch (const std::exception &e) {
    GTEST_SKIP() << "Failed to create dynamic TRT decoder: " << e.what();
  }

  std::vector<std::vector<cudaq::qec::float_t>> syndromes = {
      {0.0, 1.0, 0.0}, {1.0, 0.0, 1.0}, {0.25, 0.5, 0.75}};
  auto results = trt_decoder->decode_batch(syndromes);
  ASSERT_EQ(results.size(), syndromes.size());
  for (std::size_t r = 0; r < results.size(); ++r) {
    ASSERT_TRUE(results[r].converged);
    ASSERT_EQ(results[r].result.size(), syndromes[r].size());
    for (std::size_t c = 0; c < syndromes[r].size(); ++c)
      EXPECT_NEAR(results[r].result[c], syndromes[r][c], 1e-5);
  }
}

TEST_F(TRTDecoderTest, Uint8IdentityModelBinarizesInputAndOutput) {
  // UINT8 I/O models should hard-threshold soft syndromes before inference and
  // convert UINT8 outputs back to the decoder's floating result vector.
  if (!gpu_available())
    GTEST_SKIP() << "No CUDA GPU available";
  auto onnx_path = get_uint8_onnx_asset_path();
  if (!onnx_path || !std::filesystem::exists(*onnx_path))
    GTEST_SKIP() << "Generated uint8 ONNX fixture is unavailable";

  cudaqx::heterogeneous_map params;
  params.insert("onnx_load_path", *onnx_path);
  params.insert("use_cuda_graph", false);

  std::unique_ptr<decoder> trt_decoder;
  try {
    trt_decoder = decoder::get("trt_decoder", make_identity_h(3), params);
  } catch (const std::exception &e) {
    GTEST_SKIP() << "Failed to create uint8 TRT decoder: " << e.what();
  }

  auto result = trt_decoder->decode({0.49, 0.5, 0.75});
  ASSERT_TRUE(result.converged);
  ASSERT_EQ(result.result.size(), 3u);
  EXPECT_FLOAT_EQ(result.result[0], 0.0);
  EXPECT_FLOAT_EQ(result.result[1], 1.0);
  EXPECT_FLOAT_EQ(result.result[2], 1.0);
}

TEST_F(TRTDecoderTest, MixedDtypeCopiesOutput) {
  // Mixed UINT8 input and FLOAT output should use the engine output dtype when
  // copying and interpreting output, not the input dtype selected for dispatch.
  if (!gpu_available())
    GTEST_SKIP() << "No CUDA GPU available";
  auto onnx_path = get_uint8_to_float_onnx_asset_path();
  if (!onnx_path || !std::filesystem::exists(*onnx_path))
    GTEST_SKIP() << "Generated uint8-to-float ONNX fixture is unavailable";

  cudaqx::heterogeneous_map params;
  params.insert("onnx_load_path", *onnx_path);
  params.insert("use_cuda_graph", false);

  std::unique_ptr<decoder> trt_decoder;
  try {
    trt_decoder = decoder::get("trt_decoder", make_identity_h(3), params);
  } catch (const std::exception &e) {
    GTEST_SKIP() << "Failed to create uint8-to-float TRT decoder: " << e.what();
  }

  auto result = trt_decoder->decode({0.49, 0.5, 0.75});

  // The Cast model should execute successfully; failure here means the mixed
  // dtype path did not produce a usable decoder result.
  ASSERT_TRUE(result.converged);
  // The output has three float elements; a shorter result would show that the
  // output buffer cardinality was not preserved.
  ASSERT_EQ(result.result.size(), 3u);
  EXPECT_FLOAT_EQ(result.result[0], 0.0);
  EXPECT_FLOAT_EQ(result.result[1], 1.0);
  EXPECT_FLOAT_EQ(result.result[2], 1.0);
}

TEST_F(TRTDecoderTest, BatchFailureKeepsResultCount) {
  // A post-initialisation failure in decode_batch should preserve one result
  // per input syndrome so decode() never indexes an empty result vector.
  if (!gpu_available())
    GTEST_SKIP() << "No CUDA GPU available";
  auto onnx_path = get_dynamic_onnx_asset_path();
  if (!onnx_path || !std::filesystem::exists(*onnx_path))
    GTEST_SKIP() << "Generated dynamic ONNX fixture is unavailable";

  cudaqx::heterogeneous_map params;
  params.insert("onnx_load_path", *onnx_path);
  params.insert("batch_size", std::size_t{1});
  params.insert("use_cuda_graph", false);
  params.insert("global_decoder", std::string("single_error_lut"));
  params.insert("global_decoder_params", cudaqx::heterogeneous_map{});

  std::unique_ptr<decoder> trt_decoder;
  try {
    trt_decoder = decoder::get("trt_decoder", make_identity_h(2), params);
  } catch (const std::exception &e) {
    GTEST_SKIP() << "Failed to create mismatch TRT decoder: " << e.what();
  }

  auto results = trt_decoder->decode_batch({{1.0, 0.0, 1.0}});

  // The mismatched global decoder forces an exception before any result is
  // pushed; the fixed path must still return one placeholder for the input.
  ASSERT_EQ(results.size(), 1u);
  // The placeholder must be marked failed so callers can distinguish it from a
  // successful decode without touching out-of-range elements.
  EXPECT_FALSE(results[0].converged);
}

TEST_F(TRTDecoderTest, CompositeGlobalDecoderCombinesLogicalFrame) {
  // TRT emits [pre_L, residual syndrome]; the optional global decoder decodes
  // the residual part and XORs it with pre_L to form the final observable.
  if (!gpu_available())
    GTEST_SKIP() << "No CUDA GPU available";
  auto onnx_path = get_dynamic_onnx_asset_path();
  if (!onnx_path || !std::filesystem::exists(*onnx_path))
    GTEST_SKIP() << "Generated dynamic ONNX fixture is unavailable";

  cudaqx::tensor<uint8_t> H({2, 1});
  H.at({0, 0}) = 1;
  cudaqx::tensor<uint8_t> O({1, 1});
  O.at({0, 0}) = 1;

  cudaqx::heterogeneous_map params;
  params.insert("onnx_load_path", *onnx_path);
  params.insert("batch_size", std::size_t{2});
  params.insert("use_cuda_graph", false);
  params.insert("global_decoder", std::string("single_error_lut"));
  params.insert("global_decoder_params", cudaqx::heterogeneous_map{});
  params.insert("O", O);

  std::unique_ptr<decoder> trt_decoder;
  try {
    trt_decoder = decoder::get("trt_decoder", H, params);
  } catch (const std::exception &e) {
    GTEST_SKIP() << "Failed to create composite TRT decoder: " << e.what();
  }

  auto results = trt_decoder->decode_batch({{1.0, 1.0, 0.0}, {0.0, 1.0, 0.0}});
  ASSERT_EQ(results.size(), 2u);
  ASSERT_EQ(results[0].result.size(), 1u);
  ASSERT_EQ(results[1].result.size(), 1u);
  EXPECT_FLOAT_EQ(results[0].result[0], 0.0);
  EXPECT_FLOAT_EQ(results[1].result[0], 1.0);
}

// Note: Constructor tests and parse_precision tests are disabled because they
// require actual TensorRT/CUDA initialization which is not available in the
// test environment. Only parameter validation and utility function tests are
// enabled above.

TEST_F(TRTDecoderTest, CudaDeviceId_OutOfRangeThrows) {
  if (!gpu_available())
    GTEST_SKIP() << "No CUDA GPU available";
  std::string onnx_path = get_onnx_asset_path();
  if (!std::filesystem::exists(onnx_path))
    GTEST_SKIP() << "ONNX model not found: " << onnx_path;
  int count = 0;
  cudaGetDeviceCount(&count);
  cudaqx::tensor<uint8_t> H_small({1, 1});
  cudaqx::heterogeneous_map params;
  params.insert("onnx_load_path", onnx_path);
  params.insert("cuda_device_id", count); // one past the last valid device
  EXPECT_THROW(
      { cudaq::qec::decoder::get("trt_decoder", H_small, params); },
      std::runtime_error);
}

TEST_F(TRTDecoderTest, CudaDeviceId_DecodeAsyncOnGpu1) {
  if (!gpu_available())
    GTEST_SKIP() << "No CUDA GPU available";
  int count = 0;
  cudaGetDeviceCount(&count);
  if (count < 2)
    GTEST_SKIP() << "needs >= 2 GPUs to prove non-default pinning";
  std::string onnx_path = get_onnx_asset_path();
  if (!std::filesystem::exists(onnx_path))
    GTEST_SKIP() << "ONNX model not found: " << onnx_path;

  cudaSetDevice(0); // calling thread stays on the default device

  std::size_t num_detectors = NUM_DETECTORS;
  cudaqx::tensor<uint8_t> H_mat({num_detectors, num_detectors});
  for (std::size_t i = 0; i < num_detectors; ++i)
    H_mat.at({i, i}) = 1;

  cudaqx::heterogeneous_map params;
  params.insert("onnx_load_path", onnx_path);
  params.insert("cuda_device_id", 1); // engine/buffers must land on GPU 1

  std::unique_ptr<cudaq::qec::decoder> d;
  ASSERT_NO_THROW(
      { d = cudaq::qec::decoder::get("trt_decoder", H_mat, params); });

  // decode_async spawns a fresh thread (defaults to GPU 0). If trt did NOT
  // re-assert the device, the decode would run on GPU 0 while the engine lives
  // on GPU 1 -> error/garbage. A correct result proves the per-decode guard.
  std::vector<cudaq::qec::float_t> syndrome(TEST_INPUTS[0].begin(),
                                            TEST_INPUTS[0].end());
  auto fut = d->decode_async(syndrome);
  cudaq::qec::decoder_result res;
  ASSERT_NO_THROW({ res = fut.get(); });
  ASSERT_FALSE(res.result.empty());
  float trt_output = res.result[0];
  float expected_output = TEST_OUTPUTS[0][0];
  float error = std::abs(trt_output - expected_output);
  EXPECT_LT(error, 1e-4f) << "GPU-1 decode differs from expected: got "
                          << trt_output << ", expected " << expected_output;
}

// Two real trt decoders, each pinned to a separate GPU, run concurrently from
// two independent threads and both converge to the correct output.
TEST_F(TRTDecoderTest, TwoTrtDecodersConcurrently) {
  if (!gpu_available())
    GTEST_SKIP() << "No CUDA GPU available";
  int count = 0;
  cudaGetDeviceCount(&count);
  if (count < 2)
    GTEST_SKIP() << "needs >= 2 GPUs";
  std::string onnx_path = get_onnx_asset_path();
  if (!std::filesystem::exists(onnx_path))
    GTEST_SKIP() << "ONNX model not found: " << onnx_path;

  std::size_t num_detectors = NUM_DETECTORS;
  cudaqx::tensor<uint8_t> H_mat({num_detectors, num_detectors});
  for (std::size_t i = 0; i < num_detectors; ++i)
    H_mat.at({i, i}) = 1;

  cudaqx::heterogeneous_map params0;
  params0.insert("onnx_load_path", onnx_path);
  params0.insert("cuda_device_id", 0);
  cudaqx::heterogeneous_map params1;
  params1.insert("onnx_load_path", onnx_path);
  params1.insert("cuda_device_id", 1);

  std::unique_ptr<decoder> dec0, dec1;
  try {
    dec0 = decoder::get("trt_decoder", H_mat, params0);
    dec1 = decoder::get("trt_decoder", H_mat, params1);
  } catch (const std::exception &e) {
    GTEST_SKIP() << "TRT construction failed: " << e.what();
  }

  std::vector<cudaq::qec::float_t> syndrome(TEST_INPUTS[0].begin(),
                                            TEST_INPUTS[0].end());

  cudaq::qec::decoder_result res0, res1;
  std::exception_ptr ex0, ex1;
  std::thread t0([&] {
    try {
      res0 = dec0->decode(syndrome);
    } catch (...) {
      ex0 = std::current_exception();
    }
  });
  std::thread t1([&] {
    try {
      res1 = dec1->decode(syndrome);
    } catch (...) {
      ex1 = std::current_exception();
    }
  });
  t0.join();
  t1.join();

  if (ex0)
    std::rethrow_exception(ex0);
  if (ex1)
    std::rethrow_exception(ex1);

  float expected_output = TEST_OUTPUTS[0][0];
  EXPECT_TRUE(res0.converged);
  ASSERT_FALSE(res0.result.empty());
  EXPECT_LT(std::abs(res0.result[0] - expected_output), 1e-4f)
      << "GPU-0 decode differs from expected: got " << res0.result[0]
      << ", expected " << expected_output;
  EXPECT_TRUE(res1.converged);
  ASSERT_FALSE(res1.result.empty());
  EXPECT_LT(std::abs(res1.result[0] - expected_output), 1e-4f)
      << "GPU-1 decode differs from expected: got " << res1.result[0]
      << ", expected " << expected_output;
}

// Container runtimes commonly block the mempolicy syscalls (seccomp without
// CAP_SYS_NICE). trt_decoder::decode_batch() degrades gracefully there (warns,
// keeps going), but a test that wants to exercise the mempolicy path must skip
// instead of fail. Mirrors the guard in test_device_affinity.cpp.
TEST(TrtDecoder, DecodeBatchAppliesNumaPolicy) {
#if defined(__linux__)
  int mode = -1;
  if (syscall(SYS_get_mempolicy, &mode, nullptr, 0UL, nullptr, 0UL) != 0)
    GTEST_SKIP() << "mempolicy syscalls unavailable (container seccomp?)";
#endif
  if (!gpu_available())
    GTEST_SKIP() << "No CUDA GPU available";
  std::string onnx_path = get_onnx_asset_path();
  if (!std::filesystem::exists(onnx_path))
    GTEST_SKIP() << "ONNX model not found: " << onnx_path;

  std::size_t num_detectors = NUM_DETECTORS;
  cudaqx::tensor<uint8_t> H({num_detectors, num_detectors});
  for (std::size_t i = 0; i < num_detectors; ++i)
    H.at({i, i}) = 1;

  cudaqx::heterogeneous_map params;
  params.insert("onnx_load_path", onnx_path);
  params.insert("numa_node_id", 0);

  std::unique_ptr<decoder> trt_decoder;
  try {
    trt_decoder = decoder::get("trt_decoder", H, params);
  } catch (const std::exception &e) {
    GTEST_SKIP() << "Failed to create TRT decoder: " << e.what();
  }

  std::vector<cudaq::qec::float_t> syndrome(TEST_INPUTS[0].begin(),
                                            TEST_INPUTS[0].end());
  std::vector<decoder_result> results;
  // decode_batch() applies the NUMA guard for the duration of the call and
  // restores the thread's prior policy before returning, so this only proves
  // no crash/throw occurs with a numa_node_id knob set; the mempolicy syscall
  // behavior itself is covered at the primitive level by
  // HardwareAffinity.MempolicyBindWhenRequested.
  EXPECT_NO_THROW({ results = trt_decoder->decode_batch({syndrome}); });
  ASSERT_EQ(results.size(), 1u);
  EXPECT_TRUE(results[0].converged);
  ASSERT_FALSE(results[0].result.empty());
}

#if defined(__linux__)
// Snapshot of the calling thread's placement: CPU affinity mask plus thread
// memory-policy mode and nodemask (raw SYS_get_mempolicy, matching what the
// decode_batch guard saves/restores).
struct thread_placement {
  cpu_set_t mask;
  int mempolicy_mode = -1;
  unsigned long nodemask[16] = {0};
};

static std::optional<thread_placement> capture_thread_placement() {
  thread_placement p;
  CPU_ZERO(&p.mask);
  if (sched_getaffinity(0, sizeof(p.mask), &p.mask) != 0)
    return std::nullopt;
  if (syscall(SYS_get_mempolicy, &p.mempolicy_mode, p.nodemask,
              sizeof(p.nodemask) * 8, nullptr, 0UL) != 0)
    return std::nullopt;
  return p;
}

static void expect_same_placement(const thread_placement &before,
                                  const thread_placement &after) {
  EXPECT_TRUE(CPU_EQUAL(&before.mask, &after.mask))
      << "CPU affinity mask changed across decode_batch()";
  EXPECT_EQ(before.mempolicy_mode, after.mempolicy_mode)
      << "thread mempolicy mode changed across decode_batch()";
  EXPECT_EQ(
      0, std::memcmp(before.nodemask, after.nodemask, sizeof(before.nodemask)))
      << "thread mempolicy nodemask changed across decode_batch()";
}

// Shared skip guard for the placement tests below (GPU + ONNX asset + NUMA
// node 0 + mempolicy syscalls, which container seccomp commonly blocks).
static std::optional<std::string> placement_test_skip_reason() {
  if (!gpu_available())
    return "No CUDA GPU available";
  if (!std::filesystem::exists(get_onnx_asset_path()))
    return "ONNX model not found: " + get_onnx_asset_path();
  if (!std::filesystem::exists("/sys/devices/system/node/node0"))
    return "NUMA node 0 not present";
  int mode = -1;
  if (syscall(SYS_get_mempolicy, &mode, nullptr, 0UL, nullptr, 0UL) != 0)
    return "mempolicy syscalls unavailable (container seccomp?)";
  return std::nullopt;
}

static void expect_first_output_correct(
    const std::vector<cudaq::qec::decoder_result> &results) {
  ASSERT_EQ(results.size(), 1u);
  EXPECT_TRUE(results[0].converged);
  ASSERT_FALSE(results[0].result.empty());
  EXPECT_LT(std::abs(results[0].result[0] - TEST_OUTPUTS[0][0]), 1e-4f)
      << "decode_batch output differs from expected: got "
      << results[0].result[0] << ", expected " << TEST_OUTPUTS[0][0];
}
#endif

// A thread that called bind_current_thread() owns its placement: the
// decode_batch() guard must not fire on it. Placement is captured after the
// bind and must be bit-identical after decode_batch() — regression test for
// the pinning-defeat bug where the guard re-bound (and widened) an
// already-bound thread.
TEST(TrtDecoder, BoundThreadSkipsGuardInDecodeBatch) {
#if defined(__linux__)
  if (auto reason = placement_test_skip_reason())
    GTEST_SKIP() << *reason;

  cudaqx::heterogeneous_map params;
  params.insert("onnx_load_path", get_onnx_asset_path());
  params.insert("cuda_device_id", 0);
  params.insert("numa_node_id", 0);
  std::unique_ptr<decoder> trt_decoder;
  try {
    trt_decoder =
        decoder::get("trt_decoder", make_identity_h(NUM_DETECTORS), params);
  } catch (const std::exception &e) {
    GTEST_SKIP() << "Failed to create TRT decoder: " << e.what();
  }

  std::vector<cudaq::qec::float_t> syndrome(TEST_INPUTS[0].begin(),
                                            TEST_INPUTS[0].end());
  // bind_current_thread() pins persistently (no restore), so run on a worker
  // thread to keep the gtest main thread's placement intact for later tests.
  std::optional<thread_placement> before, after;
  std::vector<cudaq::qec::decoder_result> results;
  std::exception_ptr err;
  // A guard that fires and restores perfectly is indistinguishable from a
  // skipped guard by before/after placement alone; the set_mempolicy counter
  // is the observable difference (TRT itself never calls set_mempolicy).
  const bool have_shim = cudaqx_affinity_syscall_reset != nullptr &&
                         cudaqx_set_mempolicy_count != nullptr;
  long setmem_during_decode = 0;
  std::thread worker([&] {
    try {
      trt_decoder->bind_current_thread();
      before = capture_thread_placement();
      if (have_shim)
        cudaqx_affinity_syscall_reset();
      results = trt_decoder->decode_batch({syndrome});
      if (have_shim)
        setmem_during_decode = cudaqx_set_mempolicy_count();
      after = capture_thread_placement();
    } catch (...) {
      err = std::current_exception();
    }
  });
  worker.join();
  if (err)
    std::rethrow_exception(err);

  expect_first_output_correct(results);
  ASSERT_TRUE(before.has_value());
  ASSERT_TRUE(after.has_value());
  expect_same_placement(*before, *after);
  if (have_shim)
    EXPECT_EQ(setmem_during_decode, 0)
        << "bound decode_batch() issued set_mempolicy: the guard fired on a "
           "thread that already called bind_current_thread()";
#else
  GTEST_SKIP() << "Linux-only placement test";
#endif
}

// Unbound caller: decode_batch() applies the NUMA guard for the duration of
// the call and must restore the caller's placement exactly before returning.
TEST(TrtDecoder, UnboundDecodeBatchRestoresPlacement) {
#if defined(__linux__)
  if (auto reason = placement_test_skip_reason())
    GTEST_SKIP() << *reason;

  cudaqx::heterogeneous_map params;
  params.insert("onnx_load_path", get_onnx_asset_path());
  params.insert("cuda_device_id", 0);
  params.insert("numa_node_id", 0);
  std::unique_ptr<decoder> trt_decoder;
  try {
    trt_decoder =
        decoder::get("trt_decoder", make_identity_h(NUM_DETECTORS), params);
  } catch (const std::exception &e) {
    GTEST_SKIP() << "Failed to create TRT decoder: " << e.what();
  }

  std::vector<cudaq::qec::float_t> syndrome(TEST_INPUTS[0].begin(),
                                            TEST_INPUTS[0].end());
  auto before = capture_thread_placement();
  ASSERT_TRUE(before.has_value());
  auto results = trt_decoder->decode_batch({syndrome});
  auto after = capture_thread_placement();
  ASSERT_TRUE(after.has_value());

  expect_first_output_correct(results);
  expect_same_placement(*before, *after);
#else
  GTEST_SKIP() << "Linux-only placement test";
#endif
}

// An explicit cpu_affinity={0} bind is strictly narrower than node 0's cpuset:
// if decode_batch()'s _ScopedNuma guard fired on the bound thread it would
// widen the mask to the whole node, so "exactly CPU 0 before AND after" proves
// the explicit pin survives decode_batch().
TEST(TrtDecoder, ExplicitCpuAffinityHonoredWhenBound) {
#if defined(__linux__)
  if (auto reason = placement_test_skip_reason())
    GTEST_SKIP() << *reason;

  cudaqx::heterogeneous_map params;
  params.insert("onnx_load_path", get_onnx_asset_path());
  params.insert("numa_node_id", 0);
  params.insert("cpu_affinity", std::vector<int>{0});
  std::unique_ptr<decoder> trt_decoder;
  try {
    trt_decoder =
        decoder::get("trt_decoder", make_identity_h(NUM_DETECTORS), params);
  } catch (const std::exception &e) {
    GTEST_SKIP() << "Failed to create TRT decoder: " << e.what();
  }

  std::vector<cudaq::qec::float_t> syndrome(TEST_INPUTS[0].begin(),
                                            TEST_INPUTS[0].end());
  cpu_set_t only_cpu0;
  CPU_ZERO(&only_cpu0);
  CPU_SET(0, &only_cpu0);

  std::optional<thread_placement> before, after;
  std::vector<cudaq::qec::decoder_result> results;
  std::exception_ptr err;
  std::thread worker([&] {
    try {
      trt_decoder->bind_current_thread();
      before = capture_thread_placement();
      results = trt_decoder->decode_batch({syndrome});
      after = capture_thread_placement();
    } catch (...) {
      err = std::current_exception();
    }
  });
  worker.join();
  if (err)
    std::rethrow_exception(err);

  expect_first_output_correct(results);
  ASSERT_TRUE(before.has_value());
  ASSERT_TRUE(after.has_value());
  EXPECT_TRUE(CPU_EQUAL(&only_cpu0, &before->mask))
      << "bind_current_thread() did not honor cpu_affinity={0}";
  EXPECT_TRUE(CPU_EQUAL(&only_cpu0, &after->mask))
      << "decode_batch() widened the explicit cpu_affinity={0} pin";
  expect_same_placement(*before, *after);
#else
  GTEST_SKIP() << "Linux-only placement test";
#endif
}

// Dependency-free regression guard: decoder::get()'s own construction-time
// CudaDeviceGuard rejects an out-of-range cuda_device_id before the plugin is
// ever instantiated, so decode_batch()'s own device-guard error-checking is
// never reached for this particular failure mode. Kept as a guard on the
// overall contract (out-of-range device ids must never silently proceed).
TEST(TrtDecoder, DecodeBatchThrowsOnOutOfRangeCudaDeviceId) {
  int count = 0;
  cudaGetDeviceCount(&count);
  cudaqx::tensor<uint8_t> H({2, 2});
  H.at({0, 0}) = 1;
  H.at({1, 1}) = 1;
  cudaqx::heterogeneous_map params;
  params.insert("onnx_load_path", get_onnx_asset_path());
  params.insert("cuda_device_id", count + 5); // deliberately out of range
  EXPECT_THROW(
      { cudaq::qec::decoder::get("trt_decoder", H, params); },
      std::runtime_error);
}
