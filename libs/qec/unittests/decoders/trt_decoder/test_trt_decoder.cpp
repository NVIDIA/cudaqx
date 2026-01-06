/*******************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
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
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <random>
#include <vector>

using namespace cudaq::qec;

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
  // Check if the ONNX model file exists
  std::string onnx_path = "surface_code_decoder.onnx";
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
  // Check if the ONNX model file exists
  std::string onnx_path = "surface_code_decoder.onnx";
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
  // Check if the ONNX model file exists
  std::string onnx_path = "surface_code_decoder.onnx";
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

// Note: Constructor tests and parse_precision tests are disabled because they
// require actual TensorRT/CUDA initialization which is not available in the
// test environment. Only parameter validation and utility function tests are
// enabled above.
