/*******************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/qec/decoder.h"
#include "cudaq/qec/trt_decoder_internal.h"
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

// Note: Constructor tests and parse_precision tests are disabled because they
// require actual TensorRT/CUDA initialization which is not available in the
// test environment. Only parameter validation and utility function tests are
// enabled above.
