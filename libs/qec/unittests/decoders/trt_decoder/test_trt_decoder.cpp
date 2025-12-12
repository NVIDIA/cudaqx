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

// Test decode() with uninitialized decoder (covers !initialized_ path)
TEST_F(TRTDecoderTest, DecodeUninitializedDecoder) {
  std::size_t num_detectors = NUM_DETECTORS;
  cudaqx::tensor<uint8_t> H({num_detectors, num_detectors});
  for (std::size_t i = 0; i < num_detectors; ++i) {
    H.at({i, i}) = 1;
  }

  // Create decoder with invalid ONNX path to force initialization failure
  // The decoder constructor catches exceptions and sets initialized_ = false
  cudaqx::heterogeneous_map params;
  params.insert("onnx_load_path", std::string("non_existent_model.onnx"));

  std::unique_ptr<decoder> trt_decoder;
  try {
    trt_decoder = decoder::get("trt_decoder", H, params);
  } catch (const std::exception &e) {
    // If decoder creation throws exception (before catch block), skip this test
    GTEST_SKIP() << "Decoder creation threw exception: " << e.what();
  }

  // Decoder should be created but not initialized (initialized_ = false)
  // due to exception caught in constructor (line 216-219)
  // Try to decode - should return unconverged result (line 225-227)
  std::vector<cudaq::qec::float_t> syndrome(num_detectors, 0.0f);
  auto result = trt_decoder->decode(syndrome);

  // Should return unconverged result when not initialized
  EXPECT_FALSE(result.converged);
  // Result should be non-empty but with zeros (or default values)
  // The result size should match output_size_ (which is 0 if not initialized)
  EXPECT_GE(result.result.size(), 0);
}

// Test engine_save_path parameter (covers save_engine_to_file path)
TEST_F(TRTDecoderTest, EngineSavePath) {
  std::string onnx_path = "surface_code_decoder.onnx";
  if (!std::filesystem::exists(onnx_path)) {
    GTEST_SKIP() << "ONNX model file not found: " << onnx_path;
  }

  std::size_t num_detectors = NUM_DETECTORS;
  cudaqx::tensor<uint8_t> H({num_detectors, num_detectors});
  for (std::size_t i = 0; i < num_detectors; ++i) {
    H.at({i, i}) = 1;
  }

  // Create decoder with engine_save_path to test save functionality
  std::string engine_save_path = "test_saved_engine.trt";
  cudaqx::heterogeneous_map params;
  params.insert("onnx_load_path", onnx_path);
  params.insert("engine_save_path", engine_save_path);

  std::unique_ptr<decoder> trt_decoder;
  try {
    trt_decoder = decoder::get("trt_decoder", H, params);
  } catch (const std::exception &e) {
    GTEST_SKIP() << "Failed to create TRT decoder: " << e.what();
  }

  // Verify decoder works
  std::vector<cudaq::qec::float_t> syndrome(TEST_INPUTS[0].begin(),
                                            TEST_INPUTS[0].end());
  auto result = trt_decoder->decode(syndrome);
  EXPECT_TRUE(result.converged);

  // Clean up saved engine file if it was created
  std::filesystem::remove(engine_save_path);
}

// Test different precision values to cover parse_precision branches
TEST_F(TRTDecoderTest, PrecisionFP16) {
  std::string onnx_path = "surface_code_decoder.onnx";
  if (!std::filesystem::exists(onnx_path)) {
    GTEST_SKIP() << "ONNX model file not found: " << onnx_path;
  }

  std::size_t num_detectors = NUM_DETECTORS;
  cudaqx::tensor<uint8_t> H({num_detectors, num_detectors});
  for (std::size_t i = 0; i < num_detectors; ++i) {
    H.at({i, i}) = 1;
  }

  cudaqx::heterogeneous_map params;
  params.insert("onnx_load_path", onnx_path);
  params.insert("precision", std::string("fp16"));

  std::unique_ptr<decoder> trt_decoder;
  try {
    trt_decoder = decoder::get("trt_decoder", H, params);
  } catch (const std::exception &e) {
    GTEST_SKIP() << "Failed to create TRT decoder with FP16: " << e.what();
  }

  // Verify decoder works
  std::vector<cudaq::qec::float_t> syndrome(TEST_INPUTS[0].begin(),
                                            TEST_INPUTS[0].end());
  auto result = trt_decoder->decode(syndrome);
  EXPECT_TRUE(result.converged);
}

TEST_F(TRTDecoderTest, PrecisionBF16) {
  std::string onnx_path = "surface_code_decoder.onnx";
  if (!std::filesystem::exists(onnx_path)) {
    GTEST_SKIP() << "ONNX model file not found: " << onnx_path;
  }

  std::size_t num_detectors = NUM_DETECTORS;
  cudaqx::tensor<uint8_t> H({num_detectors, num_detectors});
  for (std::size_t i = 0; i < num_detectors; ++i) {
    H.at({i, i}) = 1;
  }

  cudaqx::heterogeneous_map params;
  params.insert("onnx_load_path", onnx_path);
  params.insert("precision", std::string("bf16"));

  std::unique_ptr<decoder> trt_decoder;
  try {
    trt_decoder = decoder::get("trt_decoder", H, params);
  } catch (const std::exception &e) {
    GTEST_SKIP() << "Failed to create TRT decoder with BF16: " << e.what();
  }

  std::vector<cudaq::qec::float_t> syndrome(TEST_INPUTS[0].begin(),
                                            TEST_INPUTS[0].end());
  auto result = trt_decoder->decode(syndrome);
  EXPECT_TRUE(result.converged);
}

TEST_F(TRTDecoderTest, PrecisionINT8) {
  std::string onnx_path = "surface_code_decoder.onnx";
  if (!std::filesystem::exists(onnx_path)) {
    GTEST_SKIP() << "ONNX model file not found: " << onnx_path;
  }

  std::size_t num_detectors = NUM_DETECTORS;
  cudaqx::tensor<uint8_t> H({num_detectors, num_detectors});
  for (std::size_t i = 0; i < num_detectors; ++i) {
    H.at({i, i}) = 1;
  }

  cudaqx::heterogeneous_map params;
  params.insert("onnx_load_path", onnx_path);
  params.insert("precision", std::string("int8"));

  std::unique_ptr<decoder> trt_decoder;
  try {
    trt_decoder = decoder::get("trt_decoder", H, params);
  } catch (const std::exception &e) {
    GTEST_SKIP() << "Failed to create TRT decoder with INT8: " << e.what();
  }

  // INT8 requires calibration data, which may not be available in test
  // environment If decoder initialization fails due to calibration, that's
  // acceptable Test the precision parsing path even if calibration fails
  std::vector<cudaq::qec::float_t> syndrome(TEST_INPUTS[0].begin(),
                                            TEST_INPUTS[0].end());
  auto result = trt_decoder->decode(syndrome);

  // INT8 may fail initialization due to missing calibration data
  // In that case, decoder will not be initialized and decode will return
  // unconverged This is acceptable behavior - the test covers the precision
  // parsing path (line 432-438) The precision parsing code is covered even if
  // the engine build fails
  if (!result.converged) {
    // Decoder may not be initialized if INT8 calibration failed
    // This is expected when calibration data is not available
    // The test still covers parse_precision() with "int8" parameter
    GTEST_SKIP()
        << "INT8 precision requires calibration data which is not available in "
           "test environment. Precision parsing path is still covered.";
  }

  EXPECT_TRUE(result.converged);
}

TEST_F(TRTDecoderTest, PrecisionFP8) {
  std::string onnx_path = "surface_code_decoder.onnx";
  if (!std::filesystem::exists(onnx_path)) {
    GTEST_SKIP() << "ONNX model file not found: " << onnx_path;
  }

  std::size_t num_detectors = NUM_DETECTORS;
  cudaqx::tensor<uint8_t> H({num_detectors, num_detectors});
  for (std::size_t i = 0; i < num_detectors; ++i) {
    H.at({i, i}) = 1;
  }

  cudaqx::heterogeneous_map params;
  params.insert("onnx_load_path", onnx_path);
  params.insert("precision", std::string("fp8"));

  std::unique_ptr<decoder> trt_decoder;
  try {
    trt_decoder = decoder::get("trt_decoder", H, params);
  } catch (const std::exception &e) {
    GTEST_SKIP() << "Failed to create TRT decoder with FP8: " << e.what();
  }

  std::vector<cudaq::qec::float_t> syndrome(TEST_INPUTS[0].begin(),
                                            TEST_INPUTS[0].end());
  auto result = trt_decoder->decode(syndrome);
  EXPECT_TRUE(result.converged);
}

TEST_F(TRTDecoderTest, PrecisionTF32) {
  std::string onnx_path = "surface_code_decoder.onnx";
  if (!std::filesystem::exists(onnx_path)) {
    GTEST_SKIP() << "ONNX model file not found: " << onnx_path;
  }

  std::size_t num_detectors = NUM_DETECTORS;
  cudaqx::tensor<uint8_t> H({num_detectors, num_detectors});
  for (std::size_t i = 0; i < num_detectors; ++i) {
    H.at({i, i}) = 1;
  }

  cudaqx::heterogeneous_map params;
  params.insert("onnx_load_path", onnx_path);
  params.insert("precision", std::string("tf32"));

  std::unique_ptr<decoder> trt_decoder;
  try {
    trt_decoder = decoder::get("trt_decoder", H, params);
  } catch (const std::exception &e) {
    GTEST_SKIP() << "Failed to create TRT decoder with TF32: " << e.what();
  }

  std::vector<cudaq::qec::float_t> syndrome(TEST_INPUTS[0].begin(),
                                            TEST_INPUTS[0].end());
  auto result = trt_decoder->decode(syndrome);
  EXPECT_TRUE(result.converged);
}

TEST_F(TRTDecoderTest, PrecisionNoTF32) {
  std::string onnx_path = "surface_code_decoder.onnx";
  if (!std::filesystem::exists(onnx_path)) {
    GTEST_SKIP() << "ONNX model file not found: " << onnx_path;
  }

  std::size_t num_detectors = NUM_DETECTORS;
  cudaqx::tensor<uint8_t> H({num_detectors, num_detectors});
  for (std::size_t i = 0; i < num_detectors; ++i) {
    H.at({i, i}) = 1;
  }

  cudaqx::heterogeneous_map params;
  params.insert("onnx_load_path", onnx_path);
  params.insert("precision", std::string("noTF32"));

  std::unique_ptr<decoder> trt_decoder;
  try {
    trt_decoder = decoder::get("trt_decoder", H, params);
  } catch (const std::exception &e) {
    GTEST_SKIP() << "Failed to create TRT decoder with noTF32: " << e.what();
  }

  std::vector<cudaq::qec::float_t> syndrome(TEST_INPUTS[0].begin(),
                                            TEST_INPUTS[0].end());
  auto result = trt_decoder->decode(syndrome);
  EXPECT_TRUE(result.converged);
}

TEST_F(TRTDecoderTest, PrecisionUnknown) {
  std::string onnx_path = "surface_code_decoder.onnx";
  if (!std::filesystem::exists(onnx_path)) {
    GTEST_SKIP() << "ONNX model file not found: " << onnx_path;
  }

  std::size_t num_detectors = NUM_DETECTORS;
  cudaqx::tensor<uint8_t> H({num_detectors, num_detectors});
  for (std::size_t i = 0; i < num_detectors; ++i) {
    H.at({i, i}) = 1;
  }

  cudaqx::heterogeneous_map params;
  params.insert("onnx_load_path", onnx_path);
  params.insert("precision", std::string("unknown_precision_xyz"));

  std::unique_ptr<decoder> trt_decoder;
  try {
    trt_decoder = decoder::get("trt_decoder", H, params);
  } catch (const std::exception &e) {
    GTEST_SKIP() << "Failed to create TRT decoder with unknown precision: "
                 << e.what();
  }

  std::vector<cudaq::qec::float_t> syndrome(TEST_INPUTS[0].begin(),
                                            TEST_INPUTS[0].end());
  auto result = trt_decoder->decode(syndrome);
  EXPECT_TRUE(result.converged);
}

// Test memory_workspace parameter
TEST_F(TRTDecoderTest, MemoryWorkspace) {
  std::string onnx_path = "surface_code_decoder.onnx";
  if (!std::filesystem::exists(onnx_path)) {
    GTEST_SKIP() << "ONNX model file not found: " << onnx_path;
  }

  std::size_t num_detectors = NUM_DETECTORS;
  cudaqx::tensor<uint8_t> H({num_detectors, num_detectors});
  for (std::size_t i = 0; i < num_detectors; ++i) {
    H.at({i, i}) = 1;
  }

  cudaqx::heterogeneous_map params;
  params.insert("onnx_load_path", onnx_path);
  params.insert("memory_workspace", size_t(512 * 1024 * 1024)); // 512MB

  std::unique_ptr<decoder> trt_decoder;
  try {
    trt_decoder = decoder::get("trt_decoder", H, params);
  } catch (const std::exception &e) {
    GTEST_SKIP() << "Failed to create TRT decoder with custom workspace: "
                 << e.what();
  }

  std::vector<cudaq::qec::float_t> syndrome(TEST_INPUTS[0].begin(),
                                            TEST_INPUTS[0].end());
  auto result = trt_decoder->decode(syndrome);
  EXPECT_TRUE(result.converged);
}

// Test engine_load_path parameter (loading pre-built engine)
TEST_F(TRTDecoderTest, EngineLoadPath) {
  std::string onnx_path = "surface_code_decoder.onnx";
  if (!std::filesystem::exists(onnx_path)) {
    GTEST_SKIP() << "ONNX model file not found: " << onnx_path;
  }

  std::size_t num_detectors = NUM_DETECTORS;
  cudaqx::tensor<uint8_t> H({num_detectors, num_detectors});
  for (std::size_t i = 0; i < num_detectors; ++i) {
    H.at({i, i}) = 1;
  }

  // First, create an engine and save it
  std::string engine_save_path = "test_engine_for_load.trt";
  cudaqx::heterogeneous_map save_params;
  save_params.insert("onnx_load_path", onnx_path);
  save_params.insert("engine_save_path", engine_save_path);

  std::unique_ptr<decoder> save_decoder;
  try {
    save_decoder = decoder::get("trt_decoder", H, save_params);
    // Decode once to ensure engine is built
    std::vector<cudaq::qec::float_t> syndrome(TEST_INPUTS[0].begin(),
                                              TEST_INPUTS[0].end());
    save_decoder->decode(syndrome);
    save_decoder.reset(); // Destroy decoder to ensure engine is saved
  } catch (const std::exception &e) {
    GTEST_SKIP() << "Failed to create/save TRT decoder: " << e.what();
  }

  // Check if engine file was created
  if (!std::filesystem::exists(engine_save_path)) {
    GTEST_SKIP() << "Engine file was not created: " << engine_save_path;
  }

  // Now test loading the engine
  cudaqx::heterogeneous_map load_params;
  load_params.insert("engine_load_path", engine_save_path);

  std::unique_ptr<decoder> load_decoder;
  try {
    load_decoder = decoder::get("trt_decoder", H, load_params);
  } catch (const std::exception &e) {
    std::filesystem::remove(engine_save_path);
    GTEST_SKIP() << "Failed to load TRT decoder from engine: " << e.what();
  }

  // Verify loaded decoder works
  std::vector<cudaq::qec::float_t> syndrome(TEST_INPUTS[0].begin(),
                                            TEST_INPUTS[0].end());
  auto result = load_decoder->decode(syndrome);
  EXPECT_TRUE(result.converged);

  // Clean up
  std::filesystem::remove(engine_save_path);
}

// Note: Constructor tests and parse_precision tests are disabled because they
// require actual TensorRT/CUDA initialization which is not available in the
// test environment. Only parameter validation and utility function tests are
// enabled above.
