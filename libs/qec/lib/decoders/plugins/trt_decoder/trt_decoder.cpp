/*******************************************************************************
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/Logger.h"
#include "cudaq/qec/decoder.h"
#include "cudaq/qec/trt_decoder_internal.h"
#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

// TensorRT headers
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "cuda_runtime_api.h"

// Remove the leading path elements from a fully qualified filename
static inline void trim_filename(std::string &filename) {
  std::size_t lastSlashPos = filename.find_last_of('/');
  if (lastSlashPos != std::string::npos)
    filename = filename.substr(lastSlashPos + 1);
}

#ifndef cudaCheckError
#define cudaCheckError()                                                       \
  do {                                                                         \
    cudaError_t e = cudaGetLastError();                                        \
    if (e != cudaSuccess) {                                                    \
      std::string filename = __FILE__;                                         \
      trim_filename(filename);                                                 \
      printf("CUDA ERROR %s:%d: '%s'\n", filename.c_str(), __LINE__,           \
             cudaGetErrorString(e));                                           \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)
#endif

#ifndef HANDLE_CUDA_ERROR
#define HANDLE_CUDA_ERROR(x)                                                   \
  do {                                                                         \
    const auto err = x;                                                        \
    if (err != cudaSuccess) {                                                  \
      std::string filename = __FILE__;                                         \
      trim_filename(filename);                                                 \
      printf("CUDA ERROR %s:%d: '%s'\n", filename.c_str(), __LINE__,           \
             cudaGetErrorString(err));                                         \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)
#endif

#ifndef HANDLE_CUDA_ERROR_NO_THROW
#define HANDLE_CUDA_ERROR_NO_THROW(x)                                          \
  do {                                                                         \
    const auto err = x;                                                        \
    if (err != cudaSuccess) {                                                  \
      std::string filename = __FILE__;                                         \
      trim_filename(filename);                                                 \
      printf("CUDA ERROR %s:%d: '%s'\n", filename.c_str(), __LINE__,           \
             cudaGetErrorString(err));                                         \
    }                                                                          \
  } while (0)
#endif

// Logger for TensorRT
class Logger : public nvinfer1::ILogger {
public:
  void log(Severity severity, const char *msg) noexcept override {
    try {
      // filter out info-level messages
      if (severity >= Severity::kWARNING) {
        CUDAQ_INFO("[TensorRT] {}", msg);
      } else {
        CUDAQ_WARN("[TensorRT] {}", msg);
      }
    } catch (...) {
      // Silently ignore - can't throw from a noexcept function
    }
  }
};

static Logger gLogger;

/// @brief TensorRT-based decoder for quantum error correction
/// This decoder leverages NVIDIA TensorRT for accelerated inference
///
/// Constructor parameters:
/// - "onnx_load_path": Path to ONNX model file (will build TensorRT engine)
/// - "engine_load_path": Path to pre-built TensorRT engine file (loads
/// directly)
/// - "engine_save_path": Path to save built TensorRT engine (optional)
/// - "precision": Precision mode for inference (optional, default: "best")
///   Options: "fp16", "bf16", "int8", "fp8", "noTF32", "best"
/// - "memory_workspace": Memory workspace size in bytes (optional, default:
/// 1GB)
///
/// Note: Only one of onnx_load_path or engine_load_path should be specified,
/// not both.
namespace cudaq::qec {

class trt_decoder : public decoder {
private:
  // TensorRT-specific members
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;
  int input_index_ = 0;
  int output_index_ = 0;
  int input_size_ = 0;
  int output_size_ = 0;
  void *buffers_[2] = {nullptr, nullptr};
  cudaStream_t stream_;
  bool initialized_ = false;

public:
  trt_decoder(const cudaqx::tensor<uint8_t> &H,
              const cudaqx::heterogeneous_map &params)
      : decoder(H), initialized_(false) {
    // Decoder-specific constructor arguments can be placed in `params`.

    try {
      // Validate parameters
      trt_decoder_internal::validate_trt_decoder_parameters(params);

      // Check if CUDA is available
      check_cuda();

      bool has_engine_path = params.contains("engine_load_path");

      if (has_engine_path) {
        // Load pre-built TensorRT engine directly
        std::string engine_path = params.get<std::string>("engine_load_path");
        auto engineData = trt_decoder_internal::load_file(engine_path);

        // Create runtime and deserialize engine
        auto runtime = std::unique_ptr<nvinfer1::IRuntime>(
            nvinfer1::createInferRuntime(gLogger));
        if (!runtime) {
          throw std::runtime_error("Failed to create TensorRT runtime");
        }

        engine_ = std::unique_ptr<nvinfer1::ICudaEngine>(
            runtime->deserializeCudaEngine(engineData.data(),
                                           engineData.size()));
        if (!engine_) {
          throw std::runtime_error(
              "Failed to deserialize TensorRT engine from: " + engine_path);
        }
      } else {
        // Load ONNX model and build engine
        std::string onnx_model_path = params.get<std::string>("onnx_load_path");
        engine_ = trt_decoder_internal::build_engine_from_onnx(onnx_model_path,
                                                               params, gLogger);

        // Save engine if requested
        if (params.contains("engine_save_path")) {
          std::string engine_save_path =
              params.get<std::string>("engine_save_path");
          trt_decoder_internal::save_engine_to_file(engine_.get(),
                                                    engine_save_path);
        }
      }

      // Create execution context
      context_ = std::unique_ptr<nvinfer1::IExecutionContext>(
          engine_->createExecutionContext());
      if (!context_) {
        throw std::runtime_error("Failed to create execution context");
      }

      // Get input/output info
      int n_bindings = engine_->getNbIOTensors();
      input_index_ = -1;
      output_index_ = -1;
      for (int i = 0; i < n_bindings; ++i) {
        const char *tensorName = engine_->getIOTensorName(i);
        if (engine_->getTensorIOMode(tensorName) ==
            nvinfer1::TensorIOMode::kINPUT) {
          input_index_ = i;
        } else {
          output_index_ = i;
        }
      }

      if (input_index_ == -1 || output_index_ == -1) {
        throw std::runtime_error("Failed to identify input/output tensors");
      }

      auto inputDims =
          engine_->getTensorShape(engine_->getIOTensorName(input_index_));
      input_size_ = 1;
      for (int j = 0; j < inputDims.nbDims; ++j)
        input_size_ *= inputDims.d[j];

      auto outputDims =
          engine_->getTensorShape(engine_->getIOTensorName(output_index_));
      output_size_ = 1;
      for (int j = 0; j < outputDims.nbDims; ++j)
        output_size_ *= outputDims.d[j];

      // Allocate GPU buffers
      HANDLE_CUDA_ERROR(
          cudaMalloc(&buffers_[input_index_], input_size_ * sizeof(float)));
      HANDLE_CUDA_ERROR(
          cudaMalloc(&buffers_[output_index_], output_size_ * sizeof(float)));

      // Create CUDA stream
      HANDLE_CUDA_ERROR(cudaStreamCreate(&stream_));

      initialized_ = true;

    } catch (const std::exception &e) {
      CUDAQ_WARN("TensorRT initialization failed: {}", e.what());
      initialized_ = false;
    }
  }

  virtual decoder_result decode(const std::vector<float_t> &syndrome) override {
    decoder_result result{false, std::vector<float_t>(output_size_, 0.0)};

    if (!initialized_) {
      // Return unconverged result if not properly initialized
      return result;
    }

    try {
      // Preprocess syndrome data for TensorRT input
      // Ensure input size matches expected TensorRT input size
      assert(syndrome.size() == input_size_);
      std::vector<float> input_host(syndrome.begin(), syndrome.end());

      // Copy input to GPU
      HANDLE_CUDA_ERROR(cudaMemcpy(buffers_[input_index_], input_host.data(),
                                   input_size_ * sizeof(float),
                                   cudaMemcpyHostToDevice));

      // Set tensor addresses for TensorRT V1 API
      context_->setTensorAddress(engine_->getIOTensorName(input_index_),
                                 buffers_[input_index_]);
      context_->setTensorAddress(engine_->getIOTensorName(output_index_),
                                 buffers_[output_index_]);

      // Run inference
      context_->enqueueV3(stream_);
      HANDLE_CUDA_ERROR(cudaStreamSynchronize(stream_));

      // Copy output back from GPU
      std::vector<float> output_host(output_size_);
      HANDLE_CUDA_ERROR(cudaMemcpy(output_host.data(), buffers_[output_index_],
                                   output_size_ * sizeof(float),
                                   cudaMemcpyDeviceToHost));

      // Postprocess output to get error probabilities
      std::transform(output_host.begin(), output_host.end(),
                     result.result.begin(),
                     [](float val) { return static_cast<float_t>(val); });

      result.converged = true;

    } catch (const std::exception &e) {
      CUDAQ_WARN("TensorRT inference failed: {}", e.what());
      result.converged = false;
    }

    return result;
  }

  virtual ~trt_decoder() {
    // Clean up TensorRT resources
    if (initialized_) {
      HANDLE_CUDA_ERROR(cudaStreamDestroy(stream_));
      HANDLE_CUDA_ERROR(cudaFree(buffers_[input_index_]));
      HANDLE_CUDA_ERROR(cudaFree(buffers_[output_index_]));
      // TensorRT engine and context will be automatically destroyed by
      // unique_ptr
    }
  }

private:
  void check_cuda() {
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    if (error != cudaSuccess || deviceCount == 0) {
      throw std::runtime_error(
          "CUDA is not available or no CUDA-capable devices found. "
          "TensorRT decoder requires CUDA to be installed and at least one "
          "CUDA-capable GPU. Error: " +
          std::string(cudaGetErrorString(error)));
    }
  }

public:
  CUDAQ_EXTENSION_CUSTOM_CREATOR_FUNCTION(
      trt_decoder, static std::unique_ptr<decoder> create(
                       const cudaqx::tensor<uint8_t> &H,
                       const cudaqx::heterogeneous_map &params) {
        return std::make_unique<trt_decoder>(H, params);
      })
};

} // namespace cudaq::qec

namespace cudaq::qec {

CUDAQ_REGISTER_TYPE(trt_decoder)

} // namespace cudaq::qec

namespace cudaq::qec::trt_decoder_internal {

// Hardware platform detection class
class HardwarePlatform {
private:
  bool has_fp16_ = false;
  bool has_int8_ = false;
  bool has_bf16_ = false;
  bool has_tf32_ = false;
  bool has_fp8_ = false;
  ;

public:
  HardwarePlatform() {
    int device;
    HANDLE_CUDA_ERROR(cudaGetDevice(&device));

    cudaDeviceProp prop;
    HANDLE_CUDA_ERROR(cudaGetDeviceProperties(&prop, device));

    // ---- FP16 ----
    if (prop.major > 6 || (prop.major == 6 && prop.minor >= 0))
      has_fp16_ = true;
    else if (prop.major == 5 && prop.minor == 3)
      has_fp16_ = false;

    // ---- INT8 ----
    if (prop.major > 6 || (prop.major == 6 && prop.minor >= 1))
      has_int8_ = true;
    else
      has_int8_ = false;

    // ---- BF16 ----
    // BF16 support is available on compute capability >= 8.0 (Ampere and later)
    if (prop.major >= 8) {
      has_bf16_ = true;
    }

    // ---- TF32 ----
    // TF32 support is available on compute capability >= 8.0 (Ampere and later)
    if (prop.major >= 8) {
      has_tf32_ = true;
    }

    // ---- FP8 ----
    // FP8 support is available on compute capability >= 9.0 (Hopper and later)
    if (prop.major >= 9) {
      has_fp8_ = true;
    }
  }

  // Getter methods for device capabilities
  bool device_has_fp16() const { return has_fp16_; }
  bool device_has_int8() const { return has_int8_; }
  bool device_has_bf16() const { return has_bf16_; }
  bool device_has_tf32() const { return has_tf32_; }
  bool device_has_fp8() const { return has_fp8_; }
};

// Free function to build TensorRT engine from ONNX model

// Utility: load binary files (ONNX models or TensorRT engines)
std::vector<char> load_file(const std::string &filename) {
  std::ifstream file(filename, std::ios::binary | std::ios::ate);
  if (!file)
    throw std::runtime_error("Failed to open file: " + filename);
  size_t size = file.tellg();
  std::vector<char> buffer(size);
  file.seekg(0, std::ios::beg);
  file.read(buffer.data(), size);
  return buffer;
}

// Implementation of parameter validation
void validate_trt_decoder_parameters(const cudaqx::heterogeneous_map &params) {
  // Check for mutually exclusive parameters
  bool has_onnx_path = params.contains("onnx_load_path");
  bool has_engine_path = params.contains("engine_load_path");

  if (has_onnx_path && has_engine_path) {
    throw std::runtime_error(
        "TensorRT decoder cannot specify both 'onnx_load_path' and "
        "'engine_load_path' parameters. Please choose one.");
  }

  if (!has_onnx_path && !has_engine_path) {
    throw std::runtime_error(
        "TensorRT decoder requires either 'onnx_load_path' or "
        "'engine_load_path' parameter");
  }
}

// Implementation of the helper function to parse and configure precision
// settings
void parse_precision(const std::string &precision,
                     nvinfer1::IBuilderConfig *config) {
  // Create hardware platform detector
  HardwarePlatform platform;

  if (precision == "fp16") {
    if (platform.device_has_fp16()) {
      config->setFlag(nvinfer1::BuilderFlag::kFP16);
    } else {
      CUDAQ_WARN("Warning: FP16 requested but not supported on this platform, "
                 "using FP32");
    }
  } else if (precision == "bf16") {
    if (platform.device_has_bf16()) {
      config->setFlag(nvinfer1::BuilderFlag::kBF16);
    } else {
      CUDAQ_WARN("Warning: BF16 requested but not supported on this platform, "
                 "using FP32");
    }
  } else if (precision == "int8") {
    if (platform.device_has_int8()) {
      config->setFlag(nvinfer1::BuilderFlag::kINT8);
    } else {
      CUDAQ_WARN("Warning: INT8 requested but not supported on this platform, "
                 "using FP32");
    }
  } else if (precision == "fp8") {
    if (platform.device_has_fp8()) {
      config->setFlag(nvinfer1::BuilderFlag::kFP8);
    } else {
      CUDAQ_WARN("Warning: FP8 requested but not supported on this platform, "
                 "using FP32");
    }
  } else if (precision == "tf32") {
    if (platform.device_has_tf32()) {
      config->setFlag(nvinfer1::BuilderFlag::kTF32);
    } else {
      CUDAQ_WARN("Warning: TF32 requested but not supported on this platform, "
                 "using FP32");
    }
  } else if (precision == "noTF32") {
    config->setFlag(nvinfer1::BuilderFlag::kDISABLE_TIMING_CACHE);
    // Note: This disables timing cache, not TF32 directly
    // TF32 is controlled by the kTF32 flag, which is enabled by default in
    // newer TensorRT versions
  } else if (precision == "best") {
    // Let TensorRT choose the best precision automatically
    // This is the default behavior, no additional flags needed
  } else {
    CUDAQ_WARN("Warning: Unknown precision '{}', using default (best)",
               precision);
  }
}

std::unique_ptr<nvinfer1::ICudaEngine>
build_engine_from_onnx(const std::string &onnx_model_path,
                       const cudaqx::heterogeneous_map &params,
                       nvinfer1::ILogger &logger) {
  // Create builder, network, and parser
  auto builder =
      std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
  if (!builder) {
    throw std::runtime_error("Failed to create TensorRT builder");
  }

  auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(
      builder->createNetworkV2(0U));
  if (!network) {
    throw std::runtime_error("Failed to create TensorRT network");
  }

  auto parser = std::unique_ptr<nvonnxparser::IParser>(
      nvonnxparser::createParser(*network, logger));
  if (!parser) {
    throw std::runtime_error("Failed to create ONNX parser");
  }

  // Parse ONNX
  auto onnx_data = load_file(onnx_model_path);
  if (!parser->parse(onnx_data.data(), onnx_data.size())) {
    throw std::runtime_error("Failed to parse ONNX model: " + onnx_model_path);
  }

  // Build config
  auto config =
      std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());

  // Set memory workspace size (default: 1GB)
  size_t workspace_size = 1ULL << 30; // 1GB default
  if (params.contains("memory_workspace")) {
    workspace_size = params.get<size_t>("memory_workspace");
  }
  config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE,
                             workspace_size);

  // Configure precision based on user preference
  std::string precision = "best"; // default
  if (params.contains("precision")) {
    precision = params.get<std::string>("precision");
  }
  parse_precision(precision, config.get());

  // Build engine
  auto engine = std::unique_ptr<nvinfer1::ICudaEngine>(
      builder->buildEngineWithConfig(*network, *config));
  if (!engine) {
    throw std::runtime_error("Failed to build TensorRT engine");
  }

  return engine;
}

// Implementation of the helper function to save TensorRT engine to file
void save_engine_to_file(nvinfer1::ICudaEngine *engine,
                         const std::string &file_path) {
  if (!engine) {
    throw std::runtime_error("Cannot save null engine");
  }

  // Serialize engine
  auto serialized_engine =
      std::unique_ptr<nvinfer1::IHostMemory>(engine->serialize());
  if (!serialized_engine) {
    throw std::runtime_error("Failed to serialize TensorRT engine");
  }

  // Write to file
  std::ofstream file(file_path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Failed to open file for writing: " + file_path);
  }

  file.write(static_cast<const char *>(serialized_engine->data()),
             serialized_engine->size());
  if (!file.good()) {
    throw std::runtime_error("Failed to write engine to file: " + file_path);
  }
  file.close();

  CUDAQ_INFO("TensorRT engine saved to: {}", file_path);
}

} // namespace cudaq::qec::trt_decoder_internal
