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
#include <variant>
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

// ============================================================================
// Executor implementations (internal)
// ============================================================================

namespace {
// Traditional TensorRT execution without CUDA graphs
struct TraditionalExecutor {
  void execute(nvinfer1::IExecutionContext *context, cudaStream_t stream,
               void *input_buffer, void *output_buffer, int input_index,
               int output_index, nvinfer1::ICudaEngine *engine) {
    context->setTensorAddress(engine->getIOTensorName(input_index),
                              input_buffer);
    context->setTensorAddress(engine->getIOTensorName(output_index),
                              output_buffer);
    context->enqueueV3(stream);
    HANDLE_CUDA_ERROR(cudaStreamSynchronize(stream));
  }
};

// CUDA graph-based execution for optimized performance
struct CudaGraphExecutor {
  cudaGraph_t graph;
  cudaGraphExec_t graph_exec;

  // Constructor now takes ownership of pre-captured graph
  CudaGraphExecutor(cudaGraph_t g, cudaGraphExec_t ge)
      : graph(g), graph_exec(ge) {}

  // Delete copy constructor and assignment to prevent double-free
  CudaGraphExecutor(const CudaGraphExecutor &) = delete;
  CudaGraphExecutor &operator=(const CudaGraphExecutor &) = delete;

  // Move constructor - transfer ownership
  CudaGraphExecutor(CudaGraphExecutor &&other) noexcept
      : graph(other.graph), graph_exec(other.graph_exec) {
    other.graph = nullptr;
    other.graph_exec = nullptr;
  }

  // Move assignment - transfer ownership
  CudaGraphExecutor &operator=(CudaGraphExecutor &&other) noexcept {
    if (this != &other) {
      // Clean up existing resources
      if (graph_exec) {
        HANDLE_CUDA_ERROR_NO_THROW(cudaGraphExecDestroy(graph_exec));
      }
      if (graph) {
        HANDLE_CUDA_ERROR_NO_THROW(cudaGraphDestroy(graph));
      }
      // Transfer ownership
      graph = other.graph;
      graph_exec = other.graph_exec;
      other.graph = nullptr;
      other.graph_exec = nullptr;
    }
    return *this;
  }

  void execute(nvinfer1::IExecutionContext *context, cudaStream_t stream,
               void *input_buffer, void *output_buffer, int input_index,
               int output_index, nvinfer1::ICudaEngine *engine) {
    // Just launch the graph - no lazy capture needed!
    HANDLE_CUDA_ERROR(cudaGraphLaunch(graph_exec, stream));
    HANDLE_CUDA_ERROR(cudaStreamSynchronize(stream));
  }

  ~CudaGraphExecutor() {
    if (graph_exec) {
      HANDLE_CUDA_ERROR_NO_THROW(cudaGraphExecDestroy(graph_exec));
    }
    if (graph) {
      HANDLE_CUDA_ERROR_NO_THROW(cudaGraphDestroy(graph));
    }
  }
};

// Result structure for CUDA graph capture attempts
struct CaptureResult {
  bool success = false;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t graph_exec = nullptr;
  std::string error_message;
};

// Attempt to capture a CUDA graph for TensorRT inference
// Uses dummy input data to perform the capture during initialization
CaptureResult try_capture_cuda_graph(nvinfer1::IExecutionContext *context,
                                     cudaStream_t stream, void *input_buffer,
                                     void *output_buffer, int input_index,
                                     int output_index,
                                     nvinfer1::ICudaEngine *engine,
                                     size_t input_size) {
  CaptureResult result;

  try {
    // Generate dummy input data (values don't matter for capture, just shape)
    std::vector<float> dummy_input(input_size, 0.0f);

    // Copy dummy data to GPU
    cudaError_t err =
        cudaMemcpy(input_buffer, dummy_input.data(), input_size * sizeof(float),
                   cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      result.error_message =
          "Failed to copy dummy data: " + std::string(cudaGetErrorString(err));
      return result;
    }

    // Attempt to capture the graph
    CUDAQ_INFO("Attempting to capture CUDA graph during initialization...");

    err = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    if (err != cudaSuccess) {
      result.error_message = "cudaStreamBeginCapture failed: " +
                             std::string(cudaGetErrorString(err));
      return result;
    }

    // Record TensorRT operations
    context->setTensorAddress(engine->getIOTensorName(input_index),
                              input_buffer);
    context->setTensorAddress(engine->getIOTensorName(output_index),
                              output_buffer);
    context->enqueueV3(stream);

    err = cudaStreamEndCapture(stream, &result.graph);
    if (err != cudaSuccess) {
      result.error_message = "cudaStreamEndCapture failed: " +
                             std::string(cudaGetErrorString(err));
      return result;
    }

    // Instantiate the graph
    err = cudaGraphInstantiate(&result.graph_exec, result.graph, 0);
    if (err != cudaSuccess) {
      result.error_message = "cudaGraphInstantiate failed: " +
                             std::string(cudaGetErrorString(err));
      if (result.graph) {
        cudaGraphDestroy(result.graph);
        result.graph = nullptr;
      }
      return result;
    }

    CUDAQ_INFO("CUDA graph captured successfully during initialization");
    result.success = true;

  } catch (const std::exception &e) {
    result.error_message = "Exception during capture: " + std::string(e.what());
    // Clean up on failure
    if (result.graph_exec) {
      cudaGraphExecDestroy(result.graph_exec);
      result.graph_exec = nullptr;
    }
    if (result.graph) {
      cudaGraphDestroy(result.graph);
      result.graph = nullptr;
    }
  }

  return result;
}

// Check if CUDA graphs are supported for this engine
bool supports_cuda_graphs(const nvinfer1::ICudaEngine *engine) {
  // Check for dynamic shapes
  for (int i = 0; i < engine->getNbIOTensors(); ++i) {
    const char *name = engine->getIOTensorName(i);
    auto dims = engine->getTensorShape(name);
    for (int j = 0; j < dims.nbDims; ++j) {
      if (dims.d[j] == -1) {
        CUDAQ_INFO(
            "Dynamic shape detected in tensor '{}', CUDA graphs not supported",
            name);
        return false;
      }
    }
  }

  // Check for multiple optimization profiles (often used with dynamic shapes)
  if (engine->getNbOptimizationProfiles() > 1) {
    CUDAQ_INFO(
        "Multiple optimization profiles detected, CUDA graphs not supported");
    return false;
  }

  return true;
}
} // anonymous namespace

// ============================================================================
// trt_decoder implementation
// ============================================================================

class trt_decoder : public decoder {
private:
  // Forward declaration of implementation
  struct Impl;
  std::unique_ptr<Impl> impl_;

  // True when decoder is fully configured and ready for inference
  bool decoder_ready_ = false;

public:
  trt_decoder(const cudaqx::tensor<uint8_t> &H,
              const cudaqx::heterogeneous_map &params);

  virtual decoder_result decode(const std::vector<float_t> &syndrome) override;

  virtual ~trt_decoder();

  CUDAQ_EXTENSION_CUSTOM_CREATOR_FUNCTION(
      trt_decoder, static std::unique_ptr<decoder> create(
                       const cudaqx::tensor<uint8_t> &H,
                       const cudaqx::heterogeneous_map &params) {
        return std::make_unique<trt_decoder>(H, params);
      })

private:
  void check_cuda();
};

// ============================================================================
// PIMPL Implementation struct
// ============================================================================

struct trt_decoder::Impl {
  // TensorRT resources
  std::unique_ptr<nvinfer1::ICudaEngine> engine;
  std::unique_ptr<nvinfer1::IExecutionContext> context;
  int input_index = 0;
  int output_index = 0;
  int input_size = 0;
  int output_size = 0;
  void *buffers[2] = {nullptr, nullptr};
  cudaStream_t stream;

  // Executor (chosen once at construction, never changes)
  std::variant<TraditionalExecutor, CudaGraphExecutor> executor;

  // Execute inference (variant dispatch)
  void execute_inference() {
    std::visit(
        [&](auto &exec) {
          exec.execute(context.get(), stream, buffers[input_index],
                       buffers[output_index], input_index, output_index,
                       engine.get());
        },
        executor);
  }

  ~Impl() {
    if (buffers[input_index]) {
      HANDLE_CUDA_ERROR_NO_THROW(cudaFree(buffers[input_index]));
    }
    if (buffers[output_index]) {
      HANDLE_CUDA_ERROR_NO_THROW(cudaFree(buffers[output_index]));
    }
    HANDLE_CUDA_ERROR_NO_THROW(cudaStreamDestroy(stream));
  }
};

// ============================================================================
// trt_decoder method implementations
// ============================================================================

trt_decoder::trt_decoder(const cudaqx::tensor<uint8_t> &H,
                         const cudaqx::heterogeneous_map &params)
    : decoder(H), decoder_ready_(false) {

  impl_ = std::make_unique<Impl>();

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

      impl_->engine = std::unique_ptr<nvinfer1::ICudaEngine>(
          runtime->deserializeCudaEngine(engineData.data(), engineData.size()));
      if (!impl_->engine) {
        throw std::runtime_error(
            "Failed to deserialize TensorRT engine from: " + engine_path);
      }
    } else {
      // Load ONNX model and build engine
      std::string onnx_model_path = params.get<std::string>("onnx_load_path");
      impl_->engine = trt_decoder_internal::build_engine_from_onnx(
          onnx_model_path, params, gLogger);

      // Save engine if requested
      if (params.contains("engine_save_path")) {
        std::string engine_save_path =
            params.get<std::string>("engine_save_path");
        trt_decoder_internal::save_engine_to_file(impl_->engine.get(),
                                                  engine_save_path);
      }
    }

    // Create execution context
    impl_->context = std::unique_ptr<nvinfer1::IExecutionContext>(
        impl_->engine->createExecutionContext());
    if (!impl_->context) {
      throw std::runtime_error("Failed to create execution context");
    }

    // Get input/output info
    int n_bindings = impl_->engine->getNbIOTensors();
    impl_->input_index = -1;
    impl_->output_index = -1;
    for (int i = 0; i < n_bindings; ++i) {
      const char *tensorName = impl_->engine->getIOTensorName(i);
      if (impl_->engine->getTensorIOMode(tensorName) ==
          nvinfer1::TensorIOMode::kINPUT) {
        impl_->input_index = i;
      } else {
        impl_->output_index = i;
      }
    }

    if (impl_->input_index == -1 || impl_->output_index == -1) {
      throw std::runtime_error("Failed to identify input/output tensors");
    }

    auto inputDims = impl_->engine->getTensorShape(
        impl_->engine->getIOTensorName(impl_->input_index));
    impl_->input_size = 1;
    for (int j = 0; j < inputDims.nbDims; ++j)
      impl_->input_size *= inputDims.d[j];

    auto outputDims = impl_->engine->getTensorShape(
        impl_->engine->getIOTensorName(impl_->output_index));
    impl_->output_size = 1;
    for (int j = 0; j < outputDims.nbDims; ++j)
      impl_->output_size *= outputDims.d[j];

    // Allocate GPU buffers
    HANDLE_CUDA_ERROR(cudaMalloc(&impl_->buffers[impl_->input_index],
                                 impl_->input_size * sizeof(float)));
    HANDLE_CUDA_ERROR(cudaMalloc(&impl_->buffers[impl_->output_index],
                                 impl_->output_size * sizeof(float)));

    // Create CUDA stream
    HANDLE_CUDA_ERROR(cudaStreamCreate(&impl_->stream));

    // ========================================================================
    // SELECT EXECUTOR (once, at construction - never changes)
    // ========================================================================
    bool use_cuda_graph = true; // default preference

    // User override
    if (params.contains("use_cuda_graph")) {
      use_cuda_graph = params.get<bool>("use_cuda_graph");
      if (!use_cuda_graph) {
        CUDAQ_INFO("CUDA graphs explicitly disabled by user");
      }
    }

    // Check engine compatibility
    if (use_cuda_graph && !supports_cuda_graphs(impl_->engine.get())) {
      CUDAQ_WARN("Model has dynamic shapes or multiple profiles, "
                 "CUDA graphs not supported. Using traditional execution.");
      use_cuda_graph = false;
    }

    // Attempt to capture CUDA graph if enabled
    if (use_cuda_graph) {
      auto capture_result = try_capture_cuda_graph(
          impl_->context.get(), impl_->stream,
          impl_->buffers[impl_->input_index],
          impl_->buffers[impl_->output_index], impl_->input_index,
          impl_->output_index, impl_->engine.get(), impl_->input_size);

      if (capture_result.success) {
        impl_->executor =
            CudaGraphExecutor{capture_result.graph, capture_result.graph_exec};
        CUDAQ_INFO("TensorRT decoder initialized with CUDA graph execution");
      } else {
        CUDAQ_WARN("CUDA graph capture failed: {}. Falling back to traditional "
                   "execution.",
                   capture_result.error_message);
        impl_->executor = TraditionalExecutor{};
      }
    } else {
      impl_->executor = TraditionalExecutor{};
      CUDAQ_INFO("TensorRT decoder initialized with traditional execution");
    }

    // Decoder is now fully configured and ready for inference
    decoder_ready_ = true;

  } catch (const std::exception &e) {
    CUDAQ_WARN("TensorRT initialization failed: {}", e.what());
    decoder_ready_ = false;
  }
}

decoder_result trt_decoder::decode(const std::vector<float_t> &syndrome) {
  decoder_result result{false, std::vector<float_t>(impl_->output_size, 0.0)};

  if (!decoder_ready_) {
    // Return unconverged result if decoder is not ready
    return result;
  }

  try {
    // Preprocess syndrome data for TensorRT input
    // Ensure input size matches expected TensorRT input size
    assert(syndrome.size() == impl_->input_size);
    std::vector<float> input_host(syndrome.begin(), syndrome.end());

    // Copy input to GPU
    HANDLE_CUDA_ERROR(
        cudaMemcpy(impl_->buffers[impl_->input_index], input_host.data(),
                   impl_->input_size * sizeof(float), cudaMemcpyHostToDevice));

    // Execute inference (variant handles both traditional and CUDA graph paths)
    impl_->execute_inference();

    // Copy output back from GPU
    std::vector<float> output_host(impl_->output_size);
    HANDLE_CUDA_ERROR(
        cudaMemcpy(output_host.data(), impl_->buffers[impl_->output_index],
                   impl_->output_size * sizeof(float), cudaMemcpyDeviceToHost));

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

trt_decoder::~trt_decoder() = default;

void trt_decoder::check_cuda() {
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
