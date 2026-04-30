/*******************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/qec/decoder.h"
#include "cudaq/qec/trt_decoder_internal.h"
#include "cudaq/runtime/logger/logger.h"
#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <type_traits>
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
    // Suppress the harmless warning about logger reuse that appears when
    // Python tests build engines with Python's trt.Logger and C++ decoders
    // load those engines. Both loggers remain valid; TensorRT uses whichever
    // was registered first. This is expected behavior for mixed Python/C++
    // usage.
    std::string_view msg_view(msg);
    if (msg_view.find("logger passed into") != std::string_view::npos &&
        msg_view.find("differs from one already registered") !=
            std::string_view::npos) {
      return; // Suppress this specific warning
    }

    // filter out info-level messages
    if (severity >= Severity::kWARNING) {
      CUDAQ_INFO("[TensorRT] {}", msg);
    } else {
      CUDAQ_WARN("[TensorRT] {}", msg);
    }
  }
};

static Logger gLogger;

/// @brief TensorRT-based decoder for quantum error correction
/// This decoder leverages NVIDIA TensorRT for accelerated inference.
///
/// The network is built with strong typing
/// (NetworkDefinitionCreationFlag::kSTRONGLY_TYPED), so TensorRT preserves the
/// data types defined in the ONNX model exactly.  I/O tensor data types
/// (float32, uint8) are automatically detected from the engine and host-side
/// double values are converted to/from the engine's native types transparently.
///
/// Constructor parameters:
/// - "onnx_load_path": Path to ONNX model file (will build TensorRT engine)
/// - "engine_load_path": Path to pre-built TensorRT engine file (loads
///   directly)
/// - "engine_save_path": Path to save built TensorRT engine (optional)
/// - "precision": (optional, default: "best")  Accepted values: "tf32",
///   "noTF32", "best".  Legacy values "fp16", "bf16", "int8", "fp8" are
///   accepted but ignored with strongly-typed networks -- export the ONNX
///   model in the desired precision instead.
/// - "memory_workspace": Memory workspace size in bytes (optional, default:
///   1GB)
/// - "batch_size": Required when the ONNX model has a dynamic batch dim
///   (-1). Used to size the optimization profile and I/O buffers.
/// - "global_decoder": Optional name of a decoder to run after TRT
///   (e.g. DEM decoder). The TRT model is assumed to have detectors as
///   inputs and either (a) residual detectors as the only output, or
///   (b) when "O" is also provided, the concatenation [pre_L,
///   residual_dets] as the only output.
/// - "global_decoder_params": Optional parameters for the global decoder. The
///   decoder is created with the same H passed to the trt_decoder constructor.
/// - "O": Observables matrix (num_observables x block_size). Calls to
///   decode() and decode_batch() will return the logical frame of the
///   observables. Requires that the TRT model emits the concatenation
///   [pre_L (num_observables entries), residual_dets (rest)] as a single
///   output. When a global_decoder is also set, the final result is
///   pre_L XOR global_decoder(residual_dets); otherwise only the pre_L
///   prefix is returned.
///
/// Note: Only one of onnx_load_path or engine_load_path should be specified,
/// not both.
namespace cudaq::qec {

// ============================================================================
// Executor implementations (internal)
// ============================================================================

namespace {

// Helpers for templated I/O: binarize TRT output (float or uint8) to 0/1
// for counting and for the decoder API (float_t).
inline bool trt_io_nonzero(float val) { return val >= 0.5f; }
inline bool trt_io_nonzero(uint8_t val) { return val != 0; }
inline float_t trt_io_to_binary(float val) {
  return (val >= 0.5f) ? static_cast<float_t>(1.0) : static_cast<float_t>(0.0);
}
inline float_t trt_io_to_binary(uint8_t val) {
  return (val != 0) ? static_cast<float_t>(1.0) : static_cast<float_t>(0.0);
}

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
                                     size_t input_byte_size) {
  CaptureResult result;

  try {
    // Zero-fill the GPU input buffer (values don't matter for capture)
    cudaError_t err = cudaMemset(input_buffer, 0, input_byte_size);
    if (err != cudaSuccess) {
      result.error_message = "Failed to zero-fill input buffer: " +
                             std::string(cudaGetErrorString(err));
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

size_t dataTypeSize(nvinfer1::DataType dtype) {
  switch (dtype) {
  case nvinfer1::DataType::kFLOAT:
    return sizeof(float);
  case nvinfer1::DataType::kHALF:
    return 2;
  case nvinfer1::DataType::kINT8:
    return 1;
  case nvinfer1::DataType::kINT32:
    return sizeof(int32_t);
  case nvinfer1::DataType::kBOOL:
    return 1;
  case nvinfer1::DataType::kUINT8:
    return sizeof(uint8_t);
  case nvinfer1::DataType::kFP8:
    return 1;
  case nvinfer1::DataType::kBF16:
    return 2;
  case nvinfer1::DataType::kINT64:
    return sizeof(int64_t);
  default:
    return sizeof(float);
  }
}

const char *dataTypeName(nvinfer1::DataType dtype) {
  switch (dtype) {
  case nvinfer1::DataType::kFLOAT:
    return "float32";
  case nvinfer1::DataType::kHALF:
    return "float16";
  case nvinfer1::DataType::kINT8:
    return "int8";
  case nvinfer1::DataType::kINT32:
    return "int32";
  case nvinfer1::DataType::kBOOL:
    return "bool";
  case nvinfer1::DataType::kUINT8:
    return "uint8";
  case nvinfer1::DataType::kFP8:
    return "fp8";
  case nvinfer1::DataType::kBF16:
    return "bfloat16";
  case nvinfer1::DataType::kINT64:
    return "int64";
  default:
    return "unknown";
  }
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

  // Batch dimension from TensorRT model (first dimension of input tensor)
  size_t model_batch_size_ = 1;

  // Per-sample sizes (without batch dimension)
  size_t syndrome_size_per_sample_ = 0;
  size_t output_size_per_sample_ = 0;

  // Optional global decoder (e.g. DEM decoder) applied after TRT + postprocess
  std::unique_ptr<decoder> global_decoder_;
  cudaqx::heterogeneous_map global_decoder_params_;

  // When true, decode()/decode_batch() return the predicted logical-frame
  // observables. The TRT model must emit the concatenation
  // [pre_L (num_observables_ entries), residual_dets (rest)] as its single
  // output. Enabled by passing the "O" (observables) parameter.
  bool decode_to_observables_ = false;
  size_t num_observables_ = 0;

public:
  trt_decoder(const cudaqx::tensor<uint8_t> &H,
              const cudaqx::heterogeneous_map &params);

  virtual decoder_result decode(const std::vector<float_t> &syndrome) override;

  virtual std::vector<decoder_result>
  decode_batch(const std::vector<std::vector<float_t>> &syndromes) override;

  virtual ~trt_decoder();

  CUDAQ_EXTENSION_CUSTOM_CREATOR_FUNCTION(
      trt_decoder, static std::unique_ptr<decoder> create(
                       const cudaqx::tensor<uint8_t> &H,
                       const cudaqx::heterogeneous_map &params) {
        return std::make_unique<trt_decoder>(H, params);
      })

private:
  void check_cuda();

  /// Typed decode_batch: IoType matches the engine's I/O dtype
  /// (currently float or uint8_t).
  template <typename IoType>
  std::vector<decoder_result>
  decode_batch_impl(const std::vector<std::vector<float_t>> &syndromes) const;
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
  nvinfer1::DataType input_dtype = nvinfer1::DataType::kFLOAT;
  nvinfer1::DataType output_dtype = nvinfer1::DataType::kFLOAT;
  size_t input_elem_size = sizeof(float);
  size_t output_elem_size = sizeof(float);
  void *buffers[2] = {nullptr, nullptr};
  cudaStream_t stream;

  // Dynamic batch: set input shape before each inference (only when dynamic)
  bool has_dynamic_batch_ = false;
  std::string input_name_;
  nvinfer1::Dims input_dims_{0, {}};

  // Executor (chosen once at construction, never changes)
  std::variant<TraditionalExecutor, CudaGraphExecutor> executor;

  /// actual_batch is used only when has_dynamic_batch_.
  void execute_inference(size_t actual_batch = 0) {
    if (has_dynamic_batch_) {
      nvinfer1::Dims dims;
      dims.nbDims = input_dims_.nbDims;
      dims.d[0] = static_cast<int32_t>(actual_batch);
      for (int i = 1; i < input_dims_.nbDims; ++i)
        dims.d[i] = input_dims_.d[i];
      if (!context->setInputShape(input_name_.c_str(), dims)) {
        throw std::runtime_error("setInputShape failed for batch size " +
                                 std::to_string(actual_batch));
      }
    }
    std::visit(
        [&](auto &exec) {
          exec.execute(context.get(), stream, buffers[input_index],
                       buffers[output_index], input_index, output_index,
                       engine.get());
        },
        executor);
  }

  ~Impl() {
    // IMPORTANT: Destroy resources in the correct order.

    // 1. Synchronise the stream so all async work completes
    if (stream) {
      cudaStreamSynchronize(stream);
    }

    // 2. Destroy the CUDA graph executor BEFORE the stream it was captured on
    executor = TraditionalExecutor{};

    // 3. Destroy TensorRT execution context and engine BEFORE freeing their
    //    underlying GPU memory
    context.reset();
    engine.reset();

    // 4. Free GPU buffers
    if (buffers[input_index]) {
      HANDLE_CUDA_ERROR_NO_THROW(cudaFree(buffers[input_index]));
      buffers[input_index] = nullptr;
    }
    if (buffers[output_index]) {
      HANDLE_CUDA_ERROR_NO_THROW(cudaFree(buffers[output_index]));
      buffers[output_index] = nullptr;
    }

    // 5. Destroy stream last
    if (stream) {
      HANDLE_CUDA_ERROR_NO_THROW(cudaStreamDestroy(stream));
      stream = nullptr;
    }
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
    if (n_bindings != 2) {
      throw std::runtime_error(
          "TensorRT decoder expects exactly 2 I/O tensors (1 input + 1 "
          "output), got " +
          std::to_string(n_bindings));
    }

    const char *inputTensorName =
        impl_->engine->getIOTensorName(impl_->input_index);
    const char *outputTensorName =
        impl_->engine->getIOTensorName(impl_->output_index);

    impl_->input_dtype = impl_->engine->getTensorDataType(inputTensorName);
    impl_->output_dtype = impl_->engine->getTensorDataType(outputTensorName);
    impl_->input_elem_size = dataTypeSize(impl_->input_dtype);
    impl_->output_elem_size = dataTypeSize(impl_->output_dtype);

    CUDAQ_INFO("TensorRT engine I/O data types: input={}, output={}",
               dataTypeName(impl_->input_dtype),
               dataTypeName(impl_->output_dtype));

    if (impl_->input_dtype != nvinfer1::DataType::kFLOAT &&
        impl_->input_dtype != nvinfer1::DataType::kUINT8) {
      throw std::runtime_error("Unsupported input tensor data type: " +
                               std::string(dataTypeName(impl_->input_dtype)) +
                               ". Supported types: float32, uint8");
    }
    if (impl_->output_dtype != nvinfer1::DataType::kFLOAT &&
        impl_->output_dtype != nvinfer1::DataType::kUINT8) {
      throw std::runtime_error("Unsupported output tensor data type: " +
                               std::string(dataTypeName(impl_->output_dtype)) +
                               ". Supported types: float32, uint8");
    }

    auto inputDims = impl_->engine->getTensorShape(inputTensorName);

    impl_->has_dynamic_batch_ = (inputDims.nbDims > 0 && inputDims.d[0] == -1);

    if (impl_->has_dynamic_batch_) {
      if (!params.contains("batch_size")) {
        // FIXME - should we just default to 1 or throw an error?
        throw std::runtime_error(
            "TensorRT decoder: model has dynamic batch dimension but "
            "'batch_size' was not set in params (required for allocation)");
      }
      model_batch_size_ = params.get<size_t>("batch_size");
      if (model_batch_size_ < 1) {
        throw std::runtime_error(
            "TensorRT decoder: batch_size must be >= 1, got " +
            std::to_string(model_batch_size_));
      }
      impl_->input_name_ = impl_->engine->getIOTensorName(impl_->input_index);
      impl_->input_dims_ = inputDims;
      syndrome_size_per_sample_ = 1;
      for (int j = 1; j < inputDims.nbDims; ++j)
        syndrome_size_per_sample_ *= inputDims.d[j];
      impl_->input_size =
          static_cast<int>(model_batch_size_ * syndrome_size_per_sample_);
    } else {
      if (inputDims.nbDims > 0 && inputDims.d[0] > 0) {
        model_batch_size_ = static_cast<size_t>(inputDims.d[0]);
      } else {
        model_batch_size_ = 1;
      }

      // Calculate total input size and per-sample size
      impl_->input_size = 1;
      for (int j = 0; j < inputDims.nbDims; ++j)
        impl_->input_size *= inputDims.d[j];
      syndrome_size_per_sample_ = impl_->input_size / model_batch_size_;
    }

    auto outputDims = impl_->engine->getTensorShape(outputTensorName);
    output_size_per_sample_ = 1;
    for (int j = 1; j < outputDims.nbDims; ++j)
      output_size_per_sample_ *= (outputDims.d[j] > 0 ? outputDims.d[j] : 1);
    if (outputDims.nbDims > 0 && outputDims.d[0] > 0) {
      impl_->output_size = 1;
      for (int j = 0; j < outputDims.nbDims; ++j)
        impl_->output_size *= outputDims.d[j];
      output_size_per_sample_ = impl_->output_size / model_batch_size_;
    } else {
      impl_->output_size =
          static_cast<int>(model_batch_size_ * output_size_per_sample_);
    }

    CUDAQ_INFO("TensorRT model configuration: batch_size={}, "
               "syndrome_size_per_sample={}, output_size_per_sample={}",
               model_batch_size_, syndrome_size_per_sample_,
               output_size_per_sample_);

    // Allocate GPU buffers (sized according to engine I/O data types)
    HANDLE_CUDA_ERROR(cudaMalloc(&impl_->buffers[impl_->input_index],
                                 impl_->input_size * impl_->input_elem_size));
    HANDLE_CUDA_ERROR(cudaMalloc(&impl_->buffers[impl_->output_index],
                                 impl_->output_size * impl_->output_elem_size));

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
          impl_->output_index, impl_->engine.get(),
          impl_->input_size * impl_->input_elem_size);

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

    // Optional global decoder (e.g. DEM decoder), similar to sliding_window's
    // inner_decoder. When set, decode_batch will run: syndrome->trainX->TRT
    // ->postprocess->global_decoder->results.
    if (params.contains("global_decoder") &&
        params.contains("global_decoder_params")) {
      std::string global_decoder_name =
          params.get<std::string>("global_decoder");
      global_decoder_params_ =
          params.get<cudaqx::heterogeneous_map>("global_decoder_params");
      if (!global_decoder_name.empty()) {
        global_decoder_ =
            decoder::get(global_decoder_name, H, global_decoder_params_);
        CUDAQ_INFO("TensorRT decoder: global_decoder '{}' attached",
                   global_decoder_name);
      }
    }

    if (params.contains("O")) {
      auto O = params.get<cudaqx::tensor<uint8_t>>("O");
      if (O.rank() != 2) {
        throw std::runtime_error(
            "trt_decoder: O must be a 2-dimensional tensor (num_observables x "
            "block_size)");
      }
      if (O.shape()[1] != block_size) {
        throw std::runtime_error(
            "trt_decoder: O second dimension must equal H block_size (got " +
            std::to_string(O.shape()[1]) + ", block_size " +
            std::to_string(block_size) + ")");
      }
      decode_to_observables_ = true;
      num_observables_ = O.shape()[0];

      // The TRT model output must encode [pre_L (num_observables_ entries),
      // residual_dets (rest)]. Validate sizing where we can.
      if (output_size_per_sample_ < num_observables_) {
        throw std::runtime_error(
            "trt_decoder: TRT output_size_per_sample (" +
            std::to_string(output_size_per_sample_) +
            ") is smaller than num_observables (" +
            std::to_string(num_observables_) +
            "); model output must be [pre_L, residual_dets].");
      }
      if (global_decoder_) {
        const size_t expected =
            num_observables_ + global_decoder_->get_syndrome_size();
        if (output_size_per_sample_ != expected) {
          throw std::runtime_error(
              "trt_decoder: TRT output_size_per_sample (" +
              std::to_string(output_size_per_sample_) +
              ") must equal num_observables + global_decoder.syndrome_size "
              "(" +
              std::to_string(expected) +
              ") for the [pre_L, residual_dets] split.");
        }
      }
      CUDAQ_INFO("TensorRT decoder: decode_to_observables enabled "
                 "(num_observables={})",
                 num_observables_);
    }

    // Decoder is now fully configured and ready for inference
    decoder_ready_ = true;

  } catch (const std::exception &e) {
    CUDAQ_WARN("TensorRT initialization failed: {}", e.what());
    decoder_ready_ = false;
  }
}

decoder_result trt_decoder::decode(const std::vector<float_t> &syndrome) {
  // Validate syndrome size
  if (syndrome.size() != syndrome_size_per_sample_) {
    throw std::runtime_error("Syndrome size mismatch: expected " +
                             std::to_string(syndrome_size_per_sample_) +
                             " but got " + std::to_string(syndrome.size()));
  }

  // For models with batch_size > 1, zero-pad to fill the batch
  // This allows decode() to work with any batch size by filling unused slots
  // with zeros
  if (model_batch_size_ > 1) {
    CUDAQ_INFO(
        "Model has batch_size={}, zero-padding single syndrome to fill batch",
        model_batch_size_);

    // Create a batch with the real syndrome plus zero-padded syndromes
    std::vector<std::vector<float_t>> padded_batch;
    padded_batch.reserve(model_batch_size_);

    // First syndrome is the real one
    padded_batch.push_back(syndrome);

    // Fill remaining batch slots with zero syndromes
    std::vector<float_t> zero_syndrome(syndrome_size_per_sample_, 0.0f);
    for (size_t i = 1; i < model_batch_size_; ++i) {
      padded_batch.push_back(zero_syndrome);
    }

    auto results = decode_batch(padded_batch);

    // Return only the first result (the real syndrome)
    return results[0];
  }

  // For batch_size == 1, directly delegate to decode_batch
  auto results = decode_batch({syndrome});
  return results[0];
}

std::vector<decoder_result>
trt_decoder::decode_batch(const std::vector<std::vector<float_t>> &syndromes) {
  // Validate that we have syndromes to decode
  if (syndromes.empty()) {
    return {};
  }

  // Validate all syndrome sizes match expected size
  for (size_t i = 0; i < syndromes.size(); ++i) {
    if (syndromes[i].size() != syndrome_size_per_sample_) {
      throw std::runtime_error(
          "Syndrome size mismatch at index " + std::to_string(i) +
          ": expected " + std::to_string(syndrome_size_per_sample_) +
          " but got " + std::to_string(syndromes[i].size()));
    }
  }

  if (!decoder_ready_) {
    // Return unconverged results if decoder is not ready
    CUDAQ_WARN(
        "Decoder not ready for inference, returning {} unconverged results. "
        "Check decoder initialization logs for errors.",
        syndromes.size());

    std::vector<decoder_result> results(syndromes.size());
    const size_t result_size =
        decode_to_observables_ ? num_observables_ : output_size_per_sample_;
    for (auto &result : results) {
      result.converged = false;
      result.result.resize(result_size, 0.0);
    }
    return results;
  }

  // Dispatch on the actual engine input dtype (uint8 or float).
  if (impl_->input_dtype == nvinfer1::DataType::kUINT8)
    return decode_batch_impl<uint8_t>(syndromes);
  return decode_batch_impl<float>(syndromes);
}

template <typename IoType>
std::vector<decoder_result> trt_decoder::decode_batch_impl(
    const std::vector<std::vector<float_t>> &syndromes) const {
  std::vector<decoder_result> results;
  results.reserve(syndromes.size());

  // Output split for the predecoder pattern: when decode_to_observables_ is
  // on the TRT output is [pre_L (num_observables_), residual_dets (rest)].
  const size_t pre_L_size = decode_to_observables_ ? num_observables_ : 0;
  const size_t residual_size = output_size_per_sample_ - pre_L_size;

  try {
    size_t total_input_nonzero = 0;
    size_t total_residual_nonzero = 0;
    const bool log_residual_counts =
        cudaq::details::should_log(cudaq::details::LogLevel::info);

    for (size_t batch_start = 0; batch_start < syndromes.size();
         batch_start += model_batch_size_) {

      const size_t actual_batch =
          std::min(model_batch_size_, syndromes.size() - batch_start);

      // Prepare input batch. For float input we preserve soft (raw) values;
      // for uint8 we binarize to 0/1.
      std::vector<IoType> input_host(impl_->input_size);
      for (size_t batch_idx = 0; batch_idx < actual_batch; ++batch_idx) {
        const auto &syndrome = syndromes[batch_start + batch_idx];
        for (size_t i = 0; i < syndrome_size_per_sample_; ++i) {
          if constexpr (std::is_same_v<IoType, float>) {
            input_host[batch_idx * syndrome_size_per_sample_ + i] =
                static_cast<IoType>(syndrome[i]);
          } else {
            input_host[batch_idx * syndrome_size_per_sample_ + i] =
                static_cast<IoType>(syndrome[i] >= 0.5f ? 1 : 0);
          }
        }
      }

      HANDLE_CUDA_ERROR(cudaMemcpy(
          impl_->buffers[impl_->input_index], input_host.data(),
          impl_->input_size * sizeof(IoType), cudaMemcpyHostToDevice));

      impl_->execute_inference(actual_batch);

      std::vector<IoType> output_host(impl_->output_size);
      HANDLE_CUDA_ERROR(cudaMemcpy(
          output_host.data(), impl_->buffers[impl_->output_index],
          impl_->output_size * sizeof(IoType), cudaMemcpyDeviceToHost));

      if (log_residual_counts) {
        const size_t input_elems = actual_batch * syndrome_size_per_sample_;
        for (size_t i = 0; i < input_elems; ++i)
          if (trt_io_nonzero(input_host[i]))
            total_input_nonzero++;
        // Count non-zero entries in just the residual portion of the output.
        for (size_t batch_idx = 0; batch_idx < actual_batch; ++batch_idx) {
          const IoType *row = output_host.data() +
                              batch_idx * output_size_per_sample_ + pre_L_size;
          for (size_t i = 0; i < residual_size; ++i)
            if (trt_io_nonzero(row[i]))
              total_residual_nonzero++;
        }
      }

      if (global_decoder_) {
        // Build the global-decoder input from the residual portion of the
        // TRT output.
        const size_t global_syndrome_size =
            global_decoder_->get_syndrome_size();
        if (residual_size != global_syndrome_size) {
          throw std::runtime_error(
              "trt_decoder: residual portion of TRT output (" +
              std::to_string(residual_size) +
              ") != global_decoder.syndrome_size (" +
              std::to_string(global_syndrome_size) + ")");
        }
        std::vector<std::vector<float_t>> residual_soft(
            actual_batch, std::vector<float_t>(global_syndrome_size, 0.0f));
        for (size_t batch_idx = 0; batch_idx < actual_batch; ++batch_idx) {
          const IoType *res = output_host.data() +
                              batch_idx * output_size_per_sample_ + pre_L_size;
          float_t *out = residual_soft[batch_idx].data();
          for (size_t i = 0; i < global_syndrome_size; ++i)
            out[i] = trt_io_to_binary(res[i]);
        }
        std::vector<decoder_result> global_results =
            global_decoder_->decode_batch(residual_soft);

        if (decode_to_observables_) {
          // Combine pre_L (the prefix of the TRT output) with the global
          // decoder's logical-frame prediction via XOR.
          for (size_t batch_idx = 0; batch_idx < actual_batch; ++batch_idx) {
            decoder_result combined;
            combined.converged = global_results[batch_idx].converged;
            combined.result.resize(num_observables_, 0.0f);
            const IoType *pre_L_row =
                output_host.data() + batch_idx * output_size_per_sample_;
            const std::vector<float_t> &g = global_results[batch_idx].result;
            for (size_t k = 0; k < num_observables_; ++k) {
              const uint8_t a = trt_io_nonzero(pre_L_row[k]) ? 1u : 0u;
              const uint8_t b = (k < g.size() && g[k] >= 0.5f) ? 1u : 0u;
              combined.result[k] = static_cast<float_t>(a ^ b);
            }
            results.push_back(std::move(combined));
          }
        } else {
          for (decoder_result &r : global_results)
            results.push_back(std::move(r));
        }
      } else {
        // No global decoder. If decode_to_observables_ is set, return only
        // the pre_L prefix; otherwise return the full TRT output.
        const size_t out_per_sample =
            decode_to_observables_ ? num_observables_ : output_size_per_sample_;
        for (size_t batch_idx = 0; batch_idx < actual_batch; ++batch_idx) {
          decoder_result result;
          result.converged = true;
          result.result.resize(out_per_sample);
          const IoType *row =
              output_host.data() + batch_idx * output_size_per_sample_;
          for (size_t i = 0; i < out_per_sample; ++i) {
            if constexpr (std::is_same_v<IoType, float>) {
              result.result[i] = static_cast<float_t>(row[i]);
            } else {
              result.result[i] = trt_io_to_binary(row[i]);
            }
          }
          results.push_back(std::move(result));
        }
      }
    }

    if (log_residual_counts) {
      CUDAQ_INFO("TRT decoder: total non-zero input detectors = {}, total "
                 "non-zero residual detectors = {}",
                 total_input_nonzero, total_residual_nonzero);
    }

  } catch (const std::exception &e) {
    CUDAQ_WARN("TensorRT batch inference failed: {}", e.what());
    // Mark all results as unconverged
    for (auto &result : results) {
      result.converged = false;
    }
  }

  return results;
}

// Explicit instantiations for the supported single-output engine I/O dtypes.
template std::vector<decoder_result> trt_decoder::decode_batch_impl<float>(
    const std::vector<std::vector<float_t>> &) const;
template std::vector<decoder_result> trt_decoder::decode_batch_impl<uint8_t>(
    const std::vector<std::vector<float_t>> &) const;

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

void parse_precision(const std::string &precision,
                     nvinfer1::IBuilderConfig *config) {
  if (precision == "tf32") {
    config->setFlag(nvinfer1::BuilderFlag::kTF32);
  } else if (precision == "noTF32") {
    config->clearFlag(nvinfer1::BuilderFlag::kTF32);
  } else if (precision == "best") {
    // With strongly-typed networks the ONNX model's native types are used.
  } else if (precision == "fp16" || precision == "bf16" ||
             precision == "int8" || precision == "fp8") {
    CUDAQ_WARN("Precision '{}' is ignored when building strongly-typed "
               "networks. TensorRT uses the data types defined in the ONNX "
               "model. To use {}, export the model with the desired types.",
               precision, precision);
  } else {
    CUDAQ_WARN("Unknown precision '{}', using default (best)", precision);
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

  auto network =
      std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(
          1U << static_cast<uint32_t>(
              nvinfer1::NetworkDefinitionCreationFlag::kSTRONGLY_TYPED)));
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

  // The following is required for cases when using .onnx file with dynamic
  // batch dimension.
  if (params.contains("batch_size") && network->getNbInputs() > 0) {
    nvinfer1::ITensor *input = network->getInput(0);
    nvinfer1::Dims dims = input->getDimensions();
    if (dims.nbDims > 0 && dims.d[0] == -1) {
      const size_t batch_size = params.get<size_t>("batch_size");
      if (batch_size < 1) {
        throw std::runtime_error("batch_size must be >= 1, got " +
                                 std::to_string(batch_size));
      }
      nvinfer1::IOptimizationProfile *profile =
          builder->createOptimizationProfile();
      nvinfer1::Dims minDims = dims;
      nvinfer1::Dims optDims = dims;
      nvinfer1::Dims maxDims = dims;
      minDims.d[0] = 1;
      optDims.d[0] = static_cast<int32_t>(batch_size);
      maxDims.d[0] = static_cast<int32_t>(batch_size);
      if (!profile->setDimensions(
              input->getName(), nvinfer1::OptProfileSelector::kMIN, minDims) ||
          !profile->setDimensions(
              input->getName(), nvinfer1::OptProfileSelector::kOPT, optDims) ||
          !profile->setDimensions(
              input->getName(), nvinfer1::OptProfileSelector::kMAX, maxDims)) {
        throw std::runtime_error(
            "Failed to set optimization profile dimensions for batch");
      }
      config->addOptimizationProfile(profile);
      CUDAQ_INFO("TensorRT optimization profile: batch min=1, opt=max={}",
                 batch_size);
    }
  }

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
