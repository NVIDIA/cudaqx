/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/qec/realtime/ai_decoder_service.h"
#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h"
#include <NvOnnxParser.h>
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#define DECODER_CUDA_CHECK(call)                                               \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      throw std::runtime_error(                                                \
          std::string("CUDA Error in ai_decoder_service: ") +                  \
          cudaGetErrorString(err));                                            \
    }                                                                          \
  } while (0)

namespace cudaq::qec::realtime::experimental {

// =============================================================================
// Gateway Kernels
// =============================================================================

__global__ void gateway_input_kernel(void **mailbox_slot_ptr,
                                     void *trt_fixed_input,
                                     size_t copy_size_bytes) {
  void *ring_buffer_data = *mailbox_slot_ptr;
  if (ring_buffer_data == nullptr)
    return;

  const char *src =
      (const char *)ring_buffer_data + sizeof(cudaq::realtime::RPCHeader);
  char *dst = (char *)trt_fixed_input;

  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < copy_size_bytes;
       i += blockDim.x * gridDim.x) {
    dst[i] = src[i];
  }
}

__global__ void gateway_output_kernel(void **mailbox_slot_ptr,
                                      const void *trt_fixed_output,
                                      size_t result_size_bytes) {
  void *ring_buffer_data = *mailbox_slot_ptr;
  if (ring_buffer_data == nullptr)
    return;

  char *dst = (char *)ring_buffer_data + sizeof(cudaq::realtime::RPCHeader);
  const char *src = (const char *)trt_fixed_output;

  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < result_size_bytes;
       i += blockDim.x * gridDim.x) {
    dst[i] = src[i];
  }

  __syncthreads();

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    auto *header = (const cudaq::realtime::RPCHeader *)ring_buffer_data;
    uint32_t rid = header->request_id;
    uint64_t pts = header->ptp_timestamp;

    auto *response = (cudaq::realtime::RPCResponse *)ring_buffer_data;
    response->magic = cudaq::realtime::RPC_MAGIC_RESPONSE;
    response->status = 0;
    response->result_len = static_cast<uint32_t>(result_size_bytes);
    response->request_id = rid;
    response->ptp_timestamp = pts;
    __threadfence_system();
  }
}

// =============================================================================
// Helpers
// =============================================================================

static size_t trt_dtype_size(nvinfer1::DataType dtype) {
  switch (dtype) {
  case nvinfer1::DataType::kFLOAT:
    return 4;
  case nvinfer1::DataType::kHALF:
    return 2;
  case nvinfer1::DataType::kINT8:
    return 1;
  case nvinfer1::DataType::kUINT8:
    return 1;
  case nvinfer1::DataType::kINT32:
    return 4;
  case nvinfer1::DataType::kINT64:
    return 8;
  case nvinfer1::DataType::kBOOL:
    return 1;
  default:
    return 4;
  }
}

static size_t tensor_volume(const nvinfer1::Dims &d) {
  size_t v = 1;
  for (int i = 0; i < d.nbDims; ++i)
    v *= (d.d[i] > 0) ? static_cast<size_t>(d.d[i]) : 1;
  return v;
}

// =============================================================================
// Class Implementation
// =============================================================================

ai_decoder_service::Logger ai_decoder_service::gLogger;

void ai_decoder_service::Logger::log(Severity severity,
                                     const char *msg) noexcept {
  if (severity <= Severity::kWARNING) {
    std::printf("[TensorRT] %s\n", msg);
  }
}

ai_decoder_service::ai_decoder_service(const std::string &model_path,
                                       void **device_mailbox_slot,
                                       const std::string &engine_save_path)
    : device_mailbox_slot_(device_mailbox_slot) {
  std::string ext = model_path.substr(model_path.find_last_of('.'));
  if (ext == ".onnx") {
    build_engine_from_onnx(model_path, engine_save_path);
  } else {
    load_engine(model_path);
  }
  setup_bindings();
  allocate_resources();
}

ai_decoder_service::ai_decoder_service(void **device_mailbox_slot,
                                       size_t input_bytes, size_t output_bytes)
    : device_mailbox_slot_(device_mailbox_slot), input_size_(input_bytes),
      output_size_(output_bytes) {
  allocate_resources();
}

std::unique_ptr<ai_decoder_service> ai_decoder_service::create_passthrough(
    void **device_mailbox_slot, size_t input_bytes, size_t output_bytes) {
  return std::unique_ptr<ai_decoder_service>(
      new ai_decoder_service(device_mailbox_slot, input_bytes, output_bytes));
}

ai_decoder_service::~ai_decoder_service() {
  if (graph_exec_)
    cudaGraphExecDestroy(graph_exec_);
  if (d_trt_input_)
    cudaFree(d_trt_input_);
  if (d_trt_output_)
    cudaFree(d_trt_output_);
  for (auto *buf : d_aux_buffers_)
    cudaFree(buf);
}

void ai_decoder_service::load_engine(const std::string &path) {
  std::ifstream file(path, std::ios::binary);
  if (!file.good())
    throw std::runtime_error("Error opening engine file: " + path);

  file.seekg(0, file.end);
  size_t size = file.tellg();
  file.seekg(0, file.beg);

  std::vector<char> engine_data(size);
  file.read(engine_data.data(), size);

  runtime_.reset(nvinfer1::createInferRuntime(gLogger));
  engine_.reset(runtime_->deserializeCudaEngine(engine_data.data(), size));
  context_.reset(engine_->createExecutionContext());
}

void ai_decoder_service::build_engine_from_onnx(
    const std::string &onnx_path, const std::string &engine_save_path) {
  runtime_.reset(nvinfer1::createInferRuntime(gLogger));

  auto builder = std::unique_ptr<nvinfer1::IBuilder>(
      nvinfer1::createInferBuilder(gLogger));
  auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(
      builder->createNetworkV2(0));
  auto config =
      std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());

  // Enable FP16 optimization for Grace Blackwell / Hopper
  if (builder->platformHasFastFp16()) {
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
    std::printf("[TensorRT] FP16 precision enabled.\n");
  } else {
    std::printf("[TensorRT] Warning: Platform does not support fast FP16. "
                "Using FP32.\n");
  }

  auto parser = std::unique_ptr<nvonnxparser::IParser>(
      nvonnxparser::createParser(*network, gLogger));

  if (!parser->parseFromFile(
          onnx_path.c_str(),
          static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
    throw std::runtime_error("Failed to parse ONNX file: " + onnx_path);
  }

  bool has_dynamic = false;
  for (int i = 0; i < network->getNbInputs(); ++i) {
    auto *input = network->getInput(i);
    auto dims = input->getDimensions();
    for (int d = 0; d < dims.nbDims; ++d) {
      if (dims.d[d] <= 0) {
        has_dynamic = true;
        break;
      }
    }
    if (has_dynamic)
      break;
  }

  if (has_dynamic) {
    auto *profile = builder->createOptimizationProfile();
    for (int i = 0; i < network->getNbInputs(); ++i) {
      auto *input = network->getInput(i);
      auto dims = input->getDimensions();
      nvinfer1::Dims fixed = dims;
      for (int d = 0; d < fixed.nbDims; ++d) {
        if (fixed.d[d] <= 0)
          fixed.d[d] = 1;
      }
      profile->setDimensions(input->getName(),
                             nvinfer1::OptProfileSelector::kMIN, fixed);
      profile->setDimensions(input->getName(),
                             nvinfer1::OptProfileSelector::kOPT, fixed);
      profile->setDimensions(input->getName(),
                             nvinfer1::OptProfileSelector::kMAX, fixed);
      std::printf("[TensorRT] Set dynamic input \"%s\" to batch=1\n",
                  input->getName());
    }
    config->addOptimizationProfile(profile);
  }

  auto plan = std::unique_ptr<nvinfer1::IHostMemory>(
      builder->buildSerializedNetwork(*network, *config));
  if (!plan)
    throw std::runtime_error("Failed to build TRT engine from ONNX");

  if (!engine_save_path.empty()) {
    std::ofstream out(engine_save_path, std::ios::binary);
    if (out.good()) {
      out.write(static_cast<const char *>(plan->data()), plan->size());
      std::printf("[TensorRT] Saved engine to: %s\n", engine_save_path.c_str());
    } else {
      std::fprintf(stderr, "[TensorRT] Warning: could not save engine to %s\n",
                   engine_save_path.c_str());
    }
  }

  engine_.reset(runtime_->deserializeCudaEngine(plan->data(), plan->size()));
  if (!engine_)
    throw std::runtime_error("Failed to deserialize built engine");

  context_.reset(engine_->createExecutionContext());

  std::printf("[TensorRT] Built engine from ONNX: %s\n", onnx_path.c_str());
}

void ai_decoder_service::setup_bindings() {
  int num_io = engine_->getNbIOTensors();
  bool found_input = false;
  bool found_output = false;

  for (int i = 0; i < num_io; ++i) {
    const char *name = engine_->getIOTensorName(i);
    auto mode = engine_->getTensorIOMode(name);
    auto dims = engine_->getTensorShape(name);
    auto dtype = engine_->getTensorDataType(name);
    size_t size_bytes = tensor_volume(dims) * trt_dtype_size(dtype);

    bool is_input = (mode == nvinfer1::TensorIOMode::kINPUT);

    std::printf("[TensorRT] Binding %d: \"%s\" %s, dtype=%d, elem_size=%zu, "
                "volume=%zu, %zu bytes\n",
                i, name, is_input ? "INPUT" : "OUTPUT", static_cast<int>(dtype),
                trt_dtype_size(dtype), tensor_volume(dims), size_bytes);

    tensor_binding binding{name, nullptr, size_bytes, is_input};

    if (is_input && !found_input) {
      input_size_ = size_bytes;
      found_input = true;
    } else if (!is_input && !found_output) {
      output_size_ = size_bytes;
      found_output = true;
    }

    all_bindings_.push_back(std::move(binding));
  }
}

void ai_decoder_service::allocate_resources() {
  if (all_bindings_.empty()) {
    // Passthrough path (no TRT bindings)
    if (cudaMalloc(&d_trt_input_, input_size_) != cudaSuccess)
      throw std::runtime_error("Failed to allocate TRT Input");
    if (cudaMalloc(&d_trt_output_, output_size_) != cudaSuccess)
      throw std::runtime_error("Failed to allocate TRT Output");
    return;
  }

  bool assigned_input = false;
  bool assigned_output = false;

  for (auto &b : all_bindings_) {
    void *buf = nullptr;
    if (cudaMalloc(&buf, b.size_bytes) != cudaSuccess)
      throw std::runtime_error("Failed to allocate buffer for " + b.name);
    cudaMemset(buf, 0, b.size_bytes);
    b.d_buffer = buf;

    if (b.is_input && !assigned_input) {
      d_trt_input_ = buf;
      assigned_input = true;
    } else if (!b.is_input && !assigned_output) {
      d_trt_output_ = buf;
      assigned_output = true;
    } else {
      d_aux_buffers_.push_back(buf);
    }
  }
}

void ai_decoder_service::capture_graph(cudaStream_t stream) {
  for (auto &b : all_bindings_) {
    context_->setTensorAddress(b.name.c_str(), b.d_buffer);
  }

  if (!context_->enqueueV3(stream))
    throw std::runtime_error(
        "TRT enqueueV3 warmup failed in ai_decoder_service");
  DECODER_CUDA_CHECK(cudaStreamSynchronize(stream));

  cudaGraph_t graph;
  DECODER_CUDA_CHECK(
      cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

  gateway_input_kernel<<<1, 128, 0, stream>>>(device_mailbox_slot_,
                                              d_trt_input_, input_size_);
  if (!context_->enqueueV3(stream))
    throw std::runtime_error(
        "TRT enqueueV3 failed during graph capture in ai_decoder_service");
  gateway_output_kernel<<<1, 128, 0, stream>>>(device_mailbox_slot_,
                                               d_trt_output_, output_size_);

  DECODER_CUDA_CHECK(cudaStreamEndCapture(stream, &graph));

  cudaError_t inst_err = cudaGraphInstantiateWithFlags(
      &graph_exec_, graph, cudaGraphInstantiateFlagDeviceLaunch);
  if (inst_err != cudaSuccess) {
    cudaGraphDestroy(graph);
    throw std::runtime_error(
        std::string(
            "cudaGraphInstantiateWithFlags failed in ai_decoder_service: ") +
        cudaGetErrorString(inst_err));
  }

  DECODER_CUDA_CHECK(cudaGraphUpload(graph_exec_, stream));
  cudaGraphDestroy(graph);
  DECODER_CUDA_CHECK(cudaStreamSynchronize(stream));
}

} // namespace cudaq::qec::realtime::experimental
