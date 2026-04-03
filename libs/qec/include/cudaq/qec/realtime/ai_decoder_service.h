/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

// TensorRT 10.12+ headers emit deprecation warnings for internal symbols
// (IPluginV2, legacy calibrator enums, IAlgorithmSelector, etc.) that are
// scheduled for removal in a future release.  These warnings originate inside
// the TensorRT headers themselves and cannot be fixed on our side.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "NvInfer.h"
#pragma GCC diagnostic pop

#include <cuda_runtime.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace cudaq::qec::realtime::experimental {

class ai_decoder_service {
public:
  class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char *msg) noexcept override;
  } static gLogger;

  /// @brief Constructor. Accepts a serialized TRT engine (.engine/.plan) or
  ///        an ONNX model (.onnx) which will be compiled to a TRT engine.
  /// @param model_path Path to the model file
  /// @param device_mailbox_slot Pointer to the specific slot in the global
  /// mailbox bank
  /// @param engine_save_path If non-empty and model_path is .onnx, save the
  ///        built engine to this path for fast reloading on subsequent runs
  ai_decoder_service(const std::string &model_path, void **device_mailbox_slot,
                     const std::string &engine_save_path = "");

  /// Create a passthrough (identity copy) instance for testing without TRT.
  static std::unique_ptr<ai_decoder_service>
  create_passthrough(void **device_mailbox_slot,
                     size_t input_bytes = 1600 * sizeof(float),
                     size_t output_bytes = 1600 * sizeof(float));

  virtual ~ai_decoder_service();

  virtual void capture_graph(cudaStream_t stream);

  cudaGraphExec_t get_executable_graph() const { return graph_exec_; }

  /// @brief Size of the primary input tensor in bytes (payload from RPC)
  size_t get_input_size() const { return input_size_; }

  /// @brief Size of the primary output tensor in bytes (forwarded to CPU)
  size_t get_output_size() const { return output_size_; }

  void *get_trt_input_ptr() const { return d_trt_input_; }

protected:
  /// Passthrough constructor (no TRT, identity copy kernel only).
  ai_decoder_service(void **device_mailbox_slot, size_t input_bytes,
                     size_t output_bytes);

  void load_engine(const std::string &path);
  void build_engine_from_onnx(const std::string &onnx_path,
                              const std::string &engine_save_path = "");
  void setup_bindings();
  void allocate_resources();

  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;

  cudaGraphExec_t graph_exec_ = nullptr;

  void **device_mailbox_slot_;
  void *d_trt_input_ = nullptr;  // Primary input buffer
  void *d_trt_output_ = nullptr; // Primary output buffer (residual_detectors)
  std::vector<void *> d_aux_buffers_; // Additional I/O buffers TRT needs

  struct tensor_binding {
    std::string name;
    void *d_buffer = nullptr;
    size_t size_bytes = 0;
    bool is_input = false;
  };
  std::vector<tensor_binding> all_bindings_;

  size_t input_size_ = 0;
  size_t output_size_ = 0;
};

} // namespace cudaq::qec::realtime::experimental
