/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file predecoder_pipeline_common.h
/// @brief Shared types and helpers for the AI predecoder + PyMatching pipeline.
///
/// Used by both the software-only benchmark
/// (test_realtime_predecoder_w_pymatching.cpp) and the FPGA bridge
/// (hololink_predecoder_bridge.cpp).

#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#ifndef CUDA_VERSION
#define CUDA_VERSION 13000
#endif

#include "cudaq/qec/realtime/pipeline.h"
#include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"
#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h"

#include "cudaq/qec/code.h"
#include "cudaq/qec/decoder.h"
#include "cudaq/qec/realtime/ai_predecoder_service.h"
#include "cudaq/qec/realtime/nvtx_helpers.h"

using namespace cudaq::qec;
using namespace cudaq::qec::realtime::experimental;
namespace rt_sdk = cudaq::realtime;
namespace rt_pipeline = cudaq::qec::realtime::experimental;

#ifndef CUDAQ_REALTIME_CPU_RELAX
#if defined(__x86_64__)
#include <immintrin.h>
#define CUDAQ_REALTIME_CPU_RELAX() _mm_pause()
#elif defined(__aarch64__)
#define CUDAQ_REALTIME_CPU_RELAX() __asm__ volatile("yield" ::: "memory")
#else
#define CUDAQ_REALTIME_CPU_RELAX()                                             \
  do {                                                                         \
  } while (0)
#endif
#endif

// =============================================================================
// Pipeline Configuration
// =============================================================================

constexpr size_t NUM_SLOTS = 16;

struct PipelineConfig {
  std::string label;
  int distance;
  int num_rounds;
  std::string onnx_filename;
  int num_predecoders;
  int num_workers;
  int num_decode_workers;

  std::string onnx_path() const {
    return std::string(ONNX_MODEL_DIR) + "/" + onnx_filename;
  }

  std::string engine_path() const {
    std::string name = onnx_filename;
    auto dot = name.rfind('.');
    if (dot != std::string::npos)
      name = name.substr(0, dot);
    return std::string(ONNX_MODEL_DIR) + "/" + name + ".engine";
  }

  static PipelineConfig d7_r7() {
    return {"d7_r7_Z", 7, 7, "model1_d7_r7_unified_Z_batch1.onnx", 16, 16, 32};
  }

  static PipelineConfig d13_r13() {
    return {"d13_r13_X", 13, 13, "predecoder_memory_d13_T13_X.onnx",
            16,          16, 32};
  }

  static PipelineConfig d13_r104() {
    return {
        "d13_r104_X", 13, 104, "predecoder_memory_d13_T104_X.onnx", 8, 8, 16};
  }

  static PipelineConfig d21_r21() {
    return {"d21_r21_Z", 21, 21, "model1_d21_r21_unified_X_batch1.onnx",
            16,          16, 32};
  }

  static PipelineConfig d31_r31() {
    return {"d31_r31_Z", 31, 31, "model1_d31_r31_unified_Z_batch1.onnx",
            16,          16, 32};
  }
};

inline size_t round_up_pow2(size_t v) {
  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  v |= v >> 32;
  return v + 1;
}

// =============================================================================
// Decoder Context (PyMatching worker pool)
// =============================================================================

struct DecoderContext {
  std::vector<std::unique_ptr<cudaq::qec::decoder>> decoders;
  std::atomic<int> next_decoder_idx{0};

  int z_stabilizers = 0;
  int spatial_slices = 0;
  int num_residual_detectors = 0;

  bool use_full_H = false;

  cudaq::qec::decoder *acquire_decoder() {
    thread_local int my_idx =
        next_decoder_idx.fetch_add(1, std::memory_order_relaxed);
    return decoders[my_idx % decoders.size()].get();
  }

  std::atomic<int64_t> total_decode_us{0};
  std::atomic<int64_t> total_worker_us{0};
  std::atomic<int> decode_count{0};

  int num_input_detectors = 0;
  std::atomic<int64_t> total_input_nonzero{0};
  std::atomic<int64_t> total_output_nonzero{0};
};

// =============================================================================
// Pre-launch DMA copy callback
// =============================================================================

struct PreLaunchCopyCtx {
  void *d_trt_input;
  size_t input_size;
  void **h_ring_ptrs;
  uint8_t *rx_data_dev_base;
  uint8_t *rx_data_host_base;
};

void pre_launch_input_copy(void *user_data, void *slot_dev,
                           cudaStream_t stream);

// =============================================================================
// Worker context
// =============================================================================

struct WorkerCtx {
  ai_predecoder_service *predecoder;
  DecoderContext *decoder_ctx;
  int32_t *decode_corrections = nullptr;
  int32_t *decode_logical_pred = nullptr;
  int max_requests = 0;
  const uint8_t *obs_row = nullptr;
  size_t obs_row_size = 0;
};

struct __attribute__((packed)) DecodeResponse {
  int32_t total_corrections;
  int32_t converged;
};

// =============================================================================
// PyMatching work queue
// =============================================================================

struct PyMatchJob {
  int origin_slot;
  uint64_t request_id;
  void *ring_buffer_ptr;
};

class PyMatchQueue {
public:
  void push(PyMatchJob &&j);
  bool pop(PyMatchJob &out);
  void shutdown();

private:
  std::mutex mtx_;
  std::condition_variable cv_;
  std::queue<PyMatchJob> jobs_;
  bool stop_ = false;
};

// =============================================================================
// Test data (pre-generated from Stim, or random)
// =============================================================================

struct TestData {
  std::vector<int32_t> detectors;
  std::vector<int32_t> observables;
  uint32_t num_samples = 0;
  uint32_t num_detectors = 0;
  uint32_t num_observables = 0;

  bool loaded() const { return num_samples > 0 && num_detectors > 0; }

  const int32_t *sample(int idx) const {
    return detectors.data() +
           (static_cast<size_t>(idx % num_samples) * num_detectors);
  }

  int32_t observable(int idx, int obs = 0) const {
    return observables[static_cast<size_t>(idx % num_samples) *
                           num_observables +
                       obs];
  }
};

bool load_binary_file(const std::string &path, uint32_t &out_rows,
                      uint32_t &out_cols, std::vector<int32_t> &data);

TestData load_test_data(const std::string &data_dir);

// =============================================================================
// Stim-derived parity check matrix loader (CSR sparse -> dense tensor)
// =============================================================================

struct SparseCSR {
  uint32_t nrows = 0, ncols = 0, nnz = 0;
  std::vector<int32_t> indptr;
  std::vector<int32_t> indices;

  bool loaded() const { return nrows > 0 && ncols > 0; }

  cudaqx::tensor<uint8_t> to_dense() const;
  std::vector<uint8_t> row_dense(uint32_t r) const;
};

struct StimData {
  SparseCSR H;
  SparseCSR O;
  std::vector<double> priors;
};

bool load_csr(const std::string &path, SparseCSR &out);
StimData load_stim_data(const std::string &data_dir);
