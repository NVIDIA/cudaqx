/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/qec/realtime/ai_decoder_service.h"
#include <atomic>
#include <cuda/atomic>

// Portable CPU Yield Macro for busy-polling (skip if already defined by
// realtime API)
#ifndef CUDAQ_REALTIME_CPU_RELAX
#if defined(__x86_64__)
#include <immintrin.h>
#define CUDAQ_REALTIME_CPU_RELAX() _mm_pause()
#elif defined(__aarch64__)
#define CUDAQ_REALTIME_CPU_RELAX() __asm__ volatile("yield" ::: "memory")
#else
#define CUDAQ_REALTIME_CPU_RELAX()                                             \
  std::atomic_thread_fence(std::memory_order_seq_cst)
#endif
#endif

namespace cudaq::qec::realtime::experimental {

struct pre_decoder_job {
  int slot_idx;    ///< Worker/slot index (for release_job; always 0)
  int origin_slot; ///< FPGA ring slot for tx_flags routing (dynamic pool)
  void *ring_buffer_ptr;
  void *inference_data; ///< Points into the pinned output (single slot)

  // Performance Tracking
  uint64_t submit_ts_ns;
  uint64_t dispatch_ts_ns;
  uint64_t poll_ts_ns;
};

class ai_predecoder_service : public ai_decoder_service {
public:
  ai_predecoder_service(const std::string &engine_path,
                        void **device_mailbox_slot, int queue_depth = 1,
                        const std::string &engine_save_path = "");

  /// Create a passthrough (identity copy) instance for testing without TRT.
  static std::unique_ptr<ai_predecoder_service>
  create_passthrough(void **device_mailbox_slot, int queue_depth = 1,
                     size_t input_bytes = 1600 * sizeof(float),
                     size_t output_bytes = 1600 * sizeof(float));

  virtual ~ai_predecoder_service();

  /// @param stream CUDA stream to use for capture and warm-up inference.
  /// @param device_launch If true, instantiate the graph for device launch.
  /// @param save_graph If true, retain a clone of the captured CUDA graph
  ///        template so it can be inspected later via
  ///        get_captured_graph() (e.g. by the free functions in
  ///        cudaq/qec/realtime/graph_resources.h).  Default is false; the
  ///        service otherwise destroys the template immediately after
  ///        instantiation.
  void capture_graph(cudaStream_t stream, bool device_launch,
                     bool save_graph = false);
  void capture_graph(cudaStream_t stream) override {
    capture_graph(stream, true, false);
  }

  bool poll_next_job(pre_decoder_job &out_job);
  void release_job(int slot_idx);

  /// Stub for device-dispatcher batch path (returns nullptr; streaming uses
  /// host dispatcher)
  int *get_device_queue_idx() const { return nullptr; }
  cuda::atomic<int, cuda::thread_scope_system> *get_device_ready_flags() const {
    return d_ready_flags_;
  }
  int *get_device_inflight_flag() const { return nullptr; }

  cuda::atomic<int, cuda::thread_scope_system> *get_host_ready_flags() const {
    return h_ready_flags_;
  }
  volatile int *get_host_queue_idx() const { return nullptr; }
  int get_queue_depth() const { return queue_depth_; }

  void **get_host_ring_ptrs() const { return h_ring_ptrs_; }

  /// @brief Return the retained clone of the captured graph template.
  /// @details Non-null only when capture_graph() was called with
  /// save_graph=true.  Ownership stays with this service; callers must
  /// NOT destroy the returned handle.  Intended for opt-in introspection
  /// (e.g. cudaq::qec::realtime::experimental::collect_graph_resources).
  cudaGraph_t get_captured_graph() const { return captured_graph_; }

private:
  /// Passthrough constructor (delegates to base passthrough constructor).
  ai_predecoder_service(void **device_mailbox_slot, int queue_depth,
                        size_t input_bytes, size_t output_bytes);

  int queue_depth_; // Always 1

  cuda::atomic<int, cuda::thread_scope_system> *h_ready_flags_ = nullptr;
  void **h_ring_ptrs_ = nullptr;
  void *h_predecoder_outputs_ = nullptr;

  cuda::atomic<int, cuda::thread_scope_system> *d_ready_flags_ = nullptr;
  void **d_ring_ptrs_ = nullptr;
  void *d_predecoder_outputs_ = nullptr;

  /// Optional clone of the captured cudaGraph_t template, retained only
  /// when capture_graph() was called with save_graph=true.  Destroyed in
  /// the destructor.  The instantiated graph_exec_ lives on the base
  /// class ai_decoder_service.
  cudaGraph_t captured_graph_ = nullptr;
};

} // namespace cudaq::qec::realtime::experimental
