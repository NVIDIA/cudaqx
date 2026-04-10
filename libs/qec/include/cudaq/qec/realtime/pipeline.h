/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.
 * All rights reserved.
 *
 * This source code and the accompanying materials are made available under
 * the terms of the Apache License 2.0 which accompanies this distribution.
 ******************************************************************************/

#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <functional>
#include <memory>
#include <string>

namespace cudaq::qec::realtime::experimental {

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

struct core_pinning {
  int dispatcher = -1; // -1 = no pinning
  int consumer = -1;
  int worker_base = -1; // workers pin to base, base+1, ...
};

struct pipeline_stage_config {
  int num_workers = 8;
  int num_slots = 32;
  size_t slot_size = 16384;
  core_pinning cores;

  /// When non-null, the pipeline uses this caller-owned ring buffer
  /// (cudaq_ringbuffer_t*) instead of allocating its own.  The caller is
  /// responsible for lifetime.  ring_buffer_injector is unavailable in
  /// this mode (the FPGA/emulator owns the producer side).
  void *external_ringbuffer = nullptr;
};

// ---------------------------------------------------------------------------
// GPU Stage Factory
// ---------------------------------------------------------------------------

struct gpu_worker_resources {
  cudaGraphExec_t graph_exec = nullptr;
  cudaStream_t stream = nullptr;
  void (*pre_launch_fn)(void *user_data, void *slot_dev,
                        cudaStream_t stream) = nullptr;
  void *pre_launch_data = nullptr;
  void (*post_launch_fn)(void *user_data, void *slot_dev,
                         cudaStream_t stream) = nullptr;
  void *post_launch_data = nullptr;
  uint32_t function_id = 0;
  void *user_context = nullptr;
};

/// Called once per worker during start(). Returns GPU resources for that
/// worker.
using gpu_stage_factory = std::function<gpu_worker_resources(int worker_id)>;

// ---------------------------------------------------------------------------
// CPU Stage Callback
// ---------------------------------------------------------------------------

/// Passed to the user's CPU stage callback on each completed GPU workload.
/// The user reads gpu_output, does post-processing, and writes the
/// result into response_buffer. No atomics are exposed.
struct cpu_stage_context {
  int worker_id;
  int origin_slot;
  const void *gpu_output;
  size_t gpu_output_size;
  void *response_buffer;
  size_t max_response_size;
  void *user_context;
};

/// Returns the number of bytes written into response_buffer.
/// Return 0 if no GPU result is ready yet (poll again).
/// Return DEFERRED_COMPLETION to release the worker immediately while
/// deferring slot completion to a later complete_deferred() call.
using cpu_stage_callback = std::function<size_t(const cpu_stage_context &ctx)>;

/// Sentinel return value from cpu_stage_callback: release the worker
/// (idle_mask) but do NOT signal slot completion (tx_flags). The caller
/// is responsible for calling realtime_pipeline::complete_deferred(slot)
/// once the deferred work (e.g. a separate decode thread) finishes.
static constexpr size_t DEFERRED_COMPLETION = SIZE_MAX;

// ---------------------------------------------------------------------------
// Completion Callback
// ---------------------------------------------------------------------------

struct completion {
  uint64_t request_id;
  int slot;
  bool success;
  int cuda_error; // 0 on success
};

/// Called by the consumer thread for each completed (or errored) request.
using completion_callback = std::function<void(const completion &c)>;

// ---------------------------------------------------------------------------
// Ring Buffer Injector (software-only test/replay data source)
// ---------------------------------------------------------------------------

/// Writes RPC-framed requests into the pipeline's ring buffer, simulating
/// FPGA DMA deposits. Created via realtime_pipeline::create_injector().
/// The parent realtime_pipeline must outlive the injector.
class ring_buffer_injector {
public:
  ~ring_buffer_injector();
  ring_buffer_injector(ring_buffer_injector &&) noexcept;
  ring_buffer_injector &operator=(ring_buffer_injector &&) noexcept;

  ring_buffer_injector(const ring_buffer_injector &) = delete;
  ring_buffer_injector &operator=(const ring_buffer_injector &) = delete;

  /// Try to submit a request. Returns true if accepted, false if
  /// backpressure (all slots busy). Non-blocking. Thread-safe.
  bool try_submit(uint32_t function_id, const void *payload,
                  size_t payload_size, uint64_t request_id);

  /// Blocking submit: spins until a slot becomes available.
  void submit(uint32_t function_id, const void *payload, size_t payload_size,
              uint64_t request_id);

  uint64_t backpressure_stalls() const;

private:
  friend class realtime_pipeline;
  struct State;
  std::unique_ptr<State> state_;
  explicit ring_buffer_injector(std::unique_ptr<State> s);
};

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

class realtime_pipeline {
public:
  explicit realtime_pipeline(const pipeline_stage_config &config);
  ~realtime_pipeline();

  realtime_pipeline(const realtime_pipeline &) = delete;
  realtime_pipeline &operator=(const realtime_pipeline &) = delete;

  /// Register the GPU stage factory (called before start).
  void set_gpu_stage(gpu_stage_factory factory);

  /// Register the CPU worker callback (called before start).
  void set_cpu_stage(cpu_stage_callback callback);

  /// Register the completion callback (called before start).
  void set_completion_handler(completion_callback handler);

  /// Allocate resources, build dispatcher config, spawn all threads.
  void start();

  /// Signal shutdown, join all threads, free resources.
  void stop();

  /// Create a software injector for testing without FPGA hardware.
  /// The pipeline must be constructed but need not be started yet.
  ring_buffer_injector create_injector();

  struct Stats {
    uint64_t submitted;
    uint64_t completed;
    uint64_t dispatched;
    uint64_t backpressure_stalls;
  };

  /// Thread-safe, lock-free stats snapshot.
  Stats stats() const;

  /// Signal that deferred processing for a slot is complete.
  /// Call this from any thread after the cpu_stage callback returned
  /// DEFERRED_COMPLETION and the deferred work has finished writing the
  /// response into the slot's ring buffer area.
  void complete_deferred(int slot);

  struct ring_buffer_bases {
    uint8_t *rx_data_host;
    uint8_t *rx_data_dev;
  };

  /// Return the host and device base addresses of the RX data ring.
  /// Useful for pre_launch callbacks that need to convert between the two.
  ring_buffer_bases ringbuffer_bases() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace cudaq::qec::realtime::experimental
