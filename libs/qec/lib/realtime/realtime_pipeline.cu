/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.
 * All rights reserved.
 *
 * This source code and the accompanying materials are made available under
 * the terms of the Apache License 2.0 which accompanies this distribution.
 ******************************************************************************/

#include "cudaq/qec/realtime/nvtx_helpers.h"
#include "cudaq/qec/realtime/pipeline.h"
#include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"
#include "cudaq/realtime/daemon/dispatcher/host_dispatcher.h"

#include <cuda/std/atomic>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <iostream>
#include <mutex>
#include <pthread.h>
#include <sched.h>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace cudaq::qec::realtime::experimental {

using atomic_uint64_sys = cuda::std::atomic<uint64_t>;
using atomic_int_sys = cuda::std::atomic<int>;

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

#define PIPELINE_CUDA_CHECK(call)                                              \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "realtime_pipeline CUDA error: " << cudaGetErrorString(err) \
                << " at " << __FILE__ << ":" << __LINE__ << std::endl;         \
      std::abort();                                                            \
    }                                                                          \
  } while (0)

static void pin_thread(std::thread &t, int core) {
  if (core < 0)
    return;
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(core, &cpuset);
  pthread_setaffinity_np(t.native_handle(), sizeof(cpu_set_t), &cpuset);
}

// ---------------------------------------------------------------------------
// GPU-only mode: completion signaling via cudaLaunchHostFunc
// ---------------------------------------------------------------------------

struct GpuOnlyWorkerCtx {
  atomic_uint64_sys *tx_flags;
  atomic_uint64_sys *idle_mask;
  int *inflight_slot_tags;
  uint8_t *rx_data_host;
  size_t slot_size;
  int worker_id;
  void (*user_post_launch_fn)(void *user_data, void *slot_dev,
                              cudaStream_t stream);
  void *user_post_launch_data;
  int origin_slot;
  uint64_t tx_value;
};

static void gpu_only_host_callback(void *user_data) {
  auto *ctx = static_cast<GpuOnlyWorkerCtx *>(user_data);
  ctx->tx_flags[ctx->origin_slot].store(ctx->tx_value,
                                        cuda::std::memory_order_release);
  ctx->idle_mask->fetch_or(1ULL << ctx->worker_id,
                           cuda::std::memory_order_release);
}

static void gpu_only_post_launch(void *user_data, void *slot_dev,
                                 cudaStream_t stream) {
  NVTX_PUSH("GPUPostLaunch");
  auto *ctx = static_cast<GpuOnlyWorkerCtx *>(user_data);

  if (ctx->user_post_launch_fn)
    ctx->user_post_launch_fn(ctx->user_post_launch_data, slot_dev, stream);

  ctx->origin_slot = ctx->inflight_slot_tags[ctx->worker_id];
  uint8_t *slot_host = ctx->rx_data_host +
                       static_cast<size_t>(ctx->origin_slot) * ctx->slot_size;
  ctx->tx_value = reinterpret_cast<uint64_t>(slot_host);

  cudaLaunchHostFunc(stream, gpu_only_host_callback, ctx);
  NVTX_POP();
}

// ---------------------------------------------------------------------------
// RingBufferManager
// ---------------------------------------------------------------------------

class RingBufferManager {
public:
  RingBufferManager(size_t num_slots, size_t slot_size)
      : num_slots_(num_slots), slot_size_(slot_size) {
    PIPELINE_CUDA_CHECK(cudaHostAlloc(
        &buf_rx_, num_slots * sizeof(atomic_uint64_sys), cudaHostAllocMapped));
    rx_flags_ = static_cast<atomic_uint64_sys *>(buf_rx_);
    for (size_t i = 0; i < num_slots; ++i)
      new (rx_flags_ + i) atomic_uint64_sys(0);

    PIPELINE_CUDA_CHECK(cudaHostAlloc(
        &buf_tx_, num_slots * sizeof(atomic_uint64_sys), cudaHostAllocMapped));
    tx_flags_ = static_cast<atomic_uint64_sys *>(buf_tx_);
    for (size_t i = 0; i < num_slots; ++i)
      new (tx_flags_ + i) atomic_uint64_sys(0);

    PIPELINE_CUDA_CHECK(cudaHostGetDevicePointer(
        reinterpret_cast<void **>(&rx_flags_dev_), buf_rx_, 0));
    PIPELINE_CUDA_CHECK(cudaHostGetDevicePointer(
        reinterpret_cast<void **>(&tx_flags_dev_), buf_tx_, 0));

    PIPELINE_CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void **>(&rx_data_host_),
                                      num_slots * slot_size,
                                      cudaHostAllocMapped));
    PIPELINE_CUDA_CHECK(cudaHostGetDevicePointer(
        reinterpret_cast<void **>(&rx_data_dev_), rx_data_host_, 0));

    rb_.rx_flags = reinterpret_cast<volatile uint64_t *>(rx_flags_);
    rb_.tx_flags = reinterpret_cast<volatile uint64_t *>(tx_flags_);
    rb_.rx_data = rx_data_dev_;
    rb_.tx_data = rx_data_dev_;
    rb_.rx_stride_sz = slot_size;
    rb_.tx_stride_sz = slot_size;
    rb_.rx_flags_host = reinterpret_cast<volatile uint64_t *>(rx_flags_);
    rb_.tx_flags_host = reinterpret_cast<volatile uint64_t *>(tx_flags_);
    rb_.rx_data_host = rx_data_host_;
    rb_.tx_data_host = rx_data_host_;
  }

  ~RingBufferManager() {
    for (size_t i = 0; i < num_slots_; ++i) {
      rx_flags_[i].~atomic_uint64_sys();
      tx_flags_[i].~atomic_uint64_sys();
    }
    cudaFreeHost(buf_rx_);
    cudaFreeHost(buf_tx_);
    cudaFreeHost(rx_data_host_);
  }

  bool slot_available(uint32_t slot) const {
    auto *flags = reinterpret_cast<const volatile uint64_t *>(rx_flags_);
    return __atomic_load_n(&flags[slot], __ATOMIC_ACQUIRE) == 0;
  }

  void write_and_signal(uint32_t slot, uint32_t function_id,
                        const void *payload, uint32_t payload_len,
                        uint32_t request_id = 0, uint64_t ptp_timestamp = 0) {
    cudaq_host_ringbuffer_write_rpc_request(&rb_, slot, function_id, payload,
                                            payload_len, request_id,
                                            ptp_timestamp);
    cudaq_host_ringbuffer_signal_slot(&rb_, slot);
  }

  cudaq_tx_status_t poll_tx(uint32_t slot, int *cuda_error) const {
    return cudaq_host_ringbuffer_poll_tx_flag(&rb_, slot, cuda_error);
  }

  void clear_slot(uint32_t slot) {
    cudaq_host_ringbuffer_clear_slot(&rb_, slot);
  }

  size_t num_slots() const { return num_slots_; }
  size_t slot_size() const { return slot_size_; }

  atomic_uint64_sys *rx_flags() { return rx_flags_; }
  atomic_uint64_sys *tx_flags() { return tx_flags_; }
  uint8_t *rx_data_host() { return rx_data_host_; }
  uint8_t *rx_data_dev() { return rx_data_dev_; }
  const cudaq_ringbuffer_t &ringbuffer() const { return rb_; }

private:
  size_t num_slots_;
  size_t slot_size_;
  void *buf_rx_ = nullptr;
  void *buf_tx_ = nullptr;
  atomic_uint64_sys *rx_flags_ = nullptr;
  atomic_uint64_sys *tx_flags_ = nullptr;
  uint64_t *rx_flags_dev_ = nullptr;
  uint64_t *tx_flags_dev_ = nullptr;
  uint8_t *rx_data_host_ = nullptr;
  uint8_t *rx_data_dev_ = nullptr;
  cudaq_ringbuffer_t rb_{};
};

// ---------------------------------------------------------------------------
// Impl
// ---------------------------------------------------------------------------

struct realtime_pipeline::Impl {
  pipeline_stage_config config;

  gpu_stage_factory gpu_factory;
  cpu_stage_callback cpu_stage;
  completion_callback completion_handler;

  // Owned infrastructure
  std::unique_ptr<RingBufferManager> ring;
  void **h_mailbox_bank = nullptr;
  void **d_mailbox_bank = nullptr;

  // Dispatcher state (hidden atomics)
  atomic_int_sys shutdown_flag{0};
  uint64_t dispatcher_stats = 0;
  atomic_uint64_sys live_dispatched{0};
  atomic_uint64_sys idle_mask{0};
  std::vector<int> inflight_slot_tags;

  // Function table
  std::vector<cudaq_function_entry_t> function_table;

  // Per-worker GPU resources (from factory)
  std::vector<gpu_worker_resources> worker_resources;

  // GPU-only mode state
  bool gpu_only = false;
  std::vector<GpuOnlyWorkerCtx> gpu_only_ctxs;

  // Slot-to-request mapping (consumer-owned)
  std::vector<uint64_t> slot_request;
  std::vector<uint8_t> slot_occupied;

  // Stats (atomic counters)
  std::atomic<uint64_t> total_submitted{0};
  std::atomic<uint64_t> total_completed{0};
  std::atomic<uint64_t> backpressure_stalls{0};

  // Thread coordination
  std::atomic<bool> producer_stop{false};
  std::atomic<bool> consumer_stop{false};

  // Threads
  std::thread dispatcher_thread;
  std::thread consumer_thread;
  std::vector<std::thread> worker_threads;

  std::atomic<bool> started{false};

  // -----------------------------------------------------------------------
  // Lifecycle
  // -----------------------------------------------------------------------

  void allocate(const pipeline_stage_config &cfg) {
    if (cfg.num_workers > 64) {
      throw std::invalid_argument("num_workers (" +
                                  std::to_string(cfg.num_workers) +
                                  ") exceeds idle_mask capacity of 64");
    }

    config = cfg;

    ring = std::make_unique<RingBufferManager>(
        static_cast<size_t>(cfg.num_slots), cfg.slot_size);

    PIPELINE_CUDA_CHECK(cudaHostAlloc(&h_mailbox_bank,
                                      cfg.num_workers * sizeof(void *),
                                      cudaHostAllocMapped));
    std::memset(h_mailbox_bank, 0, cfg.num_workers * sizeof(void *));
    PIPELINE_CUDA_CHECK(cudaHostGetDevicePointer(
        reinterpret_cast<void **>(&d_mailbox_bank), h_mailbox_bank, 0));

    inflight_slot_tags.resize(cfg.num_workers, 0);
    slot_request.resize(cfg.num_slots, 0);
    slot_occupied.resize(cfg.num_slots, 0);
  }

  void start_threads() {
    if (!gpu_factory) {
      throw std::logic_error("gpu_factory must be set before calling start()");
    }

    const int nw = config.num_workers;
    gpu_only = !cpu_stage;

    // Build GPU resources via user factory
    worker_resources.resize(nw);
    function_table.resize(nw);
    for (int i = 0; i < nw; ++i) {
      worker_resources[i] = gpu_factory(i);
      function_table[i].function_id = worker_resources[i].function_id;
      function_table[i].dispatch_mode = CUDAQ_DISPATCH_GRAPH_LAUNCH;
      function_table[i].handler.graph_exec = worker_resources[i].graph_exec;
      std::memset(&function_table[i].schema, 0,
                  sizeof(function_table[i].schema));
    }

    // In GPU-only mode, set up per-worker contexts for cudaLaunchHostFunc
    // completion signaling (chains user's post_launch_fn if provided).
    if (gpu_only) {
      gpu_only_ctxs.resize(nw);
      for (int i = 0; i < nw; ++i) {
        auto &c = gpu_only_ctxs[i];
        c.tx_flags = ring->tx_flags();
        c.idle_mask = &idle_mask;
        c.inflight_slot_tags = inflight_slot_tags.data();
        c.rx_data_host = ring->rx_data_host();
        c.slot_size = config.slot_size;
        c.worker_id = i;
        c.user_post_launch_fn = worker_resources[i].post_launch_fn;
        c.user_post_launch_data = worker_resources[i].post_launch_data;
        c.origin_slot = 0;
        c.tx_value = 0;
      }
    }

    // Initialize idle_mask with all workers free
    uint64_t initial_idle = (nw >= 64) ? ~0ULL : ((1ULL << nw) - 1);
    idle_mask.store(initial_idle, cuda::std::memory_order_release);

    // Build cudaq_host_dispatch_loop_ctx_t
    std::vector<cudaq_host_dispatch_worker_t> disp_workers(nw);
    for (int i = 0; i < nw; ++i) {
      disp_workers[i].graph_exec = worker_resources[i].graph_exec;
      disp_workers[i].stream = worker_resources[i].stream;
      disp_workers[i].function_id = worker_resources[i].function_id;
      disp_workers[i].pre_launch_fn = worker_resources[i].pre_launch_fn;
      disp_workers[i].pre_launch_data = worker_resources[i].pre_launch_data;

      if (gpu_only) {
        disp_workers[i].post_launch_fn = gpu_only_post_launch;
        disp_workers[i].post_launch_data = &gpu_only_ctxs[i];
      } else {
        disp_workers[i].post_launch_fn = worker_resources[i].post_launch_fn;
        disp_workers[i].post_launch_data = worker_resources[i].post_launch_data;
      }
    }

    cudaq_host_dispatch_loop_ctx_t disp_cfg;
    std::memset(&disp_cfg, 0, sizeof(disp_cfg));

    disp_cfg.ringbuffer = ring->ringbuffer();

    disp_cfg.config.num_slots = static_cast<uint32_t>(config.num_slots);
    disp_cfg.config.slot_size = static_cast<uint32_t>(config.slot_size);
    disp_cfg.config.dispatch_path = CUDAQ_DISPATCH_PATH_HOST;
    disp_cfg.config.dispatch_mode = CUDAQ_DISPATCH_GRAPH_LAUNCH;

    disp_cfg.function_table.entries = function_table.data();
    disp_cfg.function_table.count = static_cast<uint32_t>(nw);

    disp_cfg.workers = disp_workers.data();
    disp_cfg.num_workers = static_cast<size_t>(nw);
    disp_cfg.h_mailbox_bank = h_mailbox_bank;
    disp_cfg.shutdown_flag = static_cast<void *>(&shutdown_flag);
    disp_cfg.stats_counter = &dispatcher_stats;
    disp_cfg.live_dispatched = static_cast<void *>(&live_dispatched);
    disp_cfg.idle_mask = static_cast<void *>(&idle_mask);
    disp_cfg.inflight_slot_tags = inflight_slot_tags.data();
    disp_cfg.skip_stream_sweep = true;

    // --- Dispatcher thread ---
    // The config is copied by value into the lambda; the workers vector is
    // moved in so it outlives this scope.  Raw pointers inside cfg
    // (ringbuffer, idle_mask, shutdown_flag, etc.) remain valid because they
    // point to Impl-owned allocations that outlive the dispatcher thread.
    dispatcher_thread = std::thread(
        [cfg = disp_cfg, workers = std::move(disp_workers)]() mutable {
          cfg.workers = workers.data();
          cudaq_host_dispatcher_loop(&cfg);
        });
    pin_thread(dispatcher_thread, config.cores.dispatcher);

    // --- Worker threads (skipped in GPU-only mode) ---
    if (!gpu_only) {
      worker_threads.resize(nw);
      for (int i = 0; i < nw; ++i) {
        worker_threads[i] = std::thread([this, i]() { worker_loop(i); });
        int core =
            (config.cores.worker_base >= 0) ? config.cores.worker_base + i : -1;
        pin_thread(worker_threads[i], core);
      }
    }

    // --- Consumer thread ---
    consumer_thread = std::thread([this]() { consumer_loop(); });
    pin_thread(consumer_thread, config.cores.consumer);

    started = true;
  }

  void stop_all() {
    if (!started)
      return;

    // Signal consumer to finish pending work
    producer_stop.store(true, std::memory_order_release);

    // Grace period for in-flight requests
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (total_completed.load(std::memory_order_relaxed) <
               total_submitted.load(std::memory_order_relaxed) &&
           std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    consumer_stop.store(true, std::memory_order_release);

    // Shut down dispatcher
    shutdown_flag.store(1, cuda::std::memory_order_release);
    dispatcher_thread.join();

    // Consumer
    consumer_thread.join();

    // Workers check shutdown via consumer_stop (they spin on ready_flags,
    // which will never fire after dispatcher is gone, so we need to break
    // them out). We set consumer_stop which doubles as system_stop for
    // workers; the user's poll_next_job must eventually return false.
    for (auto &t : worker_threads) {
      if (t.joinable())
        t.join();
    }

    started = false;
  }

  void free_resources() {
    ring.reset();
    if (h_mailbox_bank) {
      cudaFreeHost(h_mailbox_bank);
      h_mailbox_bank = nullptr;
    }
  }

  // -----------------------------------------------------------------------
  // Worker loop (one per worker thread)
  // -----------------------------------------------------------------------

  void worker_loop(int worker_id) {
    auto *wr = &worker_resources[worker_id];

    // The cpu_stage callback is called in "poll mode"
    // (gpu_output == nullptr). It polls its own GPU-ready
    // mechanism and, if a result is available, processes it and
    // writes the RPC response. Returns 0 when nothing was ready,
    // >0 when a job was completed. The pipeline then handles all
    // atomic signaling (tx_flags, idle_mask).

    while (!consumer_stop.load(std::memory_order_relaxed)) {
      cpu_stage_context ctx;
      ctx.worker_id = worker_id;
      ctx.origin_slot = inflight_slot_tags[worker_id];
      ctx.gpu_output = nullptr;
      ctx.gpu_output_size = 0;
      ctx.response_buffer = nullptr;
      ctx.max_response_size = 0;
      ctx.user_context = wr->user_context;

      NVTX_PUSH("WorkerPoll");
      size_t written = cpu_stage(ctx);
      NVTX_POP();
      if (written == 0) {
        CUDAQ_REALTIME_CPU_RELAX();
        continue;
      }

      if (written == DEFERRED_COMPLETION) {
        idle_mask.fetch_or(1ULL << worker_id, cuda::std::memory_order_release);
        continue;
      }

      int origin_slot = inflight_slot_tags[worker_id];

      uint8_t *slot_host = ring->rx_data_host() +
                           static_cast<size_t>(origin_slot) * config.slot_size;
      uint64_t rx_value = reinterpret_cast<uint64_t>(slot_host);

      ring->tx_flags()[origin_slot].store(rx_value,
                                          cuda::std::memory_order_release);

      idle_mask.fetch_or(1ULL << worker_id, cuda::std::memory_order_release);
    }
  }

  // -----------------------------------------------------------------------
  // Consumer loop
  // -----------------------------------------------------------------------

  void consumer_loop() {
    const uint32_t ns = static_cast<uint32_t>(config.num_slots);

    while (true) {
      if (consumer_stop.load(std::memory_order_acquire))
        break;

      bool pdone = producer_stop.load(std::memory_order_acquire);
      uint64_t nsub = total_submitted.load(std::memory_order_acquire);
      uint64_t ncomp = total_completed.load(std::memory_order_relaxed);

      if (pdone && ncomp >= nsub)
        break;

      bool found_any = false;
      for (uint32_t s = 0; s < ns; ++s) {
        if (!slot_occupied[s])
          continue;

        int cuda_error = 0;
        cudaq_tx_status_t status = ring->poll_tx(s, &cuda_error);

        if (status == CUDAQ_TX_READY) {
          NVTX_PUSH("ConsumerComplete");
          if (completion_handler) {
            completion c;
            c.request_id = slot_request[s];
            c.slot = static_cast<int>(s);
            c.success = true;
            c.cuda_error = 0;
            completion_handler(c);
          }
          total_completed.fetch_add(1, std::memory_order_relaxed);

          // ARM memory ordering: clear occupancy BEFORE
          // clearing ring buffer flags, with a fence between.
          slot_occupied[s] = 0;
          __sync_synchronize();
          ring->clear_slot(s);
          found_any = true;
          NVTX_POP();

        } else if (status == CUDAQ_TX_ERROR) {
          if (completion_handler) {
            completion c;
            c.request_id = slot_request[s];
            c.slot = static_cast<int>(s);
            c.success = false;
            c.cuda_error = cuda_error;
            completion_handler(c);
          }
          total_completed.fetch_add(1, std::memory_order_relaxed);
          slot_occupied[s] = 0;
          __sync_synchronize();
          ring->clear_slot(s);
          found_any = true;
        }
      }

      if (!found_any)
        CUDAQ_REALTIME_CPU_RELAX();
    }
  }
};

// ---------------------------------------------------------------------------
// realtime_pipeline public API
// ---------------------------------------------------------------------------

realtime_pipeline::realtime_pipeline(const pipeline_stage_config &config)
    : impl_(std::make_unique<Impl>()) {
  impl_->allocate(config);
}

realtime_pipeline::~realtime_pipeline() {
  if (impl_->started)
    impl_->stop_all();
  impl_->free_resources();
}

void realtime_pipeline::set_gpu_stage(gpu_stage_factory factory) {
  impl_->gpu_factory = std::move(factory);
}

void realtime_pipeline::set_cpu_stage(cpu_stage_callback callback) {
  impl_->cpu_stage = std::move(callback);
}

void realtime_pipeline::set_completion_handler(completion_callback handler) {
  impl_->completion_handler = std::move(handler);
}

void realtime_pipeline::start() {
  if (impl_->started)
    return;
  impl_->start_threads();
}

void realtime_pipeline::stop() { impl_->stop_all(); }

realtime_pipeline::Stats realtime_pipeline::stats() const {
  return {impl_->total_submitted.load(std::memory_order_relaxed),
          impl_->total_completed.load(std::memory_order_relaxed),
          impl_->live_dispatched.load(cuda::std::memory_order_relaxed),
          impl_->backpressure_stalls.load(std::memory_order_relaxed)};
}

realtime_pipeline::ring_buffer_bases
realtime_pipeline::ringbuffer_bases() const {
  return {impl_->ring->rx_data_host(), impl_->ring->rx_data_dev()};
}

void realtime_pipeline::complete_deferred(int slot) {
  uint8_t *slot_host = impl_->ring->rx_data_host() +
                       static_cast<size_t>(slot) * impl_->config.slot_size;
  uint64_t rx_value = reinterpret_cast<uint64_t>(slot_host);
  impl_->ring->tx_flags()[slot].store(rx_value,
                                      cuda::std::memory_order_release);
}

// ---------------------------------------------------------------------------
// ring_buffer_injector
// ---------------------------------------------------------------------------

struct ring_buffer_injector::State {
  RingBufferManager *ring = nullptr;
  std::vector<uint64_t> *slot_request = nullptr;
  std::vector<uint8_t> *slot_occupied = nullptr;
  std::atomic<uint64_t> *total_submitted = nullptr;
  std::atomic<uint64_t> *backpressure_stalls = nullptr;
  std::atomic<bool> *producer_stop = nullptr;
  int num_slots = 0;
  std::atomic<uint32_t> next_slot{0};
};

ring_buffer_injector realtime_pipeline::create_injector() {
  auto s = std::make_unique<ring_buffer_injector::State>();
  s->ring = impl_->ring.get();
  s->slot_request = &impl_->slot_request;
  s->slot_occupied = &impl_->slot_occupied;
  s->total_submitted = &impl_->total_submitted;
  s->backpressure_stalls = &impl_->backpressure_stalls;
  s->producer_stop = &impl_->producer_stop;
  s->num_slots = impl_->config.num_slots;
  return ring_buffer_injector(std::move(s));
}

ring_buffer_injector::ring_buffer_injector(std::unique_ptr<State> s)
    : state_(std::move(s)) {}

ring_buffer_injector::~ring_buffer_injector() = default;
ring_buffer_injector::ring_buffer_injector(ring_buffer_injector &&) noexcept =
    default;
ring_buffer_injector &
ring_buffer_injector::operator=(ring_buffer_injector &&) noexcept = default;

bool ring_buffer_injector::try_submit(uint32_t function_id, const void *payload,
                                      size_t payload_size,
                                      uint64_t request_id) {
  uint32_t cur = state_->next_slot.load(std::memory_order_relaxed);
  uint32_t slot = cur % static_cast<uint32_t>(state_->num_slots);
  if ((*state_->slot_occupied)[slot])
    return false;

  if (!state_->next_slot.compare_exchange_weak(
          cur, cur + 1, std::memory_order_acq_rel, std::memory_order_relaxed))
    return false;

  NVTX_PUSH("Submit");
  state_->ring->write_and_signal(slot, function_id, payload,
                                 static_cast<uint32_t>(payload_size),
                                 static_cast<uint32_t>(request_id));

  (*state_->slot_request)[slot] = request_id;
  (*state_->slot_occupied)[slot] = 1;
  state_->total_submitted->fetch_add(1, std::memory_order_release);
  NVTX_POP();
  return true;
}

void ring_buffer_injector::submit(uint32_t function_id, const void *payload,
                                  size_t payload_size, uint64_t request_id) {
  while (!try_submit(function_id, payload, payload_size, request_id)) {
    if (state_->producer_stop &&
        state_->producer_stop->load(std::memory_order_acquire))
      return;
    state_->backpressure_stalls->fetch_add(1, std::memory_order_relaxed);
    CUDAQ_REALTIME_CPU_RELAX();
  }
}

uint64_t ring_buffer_injector::backpressure_stalls() const {
  return state_->backpressure_stalls->load(std::memory_order_relaxed);
}

} // namespace cudaq::qec::realtime::experimental
