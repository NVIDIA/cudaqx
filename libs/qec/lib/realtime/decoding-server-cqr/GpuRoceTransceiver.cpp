/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#ifdef CUDAQ_GPU_ROCE_AVAILABLE

#include "GpuRoceTransceiver.h"
#include "cudaq/qec/logger.h"
#include "cudaq/qec/realtime/decoder_rpc_wire_format.h"
#include "cudaq/qec/realtime/graph_resources.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>

// CUDAQ device-graph scheduler API (cudaq-realtime-dispatch).
#include "cudaq/realtime/hololink_bridge_common.h"

namespace cudaq::qec::decoding_server {

using cudaq::qec::decoding::rpc::kEnqueueSyndromesFunctionId;
using cudaq::qec::decoding::rpc::kGetCorrectionsFunctionId;
using cudaq::qec::decoding::rpc::kResetDecoderFunctionId;

// ---------------------------------------------------------------------------
// Internal helpers (same pattern as hololink_qldpc_graph_decoder_bridge.cpp)
// ---------------------------------------------------------------------------

namespace {

// Allocate \p bytes of CUDA pinned+mapped host memory and return both the host
// pointer and its device-mapped counterpart.  The memory is zero-initialised.
bool alloc_pinned_mapped(size_t bytes, void **host_out, void **dev_out) {
  void *h = nullptr;
  if (cudaHostAlloc(&h, bytes, cudaHostAllocMapped) != cudaSuccess)
    return false;
  void *d = nullptr;
  if (cudaHostGetDevicePointer(&d, h, 0) != cudaSuccess) {
    cudaFreeHost(h);
    return false;
  }
  std::memset(h, 0, bytes);
  *host_out = h;
  *dev_out = d;
  return true;
}

// Resolve a proprietary DEVICE_CALL populate shim via dlsym and stamp the
// function table entry.  The server process must absorb
// libcudaq-qec-realtime-cudevice-proprietary.a (WHOLE_ARCHIVE) and link with
// --export-dynamic so the symbols are visible.
using populate_fn = void (*)(void *);
bool populate_device_call(cudaq_function_entry_t &entry, const char *symbol,
                          uint32_t function_id) {
  auto fn = reinterpret_cast<populate_fn>(::dlsym(RTLD_DEFAULT, symbol));
  if (!fn) {
    CUDA_QEC_ERROR(
        "GpuRoceTransceiver: dlsym({}) failed -- the server process must "
        "absorb libcudaq-qec-realtime-cudevice-proprietary.a as WHOLE_ARCHIVE "
        "and link with --export-dynamic",
        symbol);
    return false;
  }
  fn(&entry);
  entry.function_id = function_id;
  entry.routing_key = 0;
  if (entry.dispatch_mode != CUDAQ_DISPATCH_DEVICE_CALL ||
      !entry.handler.device_fn_ptr) {
    CUDA_QEC_ERROR("GpuRoceTransceiver: {} did not produce a valid "
                   "DEVICE_CALL entry",
                   symbol);
    return false;
  }
  return true;
}

#define GPU_CUDA_CHECK(expr)                                                   \
  do {                                                                         \
    cudaError_t _err = (expr);                                                 \
    if (_err != cudaSuccess)                                                   \
      throw std::runtime_error(                                                \
          std::string("GpuRoceTransceiver CUDA error: ") +                     \
          cudaGetErrorString(_err) + " (" #expr ")");                          \
  } while (0)

} // namespace

// ---------------------------------------------------------------------------
// GpuRoceConfig::from_env
// ---------------------------------------------------------------------------

static std::string env_str(const char *name, const char *def = "") {
  const char *v = std::getenv(name);
  return v ? v : def;
}
static uint32_t env_u32(const char *name, uint32_t def) {
  const char *v = std::getenv(name);
  return v ? static_cast<uint32_t>(std::stoul(v)) : def;
}
static int env_int(const char *name, int def) {
  const char *v = std::getenv(name);
  return v ? std::stoi(v) : def;
}
static size_t env_size(const char *name, size_t def) {
  const char *v = std::getenv(name);
  return v ? static_cast<size_t>(std::stoull(v)) : def;
}

GpuRoceConfig GpuRoceConfig::from_env() {
  GpuRoceConfig c;
  c.device_name = env_str("HOLOLINK_DEVICE");
  c.peer_ip = env_str("HOLOLINK_PEER_IP");
  c.remote_qp = env_u32("HOLOLINK_REMOTE_QP", 0);
  // gpu_id is not read from the environment: the device is the decoder's
  // cuda_device_id, resolved by resolve_decode_device() at transport
  // creation.
  c.frame_size = env_size("HOLOLINK_FRAME_SIZE", 384);
  c.page_size = env_size("HOLOLINK_PAGE_SIZE", 0); // 0 → derived below
  c.num_pages = env_size("HOLOLINK_NUM_PAGES", 64);
  c.reserved_sms = env_int("HOLOLINK_RESERVED_SMS", 2);
  return c;
}

// ---------------------------------------------------------------------------
// GpuRoceTransceiver constructor
// ---------------------------------------------------------------------------

GpuRoceTransceiver::GpuRoceTransceiver(const GpuRoceConfig &config)
    : gpu_id_(config.gpu_id) {
  if (config.device_name.empty())
    throw std::runtime_error("GpuRoceTransceiver: HOLOLINK_DEVICE not set");
  if (config.peer_ip.empty())
    throw std::runtime_error("GpuRoceTransceiver: HOLOLINK_PEER_IP not set");
  if (config.remote_qp == 0)
    throw std::runtime_error("GpuRoceTransceiver: HOLOLINK_REMOTE_QP not set");

  // Derive page_size from frame_size if not overridden, then round up to the
  // 128-byte Hololink granularity.  Mirrors the derivation in
  // hololink_qldpc_graph_decoder_bridge.cpp (lines 279-282).
  size_t page_size = config.page_size ? config.page_size : config.frame_size;
  page_size = (page_size + 127) & ~static_cast<size_t>(127);

  if (page_size != 0 &&
      config.num_pages > std::numeric_limits<size_t>::max() / page_size)
    throw std::runtime_error(
        "GpuRoceTransceiver: ring size overflow for "
        "HOLOLINK_FRAME_SIZE/HOLOLINK_PAGE_SIZE=" +
        std::to_string(page_size) +
        " and HOLOLINK_NUM_PAGES=" + std::to_string(config.num_pages));
  const size_t ring_bytes = page_size * config.num_pages;
  const long host_page_size = ::sysconf(_SC_PAGESIZE);
  if (host_page_size > 0 &&
      ring_bytes % static_cast<size_t>(host_page_size) != 0)
    throw std::runtime_error(
        "GpuRoceTransceiver: ring buffer size " + std::to_string(ring_bytes) +
        " bytes is not aligned to host page size " +
        std::to_string(host_page_size) +
        " bytes; adjust HOLOLINK_NUM_PAGES or HOLOLINK_PAGE_SIZE");

  // Matches the call shape in hololink_qldpc_graph_decoder_bridge.cpp (lines
  // 288-291).
  transceiver_ = hololink_create_transceiver(
      config.device_name.c_str(),
      /*arg1=*/1, config.remote_qp, config.gpu_id, config.frame_size, page_size,
      config.num_pages, config.peer_ip.c_str(),
      /*forward=*/0,
      /*rx_only=*/1,
      /*tx_only=*/1);
  if (!transceiver_)
    throw std::runtime_error(
        "GpuRoceTransceiver: hololink_create_transceiver() failed for device=" +
        config.device_name + " peer=" + config.peer_ip);

  // Do NOT destroy a half-initialized transceiver on start failure (mirrors
  // the guard at lines 297-306 in the bridge: DOCA teardown may double-free
  // GPU memory that was never allocated, causing a segfault).
  if (!hololink_start(transceiver_))
    throw std::runtime_error(
        "GpuRoceTransceiver: hololink_start() failed (check that the IB "
        "netdev has an IPv4 address assigned for RoCE v2 GID)");

  // Adopt the DOCA ring buffer GPU VRAM pointers.
  rx_ring_data_ =
      reinterpret_cast<uint8_t *>(hololink_get_rx_ring_data_addr(transceiver_));
  rx_ring_flag_ = reinterpret_cast<volatile uint64_t *>(
      hololink_get_rx_ring_flag_addr(transceiver_));
  tx_ring_data_ =
      reinterpret_cast<uint8_t *>(hololink_get_tx_ring_data_addr(transceiver_));
  tx_ring_flag_ = reinterpret_cast<volatile uint64_t *>(
      hololink_get_tx_ring_flag_addr(transceiver_));

  if (!rx_ring_data_ || !rx_ring_flag_ || !tx_ring_data_ || !tx_ring_flag_) {
    hololink_close(transceiver_);
    hololink_destroy_transceiver(transceiver_);
    transceiver_ = {};
    throw std::runtime_error(
        "GpuRoceTransceiver: null DOCA ring pointer(s) after hololink_start");
  }

  num_pages_ = hololink_get_num_pages(transceiver_);
  page_size_ = hololink_get_page_size(transceiver_);

  CUDA_QEC_INFO("GpuRoceTransceiver: Hololink started  device={} peer={} "
                "gpu={} pages={} page_size={}  "
                "QP=0x{:X} rkey={} buf=0x{:X}  "
                "(call launch_scheduler() before run())",
                config.device_name, config.peer_ip, config.gpu_id, num_pages_,
                page_size_, hololink_get_qp_number(transceiver_),
                hololink_get_rkey(transceiver_),
                hololink_get_buffer_addr(transceiver_));
}

// ---------------------------------------------------------------------------
// launch_scheduler
// ---------------------------------------------------------------------------

void GpuRoceTransceiver::launch_scheduler(void *raw_graph_resources) {
  auto *graph_res =
      static_cast<cudaq::qec::realtime::graph_resources *>(raw_graph_resources);
  if (!graph_res || !graph_res->graph_exec)
    throw std::runtime_error(
        "GpuRoceTransceiver::launch_scheduler: null graph_exec "
        "(decoder must support_graph_dispatch() and capture_decode_graph())");

  GPU_CUDA_CHECK(cudaSetDevice(gpu_id_));

  void *ft_dev = nullptr;
  if (!alloc_pinned_mapped(3 * sizeof(cudaq_function_entry_t), &ft_host_,
                           &ft_dev))
    throw std::runtime_error("GpuRoceTransceiver::launch_scheduler: "
                             "function-table pinned alloc failed");

  auto *entries = static_cast<cudaq_function_entry_t *>(ft_host_);
  bool ok =
      populate_device_call(entries[0],
                           "cudaqx_qec_realtime_dispatch_populate_enqueue_"
                           "syndromes_device_entry",
                           kEnqueueSyndromesFunctionId) &&
      populate_device_call(
          entries[1],
          "cudaqx_qec_realtime_dispatch_populate_get_corrections_device_entry",
          kGetCorrectionsFunctionId) &&
      populate_device_call(
          entries[2],
          "cudaqx_qec_realtime_dispatch_populate_reset_decoder_device_entry",
          kResetDecoderFunctionId);
  if (!ok) {
    cudaFreeHost(ft_host_);
    ft_host_ = nullptr;
    throw std::runtime_error(
        "GpuRoceTransceiver::launch_scheduler: populate_device_call failed "
        "(see error log above)");
  }

  // Resolve dispatch graph API via dlsym; cudaq-realtime-dispatch is linked
  // into the server (not this static lib) to keep the CUDA module in one copy.
  // Signatures must match create/launch/destroy_dispatch_graph_fn_t in
  // qec_realtime_session.cpp/.h exactly — calling-convention mismatch is UB.
  using create_fn_t = cudaError_t (*)(
      volatile std::uint64_t *, volatile std::uint64_t *, std::uint8_t *,
      std::uint8_t *, std::size_t, std::size_t, cudaq_function_entry_t *,
      std::size_t, void *, volatile int *, std::uint64_t *, std::size_t,
      std::uint32_t, std::uint32_t, cudaGraphExec_t, cudaStream_t,
      cudaq_dispatch_graph_context **);
  using launch_fn_t =
      cudaError_t (*)(cudaq_dispatch_graph_context *, cudaStream_t);
  using destroy_fn_t = cudaError_t (*)(cudaq_dispatch_graph_context *);

  auto create_dispatch = reinterpret_cast<create_fn_t>(
      ::dlsym(RTLD_DEFAULT, "cudaq_create_dispatch_graph_regular"));
  auto launch_dispatch = reinterpret_cast<launch_fn_t>(
      ::dlsym(RTLD_DEFAULT, "cudaq_launch_dispatch_graph"));
  auto destroy_dispatch = reinterpret_cast<destroy_fn_t>(
      ::dlsym(RTLD_DEFAULT, "cudaq_destroy_dispatch_graph"));

  if (!create_dispatch || !launch_dispatch || !destroy_dispatch) {
    cudaFreeHost(ft_host_);
    ft_host_ = nullptr;
    CUDA_QEC_ERROR(
        "GpuRoceTransceiver: cudaq dispatch API not found via dlsym -- "
        "the server must link cudaq-realtime-dispatch with --export-dynamic");
    throw std::runtime_error(
        "GpuRoceTransceiver::launch_scheduler: cudaq dispatch API not found "
        "(cudaq_create/launch/destroy_dispatch_graph_regular); "
        "link cudaq-realtime-dispatch into the server with --export-dynamic");
  }
  fn_destroy_dispatch_graph_ = destroy_dispatch;

  void *sd_host = nullptr, *sd_dev = nullptr;
  if (!alloc_pinned_mapped(sizeof(int), &sd_host, &sd_dev)) {
    cudaFreeHost(ft_host_);
    ft_host_ = nullptr;
    throw std::runtime_error("GpuRoceTransceiver::launch_scheduler: "
                             "shutdown-flag pinned alloc failed");
  }
  shutdown_host_ = static_cast<volatile int *>(sd_host);
  shutdown_dev_ = static_cast<volatile int *>(sd_dev);

  if (cudaMalloc(&d_stats_, sizeof(uint64_t)) != cudaSuccess ||
      cudaMemset(d_stats_, 0, sizeof(uint64_t)) != cudaSuccess) {
    cudaFreeHost(ft_host_);
    ft_host_ = nullptr;
    cudaFreeHost(sd_host);
    shutdown_host_ = nullptr;
    shutdown_dev_ = nullptr;
    throw std::runtime_error(
        "GpuRoceTransceiver::launch_scheduler: d_stats_ alloc failed");
  }

  GPU_CUDA_CHECK(cudaStreamCreate(&sched_stream_));

  cudaError_t cerr = create_dispatch(
      rx_ring_flag_, tx_ring_flag_, rx_ring_data_, tx_ring_data_, page_size_,
      page_size_, static_cast<cudaq_function_entry_t *>(ft_dev),
      /*func_count=*/3,
      /*graph_io_ctx=*/nullptr, shutdown_dev_, d_stats_, num_pages_,
      /*num_blocks=*/1, /*threads_per_block=*/64, graph_res->graph_exec,
      sched_stream_, &sched_ctx_);
  if (cerr != cudaSuccess) {
    cudaStreamDestroy(sched_stream_);
    sched_stream_ = nullptr;
    cudaFree(d_stats_);
    d_stats_ = nullptr;
    cudaFreeHost(ft_host_);
    ft_host_ = nullptr;
    cudaFreeHost(sd_host);
    shutdown_host_ = nullptr;
    shutdown_dev_ = nullptr;
    throw std::runtime_error(
        std::string("GpuRoceTransceiver::launch_scheduler: "
                    "cudaq_create_dispatch_graph_regular: ") +
        cudaGetErrorString(cerr));
  }

  cerr = launch_dispatch(sched_ctx_, sched_stream_);
  if (cerr != cudaSuccess) {
    fn_destroy_dispatch_graph_(sched_ctx_);
    sched_ctx_ = nullptr;
    cudaStreamDestroy(sched_stream_);
    sched_stream_ = nullptr;
    cudaFree(d_stats_);
    d_stats_ = nullptr;
    cudaFreeHost(ft_host_);
    ft_host_ = nullptr;
    cudaFreeHost(sd_host);
    shutdown_host_ = nullptr;
    shutdown_dev_ = nullptr;
    throw std::runtime_error(
        std::string("GpuRoceTransceiver::launch_scheduler: "
                    "cudaq_launch_dispatch_graph: ") +
        cudaGetErrorString(cerr));
  }

  monitor_thread_ =
      std::thread([this] { hololink_blocking_monitor(transceiver_); });

  CUDA_QEC_INFO("GpuRoceTransceiver: GPU scheduler launched  "
                "QP=0x{:X} rkey={} buf=0x{:X}  "
                "(3 DEVICE_CALL entries, graph_exec={:p})",
                hololink_get_qp_number(transceiver_),
                hololink_get_rkey(transceiver_),
                hololink_get_buffer_addr(transceiver_),
                static_cast<void *>(graph_res->graph_exec));

  // Print RDMA target info to stdout so the orchestration script can grep it.
  // Matches the format in hololink_qldpc_graph_decoder_bridge.cpp lines
  // 441-444.
  std::cout << "QP Number: 0x" << std::hex
            << hololink_get_qp_number(transceiver_) << std::dec << "\n"
            << "RKey: " << hololink_get_rkey(transceiver_) << "\n"
            << "Buffer Addr: 0x" << std::hex
            << hololink_get_buffer_addr(transceiver_) << std::dec << "\n";
  std::cout.flush();
}

// ---------------------------------------------------------------------------
// ITransceiver interface stubs (GPU scheduler handles the data path)
// ---------------------------------------------------------------------------

RxFrame GpuRoceTransceiver::recv() {
  // The GPU device-graph scheduler handles RX→dispatch→decode→TX autonomously.
  // This method only exists so DecodingServer::run()'s recv loop blocks until
  // shutdown() is called.
  while (!stopped_.load(std::memory_order_acquire))
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  return {}; // shutdown sentinel: empty buf causes the recv loop to exit
}

void GpuRoceTransceiver::send(const PeerId & /*peer*/, const uint8_t * /*data*/,
                              size_t /*len*/) {
  throw std::logic_error(
      "GpuRoceTransceiver::send() must not be called: the CUDAQ device-graph "
      "scheduler writes TX responses directly to the Hololink ring buffer");
}

// ---------------------------------------------------------------------------
// shutdown / destructor
// ---------------------------------------------------------------------------

void GpuRoceTransceiver::shutdown() {
  if (stopped_.exchange(true, std::memory_order_acq_rel))
    return; // already stopped

  // Signal the GPU scheduler kernel to stop its self-relaunch loop.
  if (shutdown_host_)
    __atomic_store_n(shutdown_host_, 1, __ATOMIC_RELEASE);

  // Stop the Hololink RX/TX kernels to unblock hololink_blocking_monitor().
  if (transceiver_)
    hololink_close(transceiver_);
}

GpuRoceTransceiver::~GpuRoceTransceiver() {
  // Ensure clean shutdown even if the caller omitted shutdown().
  if (!stopped_.exchange(true, std::memory_order_acq_rel)) {
    if (shutdown_host_)
      __atomic_store_n(shutdown_host_, 1, __ATOMIC_RELEASE);
    if (transceiver_)
      hololink_close(transceiver_);
  }

  if (monitor_thread_.joinable())
    monitor_thread_.join();

  if (sched_stream_) {
    cudaStreamSynchronize(sched_stream_); // drain the self-relaunch chain
    if (sched_ctx_ && fn_destroy_dispatch_graph_)
      fn_destroy_dispatch_graph_(sched_ctx_);
    cudaStreamDestroy(sched_stream_);
  }

  if (ft_host_)
    cudaFreeHost(ft_host_);
  if (shutdown_host_)
    cudaFreeHost(const_cast<int *>(shutdown_host_));
  if (d_stats_)
    cudaFree(d_stats_);
  if (transceiver_)
    hololink_destroy_transceiver(transceiver_);
}

// ---------------------------------------------------------------------------
// RDMA target info
// ---------------------------------------------------------------------------

uint32_t GpuRoceTransceiver::qp_number() const {
  return hololink_get_qp_number(transceiver_);
}
uint32_t GpuRoceTransceiver::rkey() const {
  return hololink_get_rkey(transceiver_);
}
uint64_t GpuRoceTransceiver::buffer_addr() const {
  return hololink_get_buffer_addr(transceiver_);
}

} // namespace cudaq::qec::decoding_server

#endif // CUDAQ_GPU_ROCE_AVAILABLE
