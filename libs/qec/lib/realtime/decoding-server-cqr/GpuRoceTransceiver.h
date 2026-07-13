/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#ifdef CUDAQ_GPU_ROCE_AVAILABLE

#include "ITransceiver.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <thread>

#include <cuda_runtime.h>

// Hololink Sensor Bridge + CUDAQ dispatcher C API.
// Both are provided by cudaq-realtime-bridge-hololink and
// cudaq-realtime-dispatch.
#include "cudaq/realtime/daemon/bridge/hololink/hololink_wrapper.h"

// Forward-declare the opaque scheduler context so the header stays independent
// of the full cudaq_realtime.h.
struct cudaq_dispatch_graph_context;

namespace cudaq::qec::decoding_server {

/// Runtime configuration for GpuRoceTransceiver.  All fields are read from
/// environment variables so that the server can be reconfigured without a
/// rebuild.
struct GpuRoceConfig {
  std::string device_name; ///< HOLOLINK_DEVICE   (IB netdev, e.g. "mlx5_0")
  uint32_t remote_qp{0};   ///< HOLOLINK_REMOTE_QP (FPGA/emulator QP number)
  int gpu_id{0};           ///< HOLOLINK_GPU_ID
  /// Set iff HOLOLINK_GPU_ID was present in the environment (the FPGA/NIC
  /// affinity is a topology fact; absence defers to the decoder's pin).
  std::optional<int> gpu_id_env;
  size_t frame_size{384}; ///< HOLOLINK_FRAME_SIZE (max RPC frame bytes)
  size_t page_size{0};    ///< HOLOLINK_PAGE_SIZE (0 → derived from frame_size)
  size_t num_pages{64};   ///< HOLOLINK_NUM_PAGES (ring depth)
  std::string peer_ip;    ///< HOLOLINK_PEER_IP   (FPGA/emulator IPv4)
  int reserved_sms{2};    ///< HOLOLINK_RESERVED_SMS (SMs for Hololink RX/TX)

  static GpuRoceConfig from_env();
};

/// GPU RoCE transport and device-graph scheduler for the decoding server.
///
/// ## Architecture
///
/// Hololink DMA's RPC frames from the FPGA directly into DOCA GPU ring buffers.
/// `launch_scheduler()` wires those ring buffers to the CUDAQ device-graph
/// scheduler (`cudaq_create_dispatch_graph_regular`) and the captured decoder
/// CUDA graph, replicating the pattern in
/// `libs/qec/unittests/utils/hololink_qldpc_graph_decoder_bridge.cpp`.
///
/// After `launch_scheduler()` returns, the GPU handles the full
/// RX → dispatch → decode → TX loop autonomously.  No CPU `recv()` or `send()`
/// is involved in the data path; those methods are stubs that satisfy the
/// `ITransceiver` contract used by `DecodingServer::run()`.
///
/// ## Multi-decoder
///
/// Currently limited to a single decoder session (enforced by DecodingServer).
/// Multi-decoder GPU RoCE with per-session ring binding is deferred.
class GpuRoceTransceiver final : public ITransceiver {
public:
  explicit GpuRoceTransceiver(const GpuRoceConfig &config);
  ~GpuRoceTransceiver() override;

  /// Wire the DOCA ring buffers to the CUDAQ device-graph scheduler and launch
  /// the GPU dispatch loop.  Must be called exactly once after the transceiver
  /// is created and before `run()`.
  ///
  /// \p raw_graph_resources is the `void *` returned by
  /// `decoder::capture_decode_graph()`; it is cast internally to
  /// `cudaq::qec::realtime::graph_resources *` to extract `graph_exec`.
  void launch_scheduler(void *raw_graph_resources);

  /// Block until shutdown() is called.  The GPU scheduler handles RX/TX;
  /// this method only satisfies the ITransceiver contract for DecodingServer.
  RxFrame recv() override;

  /// Not used on the GPU scheduler path — the device graph kernel writes TX
  /// responses directly.  Always throws std::logic_error.
  void send(const PeerId &peer, const uint8_t *data, size_t len) override;

  void shutdown() override;

  /// RDMA target info printed after launch_scheduler() for the orchestration
  /// script (QP number, rkey, buffer address).
  uint32_t qp_number() const;
  uint32_t rkey() const;
  uint64_t buffer_addr() const;

private:
  hololink_transceiver_t transceiver_{};
  int gpu_id_{0};

  // DOCA ring buffer pointers (GPU VRAM — device addresses).
  uint8_t *rx_ring_data_{nullptr};
  volatile uint64_t *rx_ring_flag_{nullptr};
  uint8_t *tx_ring_data_{nullptr};
  volatile uint64_t *tx_ring_flag_{nullptr};
  size_t num_pages_{0};
  size_t page_size_{0};

  // CUDAQ device-graph scheduler state (set by launch_scheduler).
  cudaq_dispatch_graph_context *sched_ctx_{nullptr};
  cudaStream_t sched_stream_{nullptr};
  void *ft_host_{nullptr}; ///< pinned host ptr: function entry table
  volatile int *shutdown_host_{
      nullptr}; ///< pinned host ptr: GPU scheduler stop flag
  volatile int *shutdown_dev_{nullptr}; ///< device-mapped ptr of shutdown_host_
  uint64_t *d_stats_{nullptr};
  // Cached from launch_scheduler() so the destructor can call it without dlsym.
  cudaError_t (*fn_destroy_dispatch_graph_)(cudaq_dispatch_graph_context *){
      nullptr};

  std::atomic<bool> stopped_{false};
  std::thread monitor_thread_; ///< runs hololink_blocking_monitor()
};

} // namespace cudaq::qec::decoding_server

#endif // CUDAQ_GPU_ROCE_AVAILABLE
