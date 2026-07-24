/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#ifdef CUDAQ_QEC_DEVICE_GRAPH_AVAILABLE

#include "DeviceGraphRingConsumer.h"
#include "ITransceiver.h"

#include <memory>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <cuda_runtime.h>

// CUDA-Q realtime transport-provider interface.  The Hololink transceiver is
// behind a runtime-loaded bridge provider (libcudaq-realtime-bridge-hololink),
// so this library carries NO Hololink / DOCA link-time dependencies.
#include "cudaq/realtime/daemon/bridge/bridge_interface.h"

// Forward-declare the opaque scheduler context so the header stays independent
// of the dispatch-graph API.
struct cudaq_dispatch_graph_context;

namespace cudaq::qec::decoding_server {

/// Runtime configuration for DeviceGraphTransceiver.  Transport fields are
/// read from environment variables so that the server can be reconfigured
/// without a rebuild; gpu_id is the exception -- it is the decoder's
/// cuda_device_id, filled in at transport creation.
///
/// The named fields are OPTIONAL convenience knobs: each becomes a
/// `--<flag>=` argument to the provider only when its environment variable
/// is set (they match the built-in hololink provider's flags; an RDMA-style
/// provider will typically want the same shape).  A provider with a
/// different argument surface is configured through `extra_args`
/// (QEC_DEVICE_GRAPH_PROVIDER_ARGS, whitespace-separated tokens forwarded
/// verbatim -- the same pass-through contract as the decoding server's
/// provider args).  Providers should ignore arguments they do not
/// recognize.
struct DeviceGraphConfig {
  std::string
      device_name;       ///< QEC_DEVICE_GRAPH_DEVICE (IB netdev, e.g. "mlx5_0")
  uint32_t remote_qp{0}; ///< QEC_DEVICE_GRAPH_REMOTE_QP (FPGA/emulator QP)
  int gpu_id{0};         ///< FPGA-affine GPU; set from the decoder's
                         ///< cuda_device_id by resolve_decode_device()
  size_t frame_size{0};  ///< QEC_DEVICE_GRAPH_FRAME_SIZE (max RPC frame bytes)
  size_t page_size{0};   ///< QEC_DEVICE_GRAPH_PAGE_SIZE (0 → from frame_size)
  size_t num_pages{0};   ///< QEC_DEVICE_GRAPH_NUM_PAGES (ring depth)
  std::string peer_ip;   ///< QEC_DEVICE_GRAPH_PEER_IP (FPGA/emulator IPv4)
  std::vector<std::string>
      extra_args; ///< QEC_DEVICE_GRAPH_PROVIDER_ARGS, forwarded verbatim
  // (QEC_DEVICE_GRAPH_RESERVED_SMS is consumed by DecodingSession, where the
  // decode graph is captured.)

  static DeviceGraphConfig from_env();
};

/// Device-graph dispatch engine (transport-blind) for the decoding server.
///
/// ## Architecture
///
/// The Hololink transceiver (DOCA GPU ring buffers fed by FPGA RDMA writes)
/// is brought up through the CUDA-Q realtime bridge-provider interface: the
/// constructor loads the provider (the built-in
/// `libcudaq-realtime-bridge-hololink.so`, or the library named by
/// `CUDAQ_REALTIME_BRIDGE_LIB` when set), and adopts the provider's
/// RING_BUFFER context.  `launch_scheduler()` wires those ring buffers to the
/// CUDAQ device-graph scheduler (`cudaq_create_dispatch_graph_regular`) and
/// the captured decoder CUDA graph, then starts the provider's I/O loop.
///
/// Consequence of the provider split: this library needs only the CUDA-Q
/// realtime headers + libcudaq-realtime.so at build time; the Hololink /
/// DOCA / HSB dependency chain lives entirely inside the provider .so, which
/// is built (and dlopen'd) on the machine with the hardware.
///
/// After `launch_scheduler()` returns, the GPU handles the full
/// RX → dispatch → decode → TX loop autonomously.  No CPU `recv()` or `send()`
/// is involved in the data path; those methods are stubs that satisfy the
/// `ITransceiver` contract used by `DecodingServer::run()`.
///
/// ## Multi-decoder
///
/// Currently limited to a single decoder session (enforced by DecodingServer).
/// Multi-decoder device_graph with per-session ring binding is deferred.
class DeviceGraphTransceiver final : public ITransceiver {
public:
  explicit DeviceGraphTransceiver(const DeviceGraphConfig &config);
  ~DeviceGraphTransceiver() override;

  /// Wire the DOCA ring buffers to the CUDAQ device-graph scheduler and launch
  /// the GPU dispatch loop.  Must be called exactly once after the transceiver
  /// is created and before `run()`.
  ///
  /// \p raw_graph_resources is the `void *` returned by
  /// `decoder::capture_decode_graph()`; it is cast internally to
  /// `cudaq::qec::realtime::graph_resources *` to extract `graph_exec`.
  void launch_scheduler(void *raw_graph_resources);

  /// ITransceiver hook: forwards to launch_scheduler().
  bool launch_device_scheduler(void *raw_graph_resources) override {
    launch_scheduler(raw_graph_resources);
    return true;
  }

  /// Block until shutdown() is called.  The GPU scheduler handles RX/TX;
  /// this method only satisfies the ITransceiver contract for DecodingServer.
  RxFrame recv() override;

  /// Not used on the GPU scheduler path — the device graph kernel writes TX
  /// responses directly.  Always throws std::logic_error.
  void send(const PeerId &peer, const uint8_t *data, size_t len) override;

  void shutdown() override;

  /// The provider's one-line `key=value ...` endpoint description
  /// (bridge interface v2 get_endpoint_info), verbatim.  Whatever the wire's
  /// rendezvous data is (QP/rkey for RDMA, a port for sockets), it rides
  /// through here opaquely; this class neither parses nor interprets it.
  const std::string &endpoint_info() const { return endpoint_info_; }

private:
  cudaq_realtime_bridge_handle_t bridge_{nullptr};
  int gpu_id_{0};

  // DOCA ring buffer pointers (GPU VRAM — device addresses), adopted from the
  // provider's RING_BUFFER context.
  uint8_t *rx_ring_data_{nullptr};
  volatile uint64_t *rx_ring_flag_{nullptr};
  uint8_t *tx_ring_data_{nullptr};
  volatile uint64_t *tx_ring_flag_{nullptr};
  size_t num_pages_{0};
  size_t page_size_{0};

  // RDMA target identity from the provider's endpoint-info query (v2).
  std::string endpoint_info_;

  // The device-graph scheduler over this transceiver's rings (set by
  // launch_scheduler; owns all scheduler-side CUDA state).
  std::unique_ptr<DeviceGraphRingConsumer> consumer_;

  std::atomic<bool> stopped_{false};
};

} // namespace cudaq::qec::decoding_server

#endif // CUDAQ_QEC_DEVICE_GRAPH_AVAILABLE
