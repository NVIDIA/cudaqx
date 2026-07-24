/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#ifdef CUDAQ_QEC_DEVICE_GRAPH_AVAILABLE

#include "DeviceGraphTransceiver.h"
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
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

// CUDAQ device-graph scheduler types (cudaq-realtime-dispatch) and the
// RPCHeader wire struct the provider's --payload-size argument is defined
// against.
#include "cudaq/realtime/hololink_bridge_common.h"

namespace cudaq::qec::decoding_server {

// ---------------------------------------------------------------------------
// Internal helpers (same pattern as hololink_qldpc_graph_decoder_bridge.cpp)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// DeviceGraphConfig::from_env
// ---------------------------------------------------------------------------

// Each knob reads QEC_DEVICE_GRAPH_<NAME>; the values are forwarded to
// whatever transport provider is loaded.
static const char *env_raw(const char *name) {
  const std::string full = std::string("QEC_DEVICE_GRAPH_") + name;
  return std::getenv(full.c_str());
}
static std::string env_str(const char *name, const char *def = "") {
  const char *v = env_raw(name);
  return v ? v : def;
}
static uint32_t env_u32(const char *name, uint32_t def) {
  const char *v = env_raw(name);
  return v ? static_cast<uint32_t>(std::stoul(v)) : def;
}
static size_t env_size(const char *name, size_t def) {
  const char *v = env_raw(name);
  return v ? static_cast<size_t>(std::stoull(v)) : def;
}

DeviceGraphConfig DeviceGraphConfig::from_env() {
  DeviceGraphConfig c;
  c.device_name = env_str("DEVICE");
  c.peer_ip = env_str("PEER_IP");
  c.remote_qp = env_u32("REMOTE_QP", 0);
  // gpu_id is not read from the environment: the device is the decoder's
  // cuda_device_id, resolved by resolve_decode_device() at transport
  // creation.
  c.frame_size = env_size("FRAME_SIZE", 0);
  c.page_size = env_size("PAGE_SIZE", 0); // 0 → derived from frame_size
  c.num_pages = env_size("NUM_PAGES", 0);
  // Generic pass-through channel for providers whose argument surface does
  // not match the named knobs above: whitespace-separated tokens, forwarded
  // to the provider verbatim (after the named arguments).
  if (const char *extra = env_raw("PROVIDER_ARGS")) {
    std::istringstream in(extra);
    std::string token;
    while (in >> token)
      c.extra_args.push_back(token);
  }
  return c;
}

// ---------------------------------------------------------------------------
// DeviceGraphTransceiver constructor
// ---------------------------------------------------------------------------

DeviceGraphTransceiver::DeviceGraphTransceiver(const DeviceGraphConfig &config)
    : gpu_id_(config.gpu_id) {
  if (config.device_name.empty())
    throw std::runtime_error(
        "DeviceGraphTransceiver: QEC_DEVICE_GRAPH_DEVICE not set");
  if (config.peer_ip.empty())
    throw std::runtime_error(
        "DeviceGraphTransceiver: QEC_DEVICE_GRAPH_PEER_IP not set");
  if (config.remote_qp == 0)
    throw std::runtime_error(
        "DeviceGraphTransceiver: QEC_DEVICE_GRAPH_REMOTE_QP not set");

  // Derive page_size from frame_size if not overridden, then round up to the
  // 128-byte Hololink granularity.  Mirrors the derivation in
  // hololink_qldpc_graph_decoder_bridge.cpp (lines 279-282).
  size_t page_size = config.page_size ? config.page_size : config.frame_size;
  page_size = (page_size + 127) & ~static_cast<size_t>(127);

  if (page_size != 0 &&
      config.num_pages > std::numeric_limits<size_t>::max() / page_size)
    throw std::runtime_error(
        "DeviceGraphTransceiver: ring size overflow for "
        "QEC_DEVICE_GRAPH_FRAME_SIZE/QEC_DEVICE_GRAPH_PAGE_SIZE=" +
        std::to_string(page_size) +
        " and QEC_DEVICE_GRAPH_NUM_PAGES=" + std::to_string(config.num_pages));
  const size_t ring_bytes = page_size * config.num_pages;
  const long host_page_size = ::sysconf(_SC_PAGESIZE);
  if (host_page_size > 0 &&
      ring_bytes % static_cast<size_t>(host_page_size) != 0)
    throw std::runtime_error("DeviceGraphTransceiver: ring buffer size " +
                             std::to_string(ring_bytes) +
                             " bytes is not aligned to host page size " +
                             std::to_string(host_page_size) +
                             " bytes; adjust QEC_DEVICE_GRAPH_NUM_PAGES or "
                             "QEC_DEVICE_GRAPH_PAGE_SIZE");

  // The provider computes frame_size = sizeof(RPCHeader) + payload_size, so
  // hand it the payload remainder of our frame budget.
  if (config.frame_size < sizeof(cudaq::realtime::RPCHeader))
    throw std::runtime_error(
        "DeviceGraphTransceiver: QEC_DEVICE_GRAPH_FRAME_SIZE smaller than "
        "the RPC header");
  const size_t payload_size =
      config.frame_size - sizeof(cudaq::realtime::RPCHeader);

  // Bring the Hololink transceiver up through the bridge-provider interface:
  // create() = hololink_create_transceiver + hololink_start (3-kernel shape:
  // no --forward / --unified => rx_only + tx_only kernels, with dispatch
  // supplied by our device-graph scheduler in launch_scheduler()).
  // args[0] is a program-name placeholder: the provider's parse_bridge_args
  // follows the C argv convention and starts parsing at argv[1] -- without
  // the placeholder the first real option would be silently skipped (and the
  // bridge would fall back to its built-in device default).
  const std::vector<std::string> args = {
      "device-graph-transceiver",
      "--device=" + config.device_name,
      "--peer-ip=" + config.peer_ip,
      "--remote-qp=" + std::to_string(config.remote_qp),
      "--gpu=" + std::to_string(config.gpu_id),
      "--page-size=" + std::to_string(page_size),
      "--num-pages=" + std::to_string(config.num_pages),
      "--payload-size=" + std::to_string(payload_size),
  };
  std::vector<char *> argv;
  argv.reserve(args.size());
  for (auto &a : args)
    argv.push_back(const_cast<char *>(a.c_str()));

  // A provider is just a library name/path to the loader (cached per
  // process, keyed by that string).  Default to the hololink GPU-RoCE
  // provider shipped with cudaq-realtime; CUDAQ_REALTIME_BRIDGE_LIB names a
  // replacement library (same mechanism as the decoding server's
  // --transport=<path>.so partner drop-in).
  const char *env_lib = std::getenv("CUDAQ_REALTIME_BRIDGE_LIB");
  const std::string provider_lib =
      env_lib ? env_lib : "libcudaq-realtime-bridge-hololink.so";
  if (cudaq_bridge_create_from_library(&bridge_, provider_lib.c_str(),
                                       static_cast<int>(argv.size()),
                                       argv.data()) != CUDAQ_OK ||
      !bridge_)
    throw std::runtime_error(
        "DeviceGraphTransceiver: bridge provider create failed for device=" +
        config.device_name + " peer=" + config.peer_ip + " (is " +
        provider_lib +
        " on the loader path, and "
        "does the IB netdev have an IPv4 address assigned for RoCE v2 GID?)");

  // Adopt the DOCA ring buffer GPU VRAM pointers from the provider.
  cudaq_ringbuffer_t ring{};
  if (cudaq_bridge_get_transport_context(bridge_, RING_BUFFER, &ring) !=
      CUDAQ_OK) {
    cudaq_bridge_destroy(bridge_);
    bridge_ = nullptr;
    throw std::runtime_error(
        "DeviceGraphTransceiver: provider has no ring-buffer context");
  }
  rx_ring_data_ = ring.rx_data;
  rx_ring_flag_ = ring.rx_flags;
  tx_ring_data_ = ring.tx_data;
  tx_ring_flag_ = ring.tx_flags;
  if (!rx_ring_data_ || !rx_ring_flag_ || !tx_ring_data_ || !tx_ring_flag_) {
    cudaq_bridge_destroy(bridge_);
    bridge_ = nullptr;
    throw std::runtime_error(
        "DeviceGraphTransceiver: null DOCA ring pointer(s) from provider");
  }

  // Ring geometry and RDMA target identity come from the provider's
  // interface-v2 queries; the scheduler and the orchestration handshake both
  // depend on them, so a provider without v2 support is an error.
  uint32_t num_slots = 0, slot_size = 0;
  if (cudaq_bridge_get_ring_geometry(bridge_, &num_slots, &slot_size) !=
      CUDAQ_OK) {
    cudaq_bridge_destroy(bridge_);
    bridge_ = nullptr;
    throw std::runtime_error(
        "DeviceGraphTransceiver: provider does not report ring geometry "
        "(bridge interface v2 required)");
  }
  num_pages_ = num_slots;
  page_size_ = slot_size;

  char info[512] = {0};
  if (cudaq_bridge_get_endpoint_info(bridge_, info, sizeof(info)) != CUDAQ_OK) {
    cudaq_bridge_destroy(bridge_);
    bridge_ = nullptr;
    throw std::runtime_error(
        "DeviceGraphTransceiver: provider does not report endpoint info "
        "(bridge interface v2 required)");
  }
  endpoint_info_ = info;

  // connect(): the provider finalizes whatever rendezvous its wire needs
  // (no wire traffic for hololink; the playback tool alone programs the
  // FPGA control plane).
  if (cudaq_bridge_connect(bridge_) != CUDAQ_OK) {
    cudaq_bridge_destroy(bridge_);
    bridge_ = nullptr;
    throw std::runtime_error(
        "DeviceGraphTransceiver: provider connect() failed");
  }

  CUDA_QEC_INFO("DeviceGraphTransceiver: provider started  gpu={} pages={} "
                "page_size={}  endpoint: {}  "
                "(call launch_scheduler() before run())",
                config.gpu_id, num_pages_, page_size_, endpoint_info_);
}

// ---------------------------------------------------------------------------
// launch_scheduler
// ---------------------------------------------------------------------------

void DeviceGraphTransceiver::launch_scheduler(void *raw_graph_resources) {
  // All scheduler wiring (pinned function table + populate shims + dispatch
  // graph create/launch) lives in DeviceGraphRingConsumer; this transceiver
  // contributes only its provider's ring context and geometry.
  cudaq_ringbuffer_t ring{};
  ring.rx_flags = rx_ring_flag_;
  ring.tx_flags = tx_ring_flag_;
  ring.rx_data = rx_ring_data_;
  ring.tx_data = tx_ring_data_;
  ring.rx_stride_sz = page_size_;
  ring.tx_stride_sz = page_size_;
  consumer_ = std::make_unique<DeviceGraphRingConsumer>(
      ring, num_pages_, page_size_, gpu_id_, raw_graph_resources);

  // Start the provider's I/O loop (Hololink RX/TX kernels + monitor thread,
  // owned by the provider) now that the scheduler is polling the rings.
  if (cudaq_bridge_launch(bridge_) != CUDAQ_OK) {
    consumer_->shutdown();
    throw std::runtime_error(
        "DeviceGraphTransceiver::launch_scheduler: provider launch() failed");
  }

  CUDA_QEC_INFO("DeviceGraphTransceiver: GPU scheduler launched ({})",
                endpoint_info_);

  // Publish the provider's endpoint description VERBATIM so the
  // orchestration layer can scrape whatever rendezvous tokens its wire
  // needs (qp=/rkey=/buffer_addr= for RDMA playback, port= for sockets).
  // This class does not know or care which tokens are present.
  std::cout << "QEC_DECODING_SERVER_ENDPOINT " << endpoint_info_ << "\n";
  std::cout.flush();
}

// ---------------------------------------------------------------------------
// ITransceiver interface stubs (GPU scheduler handles the data path)
// ---------------------------------------------------------------------------

RxFrame DeviceGraphTransceiver::recv() {
  // The GPU device-graph scheduler handles RX→dispatch→decode→TX autonomously.
  // This method only exists so DecodingServer::run()'s recv loop blocks until
  // shutdown() is called.
  while (!stopped_.load(std::memory_order_acquire))
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  return {}; // shutdown sentinel: empty buf causes the recv loop to exit
}

void DeviceGraphTransceiver::send(const PeerId & /*peer*/,
                                  const uint8_t * /*data*/, size_t /*len*/) {
  throw std::logic_error(
      "DeviceGraphTransceiver::send() must not be called: the CUDAQ "
      "device-graph "
      "scheduler writes TX responses directly to the Hololink ring buffer");
}

// ---------------------------------------------------------------------------
// shutdown / destructor
// ---------------------------------------------------------------------------

void DeviceGraphTransceiver::shutdown() {
  if (stopped_.exchange(true, std::memory_order_acq_rel))
    return; // already stopped

  // Signal the GPU scheduler's self-relaunch loop to stop.
  if (consumer_)
    consumer_->shutdown();

  // Stop the Hololink RX/TX kernels and join the provider's monitor thread.
  if (bridge_)
    cudaq_bridge_disconnect(bridge_);
}

DeviceGraphTransceiver::~DeviceGraphTransceiver() {
  // Ensure clean shutdown even if the caller omitted shutdown().
  if (!stopped_.exchange(true, std::memory_order_acq_rel)) {
    if (consumer_)
      consumer_->shutdown();
    if (bridge_)
      cudaq_bridge_disconnect(bridge_);
  }
  // Drain + destroy the scheduler BEFORE the provider (it polls the
  // provider's ring memory).
  consumer_.reset();
  if (bridge_)
    cudaq_bridge_destroy(bridge_);
}

} // namespace cudaq::qec::decoding_server

#endif // CUDAQ_QEC_DEVICE_GRAPH_AVAILABLE
