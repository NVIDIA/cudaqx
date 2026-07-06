/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file qec_decoding_daemon.cpp
/// @brief Test-only standalone decoding-server process: the service end of a
/// CUDA-Q device_call transport, decoding on the CPU with whatever decoder a
/// YAML config file selects.
///
/// This is the two-process analogue of the in-process host_dispatch device
/// call tests, structured exactly like CUDA-Q's cpu_roce_test_daemon: a
/// cpu_transport transceiver owns the wire and the rings, and
/// libcudaq-realtime's HOST_CALL host-dispatcher loop is wired straight onto
/// those rings.  Both the decoder and the transport are configuration, not
/// code:
///   - decoders come from `--config=<yaml>`
///     (multi_decoder_config::from_yaml_str);
///   - the transport comes from `--transport=udp|cpu_roce`: the UDP ring
///     transceiver (loopback; runs anywhere) or the CPU RoCE RDMA transceiver
///     (requires an RDMA NIC; pairs with the caller's
///     `--cudaq-device-call=cpu_roce` channel and includes the QP/rkey TCP
///     rendezvous server).
///
/// The function table comes from the decoding-server-cqr service plugin
/// (enqueue_syndromes / get_corrections / reset_decoder) regardless of
/// transport or decoder.
///
/// Prints `QEC_DECODING_DAEMON_READY port=<P> ...` on stdout once listening
/// (for udp, P is the UDP port; for cpu_roce, P is the TCP rendezvous port and
/// the line also carries `roce_ip=<IP>`), and
/// `QEC_DECODING_DAEMON_DISPATCHED count=<N>` at shutdown (the two-process
/// stand-in for the in-process cudaqx_qec_device_call_dispatch_count()
/// assertion).
///
/// Usage:
///   qec_decoding_daemon --config=<decoders.yaml>
///                           [--transport=udp|cpu_roce] [--port=0]
///                           [--num-slots=8] [--slot-size=256] [--timeout=60]
///                           [--device=mlx5_0] [--local-ip=10.0.0.2]
///
/// NOTE: --slot-size must match the caller channel's slot size (each frame
/// occupies one full slot stride on both wires).

#include "cudaq/qec/realtime/decoding_config.h"

#include "cudaq/realtime/device_call_service.h"

#include "cudaq/realtime/cpu_transport/udp_wrapper.h"
#include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"
#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h"
#include "cudaq/realtime/daemon/dispatcher/host_dispatcher.h"

#ifdef QEC_HAVE_CPU_ROCE_TRANSPORT
#include "cudaq/realtime/cpu_transport/roce_wrapper.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>

extern "C" void cudaqx_qec_realtime_device_call_service_force_link();
extern "C" std::uint64_t cudaqx_qec_device_call_dispatch_count();
namespace cudaq::qec::decoding::host {
__attribute__((visibility("default"))) std::uint64_t
max_concurrent_decoder_workers();
}

namespace {

namespace config = cudaq::qec::decoding::config;

struct DaemonConfig {
  std::string config_path;
  std::string transport = "udp";
  std::uint16_t port = 0; // 0 => ephemeral, printed on stdout
  std::uint32_t num_slots = 8;
  std::size_t slot_size = 256;
  int timeout_sec = 60;
  // cpu_roce only:
  std::string device = "mlx5_0";
  std::string local_ip = "10.0.0.2";
};

bool starts_with(const std::string &s, const char *prefix) {
  const std::size_t n = std::strlen(prefix);
  return s.size() >= n && std::memcmp(s.data(), prefix, n) == 0;
}

bool parse_args(int argc, char **argv, DaemonConfig &cfg) {
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--help" || a == "-h") {
      std::cout << "Usage: " << argv[0]
                << " --config=<decoders.yaml> [--transport=udp|cpu_roce] "
                   "[--port=N] [--num-slots=N] [--slot-size=N] [--timeout=N] "
                   "[--device=NAME] [--local-ip=ADDR]"
                << std::endl;
      return false;
    } else if (starts_with(a, "--config="))
      cfg.config_path = a.substr(9);
    else if (starts_with(a, "--transport="))
      cfg.transport = a.substr(12);
    else if (starts_with(a, "--port="))
      cfg.port = static_cast<std::uint16_t>(std::stoul(a.substr(7)));
    else if (starts_with(a, "--num-slots="))
      cfg.num_slots = static_cast<std::uint32_t>(std::stoul(a.substr(12)));
    else if (starts_with(a, "--slot-size="))
      cfg.slot_size = std::stoull(a.substr(12));
    else if (starts_with(a, "--timeout="))
      cfg.timeout_sec = std::stoi(a.substr(10));
    else if (starts_with(a, "--device="))
      cfg.device = a.substr(9);
    else if (starts_with(a, "--local-ip="))
      cfg.local_ip = a.substr(11);
    else {
      std::cerr << "Unknown argument: " << a << " (use --help)" << std::endl;
      return false;
    }
  }
  if (cfg.config_path.empty()) {
    std::cerr << "ERROR: --config=<decoders.yaml> is required" << std::endl;
    return false;
  }
  return true;
}

std::atomic<int> g_shutdown{0};
void on_signal(int) { g_shutdown.store(1, std::memory_order_release); }

// Transport-agnostic view of one wired-up transceiver: the four ring
// addresses the dispatcher consumes, plus a teardown hook. Both transports
// provide the identical ring contract (see udp_wrapper.h / roce_wrapper.h).
struct TransportEndpoints {
  volatile std::uint64_t *rx_flags = nullptr;
  volatile std::uint64_t *tx_flags = nullptr;
  std::uint8_t *rx_data = nullptr;
  std::uint8_t *tx_data = nullptr;
  std::function<void()> shutdown;
};

// Publish the rendezvous endpoint for the test fixture. Emitted once the
// caller can start connecting (udp: socket bound; cpu_roce: TCP rendezvous
// listening).
void print_ready(std::uint16_t port, const std::string &extra) {
  std::cout << "QEC_DECODING_DAEMON_READY port=" << port
            << (extra.empty() ? "" : " ") << extra << std::endl;
  std::cout.flush();
}

bool init_udp_transport(const DaemonConfig &cfg, TransportEndpoints &tp) {
  cpu_udp_transceiver_t xcvr =
      cpu_udp_create_transceiver(cfg.slot_size, cfg.num_slots);
  if (!xcvr) {
    std::cerr << "ERROR: udp transceiver create failed" << std::endl;
    return false;
  }
  if (!cpu_udp_bind(xcvr, cfg.port) || !cpu_udp_start(xcvr)) {
    std::cerr << "ERROR: udp transceiver bind/start failed" << std::endl;
    cpu_udp_destroy_transceiver(xcvr);
    return false;
  }
  tp.rx_flags = reinterpret_cast<volatile std::uint64_t *>(
      cpu_udp_get_rx_ring_flag_addr(xcvr));
  tp.tx_flags = reinterpret_cast<volatile std::uint64_t *>(
      cpu_udp_get_tx_ring_flag_addr(xcvr));
  tp.rx_data =
      reinterpret_cast<std::uint8_t *>(cpu_udp_get_rx_ring_data_addr(xcvr));
  tp.tx_data =
      reinterpret_cast<std::uint8_t *>(cpu_udp_get_tx_ring_data_addr(xcvr));
  tp.shutdown = [xcvr] {
    cpu_udp_close(xcvr);
    cpu_udp_destroy_transceiver(xcvr);
  };
  print_ready(cpu_udp_get_port(xcvr), "transport=udp");
  return true;
}

#ifdef QEC_HAVE_CPU_ROCE_TRANSPORT

// Must match CpuRoceChannel's RendezvousInfo byte-for-byte (network order).
struct RendezvousInfo {
  std::uint32_t qp_number = 0;
  std::uint32_t rkey = 0;
  std::uint32_t roce_ipv4 = 0;
};

bool write_all(int fd, const void *buf, std::size_t len) {
  const auto *p = static_cast<const std::uint8_t *>(buf);
  while (len > 0) {
    const ssize_t n = ::write(fd, p, len);
    if (n <= 0) {
      if (n < 0 && errno == EINTR)
        continue;
      return false;
    }
    p += n;
    len -= static_cast<std::size_t>(n);
  }
  return true;
}

bool read_all(int fd, void *buf, std::size_t len) {
  auto *p = static_cast<std::uint8_t *>(buf);
  while (len > 0) {
    const ssize_t n = ::read(fd, p, len);
    if (n <= 0) {
      if (n < 0 && errno == EINTR)
        continue;
      return false;
    }
    p += n;
    len -= static_cast<std::size_t>(n);
  }
  return true;
}

// Service-end CPU RoCE bring-up, mirroring cpu_roce_test_daemon: transceiver
// setup, TCP rendezvous server (READY printed once listening; blocks in
// accept until the caller channel connects), QP/rkey swap, connect, monitor
// thread.  tx_mode=RDMA_SEND: we Send responses; the caller Writes requests.
bool init_cpu_roce_transport(const DaemonConfig &cfg, TransportEndpoints &tp) {
  cpu_roce_transceiver_t xcvr = cpu_roce_create_transceiver(
      cfg.device.c_str(), /*ib_port=*/1, /*tx_ibv_qp=*/0u,
      /*frame_size=*/cfg.slot_size, /*page_size=*/cfg.slot_size, cfg.num_slots,
      /*peer_ip=*/"0.0.0.0", /*forward=*/0, /*rx_only=*/0, /*tx_only=*/0,
      /*unified=*/0, CPU_ROCE_TX_MODE_RDMA_SEND, /*peer_rx_base_addr=*/0,
      /*peer_rx_rkey=*/0);
  if (!xcvr) {
    std::cerr << "ERROR: cpu_roce transceiver create failed" << std::endl;
    return false;
  }
  cpu_roce_set_local_ip(xcvr, cfg.local_ip.c_str());
  if (!cpu_roce_setup(xcvr)) {
    std::cerr << "ERROR: cpu_roce transceiver setup() failed" << std::endl;
    cpu_roce_destroy_transceiver(xcvr);
    return false;
  }

  // TCP rendezvous server: mirror of CpuRoceChannel::exchangeRendezvous
  // (server reads the caller's {qp, rkey, ip} first, then replies).
  const int listen_fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (listen_fd < 0) {
    std::cerr << "ERROR: rendezvous socket() failed" << std::endl;
    cpu_roce_destroy_transceiver(xcvr);
    return false;
  }
  int reuse = 1;
  ::setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
  sockaddr_in srv{};
  srv.sin_family = AF_INET;
  srv.sin_addr.s_addr = htonl(INADDR_ANY);
  srv.sin_port = htons(cfg.port);
  if (::bind(listen_fd, reinterpret_cast<sockaddr *>(&srv), sizeof(srv)) !=
          0 ||
      ::listen(listen_fd, 1) != 0) {
    std::cerr << "ERROR: rendezvous bind/listen failed" << std::endl;
    ::close(listen_fd);
    cpu_roce_destroy_transceiver(xcvr);
    return false;
  }
  socklen_t srvlen = sizeof(srv);
  ::getsockname(listen_fd, reinterpret_cast<sockaddr *>(&srv), &srvlen);
  print_ready(ntohs(srv.sin_port),
              "transport=cpu_roce roce_ip=" + cfg.local_ip);

  const int conn_fd = ::accept(listen_fd, nullptr, nullptr);
  ::close(listen_fd);
  if (conn_fd < 0) {
    std::cerr << "ERROR: rendezvous accept() failed" << std::endl;
    cpu_roce_destroy_transceiver(xcvr);
    return false;
  }
  int one = 1;
  ::setsockopt(conn_fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));

  RendezvousInfo peer{};
  in_addr local_addr{};
  ::inet_pton(AF_INET, cfg.local_ip.c_str(), &local_addr);
  const RendezvousInfo self{htonl(cpu_roce_get_qp_number(xcvr)),
                            htonl(cpu_roce_get_rkey(xcvr)),
                            local_addr.s_addr};
  if (!read_all(conn_fd, &peer, sizeof(peer)) ||
      !write_all(conn_fd, &self, sizeof(self))) {
    std::cerr << "ERROR: rendezvous exchange failed" << std::endl;
    ::close(conn_fd);
    cpu_roce_destroy_transceiver(xcvr);
    return false;
  }
  ::close(conn_fd);

  char peer_ip[INET_ADDRSTRLEN] = {0};
  in_addr peer_addr{};
  peer_addr.s_addr = peer.roce_ipv4;
  ::inet_ntop(AF_INET, &peer_addr, peer_ip, sizeof(peer_ip));
  // We Send responses (no RDMA Writes to the caller), so no peer rkey needed.
  if (!cpu_roce_connect(xcvr, ntohl(peer.qp_number), peer_ip,
                        /*peer_rx_rkey=*/0)) {
    std::cerr << "ERROR: cpu_roce transceiver connect() failed" << std::endl;
    cpu_roce_destroy_transceiver(xcvr);
    return false;
  }

  auto *monitor =
      new std::thread([xcvr] { cpu_roce_blocking_monitor(xcvr); });

  tp.rx_flags = reinterpret_cast<volatile std::uint64_t *>(
      cpu_roce_get_rx_ring_flag_addr(xcvr));
  tp.tx_flags = reinterpret_cast<volatile std::uint64_t *>(
      cpu_roce_get_tx_ring_flag_addr(xcvr));
  tp.rx_data =
      reinterpret_cast<std::uint8_t *>(cpu_roce_get_rx_ring_data_addr(xcvr));
  tp.tx_data =
      reinterpret_cast<std::uint8_t *>(cpu_roce_get_tx_ring_data_addr(xcvr));
  tp.shutdown = [xcvr, monitor] {
    cpu_roce_close(xcvr);
    if (monitor->joinable())
      monitor->join();
    delete monitor;
    cpu_roce_destroy_transceiver(xcvr);
  };
  return true;
}

#endif // QEC_HAVE_CPU_ROCE_TRANSPORT

} // namespace

int main(int argc, char **argv) {
  DaemonConfig cfg;
  if (!parse_args(argc, argv, cfg))
    return 1;

  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);

  // [1] Configure the decoders described by the YAML file in THIS process; in
  // the two-process topology decode state lives with the service, not the
  // caller.
  std::ifstream config_file(cfg.config_path);
  if (!config_file) {
    std::cerr << "ERROR: cannot open config file " << cfg.config_path
              << std::endl;
    return 1;
  }
  std::stringstream config_text;
  config_text << config_file.rdbuf();
  auto decoder_config =
      config::multi_decoder_config::from_yaml_str(config_text.str());
  if (decoder_config.decoders.empty()) {
    std::cerr << "ERROR: no decoders parsed from " << cfg.config_path
              << std::endl;
    return 1;
  }
  if (config::configure_decoders(decoder_config) != 0) {
    std::cerr << "ERROR: configure_decoders failed" << std::endl;
    return 1;
  }
  std::cout << "Configured " << decoder_config.decoders.size()
            << " decoder(s); decoder 0 type: "
            << decoder_config.decoders[0].type
            << "; transport: " << cfg.transport << std::endl;

  // [2] Pull the QEC HOST_CALL function table from the decoding-server-cqr
  // service plugin -- the same table the in-process host_dispatch test uses.
  cudaqx_qec_realtime_device_call_service_force_link();
  auto pluginInfo = cudaqGetDeviceCallServicePluginInfo();
  if (!pluginInfo.getService) {
    std::cerr << "ERROR: QEC device_call service plugin missing" << std::endl;
    return 1;
  }
  auto *service = pluginInfo.getService();
  if (!service) {
    std::cerr << "ERROR: QEC device_call service create failed" << std::endl;
    return 1;
  }
  // The session owns the function table; keep it alive for the daemon's
  // lifetime (the dispatcher loop below reads table.entries in place).
  auto session = service->createDispatchSession(
      cudaq::realtime::DeviceCallDispatchMode::Host);
  if (!session) {
    std::cerr << "ERROR: QEC device_call service does not support host "
                 "dispatch"
              << std::endl;
    return 1;
  }
  const auto &table = session->dispatchTable();
  if (!table.entries || table.count == 0) {
    std::cerr << "ERROR: QEC host dispatch table unavailable" << std::endl;
    return 1;
  }

  // [3] Bring up the selected transport (prints the READY line once the
  // caller can start connecting).
  TransportEndpoints tp;
  if (cfg.transport == "udp") {
    if (!init_udp_transport(cfg, tp))
      return 1;
  } else if (cfg.transport == "cpu_roce") {
#ifdef QEC_HAVE_CPU_ROCE_TRANSPORT
    if (!init_cpu_roce_transport(cfg, tp))
      return 1;
#else
    std::cerr << "ERROR: this daemon was built without cpu_roce transport "
                 "support (libcudaq-realtime-cpu-transport not found)"
              << std::endl;
    return 1;
#endif
  } else {
    std::cerr << "ERROR: unknown --transport=" << cfg.transport
              << " (expected udp or cpu_roce)" << std::endl;
    return 1;
  }

  // [4] Wire the libcudaq-realtime host dispatcher to the transceiver rings,
  // exactly as cpu_roce_test_daemon does. Everything from here down is
  // transport-independent.
  std::atomic<int> dispatcher_shutdown{0};
  std::uint64_t packets_dispatched = 0;
  cudaq_host_dispatch_loop_ctx_t dctx{};
  dctx.ringbuffer.rx_flags_host = tp.rx_flags;
  dctx.ringbuffer.tx_flags_host = tp.tx_flags;
  dctx.ringbuffer.rx_data_host = tp.rx_data;
  dctx.ringbuffer.tx_data_host = tp.tx_data;
  dctx.ringbuffer.rx_stride_sz = cfg.slot_size;
  dctx.ringbuffer.tx_stride_sz = cfg.slot_size;
  dctx.config.num_slots = cfg.num_slots;
  dctx.config.slot_size = static_cast<std::uint32_t>(cfg.slot_size);
  dctx.config.dispatch_path = CUDAQ_DISPATCH_PATH_HOST;
  dctx.config.dispatch_mode = CUDAQ_DISPATCH_HOST_CALL;
  dctx.config.skip_tx_markers = 1;
  dctx.function_table.entries = table.entries;
  dctx.function_table.count = table.count;
  dctx.shutdown_flag = &dispatcher_shutdown;
  dctx.stats_counter = &packets_dispatched;
  dctx.skip_stream_sweep = true;

  std::thread dispatcher_thread(
      [&dctx]() { cudaq_host_dispatcher_loop(&dctx); });

  // [5] Run until signalled or timed out.
  const auto start_time = std::chrono::steady_clock::now();
  while (g_shutdown.load(std::memory_order_acquire) == 0) {
    const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                             std::chrono::steady_clock::now() - start_time)
                             .count();
    if (elapsed > cfg.timeout_sec)
      break;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  // [6] Orderly shutdown.
  dispatcher_shutdown.store(1, std::memory_order_release);
  if (dispatcher_thread.joinable())
    dispatcher_thread.join();
  if (tp.shutdown)
    tp.shutdown();
  config::finalize_decoders();

  std::cout << "QEC_DECODING_DAEMON_DISPATCHED count="
            << cudaqx_qec_device_call_dispatch_count() << std::endl;
  // Concurrency evidence for multi-logical-qubit tests: high-water mark of
  // simultaneously-busy per-decoder execution workers.
  std::cout << "QEC_DECODING_DAEMON_MAX_CONCURRENT_DECODERS count="
            << cudaq::qec::decoding::host::max_concurrent_decoder_workers()
            << std::endl;
  return 0;
}
