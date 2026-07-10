/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file decoding_server.cpp
/// @brief Standalone decoding-server process: the service end of a
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
///   - for cpu_roce, `--qp_config=rendezvous|hsb_fpga` selects how queue pairs
///     are exchanged.  `rendezvous` (default) is the TCP QP/rkey swap with a
///     CpuRoceChannel caller.  `hsb_fpga` is the Holoscan-Sensor-Bridge FPGA
///     method: the peer QP comes from `--remote-qp` (the FPGA data-plane QP,
///     or the emulator's QP) and this server prints its own QP / RKey /
///     Buffer Addr in the canonical bridge handshake format
///     (hololink_bridge_common.h) for the orchestration script to relay to
///     the playback tool -- which alone programs the FPGA over the Hololink
///     control plane.  The server itself performs NO control-plane traffic.
///
/// The function table comes from the decoding-server-cqr service plugin
/// (enqueue_syndromes / get_corrections / reset_decoder) regardless of
/// transport or decoder.
///
/// Prints `QEC_DECODING_SERVER_READY port=<P> ...` on stdout once listening
/// (for udp, P is the UDP port; for cpu_roce, P is the TCP rendezvous port and
/// the line also carries `roce_ip=<IP>`), and
/// `QEC_DECODING_SERVER_DISPATCHED count=<N>` at shutdown (the two-process
/// stand-in for the in-process cudaqx_qec_device_call_dispatch_count()
/// assertion).
///
/// Usage:
///   decoding_server --config=<decoders.yaml>
///                           [--transport=udp|cpu_roce] [--port=0]
///                           [--num-slots=8] [--slot-size=256] [--timeout=60]
///                           [--device=mlx5_0] [--local-ip=10.0.0.2]
///                           [--qp_config=rendezvous|hsb_fpga]
///                           [--peer-ip=ADDR] [--remote-qp=0x2]
///                           [--frame-size=N]
///
/// NOTE: --slot-size must match the caller channel's slot size (each frame
/// occupies one full slot stride on both wires).  With --qp_config=hsb_fpga,
/// --slot-size is the HSB page size (ring slot stride) and --num-slots is
/// capped at 64 (the HSB WQE depth).

#include "cudaq/qec/realtime/decoding_config.h"

#include "cudaq/realtime/device_call_service.h"

#include "cudaq/realtime/cpu_transport/udp_wrapper.h"
#include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"
#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h"
#include "cudaq/realtime/daemon/dispatcher/graph_launch_engine.h"

#ifdef QEC_HAVE_CPU_ROCE_TRANSPORT
#include "cudaq/realtime/cpu_transport/roce_wrapper.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

#ifdef QEC_HAVE_GPU_ROCE_TRANSPORT
// DecodingServer.h (and GpuRoceTransceiver.h via DecodingServer.cpp) live in
// the decoding-server-cqr directory, added to include paths by CMakeLists when
// CUDAQ_GPU_ROCE_AVAILABLE is true.
#include "DecodingServer.h"
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
extern "C" std::uint64_t cudaqx_qec_decoding_server_max_concurrent();
extern "C" void cudaqx_qec_decoding_server_shutdown();

namespace {

namespace config = cudaq::qec::decoding::config;

struct ServerConfig {
  std::string config_path;
  std::string transport = "udp";
  std::uint16_t port = 0; // 0 => ephemeral, printed on stdout
  std::uint32_t num_slots = 8;
  std::size_t slot_size = 256;
  int timeout_sec = 60;
  // cpu_roce only:
  std::string device = "mlx5_0";
  std::string local_ip = "10.0.0.2";
  // cpu_roce QP exchange method (see file header).
  std::string qp_config = "rendezvous";
  // hsb_fpga only:
  std::string peer_ip;           // FPGA/emulator data-plane IPv4 (required)
  std::uint32_t remote_qp = 0x2; // FPGA data-plane QP (emulator QP in emulate)
  std::size_t frame_size = 0;    // TX SGE bytes; 0 => slot_size
};

bool starts_with(const std::string &s, const char *prefix) {
  const std::size_t n = std::strlen(prefix);
  return s.size() >= n && std::memcmp(s.data(), prefix, n) == 0;
}

bool parse_args(int argc, char **argv, ServerConfig &cfg) {
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--help" || a == "-h") {
      std::cout << "Usage: " << argv[0]
                << " --config=<decoders.yaml> "
                   "[--transport=udp|cpu_roce|gpu_roce] "
                   "[--port=N] [--num-slots=N] [--slot-size=N] [--timeout=N] "
                   "[--device=NAME] [--local-ip=ADDR] "
                   "[--qp_config=rendezvous|hsb_fpga] [--peer-ip=ADDR] "
                   "[--remote-qp=N] [--frame-size=N]"
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
    else if (starts_with(a, "--qp_config="))
      cfg.qp_config = a.substr(12);
    else if (starts_with(a, "--peer-ip="))
      cfg.peer_ip = a.substr(10);
    else if (starts_with(a, "--remote-qp="))
      // base 0: accepts both decimal and 0x-prefixed hex (QP numbers are
      // conventionally printed in hex, e.g. the FPGA's fixed 0x2).
      cfg.remote_qp =
          static_cast<std::uint32_t>(std::stoul(a.substr(12), nullptr, 0));
    else if (starts_with(a, "--frame-size="))
      cfg.frame_size = std::stoull(a.substr(13));
    else {
      std::cerr << "Unknown argument: " << a << " (use --help)" << std::endl;
      return false;
    }
  }
  if (cfg.config_path.empty()) {
    std::cerr << "ERROR: --config=<decoders.yaml> is required" << std::endl;
    return false;
  }
  if (cfg.qp_config != "rendezvous" && cfg.qp_config != "hsb_fpga") {
    std::cerr << "ERROR: unknown --qp_config=" << cfg.qp_config
              << " (expected rendezvous or hsb_fpga)" << std::endl;
    return false;
  }
  if (cfg.qp_config == "hsb_fpga") {
    if (cfg.transport != "cpu_roce") {
      std::cerr << "ERROR: --qp_config=hsb_fpga requires --transport=cpu_roce"
                << std::endl;
      return false;
    }
    if (cfg.peer_ip.empty()) {
      std::cerr << "ERROR: --qp_config=hsb_fpga requires --peer-ip=<FPGA or "
                   "emulator IPv4>"
                << std::endl;
      return false;
    }
    // The HSB receive queue is WQE_NUM=64 deep; a deeper ring would alias two
    // slots per WQE and race RX against TX (same constraint as the Hololink
    // bridges).
    constexpr std::uint32_t kHsbWqeNum = 64;
    if (cfg.num_slots > kHsbWqeNum) {
      std::cerr << "WARNING: --num-slots=" << cfg.num_slots << " exceeds the "
                << "HSB WQE depth; clamping to " << kHsbWqeNum << std::endl;
      cfg.num_slots = kHsbWqeNum;
    }
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
  std::cout << "QEC_DECODING_SERVER_READY port=" << port
            << (extra.empty() ? "" : " ") << extra << std::endl;
  std::cout.flush();
}

bool init_udp_transport(const ServerConfig &cfg, TransportEndpoints &tp) {
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
bool init_cpu_roce_transport(const ServerConfig &cfg, TransportEndpoints &tp) {
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
  if (::bind(listen_fd, reinterpret_cast<sockaddr *>(&srv), sizeof(srv)) != 0 ||
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
                            htonl(cpu_roce_get_rkey(xcvr)), local_addr.s_addr};
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

  auto *monitor = new std::thread([xcvr] { cpu_roce_blocking_monitor(xcvr); });

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

// CPU RoCE bring-up for the HSB FPGA QP-exchange method, mirroring
// cuda-quantum's hsb_bridge_cpu.cpp (the proven CPU<->FPGA precedent): the
// peer QP is a CLI input (the FPGA's fixed data-plane QP, or the emulator's),
// the transceiver is created one-shot with the peer already known
// (cpu_roce_start, no TCP rendezvous / no connect step), and this server
// publishes its own QP / RKey / Buffer Addr on stdout in the canonical bridge
// handshake format.  The orchestration script scrapes those values and hands
// them to the playback tool, which alone programs the FPGA SIF over the
// Hololink control plane (DataChannel::authenticate / configure_roce) -- this
// server performs NO control-plane traffic.
//
// tx_mode=RDMA_SEND: the FPGA/emulator posts receive WQEs for the
// server->FPGA direction and RDMA-WRITEs requests into our ring, exactly as
// with hsb_bridge_cpu.
bool init_cpu_roce_hsb_fpga_transport(const ServerConfig &cfg,
                                      TransportEndpoints &tp) {
  const std::size_t frame_size =
      cfg.frame_size ? cfg.frame_size : cfg.slot_size;

  std::cout << "HSB FPGA QP exchange:\n"
            << "  Device:     " << cfg.device << "\n"
            << "  Peer IP:    " << cfg.peer_ip << "\n"
            << "  Remote QP:  0x" << std::hex << cfg.remote_qp << std::dec
            << "\n"
            << "  Slots:      " << cfg.num_slots << "\n"
            << "  Slot size:  " << cfg.slot_size << " bytes\n"
            << "  Frame size: " << frame_size << " bytes" << std::endl;

  cpu_roce_transceiver_t xcvr = cpu_roce_create_transceiver(
      cfg.device.c_str(), /*ib_port=*/1, /*tx_ibv_qp=*/cfg.remote_qp,
      frame_size, /*page_size=*/cfg.slot_size, cfg.num_slots,
      cfg.peer_ip.c_str(), /*forward=*/0, /*rx_only=*/0, /*tx_only=*/0,
      /*unified=*/0, CPU_ROCE_TX_MODE_RDMA_SEND, /*peer_rx_base_addr=*/0,
      /*peer_rx_rkey=*/0);
  if (!xcvr) {
    std::cerr << "ERROR: cpu_roce transceiver create failed" << std::endl;
    return false;
  }
  if (!cpu_roce_start(xcvr)) {
    std::cerr << "ERROR: cpu_roce_start failed" << std::endl;
    cpu_roce_destroy_transceiver(xcvr);
    return false;
  }

  auto *monitor = new std::thread([xcvr] { cpu_roce_blocking_monitor(xcvr); });

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

  // Canonical bridge handshake.  Format MUST match hololink_bridge_common.h
  // exactly -- "  KEY: VALUE", single space after the colon -- because the
  // orchestration script parses it with strict regexes (same contract as
  // hsb_bridge_cpu.cpp and the Hololink GPU bridges).  Buffer Addr is 0 with
  // an iova=0 MR registration; the playback tool handles that.
  std::cout << "\n=== Bridge Ready ===" << std::endl;
  std::cout << "  QP Number: 0x" << std::hex << cpu_roce_get_qp_number(xcvr)
            << std::dec << std::endl;
  std::cout << "  RKey: " << cpu_roce_get_rkey(xcvr) << std::endl;
  std::cout << "  Buffer Addr: 0x" << std::hex << cpu_roce_get_buffer_addr(xcvr)
            << std::dec << std::endl;
  std::cout.flush();

  print_ready(/*port=*/0,
              "transport=cpu_roce qp_config=hsb_fpga peer_ip=" + cfg.peer_ip);
  return true;
}

#endif // QEC_HAVE_CPU_ROCE_TRANSPORT

} // namespace

int main(int argc, char **argv) {
  ServerConfig cfg;
  if (!parse_args(argc, argv, cfg))
    return 1;

  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);

  // [1] Validate the YAML and hand its path to the decoding-server service:
  // the DecodingServer (one DecodingSession worker thread per decoder) builds
  // the decoder instances itself when the dispatch session is created below.
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
  ::setenv("CUDAQ_QEC_DECODER_CONFIG", cfg.config_path.c_str(),
           /*overwrite=*/1);
  std::cout << "Configured " << decoder_config.decoders.size()
            << " decoder(s); decoder 0 type: "
            << decoder_config.decoders[0].type
            << "; transport: " << cfg.transport << std::endl;

  // [2a] GPU RoCE takes a completely different path: bypass the CQR
  // DeviceCallService / HOST_CALL dispatcher and use DecodingServer directly.
  // Must be checked before force-linking the CQR plugin (which creates a
  // DecodingServer internally for the HOST_CALL path) to avoid double-init.
#ifdef QEC_HAVE_GPU_ROCE_TRANSPORT
  if (cfg.transport == "gpu_roce") {
    // DecodingServer(config_yaml) reads the YAML, creates GpuRoceTransceiver
    // (Hololink Sensor Bridge + DOCA), loads decoder sessions, and calls
    // launch_scheduler() to wire the CUDAQ device-graph scheduler to the
    // Hololink ring buffers.  The GPU scheduler then handles
    // RX→dispatch→decode→TX autonomously; this thread just waits for signal.
    cudaq::qec::decoding_server::DecodingServer server(cfg.config_path);
    // QP/rkey/buf already printed to stdout by launch_scheduler() so the
    // orchestration script can grep them before the READY line.
    std::cout << "QEC_DECODING_SERVER_READY gpu_roce" << std::endl;
    std::cout.flush();
    std::thread server_thread([&server] { server.run(); });
    const auto start_time_gr = std::chrono::steady_clock::now();
    while (g_shutdown.load(std::memory_order_acquire) == 0) {
      const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                               std::chrono::steady_clock::now() - start_time_gr)
                               .count();
      if (elapsed > cfg.timeout_sec)
        break;
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    server.stop();
    server_thread.join();
    return 0;
  }
#endif

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
  // The session owns the function table; keep it alive for the server's
  // lifetime (the dispatcher loop below reads table.entries in place).
  // Creating it also starts the DecodingServer (decoder construction + one
  // worker thread per decoder) -- before the READY line below, so slow
  // decoder initialization never races the first client request.
  std::unique_ptr<cudaq::realtime::DeviceCallServiceSession> session;
  try {
    session = service->createDispatchSession(
        cudaq::realtime::DeviceCallDispatchMode::Host);
  } catch (const std::exception &e) {
    std::cerr << "ERROR: decoding-server startup failed: " << e.what()
              << std::endl;
    return 1;
  }
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
    if (cfg.qp_config == "hsb_fpga") {
      if (!init_cpu_roce_hsb_fpga_transport(cfg, tp))
        return 1;
    } else if (!init_cpu_roce_transport(cfg, tp))
      return 1;
#else
    std::cerr << "ERROR: this server was built without cpu_roce transport "
                 "support (libcudaq-realtime-cpu-roce-transport not found)"
              << std::endl;
    return 1;
#endif
  } else if (cfg.transport == "gpu_roce") {
    // gpu_roce is handled before the CQR plugin force-link above ([2a]).
    // Reaching here means QEC_HAVE_GPU_ROCE_TRANSPORT was not defined at
    // build time (the server was not built with GPU RoCE support).
    std::cerr << "ERROR: this server was built without gpu_roce transport "
                 "support (rebuild with HOLOSCAN_SENSOR_BRIDGE_BUILD_DIR, "
                 "DOCA, and CUDA)"
              << std::endl;
    return 1;
  } else {
    std::cerr << "ERROR: unknown --transport=" << cfg.transport
              << " (expected udp, cpu_roce, or gpu_roce)" << std::endl;
    return 1;
  }

  // [4] Wire the libcudaq-realtime host dispatcher to the transceiver rings,
  // exactly as cpu_roce_test_daemon does. Everything from here down is
  // transport-independent.
  // The dispatch table is HOST_CALL-only, so the ring loop runs the inline
  // HOST_CALL path with no GRAPH_LAUNCH engine (engine == nullptr). Mirrors the
  // HOST_CALL-only branch in qec_realtime_session.cpp.
  int dispatcher_shutdown = 0;
  std::uint64_t packets_dispatched = 0;
  cudaq_ringbuffer_t ringbuffer{};
  ringbuffer.rx_flags_host = tp.rx_flags;
  ringbuffer.tx_flags_host = tp.tx_flags;
  ringbuffer.rx_data_host = tp.rx_data;
  ringbuffer.tx_data_host = tp.tx_data;
  ringbuffer.rx_stride_sz = cfg.slot_size;
  ringbuffer.tx_stride_sz = cfg.slot_size;
  cudaq_dispatcher_config_t dispatch_config{};
  dispatch_config.num_slots = cfg.num_slots;
  dispatch_config.slot_size = static_cast<std::uint32_t>(cfg.slot_size);
  dispatch_config.dispatch_path = CUDAQ_DISPATCH_PATH_HOST;
  dispatch_config.dispatch_mode = CUDAQ_DISPATCH_HOST_CALL;
  dispatch_config.skip_tx_markers = 1;
  cudaq_function_table_t function_table{};
  function_table.entries = table.entries;
  function_table.count = table.count;

  std::thread dispatcher_thread([&]() {
    cudaq_host_ring_dispatch_loop(
        &ringbuffer, &function_table, &dispatch_config,
        /*engine=*/nullptr, &dispatcher_shutdown, &packets_dispatched);
  });

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

  // [6] Orderly shutdown.  The dispatch loop polls the flag as volatile, not
  // atomically; publish the store the same way qec_realtime_session.cpp does.
  __atomic_store_n(&dispatcher_shutdown, 1, __ATOMIC_RELEASE);
  __sync_synchronize();
  if (dispatcher_thread.joinable())
    dispatcher_thread.join();
  if (tp.shutdown)
    tp.shutdown();
  // Stop the DecodingServer receive loop and join its thread before the
  // process exits (a still-joinable static thread would std::terminate).
  cudaqx_qec_decoding_server_shutdown();

  std::cout << "QEC_DECODING_SERVER_DISPATCHED count="
            << cudaqx_qec_device_call_dispatch_count() << std::endl;
  // Concurrency evidence for multi-logical-qubit tests: high-water mark of
  // simultaneously-busy DecodingSession workers.
  std::cout << "QEC_DECODING_SERVER_MAX_CONCURRENT_DECODERS count="
            << cudaqx_qec_decoding_server_max_concurrent() << std::endl;
  return 0;
}
