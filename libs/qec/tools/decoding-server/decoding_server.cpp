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
///                           [--num-slots=8] [--slot-size=256] [--timeout=N]
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
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

#ifdef QEC_HAVE_GPU_ROCE_TRANSPORT
// DecodingServer.h (and GpuRoceTransceiver.h via DecodingServer.cpp) live in
// the decoding-server-cqr directory, added to include paths by CMakeLists when
// CUDAQ_GPU_ROCE_AVAILABLE is true.
#include "DecodingServer.h"
#endif

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

extern "C" void cudaqx_qec_realtime_device_call_service_force_link();
extern "C" std::uint64_t cudaqx_qec_device_call_dispatch_count();
extern "C" std::uint64_t cudaqx_qec_decoding_server_max_concurrent();
extern "C" void cudaqx_qec_decoding_server_print_stats();
extern "C" void cudaqx_qec_decoding_server_shutdown();
extern "C" int cudaqx_qec_decoding_server_apply_config_from_yaml(
    const char *yaml, size_t yaml_len, char *err, size_t err_len);

namespace {

namespace config = cudaq::qec::decoding::config;

constexpr const char *kDefaultEndpointFile =
    "/tmp/cudaq-qec-decoding-server.env";

std::string endpoint_file_default() { return kDefaultEndpointFile; }

std::string endpoint_file_default_for_transport(const std::string &transport) {
  if (transport == "udp")
    return kDefaultEndpointFile;
  std::string suffix = transport;
  for (char &c : suffix)
    if (!std::isalnum(static_cast<unsigned char>(c)))
      c = '-';
  return "/tmp/cudaq-qec-decoding-server-" + suffix + ".env";
}

int current_process_id() {
#ifdef _WIN32
  return _getpid();
#else
  return static_cast<int>(::getpid());
#endif
}

struct ServerConfig {
  std::string config_path;
  std::string transport = "udp";
  bool transport_explicit = false;
  std::uint16_t port = 0; // 0 => ephemeral, printed on stdout
  std::uint32_t num_slots = 8;
  std::size_t slot_size = 256;
  int timeout_sec = 0; // 0 => run until signalled
  int stats_interval_sec = 0;
  std::string endpoint_file;
  bool endpoint_file_explicit = false;
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

std::string trim_copy(std::string s) {
  auto not_space = [](unsigned char c) { return !std::isspace(c); };
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
  s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
  return s;
}

int leading_spaces(const std::string &line) {
  int count = 0;
  for (char c : line) {
    if (c == ' ')
      ++count;
    else
      break;
  }
  return count;
}

void reject_tab_indentation(const std::string &line) {
  for (char c : line) {
    if (c == '\t')
      throw std::runtime_error(
          "daemon YAML indentation must use spaces, not tabs");
    if (c != ' ')
      return;
  }
}

std::string strip_inline_comment(std::string value) {
  bool quoted = false;
  char quote = '\0';
  for (std::size_t i = 0; i < value.size(); ++i) {
    const char c = value[i];
    if ((c == '\'' || c == '"') && (i == 0 || value[i - 1] != '\\')) {
      if (!quoted) {
        quoted = true;
        quote = c;
      } else if (quote == c) {
        quoted = false;
      }
    } else if (c == '#' && !quoted) {
      return trim_copy(value.substr(0, i));
    }
  }
  return trim_copy(value);
}

std::string unquote_yaml_scalar(std::string value) {
  value = strip_inline_comment(trim_copy(std::move(value)));
  if (value.size() >= 2 && ((value.front() == '"' && value.back() == '"') ||
                            (value.front() == '\'' && value.back() == '\'')))
    return value.substr(1, value.size() - 2);
  return value;
}

bool parse_yaml_key_value(const std::string &text, std::string &key,
                          std::string &value) {
  const std::size_t colon = text.find(':');
  if (colon == std::string::npos)
    return false;
  key = trim_copy(text.substr(0, colon));
  value = unquote_yaml_scalar(text.substr(colon + 1));
  return !key.empty();
}

bool yaml_key_is(const std::string &trimmed, const char *expected) {
  std::string key;
  std::string value;
  return parse_yaml_key_value(trimmed, key, value) && key == expected;
}

std::vector<std::string> split_csv(const std::string &csv) {
  std::vector<std::string> out;
  std::stringstream ss(csv);
  std::string item;
  while (std::getline(ss, item, ',')) {
    item = trim_copy(std::move(item));
    if (!item.empty())
      out.push_back(item);
  }
  return out;
}

void apply_transport_yaml_value(ServerConfig &cfg, const std::string &key,
                                const std::string &value) {
  if (key == "type" || key == "transport") {
    cfg.transport = value;
  } else if (key == "port") {
    cfg.port = static_cast<std::uint16_t>(std::stoul(value));
  } else if (key == "num_slots") {
    cfg.num_slots = static_cast<std::uint32_t>(std::stoul(value));
  } else if (key == "slot_size") {
    cfg.slot_size = std::stoull(value);
  } else if (key == "device") {
    cfg.device = value;
  } else if (key == "local_ip") {
    cfg.local_ip = value;
  } else if (key == "qp_config") {
    cfg.qp_config = value;
  } else if (key == "peer_ip") {
    cfg.peer_ip = value;
  } else if (key == "remote_qp") {
    cfg.remote_qp = static_cast<std::uint32_t>(std::stoul(value, nullptr, 0));
  } else if (key == "frame_size") {
    cfg.frame_size = std::stoull(value);
  } else {
    throw std::runtime_error("unknown server.transports key '" + key + "'");
  }
}

std::string decoder_yaml_from_daemon_yaml(const std::string &yaml) {
  std::stringstream out;
  std::istringstream in(yaml);
  std::string line;
  bool skipping_server = false;
  while (std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r')
      line.pop_back();
    reject_tab_indentation(line);
    const std::string trimmed = trim_copy(line);
    const int indent = leading_spaces(line);
    if (!skipping_server && indent == 0 && yaml_key_is(trimmed, "server")) {
      skipping_server = true;
      continue;
    }
    if (skipping_server) {
      if (!trimmed.empty() && trimmed[0] != '#' && indent == 0) {
        skipping_server = false;
      } else {
        continue;
      }
    }
    out << line << '\n';
  }
  return out.str();
}

std::string materialize_decoder_yaml(const ServerConfig &cfg,
                                     const std::string &raw_yaml,
                                     const std::string &decoder_yaml) {
  if (raw_yaml == decoder_yaml)
    return cfg.config_path;

  const auto stamp =
      std::chrono::steady_clock::now().time_since_epoch().count();
  const auto hash = std::hash<std::string>{}(cfg.config_path);
  const std::string path = "/tmp/cudaq-qec-decoding-server-decoders-" +
                           std::to_string(hash) + "-" + std::to_string(stamp) +
                           ".yaml";
  std::ofstream out(path, std::ios::trunc);
  if (!out)
    throw std::runtime_error("failed to write stripped decoder config: " +
                             path);
  out << decoder_yaml;
  return path;
}

struct TempFileCleanup {
  std::string path;
  bool enabled = false;

  ~TempFileCleanup() {
    if (enabled && !path.empty())
      std::remove(path.c_str());
  }
};

std::vector<ServerConfig>
parse_yaml_transport_configs(const std::string &yaml,
                             const ServerConfig &base) {
  std::vector<ServerConfig> transports;
  std::istringstream in(yaml);
  std::string line;
  bool in_server = false;
  bool saw_server = false;
  bool in_transports = false;
  bool saw_transports = false;
  bool have_current = false;
  int server_indent = -1;
  int transports_indent = -1;
  ServerConfig current;

  auto finish_current = [&] {
    if (!have_current)
      return;
    if (current.transport.empty())
      throw std::runtime_error("server.transports entry is missing type");
    transports.push_back(current);
    have_current = false;
  };

  while (std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r')
      line.pop_back();
    reject_tab_indentation(line);
    const std::string trimmed = trim_copy(line);
    if (trimmed.empty() || trimmed[0] == '#')
      continue;
    const int indent = leading_spaces(line);

    if (!in_server) {
      if (indent == 0 && yaml_key_is(trimmed, "server")) {
        in_server = true;
        saw_server = true;
        server_indent = indent;
      }
      continue;
    }

    if (indent <= server_indent && !yaml_key_is(trimmed, "server")) {
      finish_current();
      break;
    }

    if (!in_transports) {
      if (indent > server_indent && yaml_key_is(trimmed, "transports")) {
        in_transports = true;
        saw_transports = true;
        transports_indent = indent;
      }
      continue;
    }

    if (indent <= transports_indent) {
      finish_current();
      in_transports = false;
      continue;
    }

    if (trimmed[0] == '-') {
      finish_current();
      current = base;
      current.transport.clear();
      current.transport_explicit = true;
      current.endpoint_file.clear();
      current.endpoint_file_explicit = false;
      have_current = true;
      const std::string rest = trim_copy(trimmed.substr(1));
      if (!rest.empty()) {
        std::string key, value;
        if (parse_yaml_key_value(rest, key, value)) {
          apply_transport_yaml_value(current, key, value);
        } else {
          current.transport = unquote_yaml_scalar(rest);
        }
      }
      continue;
    }

    if (!have_current)
      throw std::runtime_error("server.transports must be a YAML sequence");
    std::string key, value;
    if (!parse_yaml_key_value(trimmed, key, value))
      throw std::runtime_error("invalid server.transports line: " + trimmed);
    apply_transport_yaml_value(current, key, value);
  }
  finish_current();
  if (saw_server && !saw_transports)
    throw std::runtime_error(
        "top-level server block is missing required transports list");
  if (saw_transports && transports.empty())
    throw std::runtime_error(
        "server.transports must contain at least one entry");
  return transports;
}

void validate_transport_config(ServerConfig &cfg) {
  if (cfg.transport != "udp" && cfg.transport != "cpu_roce" &&
      cfg.transport != "gpu_roce")
    throw std::runtime_error("unknown transport '" + cfg.transport +
                             "' (expected udp, cpu_roce, or gpu_roce)");
  if (cfg.qp_config != "rendezvous" && cfg.qp_config != "hsb_fpga")
    throw std::runtime_error("unknown qp_config '" + cfg.qp_config +
                             "' (expected rendezvous or hsb_fpga)");
  if (cfg.qp_config == "hsb_fpga") {
    if (cfg.transport != "cpu_roce")
      throw std::runtime_error("qp_config=hsb_fpga requires cpu_roce");
    if (cfg.peer_ip.empty())
      throw std::runtime_error("qp_config=hsb_fpga requires peer_ip");
    constexpr std::uint32_t kHsbWqeNum = 64;
    if (cfg.num_slots > kHsbWqeNum) {
      std::cerr << "WARNING: --num-slots=" << cfg.num_slots << " exceeds the "
                << "HSB WQE depth; clamping to " << kHsbWqeNum << std::endl;
      cfg.num_slots = kHsbWqeNum;
    }
  }
}

void finalize_transport_configs(std::vector<ServerConfig> &configs,
                                const ServerConfig &base) {
  if (configs.empty())
    throw std::runtime_error("no transports configured");

  bool has_gpu_roce = false;
  std::set<std::string> endpoint_files;
  for (std::size_t i = 0; i < configs.size(); ++i) {
    auto &cfg = configs[i];
    validate_transport_config(cfg);
    has_gpu_roce = has_gpu_roce || cfg.transport == "gpu_roce";
    if (cfg.endpoint_file.empty()) {
      if (base.endpoint_file_explicit && (configs.size() == 1 || i == 0))
        cfg.endpoint_file = base.endpoint_file;
      else
        cfg.endpoint_file = endpoint_file_default_for_transport(cfg.transport);
    }
    if (!endpoint_files.insert(cfg.endpoint_file).second) {
      cfg.endpoint_file += "." + std::to_string(i + 1);
      endpoint_files.insert(cfg.endpoint_file);
    }
  }
  if (has_gpu_roce && configs.size() != 1)
    throw std::runtime_error(
        "gpu_roce cannot be combined with HOST_CALL transports in one daemon");
}

std::vector<ServerConfig> resolve_transport_configs(const ServerConfig &cfg,
                                                    const std::string &yaml) {
  std::vector<ServerConfig> configs = parse_yaml_transport_configs(yaml, cfg);
  if (configs.empty()) {
    for (const auto &transport : split_csv(cfg.transport)) {
      ServerConfig next = cfg;
      next.transport = transport;
      next.transport_explicit = true;
      configs.push_back(std::move(next));
    }
  }
  finalize_transport_configs(configs, cfg);
  return configs;
}

std::string transport_summary(const std::vector<ServerConfig> &configs) {
  std::string out;
  for (std::size_t i = 0; i < configs.size(); ++i) {
    if (i)
      out += ",";
    out += configs[i].transport;
  }
  return out;
}

bool parse_args(int argc, char **argv, ServerConfig &cfg) {
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--help" || a == "-h") {
      std::cout
          << "Usage: " << argv[0]
          << " --config=<decoders.yaml> "
             "[--transport=udp|cpu_roce|gpu_roce[,..]] "
             "[--port=N] [--num-slots=N] [--slot-size=N] "
             "[--timeout=N] "
             "[--stats-interval=N] [--endpoint-file=PATH] "
             "[--device=NAME] [--local-ip=ADDR] "
             "[--qp_config=rendezvous|hsb_fpga] [--peer-ip=ADDR] "
             "[--remote-qp=N] [--frame-size=N]\n"
             "YAML server.transports is authoritative when present; "
             "--transport is only a legacy fallback for decoder-only YAML. "
             "Existing decoder-only YAML files continue to default to udp. "
             "By default the daemon runs until signalled; --timeout=N is "
             "test-only/CI convenience. SIGHUP parses the current decoder "
             "config, stops existing decoders, and starts replacements; "
             "UDP/CPU RoCE listeners and GPU RoCE transport binding are "
             "process lifetime state. "
             "--stats-interval=N checks live stats every N "
             "seconds and prints QEC_DECODING_SERVER_STATS only when "
             "counters change. Endpoint files are daemon-owned discovery "
             "artifacts, not YAML config; use --endpoint-file or "
             "QEC_DECODING_SERVER_ENDPOINT_FILE only as an operator/test "
             "override. READY prints the watched config path. Rejected "
             "decoder configs keep the current config serving when possible."
          << std::endl;
      return false;
    } else if (starts_with(a, "--config="))
      cfg.config_path = a.substr(9);
    else if (starts_with(a, "--transport=")) {
      cfg.transport = a.substr(12);
      cfg.transport_explicit = true;
    } else if (starts_with(a, "--port="))
      cfg.port = static_cast<std::uint16_t>(std::stoul(a.substr(7)));
    else if (starts_with(a, "--num-slots="))
      cfg.num_slots = static_cast<std::uint32_t>(std::stoul(a.substr(12)));
    else if (starts_with(a, "--slot-size="))
      cfg.slot_size = std::stoull(a.substr(12));
    else if (starts_with(a, "--timeout="))
      cfg.timeout_sec = std::stoi(a.substr(10));
    else if (starts_with(a, "--stats-interval="))
      cfg.stats_interval_sec = std::stoi(a.substr(17));
    else if (starts_with(a, "--endpoint-file=")) {
      cfg.endpoint_file = a.substr(16);
      cfg.endpoint_file_explicit = true;
    } else if (starts_with(a, "--device="))
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
  if (cfg.stats_interval_sec < 0) {
    std::cerr << "ERROR: --stats-interval must be >= 0" << std::endl;
    return false;
  }
  if (cfg.endpoint_file.empty()) {
    if (const char *path = std::getenv("QEC_DECODING_SERVER_ENDPOINT_FILE");
        path && *path) {
      cfg.endpoint_file = path;
      cfg.endpoint_file_explicit = true;
    } else {
      cfg.endpoint_file = endpoint_file_default();
    }
  }
  if (cfg.qp_config != "rendezvous" && cfg.qp_config != "hsb_fpga") {
    std::cerr << "ERROR: unknown --qp_config=" << cfg.qp_config
              << " (expected rendezvous or hsb_fpga)" << std::endl;
    return false;
  }
  if (cfg.transport_explicit && cfg.qp_config == "hsb_fpga" &&
      cfg.transport != "cpu_roce") {
    std::cerr << "ERROR: --qp_config=hsb_fpga requires --transport=cpu_roce"
              << std::endl;
    return false;
  }
  return true;
}

std::atomic<int> g_shutdown{0};
void on_signal(int) { g_shutdown.store(1, std::memory_order_release); }

std::atomic<int> g_apply_config_requested{0};
void on_sighup(int) {
  g_apply_config_requested.store(1, std::memory_order_release);
}

std::atomic<int> g_stats_requested{0};
void on_stats_signal(int) {
  g_stats_requested.store(1, std::memory_order_release);
}
struct LiveStatsSnapshot {
  uint64_t dispatched = 0;
  uint64_t max_concurrent_decoders = 0;
};

LiveStatsSnapshot read_live_stats() {
  return {cudaqx_qec_device_call_dispatch_count(),
          cudaqx_qec_decoding_server_max_concurrent()};
}

bool stats_equal(const LiveStatsSnapshot &lhs, const LiveStatsSnapshot &rhs) {
  return lhs.dispatched == rhs.dispatched &&
         lhs.max_concurrent_decoders == rhs.max_concurrent_decoders;
}

void print_live_stats(const LiveStatsSnapshot &stats) {
  std::cout << "QEC_DECODING_SERVER_STATS dispatched=" << stats.dispatched
            << " max_concurrent_decoders=" << stats.max_concurrent_decoders
            << std::endl;
  if (const char *verbose = std::getenv("QEC_DECODING_SERVER_STATS");
      verbose && verbose[0] != '\0')
    cudaqx_qec_decoding_server_print_stats();
  std::cout.flush();
}

void print_live_stats() { print_live_stats(read_live_stats()); }

void maybe_print_periodic_stats(
    const ServerConfig &cfg,
    std::chrono::steady_clock::time_point &last_stats_time,
    LiveStatsSnapshot &last_stats, std::chrono::steady_clock::time_point now) {
  if (cfg.stats_interval_sec <= 0)
    return;
  if (now - last_stats_time < std::chrono::seconds(cfg.stats_interval_sec))
    return;
  LiveStatsSnapshot current = read_live_stats();
  if (!stats_equal(current, last_stats))
    print_live_stats(current);
  last_stats = current;
  last_stats_time = now;
}

struct ConfigApplyRequest {
  std::string yaml;
  std::size_t num_decoders = 0;
  std::string type0 = "?";
};

ConfigApplyRequest read_apply_config(const std::string &config_path) {
  std::ifstream config_file(config_path);
  if (!config_file)
    throw std::runtime_error("cannot open " + config_path);
  std::stringstream config_text;
  config_text << config_file.rdbuf();

  ConfigApplyRequest out;
  out.yaml = decoder_yaml_from_daemon_yaml(config_text.str());
  auto decoder_config = config::multi_decoder_config::from_yaml_str(out.yaml);
  if (decoder_config.decoders.empty())
    throw std::runtime_error("no decoders parsed from " + config_path);
  out.num_decoders = decoder_config.decoders.size();
  out.type0 = decoder_config.decoders[0].type;
  return out;
}

void print_config_applied(const ConfigApplyRequest &config) {
  std::cout << "QEC_DECODING_SERVER_CONFIG_APPLIED decoders="
            << config.num_decoders << " type0=" << config.type0 << std::endl;
  std::cout.flush();
}

void print_config_failed(int rc, const char *state, const std::string &why) {
  std::cout << "QEC_DECODING_SERVER_CONFIG_FAILED rc=" << rc
            << " state=" << state;
  if (!why.empty())
    std::cout << " reason=" << why;
  std::cout << std::endl;
  std::cout.flush();
}

void handle_config_apply(const std::string &config_path) {
  ConfigApplyRequest config;
  try {
    config = read_apply_config(config_path);
  } catch (const std::exception &e) {
    print_config_failed(/*rc=*/1, "serving_old", e.what());
    return;
  }
  char reason[512] = {0};
  const int rc = cudaqx_qec_decoding_server_apply_config_from_yaml(
      config.yaml.data(), config.yaml.size(), reason, sizeof(reason));
  if (rc == 0) {
    print_config_applied(config);
  } else {
    const char *state = rc == 1   ? "serving_old"
                        : rc == 2 ? "awaiting_config"
                        : rc == 3 ? "busy"
                                  : "unknown";
    print_config_failed(rc, state, reason);
  }
}
#ifdef QEC_HAVE_GPU_ROCE_TRANSPORT
void handle_gpu_roce_config_apply(
    const std::string &config_path,
    std::unique_ptr<cudaq::qec::decoding_server::DecodingServer> &server) {
  auto reject_without_rebind = [&server](const std::exception &e) {
    if (server)
      print_config_failed(/*rc=*/1, "serving_old", e.what());
    else
      print_config_failed(/*rc=*/2, "awaiting_config", e.what());
  };

  ConfigApplyRequest config;
  // Outer checks reject no-state-change errors before touching the live GPU
  // transport; reconfigure_from_yaml_str repeats committed-state checks.
  try {
    config = read_apply_config(config_path);
    auto decoder_config =
        config::multi_decoder_config::from_yaml_str(config.yaml);
    if (decoder_config.decoders.size() != 1)
      throw std::runtime_error(
          "GPU RoCE reconfigure currently requires exactly one decoder");
    const auto &decoder = decoder_config.decoders.front();
    if (decoder.transport != config::DecoderTransport::gpu_roce)
      throw std::runtime_error(
          "GPU RoCE reconfigure requires decoder transport gpu_roce");
    if (server) {
      const int requested_device =
          cudaq::qec::decoding_server::resolve_decode_device(
              decoder.cuda_device_id.value_or(-1));
      if (server->transport_cuda_device() >= 0 &&
          requested_device != server->transport_cuda_device())
        throw std::runtime_error(
            "cannot change cuda_device_id during GPU RoCE live reconfigure");
    }
  } catch (const std::exception &e) {
    reject_without_rebind(e);
    return;
  }

  try {
    if (!server)
      throw std::runtime_error("gpu_roce server is not initialized");
    server->reconfigure_from_yaml_str(config.yaml, config_path);
    print_config_applied(config);
  } catch (const std::exception &e) {
    print_config_failed(/*rc=*/2, "awaiting_config", e.what());
  }
}
#endif // QEC_HAVE_GPU_ROCE_TRANSPORT
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
void publish_endpoint_file(const ServerConfig &cfg, std::uint16_t port) {
  if (cfg.endpoint_file.empty())
    return;
  std::ofstream endpoint(cfg.endpoint_file, std::ios::trunc);
  if (!endpoint) {
    std::cerr << "WARNING: failed to write endpoint file " << cfg.endpoint_file
              << std::endl;
    return;
  }
  const std::string host =
      cfg.transport == "cpu_roce" ? cfg.local_ip : "127.0.0.1";
  endpoint << "QEC_DECODING_SERVER_HOST=" << host << "\n";
  endpoint << "QEC_DECODING_SERVER_PORT=" << port << "\n";
  endpoint << "QEC_DECODING_SERVER_TRANSPORT=" << cfg.transport << "\n";
  endpoint << "QEC_DECODING_SERVER_PID=" << current_process_id() << "\n";
}

void print_ready(const ServerConfig &cfg, std::uint16_t port,
                 const std::string &extra) {
  publish_endpoint_file(cfg, port);
  std::cout << "QEC_DECODING_SERVER_READY port=" << port
            << (extra.empty() ? "" : " ") << extra
            << " config=" << cfg.config_path << std::endl;
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
  print_ready(cfg, cpu_udp_get_port(xcvr), "transport=udp");
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

int accept_until_shutdown(int listen_fd) {
  while (g_shutdown.load(std::memory_order_acquire) == 0) {
    fd_set read_fds;
    FD_ZERO(&read_fds);
    FD_SET(listen_fd, &read_fds);
    timeval timeout{};
    timeout.tv_sec = 0;
    timeout.tv_usec = 100000;
    const int ready =
        ::select(listen_fd + 1, &read_fds, nullptr, nullptr, &timeout);
    if (ready < 0) {
      if (errno == EINTR)
        continue;
      return -1;
    }
    if (ready == 0)
      continue;
    return ::accept(listen_fd, nullptr, nullptr);
  }
  errno = EINTR;
  return -1;
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
  print_ready(cfg, ntohs(srv.sin_port),
              "transport=cpu_roce roce_ip=" + cfg.local_ip);

  const int conn_fd = accept_until_shutdown(listen_fd);
  ::close(listen_fd);
  if (conn_fd < 0) {
    if (g_shutdown.load(std::memory_order_acquire) == 0)
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

  print_ready(cfg, /*port=*/0,
              "transport=cpu_roce qp_config=hsb_fpga peer_ip=" + cfg.peer_ip);
  return true;
}

#endif // QEC_HAVE_CPU_ROCE_TRANSPORT

bool init_transport(const ServerConfig &cfg, TransportEndpoints &tp) {
  if (cfg.transport == "udp")
    return init_udp_transport(cfg, tp);
  if (cfg.transport == "cpu_roce") {
#ifdef QEC_HAVE_CPU_ROCE_TRANSPORT
    if (cfg.qp_config == "hsb_fpga")
      return init_cpu_roce_hsb_fpga_transport(cfg, tp);
    return init_cpu_roce_transport(cfg, tp);
#else
    std::cerr << "ERROR: this server was built without cpu_roce transport "
                 "support (libcudaq-realtime-cpu-roce-transport not found)"
              << std::endl;
    return false;
#endif
  }
  std::cerr << "ERROR: unknown HOST_CALL transport " << cfg.transport
            << " (expected udp or cpu_roce)" << std::endl;
  return false;
}

struct RunningTransport {
  explicit RunningTransport(ServerConfig config) : cfg(std::move(config)) {}

  ServerConfig cfg;
  TransportEndpoints endpoints;
  int dispatcher_shutdown = 0;
  std::uint64_t packets_dispatched = 0;
  cudaq_ringbuffer_t ringbuffer{};
  cudaq_dispatcher_config_t dispatch_config{};
  cudaq_function_table_t function_table{};
  std::thread supervisor_thread;
  std::thread dispatcher_thread;
  std::atomic<bool> dispatcher_started{false};
  std::atomic<bool> failed{false};
};

void stop_running_transport(RunningTransport &rt) {
  __atomic_store_n(&rt.dispatcher_shutdown, 1, __ATOMIC_RELEASE);
  __sync_synchronize();
  if (rt.dispatcher_thread.joinable())
    rt.dispatcher_thread.join();
  if (rt.endpoints.shutdown) {
    rt.endpoints.shutdown();
    rt.endpoints.shutdown = nullptr;
  }
}

void run_transport_supervisor(std::shared_ptr<RunningTransport> rt,
                              cudaq_function_table_t function_table) {
  if (!init_transport(rt->cfg, rt->endpoints)) {
    rt->failed.store(true, std::memory_order_release);
    return;
  }

  rt->ringbuffer.rx_flags_host = rt->endpoints.rx_flags;
  rt->ringbuffer.tx_flags_host = rt->endpoints.tx_flags;
  rt->ringbuffer.rx_data_host = rt->endpoints.rx_data;
  rt->ringbuffer.tx_data_host = rt->endpoints.tx_data;
  rt->ringbuffer.rx_stride_sz = rt->cfg.slot_size;
  rt->ringbuffer.tx_stride_sz = rt->cfg.slot_size;

  rt->dispatch_config.num_slots = rt->cfg.num_slots;
  rt->dispatch_config.slot_size = static_cast<std::uint32_t>(rt->cfg.slot_size);
  rt->dispatch_config.dispatch_path = CUDAQ_DISPATCH_PATH_HOST;
  rt->dispatch_config.dispatch_mode = CUDAQ_DISPATCH_HOST_CALL;
  rt->dispatch_config.skip_tx_markers = 1;
  rt->function_table = function_table;

  rt->dispatcher_thread = std::thread([rt] {
    cudaq_host_ring_dispatch_loop(
        &rt->ringbuffer, &rt->function_table, &rt->dispatch_config,
        /*engine=*/nullptr, &rt->dispatcher_shutdown, &rt->packets_dispatched);
  });
  rt->dispatcher_started.store(true, std::memory_order_release);

  while (g_shutdown.load(std::memory_order_acquire) == 0)
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

  stop_running_transport(*rt);
}

std::vector<std::shared_ptr<RunningTransport>>
start_transport_supervisors(const std::vector<ServerConfig> &configs,
                            cudaq_function_table_t function_table) {
  std::vector<std::shared_ptr<RunningTransport>> transports;
  transports.reserve(configs.size());
  for (const auto &cfg : configs) {
    auto rt = std::make_shared<RunningTransport>(cfg);
    rt->supervisor_thread = std::thread(
        [rt, function_table] { run_transport_supervisor(rt, function_table); });
    transports.push_back(std::move(rt));
  }
  return transports;
}

void stop_transport_supervisors(
    std::vector<std::shared_ptr<RunningTransport>> &transports) {
  g_shutdown.store(1, std::memory_order_release);
  for (auto &rt : transports)
    if (rt->supervisor_thread.joinable())
      rt->supervisor_thread.join();
}
} // namespace

int main(int argc, char **argv) {
  ServerConfig cfg;
  if (!parse_args(argc, argv, cfg))
    return 1;

  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);
  std::signal(SIGHUP, on_sighup);
#ifdef SIGUSR1
  std::signal(SIGUSR1, on_stats_signal);
#endif

  // [1] Validate decoder YAML, resolve daemon listeners, and hand a
  // decoder-only config path to the decoding-server service. A top-level
  // `server:` block belongs to this daemon and is stripped before existing
  // decoder parsers see the YAML.
  std::ifstream config_file(cfg.config_path);
  if (!config_file) {
    std::cerr << "ERROR: cannot open config file " << cfg.config_path
              << std::endl;
    return 1;
  }
  std::stringstream config_text;
  config_text << config_file.rdbuf();
  const std::string raw_config_yaml = config_text.str();

  std::vector<ServerConfig> transport_configs;
  config::multi_decoder_config decoder_config;
  std::string decoder_yaml;
  std::string runtime_decoder_config_path;
  TempFileCleanup runtime_decoder_config_cleanup;
  try {
    transport_configs = resolve_transport_configs(cfg, raw_config_yaml);
    decoder_yaml = decoder_yaml_from_daemon_yaml(raw_config_yaml);
    decoder_config = config::multi_decoder_config::from_yaml_str(decoder_yaml);
    runtime_decoder_config_path =
        materialize_decoder_yaml(cfg, raw_config_yaml, decoder_yaml);
    runtime_decoder_config_cleanup.path = runtime_decoder_config_path;
    runtime_decoder_config_cleanup.enabled =
        runtime_decoder_config_path != cfg.config_path;
  } catch (const std::exception &e) {
    std::cerr << "ERROR: invalid decoding_server config: " << e.what()
              << std::endl;
    return 1;
  }
  if (decoder_config.decoders.empty()) {
    std::cerr << "ERROR: no decoders parsed from " << cfg.config_path
              << std::endl;
    return 1;
  }
  ::setenv("CUDAQ_QEC_DECODER_CONFIG", runtime_decoder_config_path.c_str(),
           /*overwrite=*/1);
  std::cout << "Configured " << decoder_config.decoders.size()
            << " decoder(s); decoder 0 type: "
            << decoder_config.decoders[0].type
            << "; transport: " << transport_summary(transport_configs)
            << std::endl;
  // [2a] GPU RoCE takes a completely different path: bypass the CQR
  // DeviceCallService / HOST_CALL dispatcher and use DecodingServer directly.
  // Must be checked before force-linking the CQR plugin (which creates a
  // DecodingServer internally for the HOST_CALL path) to avoid double-init.
#ifdef QEC_HAVE_GPU_ROCE_TRANSPORT
  const bool gpu_roce_requested = transport_configs.size() == 1 &&
                                  transport_configs[0].transport == "gpu_roce";
  if (gpu_roce_requested) {
    // DecodingServer(config_yaml) reads the YAML, creates GpuRoceTransceiver
    // (Hololink Sensor Bridge + DOCA), loads decoder sessions, and calls
    // launch_scheduler() to wire the CUDAQ device-graph scheduler to the
    // Hololink ring buffers.  The GPU scheduler then handles
    // RX->dispatch->decode->TX autonomously; this thread waits for signal and
    // config-apply requests.
    //
    // Construction throws when the GPU RoCE component is not linked into
    // this binary (built against HSB/DOCA headers but without the
    // proprietary cudevice archive) or when Hololink bring-up fails.
    try {
      auto server = cudaq::qec::decoding_server::DecodingServer::from_yaml_str(
          decoder_yaml, cfg.config_path);
      // QP/rkey/buf already printed to stdout by launch_scheduler() so the
      // orchestration script can grep them before the READY line.
      std::cout << "QEC_DECODING_SERVER_READY gpu_roce config="
                << cfg.config_path << std::endl;
      std::cout.flush();
      std::thread server_thread([&server] { server->run(); });
      const auto start_time_gr = std::chrono::steady_clock::now();
      auto last_stats_time = start_time_gr;
      auto last_stats = read_live_stats();
      while (g_shutdown.load(std::memory_order_acquire) == 0) {
        const auto now = std::chrono::steady_clock::now();
        if (g_apply_config_requested.exchange(0, std::memory_order_acq_rel) !=
            0)
          handle_gpu_roce_config_apply(cfg.config_path, server);
        if (g_stats_requested.exchange(0, std::memory_order_acq_rel) != 0) {
          last_stats = read_live_stats();
          print_live_stats(last_stats);
          last_stats_time = now;
        }
        maybe_print_periodic_stats(cfg, last_stats_time, last_stats, now);
        if (cfg.timeout_sec > 0) {
          const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                                   now - start_time_gr)
                                   .count();
          if (elapsed > cfg.timeout_sec)
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }
      if (server)
        server->stop();
      if (server_thread.joinable())
        server_thread.join();
    } catch (const std::exception &e) {
      std::cerr << "ERROR: gpu_roce startup failed: " << e.what() << std::endl;
      return 1;
    }
    return 0;
  }
#endif
#ifndef QEC_HAVE_GPU_ROCE_TRANSPORT
  if (transport_configs.size() == 1 &&
      transport_configs[0].transport == "gpu_roce") {
    std::cerr << "ERROR: this server was built without gpu_roce transport "
                 "support (rebuild with HOLOSCAN_SENSOR_BRIDGE_BUILD_DIR, "
                 "DOCA, and CUDA)"
              << std::endl;
    return 1;
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

  // [3] Start every configured HOST_CALL listener. Each listener owns one
  // CUDAQ realtime ring dispatcher, but all dispatch into the same QEC
  // DeviceCallService/DecodingServer instance created above.
  cudaq_function_table_t function_table{};
  function_table.entries = table.entries;
  function_table.count = table.count;
  auto running_transports =
      start_transport_supervisors(transport_configs, function_table);

  // [4] Run until signalled or timed out; SIGHUP applies the latest decoder
  // config without rebinding process-lifetime transport listeners.
  const auto start_time = std::chrono::steady_clock::now();
  auto last_stats_time = start_time;
  auto last_stats = read_live_stats();
  bool all_transports_failed = false;
  while (g_shutdown.load(std::memory_order_acquire) == 0) {
    const auto now = std::chrono::steady_clock::now();
    if (g_apply_config_requested.exchange(0, std::memory_order_acq_rel) != 0)
      handle_config_apply(cfg.config_path);
    if (g_stats_requested.exchange(0, std::memory_order_acq_rel) != 0) {
      last_stats = read_live_stats();
      print_live_stats(last_stats);
      last_stats_time = now;
    }
    maybe_print_periodic_stats(cfg, last_stats_time, last_stats, now);

    bool all_failed = !running_transports.empty();
    for (const auto &rt : running_transports)
      all_failed = all_failed && rt->failed.load(std::memory_order_acquire);
    if (all_failed) {
      std::cerr << "ERROR: all requested HOST_CALL transports failed to start"
                << std::endl;
      all_transports_failed = true;
      break;
    }

    if (cfg.timeout_sec > 0) {
      const auto elapsed =
          std::chrono::duration_cast<std::chrono::seconds>(now - start_time)
              .count();
      if (elapsed > cfg.timeout_sec)
        break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  // [5] Orderly shutdown for all listener supervisors. Each supervisor stops
  // its dispatcher loop and tears down its transport before returning.
  stop_transport_supervisors(running_transports);

  // The counters are atomics and the per-shot get_corrections cadence means
  // they are settled by the time a client-driven run reaches shutdown; print
  // before cudaqx_qec_decoding_server_shutdown() releases the sessions.
  if (const char *stats = std::getenv("QEC_DECODING_SERVER_STATS");
      stats && stats[0] != '\0')
    cudaqx_qec_decoding_server_print_stats();
  // Stop the DecodingServer receive loop and join its thread before the
  // process exits (a still-joinable static thread would std::terminate).
  cudaqx_qec_decoding_server_shutdown();

  std::cout << "QEC_DECODING_SERVER_DISPATCHED count="
            << cudaqx_qec_device_call_dispatch_count() << std::endl;
  // Concurrency evidence for multi-logical-qubit tests: high-water mark of
  // simultaneously-busy DecodingSession workers.
  std::cout << "QEC_DECODING_SERVER_MAX_CONCURRENT_DECODERS count="
            << cudaqx_qec_decoding_server_max_concurrent() << std::endl;
  return all_transports_failed ? 1 : 0;
}
