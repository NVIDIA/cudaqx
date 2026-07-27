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
/// Both the decoder and the transport are configuration, not code:
///   - decoders come from `--config=<yaml>`
///     (multi_decoder_config::from_yaml_str);
///   - the transport comes from `--transport=<name|/path/to/lib.so>`: a CUDA-Q
///     realtime bridge PROVIDER, loaded at runtime through the transport-
///     provider interface (bridge_interface.h).  A bare name resolves to
///     `libcudaq-realtime-bridge-<name>.so` next to the CUDA-Q realtime
///     libraries (udp and cpu_roce ship there); a value containing '/' is
///     loaded verbatim, which is how a partner drops in an out-of-tree
///     transport library with NO changes to this server.
///
/// This server contains no transport-specific code: it forwards all
/// unrecognized command-line arguments to the provider's create() (e.g.
/// --port/--num-slots/--slot-size for udp; --device/--local-ip/--qp_config/
/// --peer-ip/--remote-qp/--frame-size for cpu_roce), derives the dispatcher
/// geometry from the provider's ring-geometry query, publishes readiness from
/// the provider's endpoint-info query, and drives the libcudaq-realtime
/// dispatcher object (HOST path, HOST_CALL table) over the provider's rings.
///
/// The function table comes from the decoding-server-cqr service plugin
/// (enqueue_syndromes / get_corrections / reset_decoder) regardless of
/// transport or decoder.
///
/// Prints `QEC_DECODING_SERVER_READY port=<P0> ...` on stdout once the caller
/// can start connecting. The rest of the line is the provider's endpoint
/// description (e.g. `transport=udp` or `transport=cpu_roce roce_ip=<IP>`)
/// followed by one `ring<id>=<port>` token per decoder ring, so a caller can
/// route `device_call(decoder_id, ...)` to the right endpoint. At shutdown
/// each ring reports its traffic as `QEC_DECODING_SERVER_RING decoder=<id>
/// dispatched=<n>` (the two-process stand-in for the in-process
/// cudaqx_qec_device_call_dispatch_count() assertion). The standalone
/// all-device_graph path instead prints only `QEC_DECODING_SERVER_READY
/// device_graph` and reports execution via the trigger diagnostics.
///
/// Usage:
///   decoding_server --config=<decoders.yaml>
///                   [--transport=<name|/path/to/lib.so>] [--timeout=60]
///                   [provider args, forwarded verbatim...]
///
/// NOTE: --slot-size must match the caller channel's slot size (each frame
/// occupies one full slot stride on both wires).
///
/// The dispatch SHAPE is a per-decoder property of the YAML (`dispatch:
/// host|device_graph`): host decoders run through the CQR HOST_CALL
/// dispatcher below; a device_graph decoder routes the whole server through
/// the CQR DecodingServer, whose DeviceGraphTransceiver runs the
/// self-relaunching GPU scheduler over the same kind of runtime-loaded
/// provider (the hololink library by default; the YAML transport section or
/// the --transport fallback selects another).

#include "cudaq/qec/realtime/decoding_config.h"

// Ring-consumer C ABI prototypes (weak references below are checked against
// these at compile time).
#include "../../lib/realtime/decoding-server-cqr/DeviceGraphRingConsumer.h"

#include "cudaq/realtime/device_call_service.h"

#include "cudaq/realtime/daemon/bridge/bridge_interface.h"
#include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"

#ifdef QEC_HAVE_DEVICE_GRAPH_DISPATCH
// DecodingServer.h (and DeviceGraphTransceiver.h via DecodingServer.cpp) live
// in the decoding-server-cqr directory, added to include paths by CMakeLists
// when CUDAQ_QEC_DEVICE_GRAPH_AVAILABLE is true.
#include "DecodingServer.h"
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

extern "C" void cudaqx_qec_realtime_device_call_service_force_link();
// Opaque graph resources of a decoder hosted by the CQR plugin's registry.
extern "C" void *cudaqx_qec_decoding_server_graph_resources(std::uint64_t);
// Device-graph ring-consumer C API (strong definitions live in the
// cudaq-qec-decoding-server-device-graph component; referenced WEAKLY so a
// build without the component still links and fails at runtime instead).
// DeviceGraphRingConsumer.h supplies the canonical prototypes; these
// redeclarations only add the weak attribute and must match it exactly.
extern "C" __attribute__((weak)) void *
cudaqx_qec_make_device_graph_ring_consumer(const void *ring,
                                           std::size_t num_slots,
                                           std::size_t slot_size, int gpu_id,
                                           void *graph_resources);
extern "C" __attribute__((weak)) void
cudaqx_qec_device_graph_ring_consumer_shutdown(void *consumer);
extern "C" __attribute__((weak)) std::uint64_t
cudaqx_qec_device_graph_ring_consumer_dispatched(void *consumer);
extern "C" __attribute__((weak)) void
cudaqx_qec_device_graph_ring_consumer_destroy(void *consumer);
extern "C" std::uint64_t cudaqx_qec_device_call_dispatch_count();
extern "C" std::uint64_t cudaqx_qec_decoding_server_max_concurrent();
extern "C" void cudaqx_qec_decoding_server_print_stats();
extern "C" void cudaqx_qec_decoding_server_shutdown();

namespace {

namespace config = cudaq::qec::decoding::config;

struct ServerConfig {
  std::string config_path;
  std::string transport = "udp";
  bool transport_from_cli = false;
  int timeout_sec = 60;
  // Everything the server itself does not consume, forwarded verbatim to the
  // provider's create() (providers ignore arguments they don't recognize, so
  // one command line can carry any provider's options).
  std::vector<std::string> provider_args;
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
                   "[--transport=<name|/path/to/lib.so>] [--timeout=N] "
                   "[provider args, forwarded verbatim: e.g. --port=N "
                   "--num-slots=N --slot-size=N --device=NAME "
                   "--local-ip=ADDR --qp_config=rendezvous|hsb_fpga "
                   "--peer-ip=ADDR --remote-qp=N --frame-size=N]\n"
                   "--transport applies only when the YAML transport "
                   "section names no provider (a conflict is a startup "
                   "error).\n"
                   "Providers and their args are defined by the installed "
                   "cudaq-realtime (libcudaq-realtime-bridge-<name>.so); "
                   "the names and args above are examples, not an "
                   "exhaustive list -- the installation is the source of "
                   "truth."
                << std::endl;
      return false;
    } else if (starts_with(a, "--config="))
      cfg.config_path = a.substr(9);
    else if (starts_with(a, "--transport=")) {
      cfg.transport = a.substr(12);
      cfg.transport_from_cli = true;
    } else if (starts_with(a, "--timeout=")) {
      try {
        cfg.timeout_sec = std::stoi(a.substr(10));
      } catch (const std::exception &) {
        std::cerr << "ERROR: invalid --timeout value '" << a.substr(10) << "'"
                  << std::endl;
        return false;
      }
    } else
      cfg.provider_args.push_back(a);
  }
  if (cfg.config_path.empty()) {
    std::cerr << "ERROR: --config=<decoders.yaml> is required" << std::endl;
    return false;
  }
  return true;
}

std::atomic<int> g_shutdown{0};
void on_signal(int) { g_shutdown.store(1, std::memory_order_release); }

// Resolve a --transport value to the provider shared library to load:
// anything with a '/' is a caller-supplied library path (the partner
// drop-in); a bare name maps to libcudaq-realtime-bridge-<name>.so next to
// the CUDA-Q realtime libraries (QEC_BRIDGE_PROVIDER_DIR, baked in by
// CMake), falling back to the bare soname for the dynamic loader's regular
// search path.
std::string resolve_provider_lib(const std::string &transport) {
  if (transport.find('/') != std::string::npos)
    return transport;
  const std::string soname = "libcudaq-realtime-bridge-" + transport + ".so";
#ifdef QEC_BRIDGE_PROVIDER_DIR
  const std::string candidate =
      std::string(QEC_BRIDGE_PROVIDER_DIR) + "/" + soname;
  if (std::ifstream(candidate).good())
    return candidate;
#endif
  return soname;
}

// Split a provider endpoint-info line into its port and the remaining
// tokens.
std::uint16_t split_endpoint_info(const std::string &endpoint_info,
                                  std::string &rest) {
  std::uint16_t port = 0;
  std::istringstream in(endpoint_info);
  std::string token;
  while (in >> token) {
    if (starts_with(token, "port=")) {
      try {
        port = static_cast<std::uint16_t>(std::stoul(token.substr(5)));
      } catch (const std::exception &) {
        port = 0; // malformed provider endpoint token; keep the rest
      }
    } else
      rest += (rest.empty() ? "" : " ") + token;
  }
  return port;
}

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
  config::multi_decoder_config decoder_config;
  try {
    decoder_config =
        config::multi_decoder_config::from_yaml_str(config_text.str());
  } catch (const std::exception &e) {
    std::cerr << "ERROR: failed to parse " << cfg.config_path << ": "
              << e.what() << std::endl;
    return 1;
  }
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

  // The wire's identity lives in the YAML transport section; --transport is
  // only a fallback default for configs that intentionally leave the wire
  // unspecified (one YAML reused across wires, selected per launch).  A
  // config that names a provider cannot be contradicted from the command
  // line -- that is a configuration error, not a precedence question.
  const bool yaml_names_provider =
      !decoder_config.transport.provider.empty() ||
      !decoder_config.transport.device_graph.provider.empty();
  if (cfg.transport_from_cli && yaml_names_provider) {
    std::cerr << "ERROR: --transport=" << cfg.transport
              << " conflicts with the transport section in " << cfg.config_path
              << " (the YAML names the provider; drop the CLI flag or remove "
                 "the provider from the YAML)"
              << std::endl;
    return 1;
  }

  // The dispatch SHAPE comes from the decoder config (dispatch:
  // host|device_graph); --transport only ever names the wire.  An
  // all-device_graph config takes the standalone DecodingServer path below
  // ([2a], the HSB flow); any other mix runs the composed per-decoder ring
  // loop, where each decoder's ring gets the consumer its dispatch shape
  // requires (host dispatcher thread, or device-graph scheduler).
  const bool all_device_graph =
      std::all_of(decoder_config.decoders.begin(),
                  decoder_config.decoders.end(), [](const auto &d) {
                    return d.dispatch == config::DecoderDispatch::device_graph;
                  });

  // [2a] device_graph dispatch takes a different shape (device-side
  // scheduler): bypass the CQR DeviceCallService / HOST_CALL dispatcher and
  // use DecodingServer directly.  Must be checked before force-linking the
  // CQR plugin (which creates a DecodingServer internally for the HOST_CALL
  // path) to avoid double-init.
#ifdef QEC_HAVE_DEVICE_GRAPH_DISPATCH
  if (all_device_graph) {
    // DecodingServer(config_yaml) reads the YAML, creates the
    // DeviceGraphTransceiver (which loads a bridge provider: the built-in
    // hololink one, or CUDAQ_REALTIME_BRIDGE_LIB), loads decoder sessions,
    // and calls launch_scheduler() to wire the CUDAQ device-graph scheduler
    // to the provider's GPU rings.  The GPU scheduler then handles
    // RX→dispatch→decode→TX autonomously; this thread just waits for signal.
    //
    // Construction throws when the device-graph component is not linked into
    // this binary (no proprietary cudevice archive) or when provider
    // bring-up fails.
    //
    // Provider resolution for the standalone transceiver mirrors the
    // per-ring loop below: the transport section's device_graph shape
    // override > the section's provider > the --transport CLI fallback >
    // the transceiver's built-in default (hololink).  A YAML that names a
    // provider plus a CLI --transport is rejected before reaching here.
    std::string dg_provider;
    if (!decoder_config.transport.device_graph.provider.empty())
      dg_provider = decoder_config.transport.device_graph.provider;
    else if (!decoder_config.transport.provider.empty())
      dg_provider = decoder_config.transport.provider;
    else if (cfg.transport_from_cli)
      dg_provider = cfg.transport;
    if (!dg_provider.empty())
      ::setenv("CUDAQ_REALTIME_BRIDGE_LIB",
               resolve_provider_lib(dg_provider).c_str(), /*overwrite=*/1);
    // Provider args from the transport section ride the same generic
    // pass-through the per-ring loop uses: section args first, then the
    // device_graph shape override's args.  The transceiver forwards them
    // verbatim after its named knobs (QEC_DEVICE_GRAPH_* env), so a
    // non-HSB provider is configured entirely from the YAML.
    {
      std::string dg_args;
      for (const auto &a : decoder_config.transport.args)
        dg_args += (dg_args.empty() ? "" : " ") + a;
      for (const auto &a : decoder_config.transport.device_graph.args)
        dg_args += (dg_args.empty() ? "" : " ") + a;
      if (!dg_args.empty())
        ::setenv("QEC_DEVICE_GRAPH_PROVIDER_ARGS", dg_args.c_str(),
                 /*overwrite=*/1);
    }
    try {
      cudaq::qec::decoding_server::DecodingServer server(cfg.config_path);
      // QP/rkey/buf already printed to stdout by launch_scheduler() so the
      // orchestration script can grep them before the READY line.
      std::cout << "QEC_DECODING_SERVER_READY device_graph" << std::endl;
      std::cout.flush();
      std::thread server_thread([&server] { server.run(); });
      const auto start_time_gr = std::chrono::steady_clock::now();
      while (g_shutdown.load(std::memory_order_acquire) == 0) {
        const auto elapsed =
            std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - start_time_gr)
                .count();
        if (elapsed > cfg.timeout_sec)
          break;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }
      server.stop();
      server_thread.join();
    } catch (const std::exception &e) {
      std::cerr << "ERROR: device_graph startup failed: " << e.what()
                << std::endl;
      return 1;
    }
    return 0;
  }
#else
  if (all_device_graph) {
    std::cerr << "ERROR: this server was built without device_graph dispatch "
                 "support (CUDA-Q realtime bridge API or "
                 "cudaq-realtime-dispatch not found at build time)"
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
  // lifetime (the dispatcher reads table.entries in place).  Creating it
  // also starts the DecodingServer (decoder construction + one worker thread
  // per decoder) -- before the READY line below, so slow decoder
  // initialization never races the first client request.
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

  // [3] Load the transport provider and bring up ONE RING PER DECODER: each
  // decoder in the YAML gets its own provider instance (own endpoint), its
  // own ring buffer, and its own dispatcher -- the two-process form of the
  // one-ring-per-decoder topology (callers route with device_id ==
  // decoder_id and per-device endpoint args, e.g. udp-port.<id>=<port>).
  //
  // create() takes each transport instance to the point where its endpoint
  // identity is known (bound port / QP), so readiness for ALL rings is
  // published in one line before any connect() -- which, for rendezvous-
  // style transports, BLOCKS until the caller dials in.
  const std::string default_provider = decoder_config.transport.provider.empty()
                                           ? cfg.transport
                                           : decoder_config.transport.provider;

  std::vector<char *> provider_argv;
  provider_argv.reserve(cfg.provider_args.size());
  for (auto &a : cfg.provider_args)
    provider_argv.push_back(a.data());

  struct DecoderRing {
    std::int64_t decoder_id = 0;
    bool device_graph = false;
    cudaq_realtime_bridge_handle_t bridge = nullptr;
    std::uint32_t num_slots = 0;
    std::uint32_t slot_size = 0;
    std::uint16_t port = 0;
    std::string endpoint_rest; // endpoint info minus the port token
    int shutdown_flag = 0;
    std::uint64_t dispatched = 0;
    cudaq_dispatcher_t *dispatcher = nullptr; // host consumer
    void *dg_consumer = nullptr;              // device-graph consumer
  };
  // Sized once up front: set_control hands the dispatcher pointers into
  // these elements, so their addresses must not move.
  std::vector<DecoderRing> rings(decoder_config.decoders.size());

  cudaq_dispatch_manager_t *manager = nullptr;
  const auto teardown_rings = [&]() {
    for (auto &ring : rings) {
      if (ring.dispatcher) {
        cudaq_dispatcher_stop(ring.dispatcher);
        cudaq_dispatcher_destroy(ring.dispatcher);
        ring.dispatcher = nullptr;
      }
      if (ring.dg_consumer) {
        // Consumer before bridge: the scheduler polls the provider's rings.
        cudaqx_qec_device_graph_ring_consumer_shutdown(ring.dg_consumer);
        ring.dispatched =
            cudaqx_qec_device_graph_ring_consumer_dispatched(ring.dg_consumer);
        cudaqx_qec_device_graph_ring_consumer_destroy(ring.dg_consumer);
        ring.dg_consumer = nullptr;
      }
      if (ring.bridge) {
        cudaq_bridge_disconnect(ring.bridge);
        cudaq_bridge_destroy(ring.bridge);
        ring.bridge = nullptr;
      }
    }
    if (manager) {
      cudaq_dispatch_manager_destroy(manager);
      manager = nullptr;
    }
  };

  for (std::size_t i = 0; i < rings.size(); ++i) {
    auto &ring = rings[i];
    const auto &dc = decoder_config.decoders[i];
    ring.decoder_id = dc.id;
    ring.device_graph = (dc.dispatch == config::DecoderDispatch::device_graph);

    // The wire is deployment config, resolved from the YAML's top-level
    // `transport:` section (never from decoder entries).  Per-ring
    // resolution: the section's dispatch-shape override (device_graph
    // rings) > the section's provider/args > the --transport CLI fallback
    // (which only applies when the YAML names no provider -- a conflict is
    // rejected at startup above).
    // Every provider name/path resolves the same way --transport does; the
    // bridge loader caches libraries per name, so different rings may load
    // different provider libraries in one process.
    const auto &transport_section = decoder_config.transport;
    std::string ring_provider_name = default_provider;
    std::vector<std::string> ring_extra_args = transport_section.args;
    if (ring.device_graph) {
      if (!transport_section.device_graph.provider.empty())
        ring_provider_name = transport_section.device_graph.provider;
      ring_extra_args.insert(ring_extra_args.end(),
                             transport_section.device_graph.args.begin(),
                             transport_section.device_graph.args.end());
    }
    const std::string ring_lib = resolve_provider_lib(ring_provider_name);
    std::vector<char *> ring_argv = provider_argv;
    for (auto &a : ring_extra_args)
      ring_argv.push_back(a.data());

    if (cudaq_bridge_create_from_library(&ring.bridge, ring_lib.c_str(),
                                         static_cast<int>(ring_argv.size()),
                                         ring_argv.data()) != CUDAQ_OK) {
      std::cerr << "ERROR: failed to load/create transport provider '"
                << ring_lib << "' for decoder " << ring.decoder_id << std::endl;
      teardown_rings();
      return 1;
    }
    // Dispatcher geometry comes from the provider, not from re-parsed CLI.
    if (cudaq_bridge_get_ring_geometry(ring.bridge, &ring.num_slots,
                                       &ring.slot_size) != CUDAQ_OK) {
      std::cerr << "ERROR: transport provider does not report ring geometry "
                   "(bridge interface v2 required)"
                << std::endl;
      teardown_rings();
      return 1;
    }
    char endpoint_info[512] = {0};
    if (cudaq_bridge_get_endpoint_info(ring.bridge, endpoint_info,
                                       sizeof(endpoint_info)) != CUDAQ_OK)
      std::snprintf(endpoint_info, sizeof(endpoint_info), "transport=%s",
                    ring_provider_name.c_str());
    ring.port = split_endpoint_info(endpoint_info, ring.endpoint_rest);
  }

  // One READY line for all rings.  The leading `port=` token belongs to
  // the FIRST decoder listed in the YAML (existing single-ring consumers
  // sscanf it right after the prefix); each
  // ring additionally publishes `ring<decoder_id>=<port>` for per-device
  // endpoint wiring on the caller.
  {
    std::ostringstream ready;
    ready << "QEC_DECODING_SERVER_READY port=" << rings[0].port;
    if (!rings[0].endpoint_rest.empty())
      ready << ' ' << rings[0].endpoint_rest;
    for (const auto &ring : rings)
      ready << " ring" << ring.decoder_id << '=' << ring.port;
    std::cout << ready.str() << std::endl;
    std::cout.flush();
  }

  // [4] Per ring: connect, adopt the ring context, and drive a
  // libcudaq-realtime dispatcher object over it.  The dispatch table is
  // HOST_CALL-only and SHARED by every ring (handlers route by the
  // payload's decoder_id; a decoder's ring simply only ever carries its own
  // id).
  cudaq_function_table_t function_table{};
  function_table.entries = table.entries;
  function_table.count = table.count;

  if (cudaq_dispatch_manager_create(&manager) != CUDAQ_OK) {
    std::cerr << "ERROR: dispatch manager create failed" << std::endl;
    teardown_rings();
    return 1;
  }

  for (auto &ring : rings) {
    if (cudaq_bridge_connect(ring.bridge) != CUDAQ_OK) {
      std::cerr << "ERROR: transport provider connect() failed (decoder "
                << ring.decoder_id << ")" << std::endl;
      teardown_rings();
      return 1;
    }
    cudaq_ringbuffer_t ringbuffer{};
    if (cudaq_bridge_get_transport_context(ring.bridge, RING_BUFFER,
                                           &ringbuffer) != CUDAQ_OK) {
      std::cerr << "ERROR: transport provider has no ring-buffer context "
                   "(decoder "
                << ring.decoder_id << ")" << std::endl;
      teardown_rings();
      return 1;
    }

    if (ring.device_graph) {
      // Attach the device-graph scheduler as this ring's consumer: RX ->
      // dispatch -> decode -> TX runs on the GPU over the provider's
      // (GPU-pollable) rings.  The decoder's captured decode graph comes
      // from the CQR plugin's registry (built at [2], before READY).
      if (!cudaqx_qec_make_device_graph_ring_consumer) {
        std::cerr << "ERROR: decoder " << ring.decoder_id
                  << " requests device_graph dispatch but the device-graph "
                     "component is not linked into this binary (set "
                     "CUDAQ_QEC_REALTIME_CUDEVICE_PROPRIETARY_ARCHIVE)"
                  << std::endl;
        teardown_rings();
        return 1;
      }
      void *graph_resources = cudaqx_qec_decoding_server_graph_resources(
          static_cast<std::uint64_t>(ring.decoder_id));
      if (!graph_resources) {
        std::cerr << "ERROR: decoder " << ring.decoder_id
                  << " requests device_graph dispatch but did not capture a "
                     "decode graph (decoder must support graph dispatch)"
                  << std::endl;
        teardown_rings();
        return 1;
      }
      const int gpu_id = [] {
        const char *value = std::getenv("QEC_DEVICE_GRAPH_GPU_ID");
        return value ? std::atoi(value) : 0;
      }();
      ring.dg_consumer = cudaqx_qec_make_device_graph_ring_consumer(
          &ringbuffer, ring.num_slots, ring.slot_size, gpu_id, graph_resources);
      if (!ring.dg_consumer) {
        std::cerr << "ERROR: device-graph scheduler launch failed (decoder "
                  << ring.decoder_id << "; see log above)" << std::endl;
        teardown_rings();
        return 1;
      }
      if (cudaq_bridge_launch(ring.bridge) != CUDAQ_OK) {
        std::cerr << "ERROR: transport provider launch() failed (decoder "
                  << ring.decoder_id << ")" << std::endl;
        teardown_rings();
        return 1;
      }
      continue;
    }

    cudaq_dispatcher_config_t dispatch_config{};
    dispatch_config.num_slots = ring.num_slots;
    dispatch_config.slot_size = ring.slot_size;
    dispatch_config.dispatch_path = CUDAQ_DISPATCH_PATH_HOST;
    dispatch_config.dispatch_mode = CUDAQ_DISPATCH_HOST_CALL;
    dispatch_config.kernel_type = CUDAQ_KERNEL_REGULAR;
    dispatch_config.skip_tx_markers = 1;

    if (cudaq_dispatcher_create(manager, &dispatch_config, &ring.dispatcher) !=
            CUDAQ_OK ||
        cudaq_dispatcher_set_ringbuffer(ring.dispatcher, &ringbuffer) !=
            CUDAQ_OK ||
        cudaq_dispatcher_set_function_table(ring.dispatcher, &function_table) !=
            CUDAQ_OK ||
        cudaq_dispatcher_set_control(ring.dispatcher, &ring.shutdown_flag,
                                     &ring.dispatched) != CUDAQ_OK ||
        cudaq_dispatcher_start(ring.dispatcher) != CUDAQ_OK) {
      std::cerr << "ERROR: dispatcher bring-up failed (decoder "
                << ring.decoder_id << ")" << std::endl;
      teardown_rings();
      return 1;
    }

    // Start the provider's I/O loop last, once the dispatcher is polling.
    if (cudaq_bridge_launch(ring.bridge) != CUDAQ_OK) {
      std::cerr << "ERROR: transport provider launch() failed (decoder "
                << ring.decoder_id << ")" << std::endl;
      teardown_rings();
      return 1;
    }
  }

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

  // [6] Orderly shutdown: stop each dispatcher (sets its shutdown flag and
  // joins its loop thread) and its transport, then the DecodingServer
  // receive loop (a still-joinable static thread would std::terminate).
  // NOTE: the dispatcher flushes its stats counter when its loop exits, so
  // the per-ring counts are only valid AFTER teardown (dispatcher_stop joins
  // the loop thread).  teardown_rings leaves ring.dispatched intact.
  teardown_rings();
  // The counters are atomics and the per-shot get_corrections cadence means
  // they are settled by the time a client-driven run reaches shutdown; print
  // before cudaqx_qec_decoding_server_shutdown() releases the sessions.
  if (const char *stats = std::getenv("QEC_DECODING_SERVER_STATS");
      stats && stats[0] != '\0')
    cudaqx_qec_decoding_server_print_stats();
  cudaqx_qec_decoding_server_shutdown();

  for (const auto &ring : rings)
    std::cout << "QEC_DECODING_SERVER_RING decoder=" << ring.decoder_id
              << " dispatched=" << ring.dispatched << std::endl;
  std::cout << "QEC_DECODING_SERVER_DISPATCHED count="
            << cudaqx_qec_device_call_dispatch_count() << std::endl;
  // Concurrency evidence for multi-logical-qubit tests: high-water mark of
  // simultaneously-busy DecodingSession workers.
  std::cout << "QEC_DECODING_SERVER_MAX_CONCURRENT_DECODERS count="
            << cudaqx_qec_decoding_server_max_concurrent() << std::endl;
  return 0;
}
