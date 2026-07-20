/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "ITransceiver.h"
#include "RpcDispatcher.h"
#include "SessionRegistry.h"
#include "cudaq/qec/realtime/decoding_config.h"

#include <atomic>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace cudaq::qec::decoding_server {

/// Resolve the CUDA device a decode pipeline runs on from the decoder's
/// cuda_device_id (-1 when unpinned). An unpinned decoder defaults to device 0.
int resolve_decode_device(int decoder_pin);

/// Maps function_id → non-owning ITransceiver pointer.
/// Ownership lives in DecodingServer::owned_transports_.
using TransportMap = std::unordered_map<uint32_t, ITransceiver *>;

/// Top-level server: owns the registry and dispatcher, holds the
/// transceiver(s), and runs the blocking receive loop.
class DecodingServer {
public:
  /// Config-driven constructor: reads the transport type from \p config_yaml
  /// and creates the appropriate transceiver.  Requires CUDAQ_REALTIME for
  /// RoCE transports; throws std::runtime_error if the adapters are not
  /// available.  Use the explicit-transceiver constructors for testing with
  /// LoopbackTransceiver.
  explicit DecodingServer(const std::string &config_yaml);

  /// Single-transceiver constructor: all three RPCs share one transport.
  DecodingServer(std::unique_ptr<ITransceiver> transport,
                 const std::string &config_yaml);

  /// Single-transceiver constructor from an already-parsed config -- the
  /// in-process path where the application handed the config to
  /// configure_decoders() rather than pointing at a YAML file.
  DecodingServer(
      std::unique_ptr<ITransceiver> transport,
      const cudaq::qec::decoding::config::multi_decoder_config &config);

  /// Split-transport constructor: each function_id dispatched to its own
  /// transceiver.  \p owned is moved in; \p function_transport holds raw
  /// pointers into \p owned.
  DecodingServer(std::vector<std::unique_ptr<ITransceiver>> owned,
                 TransportMap function_transport,
                 const std::string &config_yaml);

  /// Stops the transports and joins all session workers before any member is
  /// destroyed: workers drain queued items that reply through raw
  /// ITransceiver pointers into owned_transports_, so they must finish while
  /// the transports are still alive.
  ~DecodingServer();

  /// Block until stop() is called.
  void run();

  /// Thread-safe; signals the receive loop to exit after the current frame.
  void stop();

  /// Print one QEC_DECODING_SERVER_DECODER_STATS line per session to stdout
  /// (test/diagnostic evidence; callers gate on QEC_DECODING_SERVER_STATS).
  void print_session_stats() const;

private:
  void init(const std::string &config_yaml);
  void register_handlers();

  /// Create a transceiver for \p transport_type.  Throws for RoCE transports
  /// until CpuRoceTransceiverAdapter / GpuRoceTransceiverAdapter are
  /// available via CUDAQ_REALTIME.
  static std::unique_ptr<ITransceiver>
  make_transport(cudaq::qec::decoding::config::DecoderTransport transport_type,
                 int pinned_cuda_device);

  // Destruction order matters: the GPU RoCE scheduler (inside
  // owned_transports_) holds a cudaGraphExec_t captured from a session's
  // decoder.  The scheduler must be destroyed (cudaStreamSynchronize +
  // cudaq_destroy_dispatch_graph) before registry_ releases the decoder and its
  // graph resources. C++ destroys members in reverse declaration order, so
  // registry_ must be declared BEFORE owned_transports_.
  SessionRegistry registry_;
  RpcDispatcher dispatcher_;
  std::atomic<bool> shutdown_{false};
  /// Maps function_id → transceiver; used to deduplicate receiver threads.
  /// Routing within the server is by function_id, not by decoder_id.
  TransportMap function_transport_;
  std::vector<std::unique_ptr<ITransceiver>> owned_transports_;
};

} // namespace cudaq::qec::decoding_server
