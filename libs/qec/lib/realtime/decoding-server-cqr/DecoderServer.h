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

namespace cudaq::qec::decoder_server {

/// Maps function_id → non-owning ITransceiver pointer.
/// Ownership lives in DecoderServer::owned_transports_.
using TransportMap = std::unordered_map<uint32_t, ITransceiver *>;

/// Top-level server: owns the registry and dispatcher, holds the
/// transceiver(s), and runs the blocking receive loop.
class DecoderServer {
public:
  /// Config-driven constructor: reads the transport type from \p config_yaml
  /// and creates the appropriate transceiver.  Requires CUDAQ_REALTIME for
  /// RoCE transports; throws std::runtime_error if the adapters are not
  /// available.  Use the explicit-transceiver constructors for testing with
  /// LoopbackTransceiver.
  explicit DecoderServer(const std::string &config_yaml);

  /// Single-transceiver constructor: all three RPCs share one transport.
  DecoderServer(std::unique_ptr<ITransceiver> transport,
                const std::string &config_yaml);

  /// Single-transceiver constructor from an already-parsed config -- the
  /// in-process path where the application handed the config to
  /// configure_decoders() rather than pointing at a YAML file.
  DecoderServer(
      std::unique_ptr<ITransceiver> transport,
      const cudaq::qec::decoding::config::multi_decoder_config &config);

  /// Split-transport constructor: each function_id dispatched to its own
  /// transceiver.  \p owned is moved in; \p function_transport holds raw
  /// pointers into \p owned.
  DecoderServer(std::vector<std::unique_ptr<ITransceiver>> owned,
                TransportMap function_transport,
                const std::string &config_yaml);

  /// Block until stop() is called.
  void run();

  /// Thread-safe; signals the receive loop to exit after the current frame.
  void stop();

private:
  void init(const std::string &config_yaml);
  void register_handlers();

  /// Create a transceiver for \p transport_type.  Throws for RoCE transports
  /// until CpuRoceTransceiverAdapter / GpuRoceTransceiverAdapter are
  /// available via CUDAQ_REALTIME.
  static std::unique_ptr<ITransceiver>
  make_transport(cudaq::qec::decoding::config::DecoderTransport transport_type);

  std::vector<std::unique_ptr<ITransceiver>> owned_transports_;
  /// Maps function_id → transceiver; used to deduplicate receiver threads.
  /// Routing within the server is by function_id, not by decoder_id.
  TransportMap function_transport_;
  SessionRegistry registry_;
  RpcDispatcher dispatcher_;
  std::atomic<bool> shutdown_{false};
};

} // namespace cudaq::qec::decoder_server
