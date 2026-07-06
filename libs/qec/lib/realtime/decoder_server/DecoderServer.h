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
  /// Single-transceiver constructor: all three RPCs share one transport.
  explicit DecoderServer(std::unique_ptr<ITransceiver> transport,
                         const std::string &config_yaml);

  /// Split-transport constructor: each function_id dispatched to its own
  /// transceiver.  \p owned is moved in; \p dispatch_map holds raw pointers
  /// into \p owned.
  DecoderServer(std::vector<std::unique_ptr<ITransceiver>> owned,
                TransportMap dispatch_map, const std::string &config_yaml);

  /// Block until stop() is called.
  void run();

  /// Thread-safe; signals the receive loop to exit after the current frame.
  void stop();

private:
  void init(const std::string &config_yaml);

  std::vector<std::unique_ptr<ITransceiver>> owned_transports_;
  TransportMap dispatch_map_;
  SessionRegistry registry_;
  RpcDispatcher dispatcher_;
  std::atomic<bool> shutdown_{false};
};

} // namespace cudaq::qec::decoder_server
