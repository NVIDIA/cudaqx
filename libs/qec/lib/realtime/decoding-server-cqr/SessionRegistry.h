/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "DecodingSession.h"
#include "cudaq/qec/realtime/decoding_config.h"

#include <memory>
#include <string>
#include <unordered_map>

namespace cudaq::qec::decoding_server {

using cudaq::qec::decoding::config::DecoderTransport;

/// Owns all DecodingSession instances, keyed by uint64_t decoder_id.
///
/// Populated eagerly at startup from the YAML config.  The map is read-only
/// after load_from_config() returns, so no locking is required at runtime.
class SessionRegistry {
public:
  /// Parse \p yaml_path and construct one DecodingSession per decoder entry.
  /// All decoder entries must declare the same transport type.
  /// @throws std::runtime_error on duplicate id, mixed transport types,
  /// missing required fields, or decoder init failure.
  void load_from_config(const std::string &yaml_path);

  /// Same, from an already-parsed config (the in-process application path,
  /// where the config was handed to configure_decoders rather than a file).
  /// \p source_name is used in error messages only.
  void load_from_config(
      const cudaq::qec::decoding::config::multi_decoder_config &config,
      const std::string &source_name);

  DecodingSession &get(uint64_t decoder_id);
  const DecodingSession &get(uint64_t decoder_id) const;

  /// Transport type shared by all sessions; valid after load_from_config().
  DecoderTransport required_transport() const { return transport_; }

  const std::unordered_map<uint64_t, std::unique_ptr<DecodingSession>> &
  sessions() const {
    return sessions_;
  }

  /// Stop and join every session's worker thread (each drains its queued
  /// items first).  Must run while the transports the queued items reply
  /// through are still alive; the sessions themselves stay registered so
  /// decoder/graph resources can be torn down later in the required order.
  void stop_workers() {
    for (auto &[id, session] : sessions_)
      session->stop_worker();
  }

private:
  std::unordered_map<uint64_t, std::unique_ptr<DecodingSession>> sessions_;
  DecoderTransport transport_{DecoderTransport::cpu_roce};
};

} // namespace cudaq::qec::decoding_server
