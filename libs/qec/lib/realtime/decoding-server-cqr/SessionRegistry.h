/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "DecoderSession.h"
#include "cudaq/qec/realtime/decoding_config.h"

#include <memory>
#include <string>
#include <unordered_map>

namespace cudaq::qec::decoder_server {

using cudaq::qec::decoding::config::DecoderTransport;

/// Owns all DecoderSession instances, keyed by uint64_t decoder_id.
///
/// Populated eagerly at startup from the YAML config.  The map is read-only
/// after load_from_config() returns, so no locking is required at runtime.
class SessionRegistry {
public:
  /// Parse \p yaml_path and construct one DecoderSession per decoder entry.
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

  DecoderSession &get(uint64_t decoder_id);
  const DecoderSession &get(uint64_t decoder_id) const;

  /// Transport type shared by all sessions; valid after load_from_config().
  DecoderTransport required_transport() const { return transport_; }

  const std::unordered_map<uint64_t, std::unique_ptr<DecoderSession>> &
  sessions() const {
    return sessions_;
  }

private:
  std::unordered_map<uint64_t, std::unique_ptr<DecoderSession>> sessions_;
  DecoderTransport transport_{DecoderTransport::cpu_roce};
};

} // namespace cudaq::qec::decoder_server
