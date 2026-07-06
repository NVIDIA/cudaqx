/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "DecoderSession.h"

#include <memory>
#include <string>
#include <unordered_map>

namespace cudaq::qec::decoder_server {

/// Owns all DecoderSession instances, keyed by uint64_t decoder_id.
///
/// Populated eagerly at startup from the YAML config.  The map is read-only
/// after load_from_config() returns, so no locking is required at runtime.
class SessionRegistry {
public:
  /// Parse \p yaml_path and construct one DecoderSession per decoder entry.
  /// @throws std::runtime_error on duplicate id, missing required fields, or
  /// decoder init failure.
  void load_from_config(const std::string &yaml_path);

  DecoderSession &get(uint64_t decoder_id);
  const DecoderSession &get(uint64_t decoder_id) const;

  const std::unordered_map<uint64_t, std::unique_ptr<DecoderSession>> &
  sessions() const {
    return sessions_;
  }

private:
  std::unordered_map<uint64_t, std::unique_ptr<DecoderSession>> sessions_;
};

} // namespace cudaq::qec::decoder_server
