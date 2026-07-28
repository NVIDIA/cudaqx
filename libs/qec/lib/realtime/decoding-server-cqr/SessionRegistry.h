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
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace cudaq::qec::decoding_server {

using cudaq::qec::decoding::config::DecoderDispatch;

enum class ConfigApplyState {
  applied,
  unchanged,
  rejected,
  awaiting_config,
  busy,
};

struct ConfigApplyResult {
  ConfigApplyState state;
  std::string message;
};

class SessionNotReady : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

/// Owns all DecodingSession instances, keyed by uint64_t decoder_id.
///
/// Populated eagerly at startup. Live config apply is serialized by
/// DecodingServer's lifecycle lock, so request handlers never access this map
/// while host sessions are being replaced.
class SessionRegistry {
public:
  /// Parse \p yaml_path and construct one DecodingSession per decoder entry.
  /// Decoder entries may mix dispatch shapes (host and device_graph).
  /// @throws std::runtime_error on duplicate id, missing required fields, or
  /// decoder init failure.
  void load_from_config(const std::string &yaml_path);

  /// Same, from an already-parsed config (the in-process application path,
  /// where the config was handed to configure_decoders rather than a file).
  /// \p source_name is used in error messages only.
  void load_from_config(
      const cudaq::qec::decoding::config::multi_decoder_config &config,
      const std::string &source_name);

  /// Apply decoder-only changes while preserving the process-owned ring
  /// topology. Decoder ids, dispatch shapes, and transport configuration are
  /// immutable. device_graph entries are also immutable because their
  /// captured graph resources are bound to live ring consumers outside this
  /// registry. Host sessions are drained, destroyed, and reconstructed.
  ///
  /// Static/topology rejection leaves the old sessions serving. A host
  /// decoder construction failure occurs after the old host resources have
  /// been released and leaves those ids unavailable until a later successful
  /// apply.
  ConfigApplyResult
  apply_config(const cudaq::qec::decoding::config::multi_decoder_config &config,
               const std::string &source_name);
  DecodingSession &get(uint64_t decoder_id);
  const DecodingSession &get(uint64_t decoder_id) const;

  /// Dispatch shape shared by ALL sessions.  Valid after load_from_config();
  /// throws when the config mixes shapes (mixed configs are composed by the
  /// decoding_server process, which binds a consumer per decoder -- the
  /// single-transceiver DecodingServer paths require a uniform shape).
  DecoderDispatch required_dispatch() const {
    if (mixed_)
      throw std::runtime_error(
          "config mixes host and device_graph dispatch; a single-transceiver "
          "DecodingServer cannot serve it (the decoding_server process "
          "composes per-decoder consumers instead)");
    return dispatch_;
  }

  /// Dispatch shape of one decoder (valid after load_from_config()).
  DecoderDispatch dispatch_for(uint64_t decoder_id) const {
    return dispatch_by_id_.at(decoder_id);
  }

  bool mixed_dispatch() const { return mixed_; }

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
  static std::unique_ptr<DecodingSession>
  make_session(const cudaq::qec::decoding::config::decoder_config &config);

  std::unordered_map<uint64_t, std::unique_ptr<DecodingSession>> sessions_;
  std::unordered_map<uint64_t, DecoderDispatch> dispatch_by_id_;
  std::unordered_set<uint64_t> unavailable_ids_;
  cudaq::qec::decoding::config::multi_decoder_config active_config_;
  bool loaded_ = false;
  DecoderDispatch dispatch_{DecoderDispatch::host};
  bool mixed_ = false;
};

} // namespace cudaq::qec::decoding_server
