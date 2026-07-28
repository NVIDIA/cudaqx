/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "SessionRegistry.h"
#include "../realtime_decoding.h"
#include "cudaq/qec/logger.h"
#include "cudaq/qec/realtime/decoding_config.h"

#include <fstream>
#include <iterator>
#include <stdexcept>
#include <unordered_map>

namespace cudaq::qec::decoding_server {

using cudaq::qec::decoding::config::decoder_config;
using cudaq::qec::decoding::config::multi_decoder_config;

/// Build the default single-VP pass-through syndrome mapping table.
/// mapping_id=0 -> VP 0 -> empty index list (pass-through)
static SyndromeMappingTable make_default_mapping_table() {
  SyndromeMappingTable table;
  table[0] = {{}};
  return table;
}

static std::unordered_map<uint64_t, const decoder_config *>
index_config(const multi_decoder_config &config,
             const std::string &source_name) {
  if (config.decoders.empty())
    throw std::runtime_error("No decoders in " + source_name);

  config.validate_custom_args();
  std::unordered_map<uint64_t, const decoder_config *> by_id;
  for (const auto &dc : config.decoders) {
    if (dc.id < 0)
      throw std::runtime_error("Negative decoder id " + std::to_string(dc.id) +
                               " in " + source_name);
    const uint64_t id = static_cast<uint64_t>(dc.id);
    if (!by_id.emplace(id, &dc).second)
      throw std::runtime_error("Duplicate decoder id " + std::to_string(dc.id) +
                               " in " + source_name);
  }
  return by_id;
}

std::unique_ptr<DecodingSession>
SessionRegistry::make_session(const decoder_config &config) {
  CUDA_QEC_INFO("SessionRegistry: creating decoder id={} type={}", config.id,
                config.type);
  auto decoder = cudaq::qec::decoding::host::create_realtime_decoder(config);
  auto session =
      DecodingSession::create(std::move(decoder), make_default_mapping_table());
  session->start_worker();
  return session;
}

void SessionRegistry::load_from_config(const std::string &yaml_path) {
  std::ifstream f(yaml_path);
  if (!f.is_open())
    throw std::runtime_error("Cannot open config file: " + yaml_path);

  std::string yaml_str((std::istreambuf_iterator<char>(f)),
                       std::istreambuf_iterator<char>());
  load_from_config(multi_decoder_config::from_yaml_str(yaml_str), yaml_path);
}

void SessionRegistry::load_from_config(const multi_decoder_config &config,
                                       const std::string &source_name) {
  const auto by_id = index_config(config, source_name);
  std::unordered_map<uint64_t, std::unique_ptr<DecodingSession>> next_sessions;
  std::unordered_map<uint64_t, DecoderDispatch> next_dispatch;
  DecoderDispatch first_dispatch = config.decoders.front().dispatch;
  bool next_mixed = false;

  for (const auto &dc : config.decoders) {
    const uint64_t id = static_cast<uint64_t>(dc.id);
    next_dispatch[id] = dc.dispatch;
    next_mixed = next_mixed || dc.dispatch != first_dispatch;
    next_sessions.emplace(id, make_session(dc));
  }

  sessions_ = std::move(next_sessions);
  dispatch_by_id_ = std::move(next_dispatch);
  unavailable_ids_.clear();
  active_config_ = config;
  dispatch_ = first_dispatch;
  mixed_ = next_mixed;
  loaded_ = true;

  CUDA_QEC_INFO("SessionRegistry: loaded {} decoder session(s)", by_id.size());
}

ConfigApplyResult
SessionRegistry::apply_config(const multi_decoder_config &config,
                              const std::string &source_name) {
  if (!loaded_)
    return {ConfigApplyState::rejected,
            "decoder registry has not completed initial configuration"};

  std::unordered_map<uint64_t, const decoder_config *> next_by_id;
  std::unordered_map<uint64_t, const decoder_config *> active_by_id;
  try {
    next_by_id = index_config(config, source_name);
    active_by_id = index_config(active_config_, "active config");
  } catch (const std::exception &e) {
    return {ConfigApplyState::rejected, e.what()};
  }

  if (config.transport != active_config_.transport)
    return {ConfigApplyState::rejected,
            "restart required: transport provider or arguments changed"};
  if (next_by_id.size() != dispatch_by_id_.size())
    return {ConfigApplyState::rejected,
            "restart required: decoder id set changed"};

  bool changed = !unavailable_ids_.empty();
  for (const auto &[id, dispatch] : dispatch_by_id_) {
    const auto next_it = next_by_id.find(id);
    const auto active_it = active_by_id.find(id);
    if (next_it == next_by_id.end() || active_it == active_by_id.end())
      return {ConfigApplyState::rejected,
              "restart required: decoder id set changed"};
    if (next_it->second->dispatch != dispatch)
      return {ConfigApplyState::rejected,
              "restart required: dispatch shape changed for decoder " +
                  std::to_string(id)};
    if (dispatch == DecoderDispatch::device_graph &&
        *next_it->second != *active_it->second)
      return {ConfigApplyState::rejected,
              "restart required: device_graph decoder " + std::to_string(id) +
                  " changed"};
    changed = changed || *next_it->second != *active_it->second;
  }

  if (!changed)
    return {ConfigApplyState::unchanged, "configuration is already active"};

  // Release old host decoder resources before constructing replacements. This
  // avoids a temporary double allocation for large GPU-backed decoders. The
  // caller holds DecodingServer's exclusive lifecycle lock, so no new host RPC
  // can acquire a session while this transition is in progress.
  std::unordered_map<uint64_t, std::unique_ptr<DecodingSession>> retired;
  for (const auto &[id, dispatch] : dispatch_by_id_) {
    if (dispatch != DecoderDispatch::host)
      continue;
    unavailable_ids_.insert(id);
    auto it = sessions_.find(id);
    if (it != sessions_.end()) {
      retired.emplace(id, std::move(it->second));
      sessions_.erase(it);
    }
  }
  for (auto &[id, session] : retired)
    session->stop_worker();
  retired.clear();

  std::unordered_map<uint64_t, std::unique_ptr<DecodingSession>> replacements;
  try {
    for (const auto &dc : config.decoders) {
      if (dc.dispatch != DecoderDispatch::host)
        continue;
      const uint64_t id = static_cast<uint64_t>(dc.id);
      replacements.emplace(id, make_session(dc));
    }
  } catch (const std::exception &e) {
    return {ConfigApplyState::awaiting_config,
            std::string("host decoder construction failed: ") + e.what()};
  } catch (...) {
    return {ConfigApplyState::awaiting_config,
            "host decoder construction failed with a non-standard exception"};
  }

  for (auto &[id, session] : replacements) {
    sessions_.emplace(id, std::move(session));
    unavailable_ids_.erase(id);
  }
  active_config_ = config;
  CUDA_QEC_INFO("SessionRegistry: live config applied to {} host session(s)",
                replacements.size());
  return {ConfigApplyState::applied, "host decoder sessions replaced"};
}

DecodingSession &SessionRegistry::get(uint64_t decoder_id) {
  if (unavailable_ids_.count(decoder_id))
    throw SessionNotReady("Decoder " + std::to_string(decoder_id) +
                          " is awaiting a valid live configuration");
  auto it = sessions_.find(decoder_id);
  if (it == sessions_.end())
    throw std::out_of_range("Unknown decoder_id: " +
                            std::to_string(decoder_id));
  return *it->second;
}

const DecodingSession &SessionRegistry::get(uint64_t decoder_id) const {
  if (unavailable_ids_.count(decoder_id))
    throw SessionNotReady("Decoder " + std::to_string(decoder_id) +
                          " is awaiting a valid live configuration");
  auto it = sessions_.find(decoder_id);
  if (it == sessions_.end())
    throw std::out_of_range("Unknown decoder_id: " +
                            std::to_string(decoder_id));
  return *it->second;
}

} // namespace cudaq::qec::decoding_server
