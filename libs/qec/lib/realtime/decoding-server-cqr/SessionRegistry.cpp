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

namespace cudaq::qec::decoding_server {

using cudaq::qec::decoding::config::multi_decoder_config;

/// Build the default single-VP pass-through syndrome mapping table.
/// mapping_id=0 → VP 0 → empty index list (pass-through)
///
/// An empty index list signals RoundAccumulator to copy bits directly without
/// scatter.  This is correct for the nominal per-round enqueue pattern where
/// the caller sends exactly the syndromes for one round and does not need
/// index remapping.  An identity-sized index list would force every enqueue
/// to provide exactly syndrome_size bits, which breaks per-round batching.
static SyndromeMappingTable make_default_mapping_table() {
  SyndromeMappingTable table;
  table[0] = {{}}; // syndrome_mapping_id=0, VP 0, pass-through
  return table;
}

// ---------------------------------------------------------------------------
// SessionRegistry
// ---------------------------------------------------------------------------

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
  for (const auto &dc : config.decoders) {
    if (dc.id < 0)
      throw std::runtime_error("Negative decoder id " + std::to_string(dc.id) +
                               " in " + source_name);
    const uint64_t id = static_cast<uint64_t>(dc.id);
    if (sessions_.count(id))
      throw std::runtime_error("Duplicate decoder id " + std::to_string(dc.id) +
                               " in " + source_name);

    // All decoders in one server instance must share the same transport type
    // because there is one receive loop per unique transceiver.
    if (sessions_.empty()) {
      transport_ = dc.transport;
    } else if (dc.transport != transport_) {
      throw std::runtime_error(
          "Mixed transport types in " + source_name +
          ": all decoder entries must declare the same transport");
    }

    CUDA_QEC_INFO("SessionRegistry: creating decoder id={} type={}", dc.id,
                  dc.type);

    auto decoder = cudaq::qec::decoding::host::create_realtime_decoder(dc);
    auto session = DecodingSession::create(std::move(decoder),
                                           make_default_mapping_table());

    // [For follow-up] dc.transport (cpu_roce / gpu_roce) is parsed from YAML
    // but not yet used to select a transceiver here. Transport binding requires
    // CpuRoceTransceiverAdapter / GpuRoceTransceiverAdapter (gated on
    // CUDAQ_REALTIME headers); the split-transport DecodingServer constructor
    // is already in place to accept the resulting dispatch map.
    session->start_worker();
    sessions_.emplace(id, std::move(session));
  }

  CUDA_QEC_INFO("SessionRegistry: loaded {} decoder session(s)",
                sessions_.size());
}

DecodingSession &SessionRegistry::get(uint64_t decoder_id) {
  auto it = sessions_.find(decoder_id);
  if (it == sessions_.end())
    throw std::out_of_range("Unknown decoder_id: " +
                            std::to_string(decoder_id));
  return *it->second;
}

const DecodingSession &SessionRegistry::get(uint64_t decoder_id) const {
  auto it = sessions_.find(decoder_id);
  if (it == sessions_.end())
    throw std::out_of_range("Unknown decoder_id: " +
                            std::to_string(decoder_id));
  return *it->second;
}

} // namespace cudaq::qec::decoding_server
