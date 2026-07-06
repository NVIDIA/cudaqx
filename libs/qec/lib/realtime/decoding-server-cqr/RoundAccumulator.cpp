/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "RoundAccumulator.h"

#include <stdexcept>
#include <string>

namespace cudaq::qec::decoder_server {

std::size_t RoundKeyHash::operator()(const RoundKey &k) const noexcept {
  // FNV-1a style mix over the three fields.
  std::size_t h = 14695981039346656037ULL;
  auto mix = [&](uint64_t v) {
    h ^= v;
    h *= 1099511628211ULL;
  };
  mix(k.decoder_id);
  mix(k.counter);
  mix(k.syndrome_mapping_id);
  return h;
}

std::optional<CompletedRound>
RoundAccumulator::ingest(const RoundKey &key, uint32_t vp_id,
                         const uint8_t *bits, size_t num_syndromes,
                         const SyndromeMappingTable &table) {
  auto table_it = table.find(key.syndrome_mapping_id);
  if (table_it == table.end())
    throw std::invalid_argument("Unknown syndrome_mapping_id: " +
                                std::to_string(key.syndrome_mapping_id));

  const auto &vp_mappings = table_it->second;

  if (vp_id >= static_cast<uint32_t>(vp_mappings.size()))
    throw std::invalid_argument("VP " + std::to_string(vp_id) +
                                " not in syndrome mapping row");

  const auto &indices = vp_mappings[vp_id];

  // Identity row (empty index list): the fragment IS the whole round,
  // whatever its length -- the single-VP MVP where the round shape is not
  // known to the server (per-round vs whole-shot batching is the client's
  // choice and the decoder accumulates rounds itself). Pass it through
  // without staging.
  if (indices.empty()) {
    if (vp_mappings.size() != 1)
      throw std::invalid_argument(
          "Identity mapping rows are single-VP only (syndrome_mapping_id " +
          std::to_string(key.syndrome_mapping_id) + ")");
    return CompletedRound{
        .counter = key.counter,
        .syndrome_mapping_id = key.syndrome_mapping_id,
        .type = RoundType::BULK,
        .bits = std::vector<uint8_t>(bits, bits + num_syndromes),
    };
  }

  if (num_syndromes != indices.size())
    throw std::invalid_argument("Syndrome count mismatch: got " +
                                std::to_string(num_syndromes) + " expected " +
                                std::to_string(indices.size()));

  auto &round = rounds_[key];
  if (round.flat.empty()) {
    // First fragment for this round — determine flat vector size.
    uint32_t max_idx = 0;
    for (const auto &vp_map : vp_mappings)
      for (uint32_t idx : vp_map)
        if (idx > max_idx)
          max_idx = idx;
    round.flat.assign(max_idx + 1, 0);
    round.expected_vp_count = static_cast<uint32_t>(vp_mappings.size());
    round.type = RoundType::BULK;
  }

  if (round.received_vps.count(vp_id))
    throw std::invalid_argument("Duplicate VP fragment for vp_id=" +
                                std::to_string(vp_id));

  for (size_t i = 0; i < num_syndromes; ++i)
    round.flat[indices[i]] = bits[i];

  round.received_vps.insert(vp_id);

  if (round.received_vps.size() < round.expected_vp_count)
    return std::nullopt;

  CompletedRound result{
      .counter = key.counter,
      .syndrome_mapping_id = key.syndrome_mapping_id,
      .type = round.type,
      .bits = std::move(round.flat),
  };
  rounds_.erase(key);
  return result;
}

void RoundAccumulator::clear() { rounds_.clear(); }

} // namespace cudaq::qec::decoder_server
