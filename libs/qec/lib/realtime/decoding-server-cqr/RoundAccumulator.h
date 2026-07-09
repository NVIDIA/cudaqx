/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <set>
#include <unordered_map>
#include <vector>

namespace cudaq::qec::decoder_server {

/// Round type forwarded from the syndrome mapping table.
enum class RoundType : uint8_t {
  BULK = 0,  ///< Mid-circuit syndrome-extraction round
  FINAL = 1, ///< Last round (data-qubit measurements)
};

/// Completed round emitted by RoundAccumulator once all VP fragments arrive.
struct CompletedRound {
  uint64_t counter;
  uint64_t syndrome_mapping_id;
  RoundType type;
  std::vector<uint8_t>
      bits; ///< round-local flat detector vector (byte-per-bit)
};

/// Syndrome mapping table:
///   table[syndrome_mapping_id][vp_id] = vector of flat target indices
///
/// For the 0.7.0 MVP (single VP), the table has one entry per
/// syndrome_mapping_id with one VP whose indices are [0, 1, ..., N-1].
using SyndromeMappingTable =
    std::unordered_map<uint64_t, std::vector<std::vector<uint32_t>>>;

/// Key that identifies an in-progress round.
struct RoundKey {
  uint64_t decoder_id;
  uint64_t counter;
  uint64_t syndrome_mapping_id;
  bool operator==(const RoundKey &) const = default;
};

struct RoundKeyHash {
  std::size_t operator()(const RoundKey &k) const noexcept;
};

/// Assembles VP syndrome fragments into the flat detector vector required by
/// the decoder.
///
/// Owned by DecoderSession and accessed exclusively by that session's FIFO
/// worker thread.  No internal locking is required.
class RoundAccumulator {
public:
  /// Scatter bits from \p vp_id into the round buffer; returns CompletedRound
  /// once all VPs contribute, nullopt otherwise.
  /// @throws std::invalid_argument on unknown mapping id, duplicate VP, or
  /// length mismatch.
  std::optional<CompletedRound> ingest(const RoundKey &key, uint32_t vp_id,
                                       const uint8_t *bits,
                                       size_t num_syndromes,
                                       const SyndromeMappingTable &table);

  /// Discard all in-progress rounds.  Called by DecoderSession::on_reset().
  void clear();

private:
  struct InProgressRound {
    std::vector<uint8_t> flat;
    std::set<uint32_t> received_vps;
    uint32_t expected_vp_count = 0;
    RoundType type = RoundType::BULK;
  };

  std::unordered_map<RoundKey, InProgressRound, RoundKeyHash> rounds_;
};

} // namespace cudaq::qec::decoder_server
