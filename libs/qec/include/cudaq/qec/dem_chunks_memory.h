/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Memory-circuit-specific DEM chunk utilities.
//
// This file extends extended_dem.h with the three-phase init/bulk/final
// decomposition that models a repeated-round memory experiment, and the
// D_sparse detector map that encodes the XOR-of-consecutive-rounds rule those
// experiments use. These are NOT generic: a logical CNOT, lattice-surgery
// merge, or other non-memory circuit cannot reuse this decomposition as-is
// because its detector definitions are not pairs of adjacent syndrome rounds.
//
// Generic open-boundary DEM infrastructure (extended_dem, dem_stitch,
// dem_close, dem_chunk_spec, dem_chunk_from_spec, ...) lives in extended_dem.h
// and is circuit-agnostic.

#pragma once

#include "cudaq/qec/extended_dem.h"
#include <cstddef>
#include <cstdint>
#include <vector>

namespace cudaq::qec {

/// @brief The init / bulk / final phases of a repeated-round decomposition.
///
/// A run of T rounds is `init`, then `bulk` repeated T-2 times, then `final`.
/// Because `bulk` is what repeats, its incoming and outgoing seams must be the
/// same width; init has no incoming seam and final no outgoing seam.
struct dem_chunks_spec {
  dem_chunk_spec init;
  /// Optional: omit for a decomposition with no repeated middle.
  dem_chunk_spec bulk;
  dem_chunk_spec final;

  bool operator==(const dem_chunks_spec &) const = default;

  /// True when no phase has been set.
  bool is_empty() const;

  /// True when a repeated middle phase was supplied.
  bool has_bulk() const;

  /// @brief Validate each phase, then the relationships between them: init
  /// carries no incoming seam, final no outgoing seam, bulk's two seams are
  /// equally wide, the contracted seams line up, and every phase reports the
  /// same number of observables.
  /// @throws std::invalid_argument on the first violation.
  void validate() const;
};

/// @brief Expand a phase spec into the chunk sequence for a given round count.
///
/// Produces init, `num_rounds - 2` copies of bulk, then final, which is the
/// sequence dem_stitch_all() consumes to build the whole experiment. When the
/// spec has no bulk phase, only num_rounds == 2 is representable.
///
/// @param spec       Validated phase specs.
/// @param num_rounds Total rounds, counting init and final. Must be >= 2.
/// @return           num_rounds chunks, ready to stitch left to right.
/// @throws std::invalid_argument if num_rounds < 2, if bulk repeats are needed
///         but no bulk phase was supplied, or if the spec is inconsistent.
std::vector<extended_dem> dem_chunks_from_spec(const dem_chunks_spec &spec,
                                               std::size_t num_rounds);

/// @brief Extract the D_sparse measurement-to-detector map from T DEM chunks.
///
/// D_sparse[det_id] lists the raw per-round measurement bit positions (within
/// a flat T*d buffer laid out as round-0 bits 0..d-1, round-1 bits d..2d-1,
/// ...) that XOR-combine to produce detector det_id:
///   - Detector k       (r=0): bits {k}               (vs zero initial state)
///   - Detector r*d+k (r>0): bits {(r-1)*d+k, r*d+k} (syndrome difference)
///
/// This encodes the memory-experiment detector convention: consecutive syndrome
/// rounds are XOR-differenced. For a non-memory circuit whose detectors are
/// not pairs of adjacent rounds, construct D_sparse from circuit analysis
/// instead of using this function.
///
/// The return type matches the nested overload of decoder::set_D_sparse().
///
/// T is dem_chunks_to_rounds(dem_chunks), so a chunk spanning several rounds
/// contributes a measurement slot per round it carries.
///
/// @param dem_chunks Non-empty sequence of chunks in round order.
/// @throws std::invalid_argument on an empty sequence, a seam that does not
///         contract against its neighbour, a seam whose width differs from the
///         rest of the sequence, or a chunk whose interior rows are not a whole
///         number of rounds.
std::vector<std::vector<uint32_t>>
dem_chunks_to_d_sparse(const std::vector<extended_dem> &dem_chunks);

} // namespace cudaq::qec
