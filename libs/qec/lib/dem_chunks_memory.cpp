/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Implements the memory-circuit-specific DEM chunk decomposition:
//   dem_chunks_spec  — the init/bulk/final three-phase structure that models a
//                      repeated-round memory experiment.
//   dem_chunks_from_spec — expands the spec into a chunk sequence.
//   dem_chunks_to_d_sparse — builds the D matrix using the
//                      XOR-of-consecutive-rounds detector convention.
//
// These are NOT generic: they encode assumptions that hold for CSS memory
// experiments (uniform syndrome-check width per round, detectors formed by
// XOR-differencing adjacent rounds) but not for arbitrary QEC circuits.
// Generic chunk infrastructure lives in extended_dem.cpp.

#include "cudaq/qec/dem_chunks_memory.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace cudaq::qec {

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

// Number of rows described by a -1-terminated index list.
static uint32_t sparse_row_count(const std::vector<std::int64_t> &rows) {
  return static_cast<uint32_t>(std::count(rows.begin(), rows.end(), -1));
}

// ---------------------------------------------------------------------------
// dem_chunks_spec
// ---------------------------------------------------------------------------

bool dem_chunks_spec::is_empty() const {
  return init.is_empty() && bulk.is_empty() && final.is_empty();
}

bool dem_chunks_spec::has_bulk() const { return !bulk.is_empty(); }

void dem_chunks_spec::validate() const {
  if (is_empty())
    throw std::invalid_argument("dem_chunks: no phases were supplied");
  if (init.is_empty())
    throw std::invalid_argument("dem_chunks: init phase is required");
  if (final.is_empty())
    throw std::invalid_argument("dem_chunks: final phase is required");

  init.validate("dem_chunks.init");
  if (has_bulk())
    bulk.validate("dem_chunks.bulk");
  final.validate("dem_chunks.final");

  // init starts the stream: nothing feeds its incoming seam. final ends it:
  // nothing consumes its outgoing seam. Round-0 detectors for init belong in
  // H_mid_sparse, compared against the zero initial state.
  if (!init.H_in_sparse.empty())
    throw std::invalid_argument(
        "dem_chunks.init.H_in_sparse must be empty: the init phase has no "
        "incoming seam. Put the first round's detectors in H_mid_sparse.");
  if (!final.H_out_sparse.empty())
    throw std::invalid_argument(
        "dem_chunks.final.H_out_sparse must be empty: the final phase has no "
        "outgoing seam.");

  const uint32_t init_out = sparse_row_count(init.H_out_sparse);
  const uint32_t final_in = sparse_row_count(final.H_in_sparse);

  if (has_bulk()) {
    const uint32_t bulk_in = sparse_row_count(bulk.H_in_sparse);
    const uint32_t bulk_out = sparse_row_count(bulk.H_out_sparse);
    // bulk is the phase that repeats, so it has to stitch to itself.
    if (bulk_in != bulk_out)
      throw std::invalid_argument(
          "dem_chunks.bulk seams are unequal (H_in_sparse has " +
          std::to_string(bulk_in) + " rows, H_out_sparse has " +
          std::to_string(bulk_out) +
          " rows); the repeated phase must stitch to itself");
    if (init_out != bulk_in)
      throw std::invalid_argument(
          "dem_chunks seam width mismatch: init.H_out_sparse has " +
          std::to_string(init_out) + " rows but bulk.H_in_sparse has " +
          std::to_string(bulk_in));
    if (bulk_out != final_in)
      throw std::invalid_argument(
          "dem_chunks seam width mismatch: bulk.H_out_sparse has " +
          std::to_string(bulk_out) + " rows but final.H_in_sparse has " +
          std::to_string(final_in));
  } else if (init_out != final_in) {
    throw std::invalid_argument(
        "dem_chunks seam width mismatch: init.H_out_sparse has " +
        std::to_string(init_out) + " rows but final.H_in_sparse has " +
        std::to_string(final_in));
  }

  const uint32_t init_obs = sparse_row_count(init.O_sparse);
  if (has_bulk() && sparse_row_count(bulk.O_sparse) != init_obs)
    throw std::invalid_argument(
        "dem_chunks observable count mismatch: init has " +
        std::to_string(init_obs) + " but bulk has " +
        std::to_string(sparse_row_count(bulk.O_sparse)));
  if (sparse_row_count(final.O_sparse) != init_obs)
    throw std::invalid_argument(
        "dem_chunks observable count mismatch: init has " +
        std::to_string(init_obs) + " but final has " +
        std::to_string(sparse_row_count(final.O_sparse)));
} // end - dem_chunks_spec::validate()

// ---------------------------------------------------------------------------
// dem_chunks_from_spec
// ---------------------------------------------------------------------------

// spec.validate() is called first; each phase is then materialized via the
// public dem_chunk_from_spec() which re-validates that phase in isolation.
std::vector<extended_dem> dem_chunks_from_spec(const dem_chunks_spec &spec,
                                               std::size_t num_rounds) {
  if (num_rounds < 2)
    throw std::invalid_argument(
        "dem_chunks_from_spec: num_rounds must be at least 2 (init and "
        "final), got " +
        std::to_string(num_rounds));
  spec.validate();

  const std::size_t bulk_repeats = num_rounds - 2;
  if (bulk_repeats > 0 && !spec.has_bulk())
    throw std::invalid_argument("dem_chunks_from_spec: num_rounds " +
                                std::to_string(num_rounds) + " needs " +
                                std::to_string(bulk_repeats) +
                                " bulk rounds but no bulk phase was supplied");

  std::vector<extended_dem> chunks;
  chunks.reserve(num_rounds);
  chunks.push_back(dem_chunk_from_spec(spec.init, "dem_chunks.init"));
  if (bulk_repeats > 0) {
    const auto bulk = dem_chunk_from_spec(spec.bulk, "dem_chunks.bulk");
    chunks.insert(chunks.end(), bulk_repeats, bulk);
  }
  chunks.push_back(dem_chunk_from_spec(spec.final, "dem_chunks.final"));
  return chunks;
} // end - dem_chunks_from_spec()

// ---------------------------------------------------------------------------
// dem_chunks_to_d_sparse
// ---------------------------------------------------------------------------

// D_sparse[det_id] = measurement bit positions that XOR-combine to form that
// detector under the memory-experiment convention:
//   det k       (r=0): {k}               — vs. zero initial state
//   det r*d+k (r>0): {(r-1)*d+k, r*d+k} — consecutive syndrome XOR
std::vector<std::vector<uint32_t>>
dem_chunks_to_d_sparse(const std::vector<extended_dem> &dem_chunks) {
  if (dem_chunks.empty())
    throw std::invalid_argument(
        "dem_chunks_to_d_sparse: dem_chunks must be non-empty");

  const uint32_t d = dem_chunks[0].num_in_seam_rows() != 0
                         ? dem_chunks[0].num_in_seam_rows()
                         : dem_chunks[0].num_out_seam_rows();
  if (d == 0)
    throw std::invalid_argument(
        "dem_chunks_to_d_sparse: chunk 0 has no seam rows on either side, "
        "so its rounds cannot be counted");

  // dem_chunks_to_rounds validates seam widths and contractibility.
  const std::size_t T = dem_chunks_to_rounds(dem_chunks);
  const std::size_t n_det = T * static_cast<std::size_t>(d);

  if (T > 0) {
    const auto max_bit = static_cast<std::size_t>(T - 1) * d + (d - 1);
    if (max_bit > std::numeric_limits<uint32_t>::max())
      throw std::overflow_error(
          "dem_chunks_to_d_sparse: measurement bit index exceeds uint32_t "
          "max");
  }

  std::vector<std::vector<uint32_t>> d_sparse(n_det);
  for (std::size_t r = 0; r < T; ++r) {
    for (uint32_t k = 0; k < d; ++k) {
      const std::size_t det = r * d + k;
      if (r == 0) {
        d_sparse[det] = {k};
      } else {
        d_sparse[det] = {static_cast<uint32_t>((r - 1) * d + k),
                         static_cast<uint32_t>(r * d + k)};
      }
    }
  } // end - for(r)
  return d_sparse;
} // end - dem_chunks_to_d_sparse()

} // namespace cudaq::qec
