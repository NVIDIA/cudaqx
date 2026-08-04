/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// This file defines extended_dem — one chunk of a detector error model, which
// partitions the PCM into four row groups (interior detectors, observable
// flips, incoming-seam syndrome, and outgoing-seam syndrome) — together with
// the three operations that compose chunks into a flat detector_error_model:
//
//   extended_dem_from_css_matrices()  — build a one-round chunk
//   dem_stitch(a, b)                      — compose two adjacent chunks
//   dem_close(dem)                        — collapse seam rows into a flat DEM
//
// Stitching contracts a.out_syndrome and b.in_syndrome onto the same output
// rows (the seam), so each fault in A and each fault in B independently
// contribute to the seam detector. Stitching T one-round chunks and closing
// produces output identical to dem_from_css_matrices(code, noise, T).

#pragma once

#include "cudaq/qec/code_matrices.h"
#include "cudaq/qec/detector_error_model.h"
#include "cudaq/qec/sparse_binary_matrix.h"
#include <cstdint>
#include <string>
#include <vector>

namespace cudaq::qec {

/// @brief One DEM chunk, partitioned into interior, observable, and seam rows.
///
/// All four sparse matrices share the same num_cols() — the number of fault
/// mechanisms in this chunk. Row counts:
///   interior:    n_interior detectors fully inside this chunk.
///   observables: k = lz.num_rows() + lx.num_rows().
///   in_syndrome: d = hz.num_rows() + hx.num_rows() (left seam, incoming).
///   out_syndrome: d (right seam, outgoing).
///
/// For a one-round chunk, in_syndrome == out_syndrome (the raw syndrome of
/// that round participates in both the left and right seam detectors). Tags
/// identify each seam row: a.out_tags[k] must equal b.in_tags[k] for dem_stitch
/// to be valid.
///
/// The two seams need not be the same width. A phase chunk in the
/// init / bulk / final decomposition is deliberately asymmetric:
///   - init  has no incoming seam (nothing precedes it), so in_syndrome has
///     zero rows while out_syndrome carries the syndrome forward. Its own
///     detectors live in interior, because round 0 is compared against the
///     zero initial state rather than against a previous round.
///   - bulk  has both seams and usually no interior rows: its detectors are
///     the seam differences formed when it is stitched to its neighbours.
///   - final has no outgoing seam, so out_syndrome has zero rows.
/// Use num_in_seam_rows() / num_out_seam_rows() when the distinction matters;
/// num_seam_rows() reports the incoming width and is the right choice only for
/// the uniform per-round chunks the streaming helpers below expect.
struct extended_dem {
  sparse_binary_matrix interior;     ///< [n_int x n_faults] interior detectors
  sparse_binary_matrix observables;  ///< [k x n_faults] observable flips
  sparse_binary_matrix in_syndrome;  ///< [d x n_faults] left seam syndrome
  sparse_binary_matrix out_syndrome; ///< [d x n_faults] right seam syndrome
  std::vector<double> fault_priors;  ///< length n_faults
  std::vector<uint64_t> in_tags;     ///< check identity per in_syndrome row
  std::vector<uint64_t> out_tags;    ///< check identity per out_syndrome row

  /// Number of fault columns (num_cols shared by all four matrices).
  uint32_t num_faults() const;

  /// Number of interior detector rows.
  uint32_t num_interior() const;

  /// Number of observable rows.
  uint32_t num_observables() const;

  /// Number of incoming-seam rows, i.e. rows of in_syndrome.
  ///
  /// Equals num_out_seam_rows() for the uniform per-round chunks produced by
  /// extended_dem_from_css_matrices(); differs for phase chunks.
  uint32_t num_seam_rows() const;

  /// Number of incoming-seam rows. Spelled-out alias for num_seam_rows(),
  /// for call sites where the asymmetry is the point.
  uint32_t num_in_seam_rows() const;

  /// Number of outgoing-seam rows, i.e. rows of out_syndrome.
  uint32_t num_out_seam_rows() const;

  /// @brief Throw std::invalid_argument unless this chunk is internally
  /// consistent, i.e. safe to stitch, close, or merge.
  ///
  /// num_faults() reports in_syndrome's width alone, so nothing else about a
  /// chunk is self-describing: a block that disagrees with it, or a prior list
  /// that does not have one entry per fault, would otherwise be read at the
  /// wrong width and silently misalign columns (or walk past the end of a
  /// nested column list). Checks that
  ///   - every block is num_faults() columns wide, including zero-row blocks
  ///     (use a width-n empty matrix, not a default-constructed one, when the
  ///     chunk has faults),
  ///   - fault_priors has one entry per fault, and
  ///   - each tag vector has one entry per row of the seam it names.
  ///
  /// @param context Prefix for the error message, naming the caller.
  void validate(const char *context) const;
};

/// @brief Build a one-round extended_dem from CSS matrices and noise.
///
/// For each active fault (qubit with nonzero rate), emits one fault column:
///   - in_syndrome and out_syndrome rows: the raw syndrome of that fault
///     (hz[:,q] for X faults, hx[:,q] for Z faults; both for Y).
///     in_syndrome == out_syndrome for a one-round chunk because the same
///     syndrome[r] appears on both sides of the seam detectors.
///   - observables rows: lz[:,q] for X/Y faults, lx[:,q] for Z/Y faults.
///   - interior rows: empty (no syndrome differences within one round).
///
/// Tags are sequential: in_tags[k] = out_tags[k] = k.
///
/// @param code  CSS code matrices. Same constraints as dem_from_css_matrices.
/// @param noise Depolarizing noise rates. Per-qubit vectors override scalars.
/// @return      One-round extended_dem with n_seam_rows = hz.num_rows() +
///              hx.num_rows() and n_faults = |active_X|+|active_Z|+|active_Y|.
/// @throws std::invalid_argument on dimension or per-qubit size mismatch.
extended_dem extended_dem_from_css_matrices(const css_code_matrices &code,
                                            const css_noise_params &noise);

// ---------------------------------------------------------------------------
// Declarative chunk specs (the form carried in decoder configuration YAML)
// ---------------------------------------------------------------------------

/// @brief One DEM chunk written as flat row-sparse index lists.
///
/// Each vector holds the fault-column indices of a row followed by a -1
/// terminator, so `[0, 1, 5, -1, 1, 2, 6, -1]` is two rows and the row count of
/// a well-formed vector is its number of -1 entries. This is the same encoding
/// the decoder configuration already uses for H_sparse / O_sparse / D_sparse,
/// which is why the phases can be written directly in YAML.
///
/// The four matrices map onto extended_dem's row groups:
///   H_in_sparse  -> in_syndrome   (incoming seam; empty for an init phase)
///   H_mid_sparse -> interior      (detectors wholly inside this chunk)
///   H_out_sparse -> out_syndrome  (outgoing seam; empty for a final phase)
///   O_sparse     -> observables
struct dem_chunk_spec {
  /// Number of fault columns, i.e. num_cols() of all four matrices.
  uint64_t num_faults = 0;
  std::vector<std::int64_t> H_in_sparse;
  std::vector<std::int64_t> H_mid_sparse;
  std::vector<std::int64_t> H_out_sparse;
  std::vector<std::int64_t> O_sparse;
  /// Prior for each fault column; must have num_faults entries.
  std::vector<double> error_rates;

  bool operator==(const dem_chunk_spec &) const = default;

  /// True when nothing has been set, used to detect an omitted phase.
  bool is_empty() const;

  /// @brief Check internal consistency: a positive fault count that fits in
  /// uint32_t (sparse matrix column index width), one error rate per fault with
  /// each in [0, 1], and index lists that are -1 terminated with every index
  /// in [0, num_faults).
  /// @param context Prefix for error messages, e.g. "dem_chunks.init".
  /// @throws std::invalid_argument on the first violation.
  void validate(const std::string &context) const;
};

/// @brief Build an extended_dem from a dem_chunk_spec.
///
/// Seam tags are assigned sequentially (in_tags[k] = out_tags[k] = k), which is
/// what makes adjacent phases stitchable: the k-th check keeps its identity
/// across every phase boundary.
///
/// @param spec    Chunk to materialize.
/// @param context Prefix for error messages raised by spec.validate().
/// @return        The chunk as an extended_dem.
/// @throws std::invalid_argument if the spec is inconsistent.
extended_dem dem_chunk_from_spec(const dem_chunk_spec &spec,
                                 const std::string &context = "dem_chunk");

/// @brief Stitch two adjacent DEM chunks: contract a.out_syndrome with
/// b.in_syndrome.
///
/// The seam = [a.out_syndrome | b.in_syndrome] (fault columns from A then B)
/// becomes new interior rows in the result. Each fault in A and each fault in
/// B independently contribute to the same seam-detector row, which fires when
/// the syndrome changes across the chunk boundary. The stitched chunk spans
/// the fault mechanisms of both inputs.
///
/// Interior rows come out in ascending round order -- a's interior, then the
/// seam between them, then b's interior -- for any association of stitches, so
/// dem_close_all() and dem_chunks_to_detector_round() read the same round
/// layout out of a stitched chunk as they would out of the rounds pushed
/// separately.
///
/// Row counts after stitching:
///   interior:    a.num_interior() + b.num_interior() + a.num_out_seam_rows()
///   observables: a.num_observables()  (must equal b.num_observables())
///   in_syndrome: a.num_in_seam_rows()
///   out_syndrome: b.num_out_seam_rows()
///
/// The contracted seam is a's outgoing side against b's incoming side, so
/// those two widths must agree; a's incoming and b's outgoing widths are
/// carried through untouched and may be anything, including zero. That is what
/// lets an init chunk (no incoming seam) stitch to a bulk chunk, and a bulk
/// chunk to a final one (no outgoing seam).
///
/// @param a Left DEM chunk. a.out_tags must equal b.in_tags element-wise.
/// @param b Right DEM chunk.
/// @return   Stitched extended_dem.
/// @throws std::runtime_error if a's outgoing seam width differs from b's
///         incoming seam width, on tag mismatch, or on observable-count
///         mismatch.
extended_dem dem_stitch(const extended_dem &a, const extended_dem &b);

/// @brief Stitch a span of adjacent DEM chunks left-to-right.
///
/// Equivalent to dem_stitch(dem_stitch(...stitch(dem_chunks[0],
/// dem_chunks[1])...), dem_chunks[n-1]). dem_chunks must be non-empty and each
/// adjacent pair must be tag-compatible.
///
/// @param dem_chunks Non-empty sequence of one-round (or pre-stitched) chunks.
/// @return       Fully-stitched extended_dem.
/// @throws std::invalid_argument if dem_chunks is empty.
/// @throws std::runtime_error on any pairwise tag or shape mismatch.
extended_dem dem_stitch_all(const std::vector<extended_dem> &dem_chunks);

// ---------------------------------------------------------------------------
// Duplicate fault columns
// ---------------------------------------------------------------------------

/// @brief Prior-combining strategy for dem_merge_duplicate_columns().
///
/// When two or more fault columns share identical row support they are merged
/// into one. The merged prior is computed from the individual priors using
/// one of these two rules:
///
///   - or_combine (default): p_merged = 1/2 * (1 - prod_i(1 - 2 p_i))
///     Exact probability that an odd number of independent events fire, which
///     is the net GF(2) effect of identical DEM columns (even counts cancel).
///     Pairwise this is P(A xor B) = p + q - 2 p q, matching
///     detector_error_model canonicalization. Prefer this for physical fault
///     mechanisms.
///
///   - sum_combine: p_merged = min(1, sum_i(p_i))
///     Linear approximation valid when all p_i are small. The sum of several
///     larger priors can exceed 1, which is not a probability any decoder can
///     use, so the result is clamped; prefer or_combine when the priors are
///     not small.
enum class prior_combine_mode { or_combine, sum_combine };

/// @brief Merge fault columns with identical row support into single columns.
///
/// After stitching, duplicate-support columns arise when seam-time noise is
/// modelled on both sides of a measurement boundary. This function collapses
/// them: all columns whose nonzero row sets are identical (across the full
/// combined PCM: interior + observables + in_syndrome + out_syndrome) are
/// replaced by one column whose prior is the combination of the merged priors
/// under the chosen rule.
///
/// Output columns are sorted lexicographically by their row-support tuples,
/// matching the Python reference implementation. Row counts and seam tags are
/// copied unchanged; only the column space is modified.
///
/// Columns are compared over GF(2), so a column that lists a row twice matches
/// one that omits that row. This operates across columns and spans all four
/// blocks at once; sparse_binary_matrix::canonicalize() is the unrelated
/// within-column normalization of a single block.
///
/// @param dem   Input extended_dem (not modified).
/// @param mode  How to combine priors of merged columns.
/// @return      New extended_dem with unique-support columns.
extended_dem dem_merge_duplicate_columns(
    const extended_dem &dem,
    prior_combine_mode mode = prior_combine_mode::or_combine);

/// @brief Return true iff every fault column has a unique row-support set.
bool are_dem_columns_unique(const extended_dem &dem);

/// @brief Throw std::invalid_argument if any two columns share row support.
///
/// The error message names the duplicate support and the affected column
/// indices, and suggests calling dem_merge_duplicate_columns() to fix the
/// issue.
void assert_dem_columns_unique(const extended_dem &dem);

/// @brief Stitch DEM chunks left-to-right then merge duplicate columns.
///
/// Equivalent to dem_merge_duplicate_columns(dem_stitch_all(dem_chunks), mode).
/// Convenience wrapper for the common pattern of assembling a window and
/// immediately deduplicating its fault columns.
///
/// @param dem_chunks Non-empty sequence of chunks in round order.
/// @param mode   Prior-combining strategy (default: or_combine).
extended_dem
dem_stitch_merged(const std::vector<extended_dem> &dem_chunks,
                  prior_combine_mode mode = prior_combine_mode::or_combine);

// ---------------------------------------------------------------------------
// Utilities for streaming decoder integration
// ---------------------------------------------------------------------------

/// @brief How many rounds one DEM chunk spans.
///
/// A chunk built by extended_dem_from_css_matrices() spans one round and has
/// no interior rows. Stitching R of them contracts R-1 seams into interior
/// rows, so a chunk's interior row count counts its rounds after the leading
/// one: `1 + num_interior() / d`.
///
/// The leading round is the incoming seam band. An init phase chunk has none --
/// nothing precedes the first round for it to compare against -- and carries
/// round 0 in its interior instead, so it spans `num_interior() / d` rounds.
/// `d` is whichever seam the chunk has, incoming for preference.
///
/// @param dem_chunk Chunk to measure.
/// @throws std::invalid_argument if the chunk has no seam rows on either side,
///         or if its interior rows are not a whole number of rounds.
uint32_t dem_chunk_rounds(const extended_dem &dem_chunk);

/// @brief Total rounds a sequence of DEM chunks describes.
///
/// The sum of dem_chunk_rounds() over the sequence, which is the round count
/// dem_close_all(dem_chunks) produces detectors for. Only equal to
/// dem_chunks.size() when every chunk spans a single round.
///
/// @param dem_chunks Non-empty sequence of chunks in round order.
/// @throws std::invalid_argument on an empty sequence, a seam that does not
///         contract against its neighbour, a seam whose width differs from the
///         rest of the sequence, or a chunk whose interior rows are not a whole
///         number of rounds.
std::size_t dem_chunks_to_rounds(const std::vector<extended_dem> &dem_chunks);

/// @brief Extract the detector→round mapping from a sequence of DEM chunks.
///
/// Returns a vector of length T*d (where d is the sequence's seam width and
/// T is dem_chunks_to_rounds(dem_chunks)) where entry i gives the round index
/// (0..T-1) of detector i in the flat DEM produced by
/// dem_close_all(dem_chunks). Detector r*d+k belongs to round r.
///
/// This vector is the "detector_round" parameter expected by streaming
/// decoders that need to know when each detector is available so they can
/// stream inputs round-by-round.
///
/// A chunk may span several rounds: dem_stitch() keeps its interior rows in
/// ascending round order, so a stitched chunk maps to exactly the rounds its
/// pieces would have mapped to on their own.
///
/// @param dem_chunks Non-empty sequence of chunks in round order.
/// @throws std::invalid_argument on an empty sequence, a seam that does not
///         contract against its neighbour, a seam whose width differs from the
///         rest of the sequence, or a chunk whose interior rows are not a whole
///         number of rounds.
std::vector<std::int32_t>
dem_chunks_to_detector_round(const std::vector<extended_dem> &dem_chunks);

/// @brief Extract the O_sparse observable-flip map from T DEM chunks.
///
/// O_sparse[obs_id] lists the global fault column indices (across all T
/// chunks, concatenated in chunk order) that flip observable obs_id. This
/// matches the nested overload of decoder::set_O_sparse() and is compatible
/// with the observables_flips_matrix produced by dem_close_all(dem_chunks).
///
/// Keyed on fault columns rather than rounds, so how many rounds each chunk
/// spans makes no difference here.
///
/// @param dem_chunks Non-empty sequence of chunks in round order.
/// @throws std::invalid_argument if dem_chunks is empty.
std::vector<std::vector<uint32_t>>
dem_chunks_to_o_sparse(const std::vector<extended_dem> &dem_chunks);

/// @brief Build a flat detector_error_model from T DEM chunks in O(T) time.
///
/// Equivalent to dem_close(dem_stitch_all(dem_chunks)) but avoids the O(T²)
/// cost of the left-fold accumulation in dem_stitch_all. Detector rows are
/// always emitted in round order: band r*d..(r+1)*d-1 corresponds to
/// detector[r] regardless of chunk granularity. Error rates are concatenated in
/// chunk order.
///
/// Prefer this over dem_close(dem_stitch_all(dem_chunks)) whenever all T chunks
/// are available upfront and only the closed DEM is needed.
/// dem_stitch/dem_stitch_all remain the right choice when the intermediate
/// extended_dem must be inspected (e.g. seam rows before closing, or partial
/// stitching).
///
/// Seams only have to contract pairwise, so a phase decomposition works: the
/// first chunk may have no incoming seam (its interior carries round 0) and the
/// last may have no outgoing one. As with dem_close(), the last chunk's
/// out_syndrome is discarded: any detector that should appear in the closed
/// DEM must already live in some chunk's in_syndrome or interior (for a
/// phase-decomposed experiment, that means the final chunk's H_in_sparse /
/// H_mid_sparse).
///
/// @param dem_chunks Non-empty sequence of chunks in round order. Each chunk's
///               out_syndrome must match the next one's in_syndrome, and all
///               must share num_observables().
/// @return       detector_error_model ready for any decoder.
/// @throws std::invalid_argument if dem_chunks is empty or dimensions differ.
detector_error_model dem_close_all(const std::vector<extended_dem> &dem_chunks);

/// @brief Build a canonicalized, CSC-format parity-check matrix from chunks.
///
/// Equivalent to
///   sparse_binary_matrix(dem_close_all(dem_chunks).detector_error_matrix)
///       .canonicalize()
///       .to_csc()
/// expressed as a single named utility so decoders that need an H matrix
/// directly (not the full detector_error_model) don't have to repeat the
/// idiom.
///
/// @param dem_chunks Non-empty sequence of chunks in round order.
/// @return           Canonicalized CSC sparse_binary_matrix (rows = detectors,
///                   cols = fault mechanisms).
/// @throws std::invalid_argument if dem_chunks is empty.
sparse_binary_matrix
dem_chunks_to_pcm(const std::vector<extended_dem> &dem_chunks);

/// @brief Collapse an extended_dem into a flat detector_error_model.
///
/// Places in_syndrome rows first, then interior rows, to match the detector
/// row ordering produced by dem_from_css_matrices():
///   - detector_error_matrix: [in_syndrome stacked above interior]
///   - observables_flips_matrix: observables
///   - error_rates: fault_priors
///
/// out_syndrome is intentionally dropped. Closing models a terminated
/// experiment: there is no later round for the outgoing seam to differ against,
/// matching dem_from_css_matrices (final-round faults touch only the last
/// detector band). Put any detector that must survive closing into in_syndrome
/// or interior instead — for example a final data-readout boundary belongs in
/// the last chunk's in_syndrome / interior, never only in out_syndrome.
///
/// Invariant (up to canonicalization):
///   dem_close(dem_stitch_all(T one-round chunks))
///     == dem_from_css_matrices(code, noise, T)
///
/// @param dem Fully-stitched (or single-chunk) extended_dem.
/// @return    detector_error_model ready for use with any decoder.
detector_error_model dem_close(const extended_dem &dem);

} // namespace cudaq::qec
