/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Implements extended_dem construction and composition using
// sparse_binary_matrix throughout. No dense tensor intermediates appear until
// dem_close(), where the output detector_error_model requires tensor<uint8_t>.
//
// Each fault column of a one-round chunk has in_syndrome == out_syndrome
// (the raw syndrome of that round), because syndrome[r] participates in
// both seam detectors it borders: detector[r] = syndrome[r] XOR syndrome[r-1]
// (via in_syndrome) and detector[r+1] = syndrome[r+1] XOR syndrome[r]
// (via out_syndrome).
//
// Stitching is horizontal block concatenation: A's fault columns occupy the
// left half and B's the right. The seam rows are formed by placing both
// a.out_syndrome and b.in_syndrome at the same row indices so each fault
// independently contributes to the seam detector.
//
// dem_close() places in_syndrome rows first (they become detector[0] =
// syndrome[0] vs. zero initial state), then interior rows, matching the row
// order of dem_from_css_matrices(). out_syndrome is dropped: closing ends the
// experiment, so there is no later round for that seam to become a detector.

#include "cudaq/qec/extended_dem.h"
#include "cudaq/qec/code_matrices.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace cudaq::qec {

using namespace detail; // resolve_num_qubits, check_num_cols, etc.

// ---------------------------------------------------------------------------
// extended_dem accessors
// ---------------------------------------------------------------------------

uint32_t extended_dem::num_faults() const { return in_syndrome.num_cols(); }

uint32_t extended_dem::num_interior() const { return interior.num_rows(); }

uint32_t extended_dem::num_observables() const {
  return observables.num_rows();
}

uint32_t extended_dem::num_seam_rows() const { return in_syndrome.num_rows(); }

uint32_t extended_dem::num_in_seam_rows() const {
  return in_syndrome.num_rows();
}

// Everything downstream indexes fault columns by the chunk's own fault index,
// so a block that is not that wide would be scattered into the wrong columns
// (or past the end of a nested column list) with no other symptom. Zero-row
// blocks must still report num_faults() columns: dem_merge_duplicate_columns
// and dem_chunks_to_o_sparse walk every fault index through every block.
void extended_dem::validate(const char *context) const {
  const uint32_t n = num_faults();

  const auto check_width = [&](const sparse_binary_matrix &block,
                               const char *label) {
    if (block.num_cols() != n)
      throw std::invalid_argument(
          std::string(context) + ": " + label + " has " +
          std::to_string(block.num_cols()) + " columns but the chunk has " +
          std::to_string(n) +
          " faults; every block must be as wide as the chunk");
  };
  // in_syndrome is what num_faults() reports, so it defines n rather than
  // being checked against it.
  check_width(interior, "interior");
  check_width(observables, "observables");
  check_width(out_syndrome, "out_syndrome");

  if (fault_priors.size() != n)
    throw std::invalid_argument(std::string(context) + ": fault_priors has " +
                                std::to_string(fault_priors.size()) +
                                " entries but the chunk has " +
                                std::to_string(n) + " faults");

  const auto check_tags = [&context](const std::vector<uint64_t> &tags,
                                     uint32_t rows, const char *label) {
    if (tags.size() != rows)
      throw std::invalid_argument(std::string(context) + ": " + label +
                                  " has " + std::to_string(tags.size()) +
                                  " entries but the seam it names has " +
                                  std::to_string(rows) + " rows");
  };
  check_tags(in_tags, num_in_seam_rows(), "in_tags");
  check_tags(out_tags, num_out_seam_rows(), "out_tags");
} // end - extended_dem::validate()

uint32_t extended_dem::num_out_seam_rows() const {
  return out_syndrome.num_rows();
}

// ---------------------------------------------------------------------------
// Internal sparse-matrix helpers (extended_dem-specific)
// ---------------------------------------------------------------------------
// col_list, resolve_num_qubits, check_num_cols, check_per_qubit_size,
// padded_nested_csc, qubit_rate, and active_qubits come from
// cudaq::qec::detail (code_matrices.h) via `using namespace detail` above.

// [A | B]: same num_rows, concatenate column lists.
static sparse_binary_matrix hcat(const sparse_binary_matrix &a,
                                 const sparse_binary_matrix &b) {
  auto cols = a.to_nested_csc();
  const auto b_cols = b.to_nested_csc();
  cols.insert(cols.end(), b_cols.begin(), b_cols.end());
  const auto n_total = static_cast<std::size_t>(a.num_cols()) + b.num_cols();
  if (n_total > std::numeric_limits<uint32_t>::max())
    throw std::overflow_error("hcat: combined column count exceeds uint32_t");
  return sparse_binary_matrix::from_nested_csc(
      a.num_rows(), static_cast<uint32_t>(n_total), cols);
}

// [A ; B]: same num_cols, B row indices offset by a.num_rows().
// Precondition: b.num_cols() == a.num_cols(). All internal callers satisfy
// this by padding both operands to the same column count before calling.
static sparse_binary_matrix vstack(const sparse_binary_matrix &a,
                                   const sparse_binary_matrix &b) {
  if (b.num_cols() != a.num_cols())
    throw std::invalid_argument("vstack: column count mismatch (" +
                                std::to_string(a.num_cols()) + " vs " +
                                std::to_string(b.num_cols()) + ")");
  const uint32_t n_cols = a.num_cols();
  const uint32_t a_rows = a.num_rows();
  const auto total_rows = static_cast<std::size_t>(a_rows) + b.num_rows();
  if (total_rows > std::numeric_limits<uint32_t>::max())
    throw std::overflow_error("vstack: combined row count exceeds uint32_t");
  auto a_cols = a.to_nested_csc();
  const auto b_cols = b.to_nested_csc();
  col_list combined(n_cols);
  for (uint32_t c = 0; c < n_cols; ++c) {
    combined[c] = std::move(a_cols[c]);
    for (auto r : b_cols[c])
      combined[c].push_back(a_rows +
                            r); // safe: a_rows+r < total_rows ≤ UINT32_MAX
  }
  return sparse_binary_matrix::from_nested_csc(
      static_cast<uint32_t>(total_rows), n_cols, combined);
}

// Append empty columns on the right until total column count = n_total.
static sparse_binary_matrix pad_right(const sparse_binary_matrix &m,
                                      uint32_t n_total) {
  auto cols = m.to_nested_csc();
  cols.resize(n_total);
  return sparse_binary_matrix::from_nested_csc(m.num_rows(), n_total, cols);
}

// Prepend n_left empty columns, shifting existing columns to the right.
static sparse_binary_matrix pad_left(const sparse_binary_matrix &m,
                                     uint32_t n_left, uint32_t n_total) {
  const auto src = m.to_nested_csc();
  col_list cols(n_total);
  for (uint32_t c = 0; c < m.num_cols(); ++c)
    cols[n_left + c] = src[c];
  return sparse_binary_matrix::from_nested_csc(m.num_rows(), n_total, cols);
}

// All-zero sparse matrix.
static sparse_binary_matrix zero_matrix(uint32_t nrows, uint32_t ncols) {
  return sparse_binary_matrix::from_nested_csc(nrows, ncols, col_list(ncols));
}

// ---------------------------------------------------------------------------
// extended_dem_from_css_matrices
// ---------------------------------------------------------------------------

// Builds the one-round extended_dem directly from sparse column lists.
// No dense tensor intermediate: the per-qubit CSC index lists from
// hz/hx/lz/lx are assembled into syndrome and observable column lists
// in one pass, then wrapped into sparse_binary_matrix objects.
extended_dem extended_dem_from_css_matrices(const css_code_matrices &code,
                                            const css_noise_params &noise) {
  // Checked before the n == 0 early-out below so that a malformed rate is
  // reported even when the code matrices describe no qubits to apply it to.
  validate_noise_rates(noise);

  extended_dem result;
  const std::size_t n = resolve_num_qubits(code);
  if (n != 0) { // all code below depends on at least one qubit

    check_num_cols(code.hz, n, "hz");
    check_num_cols(code.hx, n, "hx");
    check_num_cols(code.lz, n, "lz");
    check_num_cols(code.lx, n, "lx");
    check_per_qubit_size(noise.px_per_qubit, n, "px_per_qubit");
    check_per_qubit_size(noise.py_per_qubit, n, "py_per_qubit");
    check_per_qubit_size(noise.pz_per_qubit, n, "pz_per_qubit");

    // Per-qubit CSC column lists: syn_csc[q] gives the syndrome rows
    // triggered by a fault on qubit q (Z-type rows first, then X-type).
    const auto hz_csc = padded_nested_csc(code.hz, n);
    const auto hx_csc = padded_nested_csc(code.hx, n);
    const auto lz_csc = padded_nested_csc(code.lz, n);
    const auto lx_csc = padded_nested_csc(code.lx, n);

    const uint32_t nz = static_cast<uint32_t>(code.hz.num_rows());
    const uint32_t nx = static_cast<uint32_t>(code.hx.num_rows());
    const uint32_t kz = static_cast<uint32_t>(code.lz.num_rows());
    const uint32_t kx = static_cast<uint32_t>(code.lx.num_rows());
    // Seam/observable widths are uint32_t matrix dimensions; reject sums that
    // would wrap before they become wrong sparse shapes.
    if (static_cast<uint64_t>(nz) + nx > std::numeric_limits<uint32_t>::max())
      throw std::invalid_argument(
          "extended_dem_from_css_matrices: hz.num_rows() + hx.num_rows() "
          "exceeds uint32_t max");
    if (static_cast<uint64_t>(kz) + kx > std::numeric_limits<uint32_t>::max())
      throw std::invalid_argument(
          "extended_dem_from_css_matrices: lz.num_rows() + lx.num_rows() "
          "exceeds uint32_t max");
    const uint32_t d = nz + nx; // seam rows (total checks per round)
    const uint32_t k = kz + kx; // observable rows

    // pm_per_check must have length d when non-empty.
    check_rate_vector_size(noise.pm_per_check, static_cast<std::size_t>(d),
                           "pm_per_check", "n_checks");

    const auto x_qubits = active_qubits(noise.px, noise.px_per_qubit, n);
    const auto z_qubits = active_qubits(noise.pz, noise.pz_per_qubit, n);
    const auto y_qubits = active_qubits(noise.py, noise.py_per_qubit, n);
    const auto m_checks = active_checks(noise.pm, noise.pm_per_check,
                                        static_cast<std::size_t>(d));
    const std::size_t n_faults_sz =
        x_qubits.size() + z_qubits.size() + y_qubits.size() + m_checks.size();
    if (n_faults_sz > std::numeric_limits<uint32_t>::max())
      throw std::invalid_argument(
          "extended_dem_from_css_matrices: active fault count exceeds "
          "uint32_t max");
    const uint32_t n_faults = static_cast<uint32_t>(n_faults_sz);

    if (n_faults != 0) { // no columns to build when all rates are zero

      // Build syndrome and observable column lists directly. For a one-round
      // chunk, syndrome rows are the raw check activations (not differenced):
      //   Z-type rows (0..nz-1): from hz[:,q]
      //   X-type rows (nz..d-1): from hx[:,q] (offset by nz)
      col_list syn_cols(n_faults);
      col_list obs_cols(n_faults);
      std::vector<double> priors;
      priors.reserve(n_faults);

      uint32_t col = 0;

      // X faults trigger Z-type syndrome rows and Z-type observable rows.
      for (const std::size_t q : x_qubits) {
        for (auto r : hz_csc[q])
          syn_cols[col].push_back(r);
        for (auto r : lz_csc[q])
          obs_cols[col].push_back(r);
        priors.push_back(qubit_rate(noise.px, noise.px_per_qubit, q));
        ++col;
      }

      // Z faults trigger X-type syndrome rows (offset nz) and X-type obs.
      for (const std::size_t q : z_qubits) {
        for (auto r : hx_csc[q])
          syn_cols[col].push_back(nz + r);
        for (auto r : lx_csc[q])
          obs_cols[col].push_back(kz + r);
        priors.push_back(qubit_rate(noise.pz, noise.pz_per_qubit, q));
        ++col;
      }

      // Y faults trigger both syndrome bands and both observable bands.
      for (const std::size_t q : y_qubits) {
        for (auto r : hz_csc[q])
          syn_cols[col].push_back(r);
        for (auto r : hx_csc[q])
          syn_cols[col].push_back(nz + r);
        for (auto r : lz_csc[q])
          obs_cols[col].push_back(r);
        for (auto r : lx_csc[q])
          obs_cols[col].push_back(kz + r);
        priors.push_back(qubit_rate(noise.py, noise.py_per_qubit, q));
        ++col;
      }

      // Measurement error faults: check k fires only row k in the syndrome
      // (one check misfires), no logical effect. in_syndrome == out_syndrome as
      // always for a one-round chunk: syndrome[r][k] participates in both
      // detector[r] (via in_syndrome) and detector[r+1] (via out_syndrome).
      for (const std::size_t ck : m_checks) {
        syn_cols[col].push_back(
            static_cast<sparse_binary_matrix::index_type>(ck));
        // obs_cols[col] stays empty — measurement errors don't flip
        // observables.
        priors.push_back(check_rate(noise.pm, noise.pm_per_check, ck));
        ++col;
      }

      // in_syndrome == out_syndrome: the same syndrome participates in both
      // the left seam (detector[r]) and the right seam (detector[r+1]).
      result.in_syndrome =
          sparse_binary_matrix::from_nested_csc(d, n_faults, syn_cols);
      result.out_syndrome = result.in_syndrome;

      result.observables =
          sparse_binary_matrix::from_nested_csc(k, n_faults, obs_cols);

      result.interior = zero_matrix(0, n_faults);
      result.fault_priors = std::move(priors);

      // Sequential check IDs: tag k == k for all checks.
      result.in_tags.resize(d);
      std::iota(result.in_tags.begin(), result.in_tags.end(), uint64_t{0});
      result.out_tags = result.in_tags;

    } // end - if (n_faults != 0)
  } // end - if (n != 0)

  return result;
} // end - extended_dem_from_css_matrices()

// ---------------------------------------------------------------------------
// dem_chunk_spec / dem_chunks_spec
// ---------------------------------------------------------------------------

// Number of rows a -1-terminated index list describes. validate() has already
// established that the list ends with a terminator, so this is exact.
static uint32_t sparse_row_count(const std::vector<std::int64_t> &rows) {
  return static_cast<uint32_t>(std::count(rows.begin(), rows.end(), -1));
}

// Split a -1-terminated index list into a sparse matrix with num_faults
// columns. Empty input yields a 0-row matrix that still carries the column
// count, which is what keeps extended_dem::num_faults() meaningful for an init
// phase whose in_syndrome has no rows at all.
static sparse_binary_matrix
sparse_from_terminated_rows(const std::vector<std::int64_t> &rows,
                            uint64_t num_faults) {
  std::vector<std::vector<sparse_binary_matrix::index_type>> nested;
  std::vector<sparse_binary_matrix::index_type> current;
  for (auto value : rows) {
    if (value == -1) {
      nested.push_back(current);
      current.clear();
      continue;
    }
    current.push_back(static_cast<sparse_binary_matrix::index_type>(value));
  }
  return sparse_binary_matrix::from_nested_csr(
      static_cast<sparse_binary_matrix::index_type>(nested.size()),
      static_cast<sparse_binary_matrix::index_type>(num_faults), nested);
}

static void validate_index_list(const std::vector<std::int64_t> &rows,
                                uint64_t num_faults, const std::string &context,
                                const std::string &field) {
  if (rows.empty())
    return;
  if (rows.back() != -1)
    throw std::invalid_argument(context + "." + field +
                                " must end with a -1 row terminator");
  for (auto value : rows) {
    if (value == -1)
      continue;
    if (value < 0 || static_cast<uint64_t>(value) >= num_faults)
      throw std::invalid_argument(
          context + "." + field + " index " + std::to_string(value) +
          " is out of range for num_faults " + std::to_string(num_faults));
  }
}

bool dem_chunk_spec::is_empty() const {
  return num_faults == 0 && H_in_sparse.empty() && H_mid_sparse.empty() &&
         H_out_sparse.empty() && O_sparse.empty() && error_rates.empty();
}

void dem_chunk_spec::validate(const std::string &context) const {
  if (num_faults == 0)
    throw std::invalid_argument(context + ".num_faults must be positive");
  // sparse_binary_matrix columns are indexed with uint32_t; a wider count
  // would wrap in sparse_from_terminated_rows and silently mis-size the chunk.
  if (num_faults > std::numeric_limits<sparse_binary_matrix::index_type>::max())
    throw std::invalid_argument(
        context + ".num_faults (" + std::to_string(num_faults) +
        ") exceeds uint32_t max (" +
        std::to_string(
            std::numeric_limits<sparse_binary_matrix::index_type>::max()) +
        ")");
  if (error_rates.size() != num_faults)
    throw std::invalid_argument(
        context + ".error_rates has " + std::to_string(error_rates.size()) +
        " entries but num_faults is " + std::to_string(num_faults));
  for (std::size_t i = 0; i < error_rates.size(); ++i)
    if (!(error_rates[i] >= 0.0 && error_rates[i] <= 1.0))
      throw std::invalid_argument(context + ".error_rates[" +
                                  std::to_string(i) +
                                  "] = " + std::to_string(error_rates[i]) +
                                  " is not a probability in [0, 1]");

  validate_index_list(H_in_sparse, num_faults, context, "H_in_sparse");
  validate_index_list(H_mid_sparse, num_faults, context, "H_mid_sparse");
  validate_index_list(H_out_sparse, num_faults, context, "H_out_sparse");
  validate_index_list(O_sparse, num_faults, context, "O_sparse");
}

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

  // init starts the stream, so nothing feeds its incoming seam; final ends it,
  // so nothing consumes its outgoing seam. init's own round-0 detectors belong
  // in H_mid_sparse, compared against the zero initial state.
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

// Build an extended_dem from a spec without validating. The caller must have
// already called spec.validate() or dem_chunks_spec::validate().
static extended_dem dem_chunk_from_spec_impl(const dem_chunk_spec &spec) {
  extended_dem dem;
  dem.interior =
      sparse_from_terminated_rows(spec.H_mid_sparse, spec.num_faults);
  dem.observables = sparse_from_terminated_rows(spec.O_sparse, spec.num_faults);
  dem.in_syndrome =
      sparse_from_terminated_rows(spec.H_in_sparse, spec.num_faults);
  dem.out_syndrome =
      sparse_from_terminated_rows(spec.H_out_sparse, spec.num_faults);
  dem.fault_priors = spec.error_rates;

  // Sequential tags give the k-th check the same identity in every phase, so
  // adjacent phases satisfy dem_stitch's a.out_tags == b.in_tags requirement.
  dem.in_tags.resize(dem.num_in_seam_rows());
  std::iota(dem.in_tags.begin(), dem.in_tags.end(), uint64_t{0});
  dem.out_tags.resize(dem.num_out_seam_rows());
  std::iota(dem.out_tags.begin(), dem.out_tags.end(), uint64_t{0});

  return dem;
}

extended_dem dem_chunk_from_spec(const dem_chunk_spec &spec,
                                 const std::string &context) {
  spec.validate(context);
  return dem_chunk_from_spec_impl(spec);
} // end - dem_chunk_from_spec()

std::vector<extended_dem> dem_chunks_from_spec(const dem_chunks_spec &spec,
                                               std::size_t num_rounds) {
  if (num_rounds < 2)
    throw std::invalid_argument(
        "dem_chunks_from_spec: num_rounds must be at least 2 (init and final), "
        "got " +
        std::to_string(num_rounds));
  spec.validate();

  const std::size_t bulk_repeats = num_rounds - 2;
  if (bulk_repeats > 0 && !spec.has_bulk())
    throw std::invalid_argument("dem_chunks_from_spec: num_rounds " +
                                std::to_string(num_rounds) + " needs " +
                                std::to_string(bulk_repeats) +
                                " bulk rounds but no bulk phase was supplied");

  // spec.validate() already checked each phase; use the impl variant to avoid
  // re-running per-phase validation for every call below.
  std::vector<extended_dem> chunks;
  chunks.reserve(num_rounds);
  chunks.push_back(dem_chunk_from_spec_impl(spec.init));
  if (bulk_repeats > 0) {
    const auto bulk = dem_chunk_from_spec_impl(spec.bulk);
    chunks.insert(chunks.end(), bulk_repeats, bulk);
  }
  chunks.push_back(dem_chunk_from_spec_impl(spec.final));
  return chunks;
} // end - dem_chunks_from_spec()

// ---------------------------------------------------------------------------
// dem_stitch
// ---------------------------------------------------------------------------

extended_dem dem_stitch(const extended_dem &a, const extended_dem &b) {
  a.validate("dem_stitch: left chunk");
  b.validate("dem_stitch: right chunk");
  if (a.num_observables() != b.num_observables())
    throw std::invalid_argument("dem_stitch: observable count mismatch (" +
                                std::to_string(a.num_observables()) + " vs " +
                                std::to_string(b.num_observables()) + ")");
  // Only the contracted seam has to line up: a's outgoing side against b's
  // incoming side. Comparing a.num_seam_rows() to b.num_seam_rows() instead
  // would test a's *incoming* width, which is not part of this seam at all and
  // is legitimately zero for an init phase chunk.
  if (a.num_out_seam_rows() != b.num_in_seam_rows())
    throw std::invalid_argument(
        "dem_stitch: seam row count mismatch (a out_syndrome " +
        std::to_string(a.num_out_seam_rows()) + " vs b in_syndrome " +
        std::to_string(b.num_in_seam_rows()) + ")");
  if (a.out_tags != b.in_tags)
    throw std::invalid_argument(
        "dem_stitch: a.out_tags != b.in_tags — check ordering is incompatible");

  const auto n_A_sz = static_cast<std::size_t>(a.num_faults());
  const auto n_total_sz = n_A_sz + b.num_faults();
  if (n_total_sz > std::numeric_limits<uint32_t>::max())
    throw std::overflow_error("dem_stitch: combined fault count (" +
                              std::to_string(n_total_sz) +
                              ") exceeds uint32_t max");
  const uint32_t n_A = static_cast<uint32_t>(n_A_sz);
  const uint32_t n_total = static_cast<uint32_t>(n_total_sz);

  extended_dem r;

  // interior = [ a.interior | 0   ]            (a's existing interior)
  //            [ a.out_syndrome | b.in_syndrome ]  (new seam row)
  //            [ 0          | b.interior ]     (b's existing interior)
  // Faults from A and B are disjoint half-columns; each contributes
  // independently to the same seam-detector row via its own half.
  //
  // The new seam sits between the two interiors, which is what keeps interior
  // rows in ascending round order however the caller associates its stitches:
  // a's rounds precede the seam they end at, which precedes b's rounds. Append
  // it after b.interior instead and a tree fold -- dem_stitch(dem_stitch(a, b),
  // dem_stitch(c, d)) -- would file the b|c seam after c's own rounds, leaving
  // dem_close_all() emitting detector rows out of round order.
  r.interior = vstack(vstack(pad_right(a.interior, n_total),
                             hcat(a.out_syndrome, b.in_syndrome)),
                      pad_left(b.interior, n_A, n_total));

  // Observables and seam boundaries: simple horizontal concat / padding.
  r.observables = hcat(a.observables, b.observables);
  r.in_syndrome = pad_right(a.in_syndrome, n_total);
  r.out_syndrome = pad_left(b.out_syndrome, n_A, n_total);

  r.fault_priors = a.fault_priors;
  r.fault_priors.insert(r.fault_priors.end(), b.fault_priors.begin(),
                        b.fault_priors.end());
  r.in_tags = a.in_tags;
  r.out_tags = b.out_tags;

  return r;
} // end - dem_stitch()

// ---------------------------------------------------------------------------
// dem_stitch_all
// ---------------------------------------------------------------------------

extended_dem dem_stitch_all(const std::vector<extended_dem> &dem_chunks) {
  if (dem_chunks.empty())
    throw std::invalid_argument("dem_stitch_all: dem_chunks must be non-empty");
  // dem_stitch validates both of its operands, which covers every chunk
  // except when there is only one and no stitching happens at all.
  if (dem_chunks.size() == 1)
    dem_chunks[0].validate("dem_stitch_all: chunk 0");
  extended_dem acc = dem_chunks[0];
  for (std::size_t i = 1; i < dem_chunks.size(); ++i)
    acc = dem_stitch(acc, dem_chunks[i]);
  return acc;
}

// ---------------------------------------------------------------------------
// dem_close
// ---------------------------------------------------------------------------

// Scatter the non-zero entries of src into dst at (row_offset, col_offset).
// write_sparse_rows is the col_offset=0 special case.
static void write_sparse_block(cudaqx::tensor<uint8_t> &dst,
                               const sparse_binary_matrix &src,
                               std::size_t row_offset, std::size_t col_offset) {
  const auto cols = src.to_nested_csc();
  const uint32_t nc = src.num_cols();
  for (uint32_t c = 0; c < nc; ++c)
    for (auto r : cols[c])
      dst.at({row_offset + r, col_offset + c}) ^= 1;
}

static void write_sparse_rows(cudaqx::tensor<uint8_t> &dst,
                              const sparse_binary_matrix &src,
                              uint32_t row_offset) {
  write_sparse_block(dst, src, row_offset, 0);
}

// dem_close() must emit a detector_error_model whose matrix fields are
// tensor<uint8_t> — that is the existing public API for all decoders.
// The sparse → dense conversion here is the only use of dense tensors
// in the extended_dem layer.
//
// out_syndrome is not written: closing ends the experiment, so there is no
// next round for that seam to become a detector against. Callers that need a
// final-boundary detector must put it in in_syndrome or interior first.
detector_error_model dem_close(const extended_dem &dem) {
  dem.validate("dem_close");
  const uint32_t n_faults = dem.num_faults();
  const uint32_t n_seam = dem.num_in_seam_rows();
  const uint32_t n_interior = dem.num_interior();
  const uint32_t n_obs = dem.num_observables();
  // Row order: in_syndrome first (detector[0] = syndrome[0] vs. zero),
  // then interior seam rows (detectors for subsequent round differences).
  const uint32_t n_det = n_seam + n_interior;

  detector_error_model result;
  result.detector_error_matrix = cudaqx::tensor<uint8_t>({n_det, n_faults});
  result.observables_flips_matrix = cudaqx::tensor<uint8_t>({n_obs, n_faults});
  result.error_rates = dem.fault_priors;

  write_sparse_rows(result.detector_error_matrix, dem.in_syndrome, 0);
  write_sparse_rows(result.detector_error_matrix, dem.interior, n_seam);
  write_sparse_rows(result.observables_flips_matrix, dem.observables, 0);

  return result;
} // end - dem_close()

// ---------------------------------------------------------------------------
// Streaming decoder integration utilities
// ---------------------------------------------------------------------------

// The width of a chunk's seam, taken from whichever side it has. A phase
// decomposition leaves the first chunk's incoming seam empty (nothing precedes
// the first round) and the last chunk's outgoing seam empty (nothing follows
// the last), so neither side can be read blindly.
static uint32_t chunk_seam_width(const extended_dem &dem_chunk) {
  return dem_chunk.num_in_seam_rows() != 0 ? dem_chunk.num_in_seam_rows()
                                           : dem_chunk.num_out_seam_rows();
}

// Adjacent chunks contract a's outgoing seam against b's incoming one, the same
// rule dem_stitch applies. Comparing num_seam_rows() across chunks instead
// would compare two incoming widths, neither of which is the seam being
// contracted, and so would reject a phase decomposition.
//
// Tags are checked alongside widths because matching row counts alone do not
// make a seam contractible: out_tags[i] and in_tags[i] have to name the same
// physical check, or the contraction pairs up unrelated rows. Skipping this
// here would leave dem_close_all() accepting sequences dem_stitch() rejects.
static void
require_contractible_seams(const std::vector<extended_dem> &dem_chunks,
                           const char *fn) {
  for (std::size_t i = 0; i + 1 < dem_chunks.size(); ++i) {
    if (dem_chunks[i].num_out_seam_rows() !=
        dem_chunks[i + 1].num_in_seam_rows())
      throw std::invalid_argument(
          std::string(fn) + ": chunk " + std::to_string(i) +
          " out_syndrome has " +
          std::to_string(dem_chunks[i].num_out_seam_rows()) +
          " rows but chunk " + std::to_string(i + 1) + " in_syndrome has " +
          std::to_string(dem_chunks[i + 1].num_in_seam_rows()));
    if (dem_chunks[i].out_tags != dem_chunks[i + 1].in_tags)
      throw std::invalid_argument(
          std::string(fn) + ": chunk " + std::to_string(i) +
          " out_tags do not match chunk " + std::to_string(i + 1) +
          " in_tags, so their seam rows describe different checks");
  } // end - for(i)
}

// Shared validation for the round-indexed maps: non-empty, seams that contract,
// one seam width shared by every round, and interior rows that come in whole
// rounds. These maps place detector r*d+k, so a chunk carrying a different
// width would silently shift every later round's detectors.
static void
validate_dem_chunk_sequence(const std::vector<extended_dem> &dem_chunks,
                            const char *fn) {
  if (dem_chunks.empty())
    throw std::invalid_argument(std::string(fn) +
                                ": dem_chunks must be non-empty");
  require_contractible_seams(dem_chunks, fn);

  const uint32_t d = chunk_seam_width(dem_chunks[0]);
  if (d == 0)
    throw std::invalid_argument(std::string(fn) +
                                ": chunk 0 has no seam rows on either side, so "
                                "its rounds cannot be counted");
  for (std::size_t i = 0; i < dem_chunks.size(); ++i) {
    // An absent seam is the open end of the sequence; a present one has to
    // agree with the rest, otherwise there is no single round width.
    for (const auto [width, side] :
         {std::pair{dem_chunks[i].num_in_seam_rows(), "in_syndrome"},
          std::pair{dem_chunks[i].num_out_seam_rows(), "out_syndrome"}})
      if (width != 0 && width != d)
        throw std::invalid_argument(
            std::string(fn) + ": chunk " + std::to_string(i) + " " + side +
            " has " + std::to_string(width) +
            " rows, not the sequence's round width " + std::to_string(d));
    if (dem_chunks[i].num_interior() % d != 0)
      throw std::invalid_argument(
          std::string(fn) + ": chunk " + std::to_string(i) + " has " +
          std::to_string(dem_chunks[i].num_interior()) +
          " interior rows, which is not a whole number of rounds of " +
          std::to_string(d) + " checks");
  }
}

// dem_chunk_rounds: how many rounds one chunk spans. Stitching R one-round
// chunks contracts R-1 seams into interior rows, so the interior row count
// gives the rounds after the leading one.
//
// The leading round is the incoming seam band, which an init phase chunk does
// not have -- it carries round 0 in its interior instead, having nothing before
// it to compare against. So the band counts as a round only when it is there,
// and a chunk with interior rows and no incoming seam (interior == d) is one
// round, not two.
uint32_t dem_chunk_rounds(const extended_dem &dem_chunk) {
  const uint32_t d = chunk_seam_width(dem_chunk);
  if (d == 0)
    throw std::invalid_argument("dem_chunk_rounds: chunk has no seam rows on "
                                "either side, so its rounds cannot be counted");
  if (dem_chunk.num_interior() % d != 0)
    throw std::invalid_argument(
        "dem_chunk_rounds: chunk has " +
        std::to_string(dem_chunk.num_interior()) +
        " interior rows, which is not a whole number of rounds of " +
        std::to_string(d) + " checks");
  const uint32_t leading = dem_chunk.num_in_seam_rows() != 0 ? 1 : 0;
  return leading + dem_chunk.num_interior() / d;
}

// dem_chunks_to_rounds: total rounds a sequence of chunks describes.
std::size_t dem_chunks_to_rounds(const std::vector<extended_dem> &dem_chunks) {
  validate_dem_chunk_sequence(dem_chunks, "dem_chunks_to_rounds");
  std::size_t rounds = 0;
  for (const auto &dem_chunk : dem_chunks)
    rounds += dem_chunk_rounds(dem_chunk);
  return rounds;
}

// dem_chunks_to_detector_round: detector r*d+k belongs to round r.
// Purely dimension-based — no CSC traversal needed.
std::vector<int32_t>
dem_chunks_to_detector_round(const std::vector<extended_dem> &dem_chunks) {
  validate_dem_chunk_sequence(dem_chunks, "dem_chunks_to_detector_round");

  const uint32_t d = chunk_seam_width(dem_chunks[0]);
  std::vector<int32_t> result;
  int32_t round = 0;

  result.reserve(dem_chunks.size() * static_cast<std::size_t>(d));
  for (const auto &dem_chunk : dem_chunks) {
    // A multi-round chunk carries its own rounds in ascending order, so rounds
    // simply keep counting up across chunk boundaries.
    const uint32_t rounds = dem_chunk_rounds(dem_chunk);
    for (uint32_t r = 0; r < rounds; ++r, ++round)
      result.insert(result.end(), d, round);
  }
  return result;
}

// dem_chunks_to_d_sparse: D_sparse[det_id] = measurement bit positions that
// XOR-combine to form that detector.
//   det k       (r=0): {k}               — compared to zero initial state
//   det r*d+k (r>0): {(r-1)*d+k, r*d+k} — consecutive syndrome XOR
std::vector<std::vector<uint32_t>>
dem_chunks_to_d_sparse(const std::vector<extended_dem> &dem_chunks) {
  validate_dem_chunk_sequence(dem_chunks, "dem_chunks_to_d_sparse");

  const uint32_t d = chunk_seam_width(dem_chunks[0]);
  // Rounds, not chunks: a chunk may carry several of them.
  const std::size_t T = dem_chunks_to_rounds(dem_chunks);
  const std::size_t n_det = T * static_cast<std::size_t>(d);

  std::vector<std::vector<uint32_t>> d_sparse(n_det);

  // Largest measurement bit index is (T-1)*d + (d-1); check it fits uint32_t.
  if (T > 0) {
    const auto max_bit = static_cast<std::size_t>(T - 1) * d + (d - 1);
    if (max_bit > std::numeric_limits<uint32_t>::max())
      throw std::overflow_error(
          "dem_chunks_to_d_sparse: measurement bit index exceeds uint32_t max");
  }

  for (std::size_t r = 0; r < T; ++r) {
    for (uint32_t k = 0; k < d; ++k) {
      const std::size_t det = r * d + k;
      if (r == 0) {
        // Round-0 detector: single measurement at position k.
        d_sparse[det] = {k};
      } else {
        // Later detector: XOR of previous and current round's measurement.
        d_sparse[det] = {static_cast<uint32_t>((r - 1) * d + k),
                         static_cast<uint32_t>(r * d + k)};
      }
    }
  }
  return d_sparse;
}

// dem_chunks_to_o_sparse: O_sparse[obs_id] = global fault column indices that
// flip observable obs_id. Built by transposing each chunk's observables
// sparse matrix with the appropriate column offset applied.
std::vector<std::vector<uint32_t>>
dem_chunks_to_o_sparse(const std::vector<extended_dem> &dem_chunks) {
  if (dem_chunks.empty())
    throw std::invalid_argument(
        "dem_chunks_to_o_sparse: dem_chunks must be non-empty");

  for (std::size_t i = 0; i < dem_chunks.size(); ++i)
    dem_chunks[i].validate(
        ("dem_chunks_to_o_sparse: chunk " + std::to_string(i)).c_str());

  const uint32_t k_obs = dem_chunks[0].num_observables();

  // Validate consistent observable count across chunks.
  for (std::size_t i = 1; i < dem_chunks.size(); ++i)
    if (dem_chunks[i].num_observables() != k_obs)
      throw std::invalid_argument(
          "dem_chunks_to_o_sparse: chunk " + std::to_string(i) + " has " +
          std::to_string(dem_chunks[i].num_observables()) +
          " observables but chunk 0 has " + std::to_string(k_obs));

  std::vector<std::vector<uint32_t>> o_sparse(k_obs);

  std::size_t col_off = 0;
  for (const auto &c : dem_chunks) {
    const uint32_t nc = c.num_faults();
    // Check that the highest global column index in this chunk fits uint32_t.
    if (nc > 0 && col_off + nc - 1 > std::numeric_limits<uint32_t>::max())
      throw std::overflow_error("dem_chunks_to_o_sparse: global fault column "
                                "index exceeds uint32_t max");
    // obs is stored as [k_obs × n_faults] in CSC. to_nested_csc() gives
    // outer index = fault column, inner = observable rows that flip.
    // We need the transpose: observable → fault columns. validate() above
    // already required observables.num_cols() == nc.
    const auto obs_csc = c.observables.to_nested_csc();
    for (uint32_t fc = 0; fc < nc; ++fc)
      for (auto obs_row : obs_csc[fc])
        o_sparse[obs_row].push_back(static_cast<uint32_t>(col_off + fc));
    col_off += nc;
  }
  return o_sparse;
}

// ---------------------------------------------------------------------------
// dem_close_all
// ---------------------------------------------------------------------------

// dem_close_all builds the closed DEM in a single forward pass over the chunk
// list. Each fault column is written at most twice: once into the detector
// band for the seam on its left (via in_syndrome) and once into the seam on
// its right (via out_syndrome). The last chunk's out_syndrome is dropped on
// purpose — same terminal-boundary rule as dem_close() — so a detector that
// should appear in the closed DEM must already be in some in_syndrome or
// interior. For single-round chunks (num_interior==0), the output is
// byte-identical to dem_close(dem_stitch_all(dem_chunks)).
//
// Detector row layout (round order, regardless of chunk granularity):
//   rows 0..d-1:             dem_chunks[0].in_syndrome   (detector[0])
//   rows d..d+n_int[0]-1:    dem_chunks[0].interior
//   rows d+n_int[0]...:      seam between chunk 0 and chunk 1
//   rows ...:                 dem_chunks[1].interior
//   ...
// where d = num_seam_rows() and n_int[i] = dem_chunks[i].num_interior().
detector_error_model
dem_close_all(const std::vector<extended_dem> &dem_chunks) {
  if (dem_chunks.empty())
    throw std::invalid_argument("dem_close_all: dem_chunks must be non-empty");

  for (std::size_t i = 0; i < dem_chunks.size(); ++i)
    dem_chunks[i].validate(
        ("dem_close_all: chunk " + std::to_string(i)).c_str());

  const uint32_t k = dem_chunks[0].num_observables();
  const std::size_t T = dem_chunks.size();

  require_contractible_seams(dem_chunks, "dem_close_all");
  for (std::size_t i = 1; i < T; ++i)
    if (dem_chunks[i].num_observables() != k)
      throw std::invalid_argument(
          "dem_close_all: chunk " + std::to_string(i) + " has " +
          std::to_string(dem_chunks[i].num_observables()) +
          " observables but chunk 0 has " + std::to_string(k));

  // Detector rows open with chunk 0's incoming seam, which is round 0 compared
  // against the initial state. An init phase chunk has no such band and carries
  // round 0 in its interior instead, so this is zero for a phase sequence.
  const std::size_t lead = dem_chunks[0].num_in_seam_rows();

  // Cumulative fault column offsets.
  std::vector<std::size_t> col_off(T + 1, 0);
  for (std::size_t i = 0; i < T; ++i)
    col_off[i + 1] =
        col_off[i] + static_cast<std::size_t>(dem_chunks[i].num_faults());

  const std::size_t n_faults = col_off.back();
  detector_error_model result;

  if (n_faults != 0) {
    std::size_t total_interior = 0;
    for (const auto &c : dem_chunks)
      total_interior += c.num_interior();
    // One seam per adjacent pair, each as wide as the sides it contracts. A
    // uniform sequence makes every term d, recovering lead + interior + (T-1)d.
    std::size_t total_seam = 0;
    for (std::size_t i = 0; i + 1 < T; ++i)
      total_seam += dem_chunks[i].num_out_seam_rows();
    const std::size_t n_detectors = lead             // in_syndrome of chunk 0
                                    + total_interior // across all chunks
                                    + total_seam;

    result.detector_error_matrix =
        cudaqx::tensor<uint8_t>({n_detectors, n_faults});
    result.observables_flips_matrix =
        cudaqx::tensor<uint8_t>({static_cast<std::size_t>(k), n_faults});
    result.error_rates.reserve(n_faults);

    // detector[0]: in_syndrome of chunk 0, at its own column range.
    write_sparse_block(result.detector_error_matrix, dem_chunks[0].in_syndrome,
                       0, col_off[0]);

    std::size_t row_cursor = lead;

    for (std::size_t i = 0; i < T; ++i) {
      // Interior rows already closed within chunk i (seams inside this chunk).
      if (dem_chunks[i].num_interior() > 0) {
        write_sparse_block(result.detector_error_matrix, dem_chunks[i].interior,
                           row_cursor, col_off[i]);
        row_cursor += dem_chunks[i].num_interior();
      }

      // Seam between chunk i and chunk i+1: left side from i's out_syndrome,
      // right side from (i+1)'s in_syndrome. Dropped for the last chunk.
      if (i + 1 < T) {
        write_sparse_block(result.detector_error_matrix,
                           dem_chunks[i].out_syndrome, row_cursor, col_off[i]);
        write_sparse_block(result.detector_error_matrix,
                           dem_chunks[i + 1].in_syndrome, row_cursor,
                           col_off[i + 1]);
        row_cursor += dem_chunks[i].num_out_seam_rows();
      }

      // Observables: chunk i's faults go into its own column range.
      write_sparse_block(result.observables_flips_matrix,
                         dem_chunks[i].observables, 0, col_off[i]);
    } // end - for(i)

    for (const auto &c : dem_chunks)
      result.error_rates.insert(result.error_rates.end(),
                                c.fault_priors.begin(), c.fault_priors.end());

  } // end - if (n_faults != 0)

  return result;
} // end - dem_close_all()

sparse_binary_matrix
dem_chunks_to_pcm(const std::vector<extended_dem> &dem_chunks) {
  return sparse_binary_matrix(dem_close_all(dem_chunks).detector_error_matrix)
      .canonicalize()
      .to_csc();
} // end - dem_chunks_to_pcm()

// ---------------------------------------------------------------------------
// dem_merge_duplicate_columns — merge fault columns with identical row support
// ---------------------------------------------------------------------------

namespace {

// Per-column row lists, sorted ascending with duplicate rows GF(2)-collapsed.
// sparse_binary_matrix stores index lists as handed to it, so a source may list
// a row twice in one column; over GF(2) that is the same column as one omitting
// the row, and the two must produce the same support key. canonicalize()
// collapses per compressed group, so the matrix has to be CSC for those groups
// to be columns.
col_list canonical_columns(const sparse_binary_matrix &mat) {
  if (mat.layout() == sparse_binary_matrix_layout::csc)
    return mat.canonicalize().to_nested_csc();
  return mat.to_csc().canonicalize().to_nested_csc();
}

// The four blocks of a DEM as per-column row lists over one shared row space:
//   [0, n_int)          interior
//   [n_int, n_int+k)    observables
//   [..., +d)           in_syndrome
//   [..., +d)           out_syndrome
struct dem_column_view {
  col_list interior, observables, in_syndrome, out_syndrome;
  uint32_t obs_base, in_base, out_base;

  explicit dem_column_view(const extended_dem &dem)
      : interior(canonical_columns(dem.interior)),
        observables(canonical_columns(dem.observables)),
        in_syndrome(canonical_columns(dem.in_syndrome)),
        out_syndrome(canonical_columns(dem.out_syndrome)),
        obs_base(dem.num_interior()), in_base(obs_base + dem.num_observables()),
        out_base(in_base + dem.num_seam_rows()) {}

  // Row support of fault column j in the shared row space. Already ascending:
  // every block's list is sorted and the bases increase block by block.
  std::vector<uint32_t> support(std::size_t j) const {
    std::vector<uint32_t> sup;
    sup.reserve(interior[j].size() + observables[j].size() +
                in_syndrome[j].size() + out_syndrome[j].size());
    for (auto r : interior[j])
      sup.push_back(static_cast<uint32_t>(r));
    for (auto r : observables[j])
      sup.push_back(obs_base + static_cast<uint32_t>(r));
    for (auto r : in_syndrome[j])
      sup.push_back(in_base + static_cast<uint32_t>(r));
    for (auto r : out_syndrome[j])
      sup.push_back(out_base + static_cast<uint32_t>(r));
    return sup;
  }
};

} // namespace

extended_dem dem_merge_duplicate_columns(const extended_dem &dem,
                                         prior_combine_mode mode) {
  dem.validate("dem_merge_duplicate_columns");
  const std::size_t n = dem.num_faults();
  if (n == 0)
    return dem;

  const dem_column_view cols(dem);

  // Map support → the group of columns sharing it. std::map gives free lex
  // ordering so the output matches the Python reference.
  struct column_group {
    std::size_t representative;
    // prod(1-2p_i) for XOR (or_combine) mode, sum(p_i) for SUM mode.
    double accumulator;
    std::size_t members;
  };
  std::map<std::vector<uint32_t>, column_group> sup_map;

  for (std::size_t j = 0; j < n; ++j) {
    auto sup = cols.support(j);
    const double p = dem.fault_priors[j];
    auto it = sup_map.find(sup);
    if (it == sup_map.end()) {
      // XOR: track prod(1-2p). SUM: track sum(p).
      const double init =
          (mode == prior_combine_mode::or_combine) ? (1.0 - 2.0 * p) : p;
      sup_map.emplace(std::move(sup), column_group{j, init, 1});
    } else {
      if (mode == prior_combine_mode::or_combine)
        it->second.accumulator *= (1.0 - 2.0 * p);
      else
        it->second.accumulator += p;
      ++it->second.members;
    }
  }

  // Build new column lists and priors from the merged unique supports.
  const auto new_n = static_cast<uint32_t>(sup_map.size());
  col_list new_int(new_n), new_obs(new_n), new_ins(new_n), new_out(new_n);
  std::vector<double> new_priors;
  new_priors.reserve(new_n);

  std::size_t k = 0;
  for (const auto &[sup, group] : sup_map) {
    const std::size_t rep = group.representative;
    new_int[k] = cols.interior[rep];
    new_obs[k] = cols.observables[rep];
    new_ins[k] = cols.in_syndrome[rep];
    new_out[k] = cols.out_syndrome[rep];
    // A column with nothing to merge keeps its prior bit for bit:
    // 0.5*(1-(1-2p)) is not always p in floating point, and merging a DEM
    // whose columns are already unique must not perturb its weights.
    //
    // sum_combine is a small-p linear approximation, so its sum can leave the
    // unit interval where or_combine (XOR) cannot. Clamp it: a prior above 1
    // is not a probability, and every consumer of fault_priors treats it as
    // one.
    const double fp = group.members == 1
                          ? dem.fault_priors[rep]
                          : (mode == prior_combine_mode::or_combine
                                 ? 0.5 * (1.0 - group.accumulator)
                                 : std::min(1.0, group.accumulator));
    new_priors.push_back(fp);
    ++k;
  }

  extended_dem result;
  result.interior =
      sparse_binary_matrix::from_nested_csc(dem.num_interior(), new_n, new_int);
  result.observables = sparse_binary_matrix::from_nested_csc(
      dem.num_observables(), new_n, new_obs);
  result.in_syndrome = sparse_binary_matrix::from_nested_csc(
      dem.num_seam_rows(), new_n, new_ins);
  result.out_syndrome = sparse_binary_matrix::from_nested_csc(
      dem.num_out_seam_rows(), new_n, new_out);
  result.fault_priors = std::move(new_priors);
  result.in_tags = dem.in_tags;
  result.out_tags = dem.out_tags;
  return result;
} // end - dem_merge_duplicate_columns()

bool are_dem_columns_unique(const extended_dem &dem) {
  dem.validate("are_dem_columns_unique");
  const std::size_t n = dem.num_faults();
  const dem_column_view cols(dem);

  std::map<std::vector<uint32_t>, std::size_t> seen;
  for (std::size_t j = 0; j < n; ++j) {
    if (!seen.emplace(cols.support(j), j).second)
      return false;
  }
  return true;
} // end - are_dem_columns_unique()

void assert_dem_columns_unique(const extended_dem &dem) {
  dem.validate("assert_dem_columns_unique");
  const std::size_t n = dem.num_faults();
  const dem_column_view cols(dem);

  std::map<std::vector<uint32_t>, std::vector<std::size_t>> groups;
  for (std::size_t j = 0; j < n; ++j)
    groups[cols.support(j)].push_back(j);

  std::size_t n_dup = 0;
  const std::vector<uint32_t> *first_sup = nullptr;
  const std::vector<std::size_t> *first_cols = nullptr;
  for (const auto &[sup, cols] : groups) {
    if (cols.size() > 1) {
      ++n_dup;
      if (!first_sup) {
        first_sup = &sup;
        first_cols = &cols;
      }
    }
  }

  if (n_dup == 0)
    return;

  std::string msg =
      "extended_dem has duplicate fault columns: " + std::to_string(n_dup) +
      " duplicate support set(s); for example, columns {";
  for (std::size_t i = 0; i < first_cols->size(); ++i) {
    if (i)
      msg += ", ";
    msg += std::to_string((*first_cols)[i]);
  }
  msg += "} all share support {";
  for (std::size_t i = 0; i < first_sup->size(); ++i) {
    if (i)
      msg += ", ";
    msg += std::to_string((*first_sup)[i]);
  }
  msg += "}. Call dem_merge_duplicate_columns() to merge them.";
  throw std::invalid_argument(msg);
} // end - assert_dem_columns_unique()

extended_dem dem_stitch_merged(const std::vector<extended_dem> &dem_chunks,
                               prior_combine_mode mode) {
  return dem_merge_duplicate_columns(dem_stitch_all(dem_chunks), mode);
} // end - dem_stitch_merged()

} // namespace cudaq::qec
