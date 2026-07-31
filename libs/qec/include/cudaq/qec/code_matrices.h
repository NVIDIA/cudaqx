/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// This file defines the css_code_matrices and css_noise_params types used
// to describe a Calderbank-Shor-Steane (CSS) quantum error-correcting code
// in terms of its sparse binary generator matrices and a depolarizing noise
// model. These types serve as the primary input interface for
// dem_from_css_matrices() and are intended to be derived from a code object
// via css_matrices_from_code(), or constructed directly from raw matrix data.

#pragma once

#include "cudaq/qec/sparse_binary_matrix.h"
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace cudaq::qec {

/// @brief CSS code generator matrices: parity-check and logical-operator
/// matrices for a Calderbank-Shor-Steane (CSS) quantum error-correcting code.
///
/// All matrices are sparse binary (GF(2)). Each matrix column corresponds to
/// one data qubit, so every non-empty matrix must share the same num_cols()
/// value (the number of data qubits, n). A matrix with zero rows is a valid
/// empty block; its num_cols() may be 0 (default-constructed) or n.
///
/// Row semantics:
///   hz  — Z-type stabilizer generators [n_z_checks x n_qubits].
///          Row i is the support of the i-th Z-stabilizer generator.
///          A Z-stabilizer anticommutes with X errors, so hz[:,q] gives
///          the set of Z-checks triggered by an X error on qubit q.
///   hx  — X-type stabilizer generators [n_x_checks x n_qubits].
///          Detects Z errors symmetrically; hx[:,q] gives the X-checks
///          triggered by a Z error on qubit q.
///   lz  — Z-type logical operators [k x n_qubits].
///          lz[r, q] = 1 iff the r-th Z logical observable is flipped
///          by an X error on qubit q.
///   lx  — X-type logical operators [k x n_qubits].
///          lx[r, q] = 1 iff the r-th X logical observable is flipped
///          by a Z error on qubit q.
struct css_code_matrices {
  sparse_binary_matrix hz; ///< Z stabilizers [n_z_checks x n_qubits]
  sparse_binary_matrix hx; ///< X stabilizers [n_x_checks x n_qubits]
  sparse_binary_matrix lz; ///< Z logical operators [k x n_qubits]
  sparse_binary_matrix lx; ///< X logical operators [k x n_qubits]
};

/// @brief Noise parameters for the phenomenological noise model.
///
/// Extends the code-capacity (data-qubit-only) model with syndrome
/// measurement errors, which are the most important circuit-level effect
/// for CSS codes: each stabilizer measurement can produce the wrong outcome
/// with probability pm, creating a time-local detector event that spans
/// two consecutive rounds without flipping any logical observable.
///
/// Scalar rates (px, py, pz, pm) apply uniformly. Per-element override
/// vectors take priority when non-empty; their length must equal n_qubits
/// (for data-qubit rates) or n_checks = hz.num_rows() + hx.num_rows()
/// (for pm_per_check, Z-checks first then X-checks). Elements with
/// effective rate zero produce no DEM column.
struct css_noise_params {
  double px = 0.0; ///< Uniform X data-qubit rate; overridden by px_per_qubit
  double py = 0.0; ///< Uniform Y data-qubit rate; overridden by py_per_qubit
  double pz = 0.0; ///< Uniform Z data-qubit rate; overridden by pz_per_qubit
  double pm = 0.0; ///< Uniform syndrome measurement error rate per check per
                   ///< round; overridden by pm_per_check if set

  /// Per-qubit X rates. If non-empty, length must equal n_qubits.
  std::vector<double> px_per_qubit;
  /// Per-qubit Y rates. If non-empty, length must equal n_qubits.
  std::vector<double> py_per_qubit;
  /// Per-qubit Z rates. If non-empty, length must equal n_qubits.
  std::vector<double> pz_per_qubit;
  /// Per-check measurement error rates. If non-empty, length must equal
  /// n_checks (= hz.num_rows() + hx.num_rows()); Z-checks first.
  std::vector<double> pm_per_check;
};

// ---------------------------------------------------------------------------
// Implementation details used by dem_construction.cpp and extended_dem.cpp.
// Not part of the public API.
// ---------------------------------------------------------------------------

namespace detail {

using col_list = std::vector<std::vector<sparse_binary_matrix::index_type>>;

/// Determine n_qubits from the first matrix that has a nonzero column count.
/// Returns 0 when all matrices are default-constructed.
inline std::size_t resolve_num_qubits(const css_code_matrices &code) {
  if (code.hz.num_cols() > 0)
    return static_cast<std::size_t>(code.hz.num_cols());
  if (code.hx.num_cols() > 0)
    return static_cast<std::size_t>(code.hx.num_cols());
  if (code.lz.num_cols() > 0)
    return static_cast<std::size_t>(code.lz.num_cols());
  if (code.lx.num_cols() > 0)
    return static_cast<std::size_t>(code.lx.num_cols());
  return 0;
}

/// Throw if m has nonzero rows but a column count that disagrees with n.
inline void check_num_cols(const sparse_binary_matrix &m, std::size_t n,
                           const char *label) {
  if (m.num_rows() != 0 && static_cast<std::size_t>(m.num_cols()) != n)
    throw std::invalid_argument(
        std::string(label) + " num_cols (" + std::to_string(m.num_cols()) +
        ") does not match n_qubits (" + std::to_string(n) + ")");
}

/// Throw if per_qubit is non-empty with a length that does not match n.
inline void check_per_qubit_size(const std::vector<double> &per_qubit,
                                 std::size_t n, const char *label) {
  if (!per_qubit.empty() && per_qubit.size() != n)
    throw std::invalid_argument(std::string(label) + " has " +
                                std::to_string(per_qubit.size()) +
                                " entries but n_qubits=" + std::to_string(n));
}

/// Nested CSC for m padded to outer size n; new entries are empty vectors.
inline col_list padded_nested_csc(const sparse_binary_matrix &m,
                                  std::size_t n) {
  auto cols = m.to_nested_csc();
  cols.resize(n);
  return cols;
}

/// Effective rate for qubit q: per_qubit[q] when set, else uniform.
inline double qubit_rate(double uniform, const std::vector<double> &per_qubit,
                         std::size_t q) {
  return per_qubit.empty() ? uniform : per_qubit[q];
}

/// Qubit indices (ascending) whose effective rate is nonzero.
inline std::vector<std::size_t>
active_qubits(double uniform, const std::vector<double> &per_qubit,
              std::size_t n) {
  std::vector<std::size_t> out;
  out.reserve(n);
  for (std::size_t q = 0; q < n; ++q)
    if (qubit_rate(uniform, per_qubit, q) > 0.0)
      out.push_back(q);
  return out;
}

/// Effective measurement error rate for check k (k indexes all checks,
/// Z-type first then X-type): pm_per_check[k] when set, else uniform pm.
inline double check_rate(double uniform_pm,
                         const std::vector<double> &pm_per_check,
                         std::size_t k) {
  return pm_per_check.empty() ? uniform_pm : pm_per_check[k];
}

/// Check indices (ascending, 0..n_checks-1) with a nonzero effective rate.
inline std::vector<std::size_t>
active_checks(double uniform_pm, const std::vector<double> &pm_per_check,
              std::size_t n_checks) {
  std::vector<std::size_t> out;
  out.reserve(n_checks);
  for (std::size_t k = 0; k < n_checks; ++k)
    if (check_rate(uniform_pm, pm_per_check, k) > 0.0)
      out.push_back(k);
  return out;
}

} // namespace detail

} // namespace cudaq::qec
