/****************************************************************-*- C++ -*-****
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "cuda-qx/core/tensor.h"
#include <optional>

namespace cudaq::qec {

/// A detector error model (DEM) for a quantum error correction circuit. A
/// DEM can be created from a QEC circuit and a noise model. It contains
/// information about which errors flip which detectors. This is used by the
/// decoder to help make predictions about observables flips.
///
/// Shared size parameters among the matrix types.
/// - \p detector_error_matrix: num_detectors x num_error_mechanisms [d, e]
/// - \p error_rates: num_error_mechanisms
/// - \p observables_flips_matrix: num_observables x num_error_mechanisms [k, e]
///
/// @note The C++ API for this class may change in the future. The Python API is
/// more likely to be backwards compatible.
struct detector_error_model {
  /// The detector error matrix is a specific kind of circuit-level parity-check
  /// matrix where each row represents a detector, and each column represents
  /// an error mechanism. The entries of this matrix are H[i,j] = 1 if detector
  /// i is triggered by error mechanism j, and 0 otherwise.
  cudaqx::tensor<uint8_t> detector_error_matrix;

  /// The list of weights has length equal to the number of columns of
  /// \p detector_error_matrix, which assigns a likelihood to each error
  /// mechanism.
  std::vector<double> error_rates;

  /// The observables flips matrix is a specific kind of circuit-level parity-
  /// check matrix where each row represents a Pauli observable, and each
  /// column represents an error mechanism. The entries of this matrix are
  /// O[i,j] = 1 if Pauli observable i is flipped by error mechanism j, and 0
  /// otherwise.
  cudaqx::tensor<uint8_t> observables_flips_matrix;

  /// Error mechanism ID. From a probability perspective, each error mechanism
  /// ID is independent of all other error mechanism ID. For all errors with
  /// the *same* ID, only one of them can happen. That is - the errors
  /// containing the same ID are correlated with each other.
  std::optional<std::vector<std::size_t>> error_ids;

  /// Return the number of rows in the detector_error_matrix.
  std::size_t num_detectors() const;

  /// Return the number of columns in the detector_error_matrix, error_rates,
  /// and observables_flips_matrix.
  std::size_t num_error_mechanisms() const;

  /// Return the number of rows in the observables_flips_matrix.
  std::size_t num_observables() const;

  /// Put the detector_error_matrix into canonical form, where the rows and
  /// columns are ordered in a way that is amenable to the round-based decoding
  /// process.
  void canonicalize_for_rounds(uint32_t num_syndromes_per_round);
};

} // namespace cudaq::qec
