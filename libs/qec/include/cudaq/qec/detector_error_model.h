/****************************************************************-*- C++ -*-****
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "cuda-qx/core/tensor.h"

namespace cudaq::qec {

struct detector_error_model {
  /// Detector error models (DEMs) can be represented in a variety of ways,
  /// here a dense matrix representation is chosen, but this is not
  /// required as long as we can transform to a dense matrix
  /// representation to initialize our decoders.
  cudaqx::tensor<uint8_t> detector_error_matrix;

  /// The vector of weights along with the detector_error_matrix form
  /// the two components needs for the DEM.
  std::vector<double> error_rates;

  /// While not required, most usecases will want a notion of Pauli observables
  /// at the circuit-level
  cudaqx::tensor<uint8_t> observables_flips_matrix;

  // Shared size parameters among the matrix types.
  // - detector_error_matrix: num_detectors x num_error_mechanisms [d, e]
  // - error_rates: num_error_mechanisms
  // - observables_flips_matrix: num_observables x num_error_mechanisms [k, e]

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
