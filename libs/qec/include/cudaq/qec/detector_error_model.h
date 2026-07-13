/****************************************************************-*- C++ -*-****
 * Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "cuda-qx/core/tensor.h"
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace cudaq::qec {

/// A detector error model (DEM) for a quantum error correction circuit. A
/// DEM can be created from a QEC circuit and a noise model. It contains
/// information about which errors flip which detectors. This is used by the
/// decoder to help make predictions about observables flips.
///
/// Shared size parameters among the matrix types.
/// - `detector_error_matrix`: num_detectors x num_error_mechanisms [d, e]
/// - `error_rates`: num_error_mechanisms
/// - `observables_flips_matrix`: num_observables x num_error_mechanisms [k, e]
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
  /// `detector_error_matrix`, which assigns a likelihood to each error
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
  /// process. Columns sharing the same detector AND observable signature are
  /// merged, with their rates composed so the resulting model matches the
  /// input model. By default, zero-syndrome columns that still flip an
  /// observable (undetectable logical errors) are retained so the model's
  /// observable-flip probability is preserved. Set @p
  /// remove_zero_syndrome_errors to true to drop all columns with no detector
  /// signature, which is appropriate when the canonicalized DEM is consumed
  /// only for round-based decoding (where such columns carry no syndrome).
  ///
  /// @note Canonicalization does not preserve cross-column exclusivity
  /// structure. Each output column is given a fresh unique error id and is
  /// treated as independent of every other column; any `error_ids` correlation
  /// present in the input model is discarded.
  void canonicalize_for_rounds(uint32_t num_syndromes_per_round,
                               bool remove_zero_syndrome_errors = false);
};

namespace details {

/// Validate one optional inlined-feedback tensor. Empty tensors represent no
/// feedback. Non-empty tensors must have the requested shape and contain only
/// exact binary values so that no caller can silently reinterpret an arbitrary
/// nonzero uint8_t as a measurement-record dependency.
inline void validate_inlined_feedback_tensor(
    const cudaqx::tensor<uint8_t> &tensor, std::size_t expected_rows,
    std::size_t expected_cols, const std::string &name) {
  if (tensor.size() == 0)
    return;

  if (tensor.rank() != 2 || tensor.shape()[0] != expected_rows ||
      tensor.shape()[1] != expected_cols) {
    std::string actual;
    for (std::size_t d = 0; d < tensor.rank(); ++d)
      actual += (d ? ", " : "") + std::to_string(tensor.shape()[d]);
    throw std::runtime_error(name + " has invalid shape [" + actual +
                             "] - expected [" + std::to_string(expected_rows) +
                             ", " + std::to_string(expected_cols) +
                             "] or an empty tensor.");
  }

  for (std::size_t i = 0; i < tensor.size(); ++i) {
    const auto value = tensor.data()[i];
    if (value > 1)
      throw std::runtime_error(
          name + " has non-binary value " +
          std::to_string(static_cast<unsigned int>(value)) + " at flat index " +
          std::to_string(i) + "; entries must be exactly 0 or 1.");
  }
}

} // namespace details

/// @brief Record-to-detector composition layout derived from a code's
/// declared inlined-feedback matrices (CSR-style).
///
/// Entry range [detector_offsets[j], detector_offsets[j+1]) of
/// detector_indices lists the herald record columns XOR-ed into record j's
/// cross-round detector (records of the earlier round) and into its final
/// boundary detector (records of the last round). Entry range
/// [observable_offsets[m], observable_offsets[m+1]) of observable_indices
/// lists the record columns XOR-ed into logical observable m on every round.
/// Empty offsets vectors mean no feedback is declared for that target
/// (identity detector/observable structure).
struct inlined_feedback_layout {
  std::vector<std::size_t> detector_indices;
  std::vector<std::size_t> detector_offsets;
  std::vector<std::size_t> observable_indices;
  std::vector<std::size_t> observable_offsets;
};

/// @brief Build the record-to-detector composition layout from a code's
/// declared inlined-feedback matrices.
///
/// @param feedback Detector feedback matrix, shape
///        [num_syndromes_per_round x num_syndromes_per_round] with 0/1
///        entries, or an empty tensor for none. Entry (j, k) = 1 means the
///        cross-round detector for record j additionally XORs record k of
///        the earlier round, and the final boundary detector for record j
///        additionally XORs record k of the last round.
/// @param obs_feedback Observable feedback matrix, shape
///        [num_observables x num_syndromes_per_round] with 0/1 entries, or
///        an empty tensor for none. Entry (m, k) = 1 means logical
///        observable m additionally XORs record k of every round.
/// @param num_syndromes_per_round Records per round (number of ancilla
///        qubits).
/// @param num_observables Number of logical observables.
/// @return The CSR layout; see inlined_feedback_layout.
/// @throws std::runtime_error if a non-empty tensor's shape does not match
///         the expected dimensions or if an entry is not exactly 0 or 1.
inlined_feedback_layout
build_inlined_feedback_layout(const cudaqx::tensor<uint8_t> &feedback,
                              const cudaqx::tensor<uint8_t> &obs_feedback,
                              std::size_t num_syndromes_per_round,
                              std::size_t num_observables);

/// Parse the Stim DEM string @p dem_text into detector/observable flip
/// matrices and error rates. DEM-native decoders should consume raw DEM text
/// instead. By default (@p use_decomp_suggestions = false) the '^' separators
/// are ignored and each error instruction produces a single column. If
/// @p use_decomp_suggestions is true, error mechanisms that carry an explicit
/// graphlike decomposition (components separated by '^') are expanded into one
/// column per component, each inheriting the probability of the parent
/// instruction. @p error_ids is always left as nullopt. Note that this is a
/// lossy approximation of the original DEM.
detector_error_model dem_from_stim_text(const std::string &dem_text,
                                        bool use_decomp_suggestions = false);

} // namespace cudaq::qec
