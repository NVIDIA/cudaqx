/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// This file declares dem_from_css_matrices(), which builds a T-round
// code-capacity detector_error_model directly from CSS generator matrices
// and a depolarizing noise model, without requiring a stabilizer circuit or
// Stim round-trip. num_rounds defaults to 1 for the single-round case.

#pragma once

#include "cudaq/qec/code_matrices.h"
#include "cudaq/qec/detector_error_model.h"

namespace cudaq::qec {

/// @brief Build a T-round code-capacity detector_error_model from CSS
/// generator matrices and a depolarizing noise model.
///
/// Models T independent syndrome measurement rounds. Detectors are syndrome
/// *differences* between consecutive rounds: detector[r] fires when the
/// syndrome in round r differs from the syndrome in round r-1 (round 0 is
/// compared to the zero initial state). A data-qubit fault in round r
/// therefore spans two adjacent detector bands — round r and round r+1 —
/// because it changes the syndrome in round r but not in round r-1 or r+1.
/// Faults in the final round (r = T-1) span only that round's band since
/// there is no round T.
///
/// With num_rounds = 1 (the default), the model reduces to a flat
/// single-round code-capacity DEM: each fault spans only one detector band.
///
/// @param code       CSS code matrices. All non-empty matrices must share the
///                   same num_cols() value (n_qubits). Default-constructed
///                   (zero-row, zero-column) matrices are treated as empty.
/// @param noise      Depolarizing noise rates applied identically to every
///                   round. Per-qubit vectors override the scalar rates;
///                   qubits with effective rate 0 produce no column.
/// @param num_rounds Number of syndrome measurement rounds T (default 1,
///                   must be >= 1).
/// @return           detector_error_model with:
///   - detector_error_matrix: [T*d x e] where
///       d = hz.num_rows() + hx.num_rows(),
///       e = T * (|active_X| + |active_Z| + |active_Y|).
///     Round r occupies rows r*d .. (r+1)*d-1. Within each round:
///       rows 0 .. hz.num_rows()-1 are Z-type detectors (X and Y faults);
///       rows hz.num_rows() .. d-1 are X-type detectors (Z and Y faults).
///   - observables_flips_matrix: [k x e],
///       k = lz.num_rows() + lx.num_rows().
///     Faults in any round flip the same observable rows (the logical
///     measurement is taken once at the end of the experiment).
///   - error_rates: column layout is
///     [round 0 faults | ... | round T-1 faults]; within each round:
///     [active X qubits | active Z qubits | active Y qubits], each in
///     ascending qubit-index order.
/// @throws std::invalid_argument if num_rounds is 0, if two non-empty
///         matrices have inconsistent num_cols() values, or if a per-qubit
///         rate vector length does not equal n_qubits.
detector_error_model dem_from_css_matrices(const css_code_matrices &code,
                                           const css_noise_params &noise,
                                           std::size_t num_rounds = 1);

} // namespace cudaq::qec
