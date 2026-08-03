/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/qec/decoder_inputs.h"
#include "dem_sparse_projection.h"
#include <stdexcept>
#include <utility>

namespace cudaq::qec {

struct decoder_inputs::impl {
  decoder_model_source source = decoder_model_source::matrices;
  std::size_t num_detectors = 0;
  std::size_t num_error_mechanisms = 0;
  std::size_t num_observables = 0;
  sparse_binary_matrix H;
  /// Absent when the model supplies no observable mapping. A present but
  /// zero-row O is a supplied model, not an absent one.
  std::optional<sparse_binary_matrix> O;
  std::vector<double> rates;
  std::optional<std::vector<std::size_t>> ids;
  std::optional<sparse_binary_matrix> D;
  std::optional<std::string> raw_stim_dem;
};

namespace {

void validate_model(const sparse_binary_matrix &H,
                    const std::optional<sparse_binary_matrix> &O,
                    const std::vector<double> &rates,
                    const std::optional<std::vector<std::size_t>> &ids,
                    const std::optional<sparse_binary_matrix> &D) {
  if (O && O->num_cols() != H.num_cols())
    throw std::invalid_argument(
        "decoder_inputs: O column count must match H column count");
  if (!rates.empty() && rates.size() != H.num_cols())
    throw std::invalid_argument(
        "decoder_inputs: error_rates size must match H column count");
  if (ids && ids->size() != H.num_cols())
    throw std::invalid_argument(
        "decoder_inputs: error_ids size must match H column count");
  if (D && D->num_rows() != H.num_rows())
    throw std::invalid_argument(
        "decoder_inputs: D row count must match H row count");
}

} // namespace

std::shared_ptr<const decoder_inputs::impl> decoder_inputs::make_matrix_state(
    decoder_model_source source, sparse_binary_matrix H,
    std::optional<sparse_binary_matrix> O, std::vector<double> rates,
    std::optional<std::vector<std::size_t>> ids,
    std::optional<sparse_binary_matrix> D,
    std::optional<std::string> raw_stim_dem) {
  H = H.to_csc();
  if (O)
    *O = O->to_csr();
  if (D)
    *D = D->to_csr();
  validate_model(H, O, rates, ids, D);

  auto state = std::make_shared<decoder_inputs::impl>();
  state->source = source;
  state->num_detectors = H.num_rows();
  state->num_error_mechanisms = H.num_cols();
  state->num_observables = O ? O->num_rows() : 0;
  state->H = std::move(H);
  state->O = std::move(O);
  state->rates = std::move(rates);
  state->ids = std::move(ids);
  state->D = std::move(D);
  state->raw_stim_dem = std::move(raw_stim_dem);
  return state;
}

decoder_inputs::decoder_inputs(sparse_binary_matrix H)
    : decoder_inputs(make_matrix_state(decoder_model_source::matrices,
                                       std::move(H), std::nullopt, {},
                                       std::nullopt, std::nullopt)) {}

decoder_inputs::decoder_inputs(
    sparse_binary_matrix H, std::optional<sparse_binary_matrix> O,
    std::vector<double> error_rates,
    std::optional<sparse_binary_matrix> measurement_to_detectors,
    std::optional<std::vector<std::size_t>> error_ids)
    : decoder_inputs(make_matrix_state(
          decoder_model_source::matrices, std::move(H), std::move(O),
          std::move(error_rates), std::move(error_ids),
          std::move(measurement_to_detectors))) {}

decoder_inputs::decoder_inputs(
    detector_error_model model,
    std::optional<sparse_binary_matrix> measurement_to_detectors)
    : decoder_inputs(make_matrix_state(
          decoder_model_source::matrices,
          sparse_binary_matrix(model.detector_error_matrix),
          sparse_binary_matrix(model.observables_flips_matrix),
          std::move(model.error_rates), std::move(model.error_ids),
          std::move(measurement_to_detectors))) {}

decoder_inputs decoder_inputs::from_stim_dem(
    std::string stim_dem_text,
    std::optional<sparse_binary_matrix> measurement_to_detectors) {
  // Project straight to sparse. Going through the materialized
  // detector_error_model would allocate a dense detectors x mechanisms tensor
  // only to scan it back out again: ~98 MiB for a distance-13 model whose
  // sparse form is under 1 MiB, and wasted entirely for a DEM-native decoder.
  auto [H, O, error_rates] = details::sparse_dem_from_stim_text(stim_dem_text);
  return decoder_inputs(make_matrix_state(
      decoder_model_source::stim_dem, std::move(H), std::move(O),
      std::move(error_rates), std::nullopt, std::move(measurement_to_detectors),
      std::move(stim_dem_text)));
}

decoder_inputs::decoder_inputs(std::shared_ptr<const impl> state)
    : state_(std::move(state)) {}

decoder_inputs::decoder_inputs(const decoder_inputs &) noexcept = default;
decoder_inputs::decoder_inputs(decoder_inputs &&) noexcept = default;
decoder_inputs &
decoder_inputs::operator=(const decoder_inputs &) noexcept = default;
decoder_inputs &decoder_inputs::operator=(decoder_inputs &&) noexcept = default;
decoder_inputs::~decoder_inputs() = default;

decoder_model_source decoder_inputs::source() const noexcept {
  return state_->source;
}

const sparse_binary_matrix &decoder_inputs::detector_error_matrix() const {
  return state_->H;
}

bool decoder_inputs::has_observable_model() const noexcept {
  return state_->O.has_value();
}

const sparse_binary_matrix &decoder_inputs::observable_flips_matrix() const {
  if (!state_->O)
    throw std::logic_error(
        "decoder_inputs: no observable mapping was supplied");
  return *state_->O;
}

const std::vector<double> &decoder_inputs::error_rates() const {
  return state_->rates;
}

const std::optional<std::vector<std::size_t>> &
decoder_inputs::error_ids() const {
  return state_->ids;
}

const sparse_binary_matrix *
decoder_inputs::measurement_to_detectors() const noexcept {
  return state_->D ? &*state_->D : nullptr;
}

decoder_inputs decoder_inputs::canonicalized() const {
  auto H = state_->H.canonicalize().to_csc();
  return decoder_inputs(make_matrix_state(state_->source, std::move(H),
                                          state_->O, state_->rates, state_->ids,
                                          state_->D, state_->raw_stim_dem));
}

decoder_inputs decoder_inputs::without_measurement_to_detectors() const {
  auto state = std::make_shared<impl>(*state_);
  state->D.reset();
  return decoder_inputs(std::move(state));
}

decoder_inputs decoder_inputs::derive_with_changed_basis(
    sparse_binary_matrix H, std::optional<sparse_binary_matrix> O,
    std::vector<double> error_rates,
    std::optional<std::vector<std::size_t>> error_ids,
    std::optional<sparse_binary_matrix> measurement_to_detectors) const {
  // No raw source is carried through: it indexes the parent's detectors and
  // error mechanisms, which these matrices have re-indexed.
  return decoder_inputs(make_matrix_state(
      decoder_model_source::matrices, std::move(H), std::move(O),
      std::move(error_rates), std::move(error_ids),
      std::move(measurement_to_detectors)));
}

bool decoder_inputs::has_stim_dem() const noexcept {
  return state_->raw_stim_dem.has_value();
}

const std::string &decoder_inputs::stim_dem() const {
  if (!state_->raw_stim_dem)
    throw std::logic_error(
        "decoder_inputs: authoritative source is not a Stim DEM");
  return *state_->raw_stim_dem;
}

detector_error_model decoder_inputs::materialize_detector_error_model() const {
  detector_error_model model;
  model.detector_error_matrix = state_->H.to_dense();
  // A model with no observable mapping materializes as zero observable rows,
  // matching a DEM that declares no observables.
  model.observables_flips_matrix =
      state_->O ? state_->O->to_dense()
                : cudaqx::tensor<uint8_t>(
                      {std::size_t{0}, state_->num_error_mechanisms});
  model.error_rates = state_->rates;
  model.error_ids = state_->ids;
  return model;
}

std::size_t decoder_inputs::num_detectors() const noexcept {
  return state_->num_detectors;
}

std::size_t decoder_inputs::num_error_mechanisms() const noexcept {
  return state_->num_error_mechanisms;
}

std::size_t decoder_inputs::num_observables() const noexcept {
  return state_->num_observables;
}

} // namespace cudaq::qec
