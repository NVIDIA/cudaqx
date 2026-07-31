/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/qec/detector_error_model.h"
#include "cudaq/qec/sparse_binary_matrix.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace cudaq::qec {

/// @brief Authoritative representation from which a decoder model originates.
///
/// Matrix and Stim sources are supported now. `dem_chunks` names the compact
/// repeated-round representation being introduced by the dynamic DEM APIs.
/// Adding its typed constructor and accessor does not change the
/// `decoder_inputs` object layout or decoder factory signature.
enum class decoder_model_source : std::uint8_t {
  matrices,
  stim_dem,
  dem_chunks,
};

/// @brief Stable, owning input contract shared by offline and server decoders.
///
/// This is a small immutable value handle. Copies share the same model state;
/// the decoder factory takes the handle by value and the decoder base retains
/// it. Source-specific data is authoritative and the common matrix accessors
/// expose the projection stored when the handle is constructed. Model matrices
/// are stored sparsely instead of composing detector_error_model, whose matrix
/// fields are dense tensors.
class decoder_inputs {
public:
  /// @brief Construct an H-only matrix model.
  explicit decoder_inputs(sparse_binary_matrix detector_error_matrix);

  /// @brief Construct a materialized matrix model.
  /// @param detector_error_matrix H, with shape detectors x error mechanisms.
  /// @param observable_flips_matrix O, with shape observables x error
  /// mechanisms. Its row count is retained even when a row has no nonzeros.
  /// @param error_rates Optional rate per error mechanism.
  /// @param measurement_to_detectors Optional D, with shape detectors x raw
  /// measurements.
  /// @param error_ids Optional correlation ID per error mechanism.
  decoder_inputs(
      sparse_binary_matrix detector_error_matrix,
      sparse_binary_matrix observable_flips_matrix,
      std::vector<double> error_rates = {},
      std::optional<sparse_binary_matrix> measurement_to_detectors =
          std::nullopt,
      std::optional<std::vector<std::size_t>> error_ids = std::nullopt);

  /// @brief Construct from the existing materialized detector-error model.
  explicit decoder_inputs(detector_error_model model,
                          std::optional<sparse_binary_matrix>
                              measurement_to_detectors = std::nullopt);

  /// @brief Construct from authoritative raw Stim DEM text.
  ///
  /// Matrix accessors expose the common lossy projection produced by
  /// `dem_from_stim_text`; DEM-native decoders should consume `stim_dem()`.
  static decoder_inputs
  from_stim_dem(std::string stim_dem_text,
                std::optional<sparse_binary_matrix> measurement_to_detectors =
                    std::nullopt);

  decoder_inputs(const decoder_inputs &) noexcept;
  /// @brief Move construction leaves the source valid only for destruction or
  /// assignment.
  decoder_inputs(decoder_inputs &&) noexcept;
  decoder_inputs &operator=(const decoder_inputs &) noexcept;
  /// @brief Move assignment leaves the source valid only for destruction or
  /// assignment.
  decoder_inputs &operator=(decoder_inputs &&) noexcept;
  ~decoder_inputs();

  decoder_model_source source() const noexcept;

  /// @brief Return the stored common H projection.
  const sparse_binary_matrix &detector_error_matrix() const;

  /// @brief Return the stored common O projection.
  const sparse_binary_matrix &observable_flips_matrix() const;

  const std::vector<double> &error_rates() const;
  const std::optional<std::vector<std::size_t>> &error_ids() const;

  /// @brief Return D, or nullptr when input syndromes are already detectors.
  const sparse_binary_matrix *measurement_to_detectors() const noexcept;

  bool has_stim_dem() const noexcept;

  /// @throws std::logic_error if the authoritative source is not a Stim DEM.
  const std::string &stim_dem() const;

  /// @brief Materialize the common detector-error-model view.
  detector_error_model materialize_detector_error_model() const;

  /// Dimensions are stored as source metadata so these accessors never need to
  /// request H or O. For matrix sources they intentionally duplicate the O(1)
  /// matrix shape values in preparation for compact source alternatives.
  std::size_t num_detectors() const noexcept;
  std::size_t num_error_mechanisms() const noexcept;
  std::size_t num_observables() const noexcept;

private:
  struct impl;
  static std::shared_ptr<const impl> make_matrix_state(
      decoder_model_source source, sparse_binary_matrix detector_error_matrix,
      sparse_binary_matrix observable_flips_matrix,
      std::vector<double> error_rates,
      std::optional<std::vector<std::size_t>> error_ids,
      std::optional<sparse_binary_matrix> measurement_to_detectors,
      std::optional<std::string> raw_stim_dem = std::nullopt);
  explicit decoder_inputs(std::shared_ptr<const impl> state);
  std::shared_ptr<const impl> state_;
};

} // namespace cudaq::qec
