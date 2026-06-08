/*******************************************************************************
 * Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/qec/decoder.h"
#include "cudaq/qec/detector_error_model.h"

#include <stdexcept>
#include <string>

namespace cudaq::qec {

dem_default_values dem_defaults_for_missing_keys(
    const std::function<bool(const std::string &)> &contains_user_key,
    const detector_error_model &dem) {
  dem_default_values out;
  if (!contains_user_key("O") && dem.num_observables() > 0)
    out.O = &dem.observables_flips_matrix;
  if (!contains_user_key("error_rate_vec"))
    out.error_rate_vec = &dem.error_rates;
  return out;
}

std::string_view require_dem_text(const decoder_init &init) {
  if (const auto *dem_text = std::get_if<std::string>(&init))
    return std::string_view{*dem_text};
  throw std::runtime_error(
      "This decoder requires a Stim detector error model string; a "
      "parity-check matrix cannot be used to reconstruct the detector "
      "annotations it needs.");
}

} // namespace cudaq::qec
