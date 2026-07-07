/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "chromobius/decode/decoder.h"
#include "stim.h"
#include "cudaq/qec/decoder.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace cudaq::qec {

namespace {

struct chromobius_init_data {
  stim::DetectorErrorModel dem;
  cudaq::qec::sparse_binary_matrix base_H;
};

cudaq::qec::sparse_binary_matrix
make_empty_base_H(std::size_t num_detectors, std::size_t num_observables) {
  using index_type = cudaq::qec::sparse_binary_matrix::index_type;
  constexpr auto max_index = std::numeric_limits<index_type>::max();
  if (num_detectors > max_index)
    throw std::runtime_error(
        "Chromobius DEM has too many detectors for CUDA-Q QEC");
  if (num_observables > max_index)
    throw std::runtime_error(
        "Chromobius DEM has too many observables for CUDA-Q QEC");

  return cudaq::qec::sparse_binary_matrix::from_csc(
      static_cast<index_type>(num_detectors),
      static_cast<index_type>(num_observables),
      std::vector<index_type>(num_observables + 1, 0), {});
}

chromobius_init_data make_chromobius_init_data(const decoder_init &init) {
  const auto *dem_text = std::get_if<std::string>(&init);
  if (!dem_text) {
    throw std::runtime_error(
        "Chromobius decoder requires a Stim detector error model string as "
        "decoder input. Use get_decoder(\"chromobius\", dem_text, params).");
  }

  stim::DetectorErrorModel dem;
  try {
    dem = stim::DetectorErrorModel(*dem_text);
  } catch (const std::exception &e) {
    throw std::runtime_error(std::string("Chromobius Stim DEM parse failed: ") +
                             e.what());
  }

  const auto num_detectors = static_cast<std::size_t>(dem.count_detectors());
  const auto num_observables =
      static_cast<std::size_t>(dem.count_observables());
  if (num_observables > 64) {
    throw std::runtime_error(
        "Chromobius currently returns observable flips as a 64-bit mask; "
        "CUDA-Q QEC wrapper supports at most 64 observables.");
  }

  return chromobius_init_data{
      std::move(dem), make_empty_base_H(num_detectors, num_observables)};
}

std::vector<std::vector<uint32_t>> identity_sparse(std::size_t size) {
  std::vector<std::vector<uint32_t>> result(size);
  for (std::size_t i = 0; i < size; ++i)
    result[i].push_back(static_cast<uint32_t>(i));
  return result;
}

bool get_bool_param(const cudaqx::heterogeneous_map &params,
                    const std::string &key, bool default_value) {
  return params.contains(key) ? params.get<bool>(key) : default_value;
}

} // namespace

/// @brief Wrapper around the Chromobius Mobius decoder for color-code detector
/// error models.
class chromobius : public decoder {
private:
  stim::DetectorErrorModel dem;
  ::chromobius::Decoder chromobius_decoder;
  std::size_t num_detector_bytes = 0;
  std::size_t num_observables = 0;
  bool return_weight = false;

  std::vector<uint8_t> hard_syndrome;
  std::vector<uint8_t> packed_detection_events;

public:
  chromobius(chromobius_init_data init_data,
             const cudaqx::heterogeneous_map &params)
      : decoder(std::move(init_data.base_H)), dem(std::move(init_data.dem)) {
    ::chromobius::DecoderConfigOptions options;
    options.drop_mobius_errors_involving_remnant_errors =
        get_bool_param(params, "drop_mobius_errors_involving_remnant_errors",
                       options.drop_mobius_errors_involving_remnant_errors);
    options.ignore_decomposition_failures =
        get_bool_param(params, "ignore_decomposition_failures",
                       options.ignore_decomposition_failures);
    options.include_coords_in_mobius_dem =
        get_bool_param(params, "include_coords_in_mobius_dem",
                       options.include_coords_in_mobius_dem);

    return_weight = get_bool_param(params, "return_weight", false);

    chromobius_decoder =
        ::chromobius::Decoder::from_dem(dem, std::move(options));
    chromobius_decoder.write_mobius_match_to_std_err =
        get_bool_param(params, "write_mobius_match_to_stderr", false);

    syndrome_size = static_cast<std::size_t>(dem.count_detectors());
    num_observables = static_cast<std::size_t>(dem.count_observables());
    if (num_observables > 64) {
      throw std::runtime_error(
          "Chromobius currently returns observable flips as a 64-bit mask; "
          "CUDA-Q QEC wrapper supports at most 64 observables.");
    }

    block_size = num_observables;
    num_detector_bytes = (syndrome_size + 7) / 8;
    hard_syndrome.resize(syndrome_size);
    packed_detection_events.resize(num_detector_bytes);

    // Chromobius directly predicts observables. Make the base realtime
    // observable-reduction logic treat each predicted bit as its own observable
    // correction.
    this->set_O_sparse(identity_sparse(num_observables));
  }

  decoder_result decode(const std::vector<float_t> &syndrome) override {
    if (syndrome.size() != syndrome_size) {
      throw std::runtime_error(
          "Chromobius syndrome length must match the number of detectors");
    }

    std::fill(packed_detection_events.begin(), packed_detection_events.end(),
              uint8_t{0});
    cudaq::qec::convert_vec_soft_to_hard(syndrome, hard_syndrome);
    for (std::size_t i = 0; i < syndrome.size(); ++i) {
      if (hard_syndrome[i])
        packed_detection_events[i >> 3] |= static_cast<uint8_t>(1u << (i & 7));
    }

    float weight = 0.0f;
    auto prediction = chromobius_decoder.decode_detection_events(
        std::span<const uint8_t>(packed_detection_events.data(),
                                 packed_detection_events.size()),
        return_weight ? &weight : nullptr);

    decoder_result result{true, std::vector<float_t>(num_observables, 0.0)};
    for (std::size_t i = 0; i < num_observables; ++i)
      result.result[i] = static_cast<float_t>((prediction >> i) & 1);

    if (return_weight) {
      cudaqx::heterogeneous_map opt_results;
      opt_results.insert("weight", static_cast<double>(weight));
      result.opt_results = opt_results;
    }
    return result;
  }

  std::string get_version() const override {
    return "CUDA-Q QEC Chromobius Decoder wrapper";
  }

  CUDAQ_EXTENSION_CUSTOM_CREATOR_FUNCTION(
      chromobius, static std::unique_ptr<decoder> create(
                      const cudaq::qec::decoder_init &init,
                      const cudaqx::heterogeneous_map &params) {
        return std::make_unique<chromobius>(make_chromobius_init_data(init),
                                            params);
      })
};

CUDAQ_EXT_PT_REGISTER_TYPE(chromobius)

} // namespace cudaq::qec
