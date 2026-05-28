/*******************************************************************************
 * Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "stim.h"
#include "cudaq/qec/decoder.h"

#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace cudaq::qec {

namespace {

std::pair<std::recursive_mutex &,
          std::unordered_map<std::string, stim_dem_decoder_creator> &>
get_stim_dem_registry() {
  static std::recursive_mutex *mutex = new std::recursive_mutex();
  static auto *registry =
      new std::unordered_map<std::string, stim_dem_decoder_creator>();
  return {*mutex, *registry};
}

struct extracted_dem {
  cudaqx::tensor<uint8_t> H;
  cudaqx::tensor<uint8_t> O;
  std::vector<double> error_rates;
};

extracted_dem extract_from_stim_dem(const std::string &dem_text) {
  stim::DetectorErrorModel dem(dem_text);
  const std::size_t num_detectors =
      static_cast<std::size_t>(dem.count_detectors());
  const std::size_t num_observables =
      static_cast<std::size_t>(dem.count_observables());

  std::vector<std::vector<std::size_t>> detector_hits;
  std::vector<std::vector<std::size_t>> observable_hits;
  std::vector<double> rates;

  dem.iter_flatten_error_instructions([&](const stim::DemInstruction &inst) {
    if (inst.arg_data.size() == 0)
      throw std::runtime_error(
          "Stim DEM error instruction missing probability argument");
    const double prob = inst.arg_data[0];
    std::vector<std::size_t> dets;
    std::vector<std::size_t> obs;
    for (const auto &target : inst.target_data) {
      if (target.is_separator())
        continue;
      if (target.is_relative_detector_id()) {
        dets.push_back(static_cast<std::size_t>(target.val()));
      } else if (target.is_observable_id()) {
        obs.push_back(static_cast<std::size_t>(target.val()));
      }
    }
    detector_hits.push_back(std::move(dets));
    observable_hits.push_back(std::move(obs));
    rates.push_back(prob);
  });

  const std::size_t num_errors = rates.size();
  extracted_dem result;
  result.H = cudaqx::tensor<uint8_t>({num_detectors, num_errors});
  result.O = cudaqx::tensor<uint8_t>({num_observables, num_errors});
  result.error_rates = std::move(rates);

  for (std::size_t err = 0; err < num_errors; ++err) {
    for (auto det : detector_hits[err]) {
      if (det >= num_detectors)
        throw std::runtime_error(
            "Stim DEM detector id out of range while extracting H");
      result.H.at({det, err}) = 1;
    }
    for (auto ob : observable_hits[err]) {
      if (ob >= num_observables)
        throw std::runtime_error(
            "Stim DEM observable id out of range while extracting O");
      result.O.at({ob, err}) = 1;
    }
  }

  return result;
}

} // namespace

void register_stim_dem_decoder_creator(const std::string &name,
                                       stim_dem_decoder_creator creator) {
  auto [mutex, registry] = get_stim_dem_registry();
  std::lock_guard<std::recursive_mutex> lock(mutex);
  registry[name] = std::move(creator);
}

std::unique_ptr<decoder>
get_decoder_from_stim_dem(const std::string &name,
                          const std::string &stim_dem_text,
                          const cudaqx::heterogeneous_map options) {
  {
    auto [mutex, registry] = get_stim_dem_registry();
    std::lock_guard<std::recursive_mutex> lock(mutex);
    auto iter = registry.find(name);
    if (iter != registry.end())
      return iter->second(stim_dem_text, options);
  }

  auto extracted = extract_from_stim_dem(stim_dem_text);

  cudaqx::heterogeneous_map merged = options;
  if (!merged.contains("O"))
    merged.insert("O", extracted.O);
  if (!merged.contains("error_rate_vec"))
    merged.insert("error_rate_vec", extracted.error_rates);

  return decoder::get(name, extracted.H, merged);
}

} // namespace cudaq::qec
