/*******************************************************************************
 * Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/qec/decoder.h"
#include "cudaq/qec/detector_error_model.h"

#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace cudaq::qec {

namespace {

struct stim_dem_registry {
  std::recursive_mutex &mutex;
  std::unordered_map<std::string, stim_dem_decoder_creator> &map;
};
stim_dem_registry get_stim_dem_registry() {
  static std::recursive_mutex *mutex = new std::recursive_mutex();
  static auto *map =
      new std::unordered_map<std::string, stim_dem_decoder_creator>();
  return {*mutex, *map};
}

} // namespace

void register_stim_dem_decoder_creator(const std::string &name,
                                       stim_dem_decoder_creator creator) {
  auto reg = get_stim_dem_registry();
  std::lock_guard<std::recursive_mutex> lock(reg.mutex);
  reg.map[name] = std::move(creator);
}

void unregister_stim_dem_decoder_creator(const std::string &name) {
  auto reg = get_stim_dem_registry();
  std::lock_guard<std::recursive_mutex> lock(reg.mutex);
  reg.map.erase(name);
}

std::unique_ptr<decoder>
get_decoder_from_stim_dem(const std::string &name,
                          const std::string &stim_dem_text,
                          const cudaqx::heterogeneous_map options) {
  stim_dem_decoder_creator creator;
  {
    auto reg = get_stim_dem_registry();
    std::lock_guard<std::recursive_mutex> lock(reg.mutex);
    auto iter = reg.map.find(name);
    if (iter != reg.map.end())
      creator = iter->second;
  }
  if (creator)
    return creator(stim_dem_text, options);

  if (!decoder::is_registered(name))
    throw std::runtime_error(
        "get_decoder_from_stim_dem: decoder \"" + name +
        "\" is not registered. Run with CUDAQ_LOG_LEVEL=info to see plugin "
        "diagnostics at startup.");

  auto dem = dem_from_stim_text(stim_dem_text);

  cudaqx::heterogeneous_map merged = options;
  if (!merged.contains("O"))
    merged.insert("O", dem.observables_flips_matrix);
  if (!merged.contains("error_rate_vec"))
    merged.insert("error_rate_vec", dem.error_rates);

  return decoder::get(name, dem.detector_error_matrix, merged);
}

} // namespace cudaq::qec
