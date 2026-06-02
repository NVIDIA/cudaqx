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

// std::mutex is enough: the factory copies the creator out before
// invoking, so creators cannot re-enter the registry.
struct stim_dem_registry {
  std::mutex &mutex;
  std::unordered_map<std::string, stim_dem_decoder_creator> &map;
};
stim_dem_registry get_stim_dem_registry() {
  // Heap-allocated to outlive static destructors (plugin dlclose unregister
  // path); matches the cudaqx extension_point pattern. See extension_point.h.
  static std::mutex *mutex = new std::mutex();
  static auto *map =
      new std::unordered_map<std::string, stim_dem_decoder_creator>();
  return {*mutex, *map};
}

} // namespace

dem_default_values dem_defaults_for_missing_keys(
    const std::function<bool(const std::string &)> &contains_user_key,
    const detector_error_model &dem) {
  dem_default_values out;
  if (!contains_user_key("O"))
    out.O = &dem.observables_flips_matrix;
  if (!contains_user_key("error_rate_vec"))
    out.error_rate_vec = &dem.error_rates;
  return out;
}

void register_stim_dem_decoder_creator(const std::string &name,
                                       stim_dem_decoder_creator creator) {
  auto reg = get_stim_dem_registry();
  std::lock_guard<std::mutex> lock(reg.mutex);
  reg.map[name] = std::move(creator);
}

void unregister_stim_dem_decoder_creator(const std::string &name) {
  auto reg = get_stim_dem_registry();
  std::lock_guard<std::mutex> lock(reg.mutex);
  reg.map.erase(name);
}

std::unique_ptr<decoder>
get_decoder_from_stim_dem(const std::string &name,
                          const std::string &stim_dem_text,
                          const cudaqx::heterogeneous_map &options) {
  stim_dem_decoder_creator creator;
  {
    auto reg = get_stim_dem_registry();
    std::lock_guard<std::mutex> lock(reg.mutex);
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
  // Keep in sync with the Python binding in py_decoder.cpp.
  auto defaults = dem_defaults_for_missing_keys(
      [&](const std::string &key) { return merged.contains(key); }, dem);
  if (defaults.O)
    merged.insert("O", *defaults.O);
  if (defaults.error_rate_vec)
    merged.insert("error_rate_vec", *defaults.error_rate_vec);

  return decoder::get(name, dem.detector_error_matrix, merged);
}

} // namespace cudaq::qec
