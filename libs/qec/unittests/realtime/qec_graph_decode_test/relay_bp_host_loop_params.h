/****************************************************************-*- C++ -*-****
 * Copyright (c) 2025-2026 NVIDIA Corporation & Affiliates.                    *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cuda-qx/core/heterogeneous_map.h"

#include <cstddef>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace test_realtime_qldpc {
namespace detail {

inline std::string trim(std::string value) {
  const auto first = value.find_first_not_of(" \t\r\n");
  if (first == std::string::npos)
    return {};
  const auto last = value.find_last_not_of(" \t\r\n");
  return value.substr(first, last - first + 1);
}

inline std::string scalar(const std::string &yaml, const std::string &key) {
  const std::regex pattern("(?:^|\\n)[ \\t]*" + key +
                           "[ \\t]*:[ \\t]*([^\\r\\n#]+)");
  std::smatch match;
  if (!std::regex_search(yaml, match, pattern))
    throw std::runtime_error("Missing Relay BP parameter: " + key);
  return trim(match[1].str());
}

inline std::vector<double> double_list(const std::string &yaml,
                                       const std::string &key) {
  const std::regex pattern(key + "[ \\t]*:[ \\t]*\\[([^\\]]*)\\]");
  std::smatch match;
  if (!std::regex_search(yaml, match, pattern))
    throw std::runtime_error("Missing Relay BP parameter: " + key);

  std::vector<double> values;
  std::istringstream input(match[1].str());
  std::string value;
  while (std::getline(input, value, ','))
    values.push_back(std::stod(trim(value)));
  return values;
}

inline bool boolean(const std::string &yaml, const std::string &key) {
  const auto value = scalar(yaml, key);
  if (value == "true")
    return true;
  if (value == "false")
    return false;
  throw std::runtime_error("Invalid boolean Relay BP parameter " + key + ": " +
                           value);
}

} // namespace detail

// Parse plugin arguments separately when the decoder has no registered schema.
inline std::string relay_bp_host_loop_config_yaml(const std::string &yaml) {
  std::istringstream input(yaml);
  std::ostringstream output;
  std::string line;
  bool skipping = false;
  std::size_t custom_args_indent = 0;

  while (std::getline(input, line)) {
    const auto first = line.find_first_not_of(" \t");
    if (!skipping && first != std::string::npos &&
        line.compare(first, 20, "decoder_custom_args:") == 0) {
      skipping = true;
      custom_args_indent = first;
      continue;
    }

    if (skipping) {
      if (first == std::string::npos || first > custom_args_indent)
        continue;
      skipping = false;
    }

    output << line << '\n';
  }

  return output.str();
}

inline cudaqx::heterogeneous_map
relay_bp_host_loop_params(const std::string &yaml) {
  cudaqx::heterogeneous_map params;
  params.insert("use_sparsity", detail::boolean(yaml, "use_sparsity"));
  params.insert("error_rate_vec", detail::double_list(yaml, "error_rate_vec"));
  params.insert("max_iterations",
                std::stoi(detail::scalar(yaml, "max_iterations")));
  params.insert("bp_method", std::stoi(detail::scalar(yaml, "bp_method")));
  params.insert("composition", std::stoi(detail::scalar(yaml, "composition")));
  params.insert("gamma0", std::stod(detail::scalar(yaml, "gamma0")));
  params.insert("clip_value", std::stod(detail::scalar(yaml, "clip_value")));
  params.insert("repeatable", detail::boolean(yaml, "repeatable"));

  cudaqx::heterogeneous_map srelay_params;
  srelay_params.insert("pre_iter", static_cast<std::size_t>(std::stoull(
                                       detail::scalar(yaml, "pre_iter"))));
  srelay_params.insert("num_sets", static_cast<std::size_t>(std::stoull(
                                       detail::scalar(yaml, "num_sets"))));
  srelay_params.insert("stopping_criterion",
                       detail::scalar(yaml, "stopping_criterion"));
  srelay_params.insert("stop_nconv", static_cast<std::size_t>(std::stoull(
                                         detail::scalar(yaml, "stop_nconv"))));
  params.insert("srelay_config", srelay_params);
  params.insert("gamma_dist", detail::double_list(yaml, "gamma_dist"));
  return params;
}

} // namespace test_realtime_qldpc
