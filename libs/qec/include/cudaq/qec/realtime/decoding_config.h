/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cuda-qx/core/heterogeneous_map.h"
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace cudaq::qec::decoding::config {

struct srelay_bp_config {
  std::optional<std::size_t> pre_iter;
  std::optional<std::size_t> num_sets;
  std::optional<std::string> stopping_criterion;
  std::optional<std::size_t> stop_nconv;

  bool operator==(const srelay_bp_config &) const = default;

  __attribute__((visibility("default"))) cudaqx::heterogeneous_map
  to_heterogeneous_map() const;

  __attribute__((visibility("default"))) static srelay_bp_config
  from_heterogeneous_map(const cudaqx::heterogeneous_map &map);
};

struct nv_qldpc_decoder_config {
  std::optional<bool> use_sparsity;
  std::optional<double> error_rate;
  std::optional<std::vector<double>> error_rate_vec;
  std::optional<int> max_iterations;
  std::optional<int> n_threads;
  std::optional<bool> use_osd;
  std::optional<int> osd_method;
  std::optional<int> osd_order;
  std::optional<int> bp_batch_size;
  std::optional<int> osd_batch_size;
  std::optional<int> iter_per_check;
  std::optional<double> clip_value;
  std::optional<int> bp_method;
  std::optional<double> scale_factor;
  std::optional<std::string> proc_float;
  std::optional<double> gamma0;
  std::optional<std::vector<double>> gamma_dist;
  std::optional<std::vector<std::vector<double>>> explicit_gammas;
  std::optional<srelay_bp_config> srelay_config;
  std::optional<int> bp_seed;
  std::optional<int> composition;

  bool operator==(const nv_qldpc_decoder_config &) const = default;
  // opt_results is currently not supported for real-time decoding.

  __attribute__((visibility("default"))) cudaqx::heterogeneous_map
  to_heterogeneous_map() const;

  __attribute__((visibility("default"))) static nv_qldpc_decoder_config
  from_heterogeneous_map(const cudaqx::heterogeneous_map &map);
};

struct multi_error_lut_config {
  std::optional<int> lut_error_depth;
  // error_rate_vec is currently not supported for real-time decoding.
  // opt_results is currently not supported for real-time decoding.

  bool operator==(const multi_error_lut_config &) const = default;

  __attribute__((visibility("default"))) cudaqx::heterogeneous_map
  to_heterogeneous_map() const;

  __attribute__((visibility("default"))) static multi_error_lut_config
  from_heterogeneous_map(const cudaqx::heterogeneous_map &map);
};

struct single_error_lut_config {
  bool operator==(const single_error_lut_config &) const = default;

  __attribute__((visibility("default"))) cudaqx::heterogeneous_map
  to_heterogeneous_map() const;

  __attribute__((visibility("default"))) static single_error_lut_config
  from_heterogeneous_map(const cudaqx::heterogeneous_map &map);
};

struct sliding_window_config {
  std::optional<std::size_t> window_size;
  std::optional<std::size_t> step_size;
  std::optional<std::size_t> num_syndromes_per_round;
  std::optional<bool> straddle_start_round;
  std::optional<bool> straddle_end_round;
  std::vector<double> error_rate_vec;
  std::string inner_decoder_name;

  // Concrete inner decoder configurations (only one should be set based on
  // inner_decoder_name)
  std::optional<single_error_lut_config> single_error_lut_params;
  std::optional<multi_error_lut_config> multi_error_lut_params;
  std::optional<nv_qldpc_decoder_config> nv_qldpc_decoder_params;

  bool operator==(const sliding_window_config &) const = default;

  __attribute__((visibility("default"))) cudaqx::heterogeneous_map
  to_heterogeneous_map() const;

  __attribute__((visibility("default"))) static sliding_window_config
  from_heterogeneous_map(const cudaqx::heterogeneous_map &map);
};

/// @brief Configuration structure for decoder options.
struct decoder_config {
  int64_t id = 0;
  std::string type;
  uint64_t block_size = 0;
  uint64_t syndrome_size = 0;
  std::vector<std::int64_t> H_sparse;
  std::vector<std::int64_t> O_sparse;
  std::optional<std::vector<std::int64_t>> D_sparse;
  std::variant<single_error_lut_config, multi_error_lut_config,
               nv_qldpc_decoder_config, sliding_window_config>
      decoder_custom_args;

  bool operator==(const decoder_config &) const = default;

  __attribute__((visibility("default"))) cudaqx::heterogeneous_map
  decoder_custom_args_to_heterogeneous_map() const {
    if (std::holds_alternative<single_error_lut_config>(decoder_custom_args)) {
      return std::get<single_error_lut_config>(decoder_custom_args)
          .to_heterogeneous_map();
    } else if (std::holds_alternative<multi_error_lut_config>(
                   decoder_custom_args)) {
      return std::get<multi_error_lut_config>(decoder_custom_args)
          .to_heterogeneous_map();
    } else if (std::holds_alternative<nv_qldpc_decoder_config>(
                   decoder_custom_args)) {
      return std::get<nv_qldpc_decoder_config>(decoder_custom_args)
          .to_heterogeneous_map();
    } else if (std::holds_alternative<sliding_window_config>(
                   decoder_custom_args)) {
      return std::get<sliding_window_config>(decoder_custom_args)
          .to_heterogeneous_map();
    }
    return cudaqx::heterogeneous_map();
  }

  __attribute__((visibility("default"))) void
  set_decoder_custom_args_from_heterogeneous_map(
      const cudaqx::heterogeneous_map &map) {
    if (type == "single_error_lut") {
      decoder_custom_args =
          single_error_lut_config::from_heterogeneous_map(map);
    } else if (type == "multi_error_lut") {
      decoder_custom_args = multi_error_lut_config::from_heterogeneous_map(map);
    } else if (type == "nv-qldpc-decoder") {
      decoder_custom_args =
          nv_qldpc_decoder_config::from_heterogeneous_map(map);
    } else if (type == "sliding_window") {
      decoder_custom_args = sliding_window_config::from_heterogeneous_map(map);
    }
  }

  __attribute__((visibility("default"))) std::string
  to_yaml_str(int column_wrap = 80);
  __attribute__((visibility("default"))) static decoder_config
  from_yaml_str(const std::string &yaml_str);
};

class multi_decoder_config {
public:
  std::vector<decoder_config> decoders;

  bool operator==(const multi_decoder_config &) const = default;

  __attribute__((visibility("default"))) std::string
  to_yaml_str(int column_wrap = 80);
  __attribute__((visibility("default"))) static multi_decoder_config
  from_yaml_str(const std::string_view yaml_str);
};

/// @brief Configure the decoders (`multi_decoder_config` variant). This
/// function configures both local decoders, and if running on remote target
/// hardware, will submit the configuration to the remote target for further
/// processing.
/// @param config The configuration to use.
/// @return 0 on success, non-zero on failure.
__attribute__((visibility("default"))) int
configure_decoders(multi_decoder_config &config);

/// @brief Configure the decoders from a file. This function configures both
/// local decoders, and if running on remote target hardware, will submit the
/// configuration to the remote target for further processing.
/// @param config_file The file to read the configuration from.
/// @return 0 on success, non-zero on failure.
__attribute__((visibility("default"))) int
configure_decoders_from_file(const char *config_file);

/// @brief Configure the decoders from a string. This function configures both
/// local decoders, and if running on remote target hardware, will submit the
/// configuration to the remote target for further processing.
/// @param config_str The string to read the configuration from.
/// @return 0 on success, non-zero on failure.
__attribute__((visibility("default"))) int
configure_decoders_from_str(const char *config_str);

/// @brief Finalize the decoders. This function finalizes local decoders.
__attribute__((visibility("default"))) void finalize_decoders();

} // namespace cudaq::qec::decoding::config
