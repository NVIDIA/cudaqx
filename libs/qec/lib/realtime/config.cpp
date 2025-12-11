/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/ExtraPayloadProvider.h"
#include "common/Logger.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Base64.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Support/raw_ostream.h"
#include "realtime_decoding.h"
#include "cudaq/qec/realtime/decoding_config.h"
#include <filesystem>
#include <fstream>
#include <type_traits>

// Helper function(s) to remove the optional wrapper from a type.
// Primary template for non-optional types
template <typename T>
struct remove_optional {
  using type = T;
};
// Partial specialization for std::optional<T>
template <typename T>
struct remove_optional<std::optional<T>> {
  using type = T;
};
// Alias
template <typename T>
using remove_optional_t = typename remove_optional<T>::type;

namespace cudaq::qec::decoding::config {

#define INSERT_ARG(arg_name)                                                   \
  do {                                                                         \
    if (this->arg_name.has_value()) {                                          \
      config_map.insert(#arg_name, this->arg_name.value());                    \
    }                                                                          \
  } while (false)

#define INSERT_ARG_PLAIN(arg_name)                                             \
  do {                                                                         \
    config_map.insert(#arg_name, this->arg_name);                              \
  } while (false)

#define GET_ARG(arg_name)                                                      \
  do {                                                                         \
    if (map.contains(#arg_name)) {                                             \
      config.arg_name =                                                        \
          map.get<remove_optional_t<decltype(config.arg_name)>>(#arg_name);    \
    }                                                                          \
  } while (false)

#define GET_ARG_PLAIN(arg_name)                                                \
  do {                                                                         \
    if (map.contains(#arg_name)) {                                             \
      config.arg_name =                                                        \
          map.get<std::decay_t<decltype(config.arg_name)>>(#arg_name);         \
    }                                                                          \
  } while (false)

// ------ relay_bp_config ------
cudaqx::heterogeneous_map srelay_bp_config::to_heterogeneous_map() const {
  cudaqx::heterogeneous_map config_map;

  INSERT_ARG(pre_iter);
  INSERT_ARG(num_sets);
  INSERT_ARG(stopping_criterion);
  INSERT_ARG(stop_nconv);

  return config_map;
}

srelay_bp_config
srelay_bp_config::from_heterogeneous_map(const cudaqx::heterogeneous_map &map) {
  srelay_bp_config config;
  GET_ARG(pre_iter);
  GET_ARG(num_sets);
  GET_ARG(stopping_criterion);
  GET_ARG(stop_nconv);

  return config;
}

// ------ nv_qldpc_decoder_config ------
cudaqx::heterogeneous_map
nv_qldpc_decoder_config::to_heterogeneous_map() const {
  cudaqx::heterogeneous_map config_map;

  INSERT_ARG(use_sparsity);
  INSERT_ARG(error_rate);
  INSERT_ARG(error_rate_vec);
  INSERT_ARG(max_iterations);
  INSERT_ARG(n_threads);
  INSERT_ARG(use_osd);
  INSERT_ARG(osd_method);
  INSERT_ARG(osd_order);
  INSERT_ARG(bp_batch_size);
  INSERT_ARG(osd_batch_size);
  INSERT_ARG(iter_per_check);
  INSERT_ARG(clip_value);
  INSERT_ARG(bp_method);
  INSERT_ARG(scale_factor);
  INSERT_ARG(proc_float);
  INSERT_ARG(gamma0);
  INSERT_ARG(gamma_dist);
  INSERT_ARG(explicit_gammas);
  INSERT_ARG(bp_seed);
  INSERT_ARG(composition);
  // srelay_config needs to be converted to heterogeneous_map for decoder
  // compatibility
  if (srelay_config.has_value()) {
    config_map.insert("srelay_config", srelay_config->to_heterogeneous_map());
  }

  return config_map;
}

nv_qldpc_decoder_config nv_qldpc_decoder_config::from_heterogeneous_map(
    const cudaqx::heterogeneous_map &map) {
  nv_qldpc_decoder_config config;
  GET_ARG(use_sparsity);
  GET_ARG(error_rate);
  GET_ARG(error_rate_vec);
  GET_ARG(max_iterations);
  GET_ARG(n_threads);
  GET_ARG(use_osd);
  GET_ARG(osd_method);
  GET_ARG(osd_order);
  GET_ARG(bp_batch_size);
  GET_ARG(osd_batch_size);
  GET_ARG(iter_per_check);
  GET_ARG(clip_value);
  GET_ARG(bp_method);
  GET_ARG(scale_factor);
  GET_ARG(proc_float);
  GET_ARG(gamma0);
  GET_ARG(gamma_dist);
  GET_ARG(explicit_gammas);
  GET_ARG(bp_seed);
  GET_ARG(composition);
  // Special handling for srelay_config - it might be stored as a nested
  // heterogeneous_map
  if (map.contains("srelay_config")) {
    try {
      // Try to get it as a srelay_bp_config first (for C++ -> C++)
      config.srelay_config = map.get<srelay_bp_config>("srelay_config");
    } catch (...) {
      // If that fails, try to get it as a heterogeneous_map (for Python
      // round-trip)
      auto nested_map = map.get<cudaqx::heterogeneous_map>("srelay_config");
      config.srelay_config =
          srelay_bp_config::from_heterogeneous_map(nested_map);
    }
  }

  return config;
}

// ------ multi_error_lut_config ------
cudaqx::heterogeneous_map multi_error_lut_config::to_heterogeneous_map() const {
  cudaqx::heterogeneous_map config_map;
  INSERT_ARG(lut_error_depth);
  return config_map;
}

multi_error_lut_config multi_error_lut_config::from_heterogeneous_map(
    const cudaqx::heterogeneous_map &map) {
  multi_error_lut_config config;
  GET_ARG(lut_error_depth);
  return config;
}

// ------ single_error_lut_config ------
cudaqx::heterogeneous_map
single_error_lut_config::to_heterogeneous_map() const {
  cudaqx::heterogeneous_map config_map;
  return config_map;
}

single_error_lut_config single_error_lut_config::from_heterogeneous_map(
    const cudaqx::heterogeneous_map &map) {
  single_error_lut_config config;
  return config;
}

// ------ trt_decoder_config ------
cudaqx::heterogeneous_map trt_decoder_config::to_heterogeneous_map() const {
  cudaqx::heterogeneous_map config_map;

  INSERT_ARG(onnx_load_path);
  INSERT_ARG(engine_load_path);
  INSERT_ARG(engine_save_path);
  INSERT_ARG(precision);
  INSERT_ARG(memory_workspace);

  return config_map;
}

trt_decoder_config trt_decoder_config::from_heterogeneous_map(
    const cudaqx::heterogeneous_map &map) {
  trt_decoder_config config;
  GET_ARG(onnx_load_path);
  GET_ARG(engine_load_path);
  GET_ARG(engine_save_path);
  GET_ARG(precision);
  GET_ARG(memory_workspace);

  return config;
}

// ------ sliding_window_config ------
cudaqx::heterogeneous_map sliding_window_config::to_heterogeneous_map() const {
  cudaqx::heterogeneous_map config_map;
  INSERT_ARG(window_size);
  INSERT_ARG(step_size);
  INSERT_ARG(num_syndromes_per_round);
  INSERT_ARG(straddle_start_round);
  INSERT_ARG(straddle_end_round);
  INSERT_ARG_PLAIN(error_rate_vec);
  INSERT_ARG_PLAIN(inner_decoder_name);

  // Handle concrete inner decoder configs
  cudaqx::heterogeneous_map inner_decoder_params;
  if (single_error_lut_params.has_value()) {
    inner_decoder_params =
        single_error_lut_params.value().to_heterogeneous_map();
  } else if (multi_error_lut_params.has_value()) {
    inner_decoder_params =
        multi_error_lut_params.value().to_heterogeneous_map();
  } else if (nv_qldpc_decoder_params.has_value()) {
    inner_decoder_params =
        nv_qldpc_decoder_params.value().to_heterogeneous_map();
  }
  if (!inner_decoder_params.empty()) {
    config_map.insert("inner_decoder_params", inner_decoder_params);
  }

  return config_map;
}

sliding_window_config sliding_window_config::from_heterogeneous_map(
    const cudaqx::heterogeneous_map &map) {
  sliding_window_config config;
  GET_ARG(window_size);
  GET_ARG(step_size);
  GET_ARG(num_syndromes_per_round);
  GET_ARG(straddle_start_round);
  GET_ARG(straddle_end_round);
  GET_ARG_PLAIN(error_rate_vec);
  GET_ARG_PLAIN(inner_decoder_name);

  // Reconstruct inner decoder configs based on the decoder type
  if (!config.inner_decoder_name.empty() &&
      map.contains("inner_decoder_params")) {
    const auto &inner_decoder_params =
        map.get<cudaqx::heterogeneous_map>("inner_decoder_params");
    const std::string &decoder_name = config.inner_decoder_name;

    if (decoder_name == "single_error_lut") {
      config.single_error_lut_params =
          single_error_lut_config::from_heterogeneous_map(inner_decoder_params);
    } else if (decoder_name == "multi_error_lut") {
      config.multi_error_lut_params =
          multi_error_lut_config::from_heterogeneous_map(inner_decoder_params);
    } else if (decoder_name == "nv-qldpc-decoder") {
      config.nv_qldpc_decoder_params =
          nv_qldpc_decoder_config::from_heterogeneous_map(inner_decoder_params);
    }
  }

  return config;
}

#undef INSERT_ARG
#undef GET_ARG

} // namespace cudaq::qec::decoding::config

LLVM_YAML_IS_SEQUENCE_VECTOR(std::vector<double>)
LLVM_YAML_IS_SEQUENCE_VECTOR(cudaq::qec::decoding::config::decoder_config)

namespace llvm::yaml {

template <>
struct MappingTraits<cudaq::qec::decoding::config::srelay_bp_config> {
  static void mapping(IO &io,
                      cudaq::qec::decoding::config::srelay_bp_config &config) {
    io.mapOptional("pre_iter", config.pre_iter);
    io.mapOptional("num_sets", config.num_sets);
    io.mapOptional("stopping_criterion", config.stopping_criterion);
    io.mapOptional("stop_nconv", config.stop_nconv);
  }
};

template <>
struct MappingTraits<cudaq::qec::decoding::config::nv_qldpc_decoder_config> {
  static void
  mapping(IO &io,
          cudaq::qec::decoding::config::nv_qldpc_decoder_config &config) {
    io.mapOptional("use_sparsity", config.use_sparsity);
    io.mapOptional("error_rate", config.error_rate);
    io.mapOptional("error_rate_vec", config.error_rate_vec);
    io.mapOptional("max_iterations", config.max_iterations);
    io.mapOptional("n_threads", config.n_threads);
    io.mapOptional("use_osd", config.use_osd);
    io.mapOptional("osd_method", config.osd_method);
    io.mapOptional("osd_order", config.osd_order);
    io.mapOptional("bp_batch_size", config.bp_batch_size);
    io.mapOptional("osd_batch_size", config.osd_batch_size);
    io.mapOptional("iter_per_check", config.iter_per_check);
    io.mapOptional("clip_value", config.clip_value);
    io.mapOptional("bp_method", config.bp_method);
    io.mapOptional("scale_factor", config.scale_factor);
    io.mapOptional("proc_float", config.proc_float);
    io.mapOptional("gamma0", config.gamma0);
    io.mapOptional("gamma_dist", config.gamma_dist);
    io.mapOptional("explicit_gammas", config.explicit_gammas);
    io.mapOptional("bp_seed", config.bp_seed);
    io.mapOptional("srelay_config", config.srelay_config);
    io.mapOptional("composition", config.composition);
  }
};

template <>
struct MappingTraits<cudaq::qec::decoding::config::multi_error_lut_config> {
  static void
  mapping(IO &io,
          cudaq::qec::decoding::config::multi_error_lut_config &config) {
    io.mapOptional("lut_error_depth", config.lut_error_depth);
  }
};

template <>
struct MappingTraits<cudaq::qec::decoding::config::single_error_lut_config> {
  static void
  mapping(IO &io,
          cudaq::qec::decoding::config::single_error_lut_config &config) {}
};

template <>
struct MappingTraits<cudaq::qec::decoding::config::trt_decoder_config> {
  static void
  mapping(IO &io, cudaq::qec::decoding::config::trt_decoder_config &config) {
    io.mapOptional("onnx_load_path", config.onnx_load_path);
    io.mapOptional("engine_load_path", config.engine_load_path);
    io.mapOptional("engine_save_path", config.engine_save_path);
    io.mapOptional("precision", config.precision);
    io.mapOptional("memory_workspace", config.memory_workspace);
  }
};

template <>
struct MappingTraits<cudaq::qec::decoding::config::sliding_window_config> {
  static void
  mapping(IO &io, cudaq::qec::decoding::config::sliding_window_config &config) {
    io.mapOptional("window_size", config.window_size);
    io.mapOptional("step_size", config.step_size);
    io.mapOptional("num_syndromes_per_round", config.num_syndromes_per_round);
    io.mapOptional("straddle_start_round", config.straddle_start_round);
    io.mapOptional("straddle_end_round", config.straddle_end_round);
    io.mapRequired("error_rate_vec", config.error_rate_vec);
    io.mapRequired("inner_decoder_name", config.inner_decoder_name);

    // Concrete inner decoder configurations
    if (config.inner_decoder_name == "single_error_lut") {
      io.mapOptional("inner_decoder_params", config.single_error_lut_params);
    } else if (config.inner_decoder_name == "multi_error_lut") {
      io.mapOptional("inner_decoder_params", config.multi_error_lut_params);
    } else if (config.inner_decoder_name == "nv-qldpc-decoder") {
      io.mapOptional("inner_decoder_params", config.nv_qldpc_decoder_params);
    }
  }
};

template <>
struct MappingTraits<cudaq::qec::decoding::config::decoder_config> {
  static void mapping(IO &io,
                      cudaq::qec::decoding::config::decoder_config &config) {
    io.mapRequired("id", config.id);
    io.mapRequired("type", config.type);
    io.mapRequired("block_size", config.block_size);
    io.mapRequired("syndrome_size", config.syndrome_size);
    io.mapRequired("H_sparse", config.H_sparse);
    io.mapRequired("O_sparse", config.O_sparse);
    io.mapRequired("D_sparse", config.D_sparse);

    // Validate that the number of rows in the H_sparse vector is equal to
    // syndrome_size.
    auto num_H_rows =
        std::count(config.H_sparse.begin(), config.H_sparse.end(), -1);
    if (num_H_rows != config.syndrome_size) {
      throw std::runtime_error(
          "Number of rows in H_sparse vector is not equal to syndrome_size: " +
          std::to_string(num_H_rows) +
          " != " + std::to_string(config.syndrome_size));
    }

    // Validate that no values in the H_sparse vector are out of range.
    for (auto value : config.H_sparse) {
      if (value < -1 || (value >= 0 && value >= config.block_size)) {
        throw std::runtime_error("Value in H_sparse vector is out of range: " +
                                 std::to_string(value));
      }
    }

    // Validate that no values in the O_sparse vector are out of range.
    for (auto value : config.O_sparse) {
      if (value < -1 || (value >= 0 && value >= config.block_size)) {
        throw std::runtime_error("Value in O_sparse vector is out of range: " +
                                 std::to_string(value));
      }
    }

    // Validate that if the D_sparse is provided, it is a valid D matrix. That
    // means that the number of rows in the D_sparse matrix should be equal to
    // the number of rows in the H_sparse matrix, and no row should be empty.
    if (!config.D_sparse.empty()) {
      auto num_D_rows =
          std::count(config.D_sparse.begin(), config.D_sparse.end(), -1);
      if (num_D_rows != config.syndrome_size) {
        throw std::runtime_error("Number of rows in D_sparse vector is not "
                                 "equal to syndrome_size: " +
                                 std::to_string(num_D_rows) +
                                 " != " + std::to_string(config.syndrome_size));
      }
      // No row should be empty, which means that there should be no
      // back-to-back -1 values.
      for (std::size_t i = 0; i < config.D_sparse.size() - 1; ++i) {
        if (config.D_sparse.at(i) == -1 && config.D_sparse.at(i + 1) == -1) {
          throw std::runtime_error("D_sparse row is empty for decoder " +
                                   std::to_string(config.id));
        }
      }
    }
#define INIT_AND_MAP_DECODER_CUSTOM_ARGS(type)                                 \
  do {                                                                         \
    if (!std::holds_alternative<type>(config.decoder_custom_args)) {           \
      config.decoder_custom_args = type();                                     \
    }                                                                          \
    io.mapOptional("decoder_custom_args",                                      \
                   std::get<type>(config.decoder_custom_args));                \
  } while (false)

    if (config.type == "nv-qldpc-decoder") {
      INIT_AND_MAP_DECODER_CUSTOM_ARGS(
          cudaq::qec::decoding::config::nv_qldpc_decoder_config);
    } else if (config.type == "multi_error_lut") {
      INIT_AND_MAP_DECODER_CUSTOM_ARGS(
          cudaq::qec::decoding::config::multi_error_lut_config);
    } else if (config.type == "single_error_lut") {
      INIT_AND_MAP_DECODER_CUSTOM_ARGS(
          cudaq::qec::decoding::config::single_error_lut_config);
    } else if (config.type == "trt_decoder") {
      INIT_AND_MAP_DECODER_CUSTOM_ARGS(
          cudaq::qec::decoding::config::trt_decoder_config);
    } else if (config.type == "sliding_window") {
      INIT_AND_MAP_DECODER_CUSTOM_ARGS(
          cudaq::qec::decoding::config::sliding_window_config);
    }
  }
};

// multi_decoder_config mapping traits
template <>
struct MappingTraits<cudaq::qec::decoding::config::multi_decoder_config> {
  static void
  mapping(IO &io, cudaq::qec::decoding::config::multi_decoder_config &config) {
    io.mapRequired("decoders", config.decoders);
  }
};

} // namespace llvm::yaml

// Static method to convert a YAML string to a multi_decoder_config.
cudaq::qec::decoding::config::multi_decoder_config
cudaq::qec::decoding::config::multi_decoder_config::from_yaml_str(
    const std::string_view yaml_str) {
  multi_decoder_config config;
  llvm::yaml::Input yaml_in(yaml_str);
  yaml_in >> config;
  return config;
}

std::string cudaq::qec::decoding::config::multi_decoder_config::to_yaml_str(
    int column_wrap) {
  std::string yaml_str;
  llvm::raw_string_ostream yaml_stream(yaml_str);
  llvm::yaml::Output yaml_out(yaml_stream, nullptr, column_wrap);
  yaml_out << *this;
  return yaml_str;
}

cudaq::qec::decoding::config::decoder_config
cudaq::qec::decoding::config::decoder_config::from_yaml_str(
    const std::string &yaml_str) {
  decoder_config config;
  llvm::yaml::Input yaml_in(yaml_str);
  yaml_in >> config;
  return config;
}

std::string
cudaq::qec::decoding::config::decoder_config::to_yaml_str(int column_wrap) {
  std::string yaml_str;
  llvm::raw_string_ostream yaml_stream(yaml_str);
  llvm::yaml::Output yaml_out(yaml_stream, nullptr, column_wrap);
  yaml_out << *this;
  return yaml_str;
}

namespace {
/// \brief Provides extra payload for decoder-related messages.
class decoder_provider : public cudaq::ExtraPayloadProvider {
  /// The configuration YML string for the decoder to be injected to job
  /// requests.
  // Note: we convert the multi_decoder_config to a YAML string so that it can
  // be reused across multiple requests without needing to re-parse the YAML
  // each time.
  std::string decoderConfigYmlStr;

public:
  decoder_provider(cudaq::qec::decoding::config::multi_decoder_config &config)
      : decoderConfigYmlStr(config.to_yaml_str()) {}
  virtual ~decoder_provider() = default;
  virtual std::string name() const override { return "decoder"; }
  virtual std::string getPayloadType() const override {
    return "gpu_decoder_config";
  }
  virtual std::string
  getExtraPayload(const cudaq::RuntimeTarget &target) override {
    return decoderConfigYmlStr;
  }
};
} // namespace

namespace cudaq::qec::decoding::config {
int configure_decoders(multi_decoder_config &config) {
  CUDAQ_INFO("Initializing realtime decoding library with config object");
  // Register the decoder provider to inject the decoder configuration into
  // the job requests.
  cudaq::registerExtraPayloadProvider(
      std::make_unique<decoder_provider>(config));
  return cudaq::qec::decoding::host::configure_decoders(config);
}

void log_config(const char *config_str, bool from_file) {
  const bool dump_config = []() {
    if (auto *ch = std::getenv("CUDAQ_QEC_DEBUG_DECODER"))
      if (ch[0] == '1' || ch[0] == 'y' || ch[0] == 'Y')
        return true;
    return false;
  }();

  if (dump_config) {
    if (cudaq::details::should_log(cudaq::details::LogLevel::info)) {
      CUDAQ_INFO(
          "Initializing realtime decoding library with config string: {}",
          config_str);
    } else {
      printf("Initializing realtime decoding library with config string: %s\n",
             config_str);
    }
  }
}

int configure_decoders_from_file(const char *config_file) {
  std::string config_file_str(config_file);
  CUDAQ_INFO("Initializing realtime decoding library with config file: {}",
             config_file_str);

  // Verify that the file exists.
  if (!std::filesystem::exists(config_file_str)) {
    CUDAQ_WARN("Config file does not exist: {}", config_file_str);
    return 1;
  }

  // Read the config file into a string.
  std::string config_str;
  std::ifstream config_file_stream(config_file_str);
  config_str = std::string(std::istreambuf_iterator<char>(config_file_stream),
                           std::istreambuf_iterator<char>());
  log_config(config_str.c_str(), /*from_file=*/true);
  auto config = multi_decoder_config::from_yaml_str(config_str);
  return configure_decoders(config);
}

int configure_decoders_from_str(const char *config_str) {
  CUDAQ_INFO("Initializing realtime decoding library with raw config string");
  log_config(config_str, /*from_file=*/false);
  auto config = multi_decoder_config::from_yaml_str(config_str);
  return configure_decoders(config);
}

void finalize_decoders() { cudaq::qec::decoding::host::finalize_decoders(); }

} // namespace cudaq::qec::decoding::config
