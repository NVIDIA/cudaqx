/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cuda-qx/core/heterogeneous_map.h"
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace cudaq::qec::decoding::config {

/// Transport type for a decoder session.
/// cpu_roce: CpuRoceTransceiver / SoftRoCE (dev, CI, no GPU required)
/// gpu_roce: GpuRoceTransceiver / DOCA (production, real ConnectX)
enum class DecoderTransport { cpu_roce, gpu_roce };

/// @brief Decoder-specific constructor arguments, stored as a
/// `cudaqx::heterogeneous_map` -- the form every decoder's constructor
/// consumes. YAML conversion and key validation are driven by the parameter
/// schema the decoder registered (see cudaq/qec/decoder_config_schema.h), so
/// out-of-tree decoders participate without any framework changes.
class decoder_custom_args_t {
public:
  decoder_custom_args_t() = default;
  decoder_custom_args_t(const cudaqx::heterogeneous_map &m) : map_(m) {}

  decoder_custom_args_t &operator=(const cudaqx::heterogeneous_map &m) {
    map_ = m;
    return *this;
  }

  cudaqx::heterogeneous_map &map() { return map_; }
  const cudaqx::heterogeneous_map &map() const { return map_; }
  bool empty() const { return map_.empty(); }

  /// Deep equality over the canonical custom-args value kinds.
  __attribute__((visibility("default"))) bool
  operator==(const decoder_custom_args_t &other) const;

private:
  cudaqx::heterogeneous_map map_;
};

/// @brief Configuration structure for decoder options.
struct decoder_config {
  int64_t id = 0;
  std::string type;
  /// Transport used to receive syndromes and send corrections for this decoder.
  /// Defaults to cpu_roce.  Set to gpu_roce for decoders where syndrome bits
  /// are DMA'd directly to GPU VRAM (e.g. nv_qldpc_decoder with RelayBP).
  DecoderTransport transport = DecoderTransport::cpu_roce;
  /// CUDA device this decoder is pinned to at construction (see the
  /// "cuda_device_id" decoder parameter). Placement knob common to any
  /// GPU-accelerated decoder, hence at this level rather than inside the
  /// per-decoder custom args. Unset = unpinned.
  std::optional<int> cuda_device_id;
  uint64_t block_size = 0;
  uint64_t syndrome_size = 0;
  std::vector<std::int64_t> H_sparse;
  std::vector<std::int64_t> O_sparse;
  std::vector<std::int64_t> D_sparse;
  decoder_custom_args_t decoder_custom_args;

  bool operator==(const decoder_config &) const = default;

  /// Return the parameter map a decoder's constructor should receive: the
  /// stored custom args with schema-declared defaults materialized (see
  /// materialize_default_args in cudaq/qec/decoder_config_schema.h) when a
  /// schema is registered for `type`, so programmatically built configs get
  /// the same defaulting the YAML parse path applies.
  __attribute__((visibility("default"))) cudaqx::heterogeneous_map
  decoder_custom_args_to_heterogeneous_map() const;

  /// Validate `decoder_custom_args` against the parameter schema registered
  /// for `type`: unknown keys, missing required keys, and the schema's own
  /// validate hook (if any). Throws std::runtime_error on the first
  /// violation. YAML parsing applies the same checks automatically; call this
  /// to vet a configuration built programmatically before using it.
  __attribute__((visibility("default"))) void validate_custom_args() const;

  __attribute__((visibility("default"))) std::string
  to_yaml_str(int column_wrap = 80);
  __attribute__((visibility("default"))) static decoder_config
  from_yaml_str(const std::string &yaml_str);
};

class multi_decoder_config {
public:
  std::vector<decoder_config> decoders;

  bool operator==(const multi_decoder_config &) const = default;

  /// Validate every decoder's custom args (see
  /// decoder_config::validate_custom_args).
  __attribute__((visibility("default"))) void validate_custom_args() const;

  __attribute__((visibility("default"))) std::string
  to_yaml_str(int column_wrap = 80);
  __attribute__((visibility("default"))) static multi_decoder_config
  from_yaml_str(const std::string_view yaml_str);
};

/// @brief Generate a JSON Schema (draft 2020-12) document describing valid
/// `multi_decoder_config` YAML files, so third-party tools (check-jsonschema,
/// python jsonschema, yaml-language-server, ...) can validate user-provided
/// configurations offline. The per-decoder `decoder_custom_args` sections are
/// generated from the parameter schemas registered at call time, so decoder
/// plugins (in-tree and out-of-tree alike) appear automatically once their
/// library is loaded. Schema `validate` hooks are arbitrary C++ and are not
/// representable; a document that passes the JSON Schema may still be
/// rejected by those hooks when parsed.
__attribute__((visibility("default"))) std::string decoder_config_json_schema();

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

/// @brief Return the most recently passed multi_decoder_config, or an empty
/// pointer if configure_decoders() has not been called in this process.
/// Used by the decoding-server DeviceCallService plugin to build
/// DecodingSessions on the in-process host_dispatch path without requiring
/// CUDAQ_QEC_DECODER_CONFIG.  Returns shared ownership: a concurrent
/// configure_decoders() replaces the stored config but cannot free it out
/// from under the caller.
__attribute__((visibility("default")))
std::shared_ptr<const multi_decoder_config>
last_configured_multi_decoder_config();

} // namespace cudaq::qec::decoding::config
