/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cuda-qx/core/heterogeneous_map.h"
#include "cudaq/qec/extended_dem.h"
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace cudaq::qec::decoding::config {

/// Dispatch shape for a decoder session -- HOW its RPCs are executed, not
/// which wire the bytes arrive on (the wire is a server-level transport
/// provider, selected independently).
/// host:         requests are dispatched on the CPU (HOST_CALL); works with
///               any transport provider (dev, CI, no GPU required).
/// device_graph: requests are dispatched on the GPU by the self-relaunching
///               device-graph scheduler (DeviceGraphTransceiver); requires a
///               decoder with a captured decode graph and a provider whose
///               rings are GPU-visible (e.g. GpuRoceTransceiver/DOCA).
/// YAML key: `dispatch: host|device_graph`.
enum class DecoderDispatch { host, device_graph };

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
  /// Dispatch shape for this decoder's RPCs.  Defaults to host.  Set to
  /// device_graph for decoders where syndrome bits are DMA'd directly to GPU
  /// VRAM and decoded by a captured CUDA graph (e.g. nv_qldpc_decoder with
  /// RelayBP).
  DecoderDispatch dispatch = DecoderDispatch::host;
  /// CUDA device this decoder is pinned to at construction (see the
  /// "cuda_device_id" decoder parameter). Placement knob common to any
  /// GPU-accelerated decoder, hence at this level rather than inside the
  /// per-decoder custom args. Unset = unpinned.
  std::optional<int> cuda_device_id;
  /// The five fields below describe the DEM two alternative ways, and exactly
  /// one of them applies:
  ///
  ///   - Flat form: H_sparse plus block_size, syndrome_size, O_sparse and
  ///     D_sparse, all sized for the whole experiment.
  ///   - Chunk form: dem_chunks plus num_rounds. The other five are derived by
  ///     expanding the phases num_rounds times, and must be omitted.
  ///
  /// See expand_dem_chunks() for the derivation, which runs at decoder
  /// construction so the rest of the pipeline only ever sees the flat form.
  uint64_t block_size = 0;
  uint64_t syndrome_size = 0;
  std::vector<std::int64_t> H_sparse;
  std::vector<std::int64_t> O_sparse;
  std::vector<std::int64_t> D_sparse;
  /// Optional per-phase DEM for a streaming, repeated-round decomposition.
  /// H_sparse above describes the whole experiment as one flat matrix, which
  /// requires knowing the round count up front; these phases describe one
  /// round each so the round count can be chosen (or grown) at run time. See
  /// cudaq::qec::dem_chunks_from_spec() for expansion to a chunk sequence.
  ///
  /// A configuration that also has a nonempty H_sparse is flat, and that
  /// matrix is the one decoders are built from -- the phases are then only a
  /// record of where it came from. Form selection keys off H_sparse.empty(),
  /// so an omitted H_sparse and an explicit empty list both count as chunk
  /// form. Nonempty H_sparse is exactly the state expand_dem_chunks() leaves
  /// behind, which is what lets an expanded configuration round-trip through
  /// YAML; it is not a way to override individual rounds.
  std::optional<dem_chunks_spec> dem_chunks;
  /// How many rounds to expand `dem_chunks` into. Required with dem_chunks and
  /// rejected without it. This is the round count the flat form would otherwise
  /// have baked into its matrix dimensions; naming it here is what lets the
  /// same phase description serve experiments of different lengths.
  ///
  /// This is the *total* round count, so the expansion is init, `num_rounds-2`
  /// bulk copies, then final, and the minimum is 2. Note that the
  /// decoder-tasking spec writes the same construction as `S_R` where `R`
  /// counts bulk copies only: this field is that `R` plus 2.
  std::optional<uint64_t> num_rounds;
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

/// Transport override applied to the rings of one dispatch shape (see the
/// `device_graph` member of `transport_config`).
struct transport_shape_override {
  /// Provider name (e.g. udp, cpu_roce, gpu_roce) or /path/to/lib.so.
  /// A bare name resolves to the CUDA-Q realtime provider library whose
  /// soname is "libcudaq-realtime-bridge-" + name + ".so", with '_' in the
  /// name mapping to '-' to match the shipped hyphenated sonames (so
  /// gpu_roce loads libcudaq-realtime-bridge-gpu-roce.so).
  /// Empty = inherit the section/CLI default.
  std::string provider;
  /// Extra provider arguments appended for this shape's rings.
  std::vector<std::string> args;

  bool operator==(const transport_shape_override &) const = default;
};

/// Server-level transport section: the WIRE is deployment configuration and
/// lives OUTSIDE the decoders list.  Transports differ between rings only by
/// dispatch shape (a device_graph ring must be GPU-pollable), so the only
/// override is shape-keyed -- decoder entries carry no transport
/// information.
///
///   transport:
///     provider: udp
///     args: [--slot-size=256]
///     device_graph:
///       provider: udp          # "gpu_roce" on an HSB rig
///       args: [--pinned-rings]
///
/// Resolution per ring: shape override (device_graph rings) > this
/// section's provider/args > the server's --transport CLI fallback.  The
/// CLI flag only applies when this section names no provider; a config
/// that names one plus an explicit --transport is rejected at startup
/// (the deployment file is the source of truth for the wire).
struct transport_config {
  std::string provider;
  std::vector<std::string> args;
  transport_shape_override device_graph;

  bool operator==(const transport_config &) const = default;
};

class multi_decoder_config {
public:
  std::vector<decoder_config> decoders;
  /// Optional server-level transport section (empty provider/args = not
  /// specified; the server's CLI defaults apply).
  transport_config transport;

  bool operator==(const multi_decoder_config &) const = default;

  /// Validate every decoder's custom args (see
  /// decoder_config::validate_custom_args).
  __attribute__((visibility("default"))) void validate_custom_args() const;

  __attribute__((visibility("default"))) std::string
  to_yaml_str(int column_wrap = 80);
  __attribute__((visibility("default"))) static multi_decoder_config
  from_yaml_str(const std::string_view yaml_str);
};

/// @brief Rewrite a chunk-form configuration into the equivalent flat form,
/// filling block_size, syndrome_size, H_sparse, O_sparse and D_sparse from
/// `dem_chunks` expanded `num_rounds` times. Everything downstream of this
/// therefore only has to understand the flat form.
///
/// Does nothing to a configuration that is already flat (one whose `H_sparse`
/// is nonempty, or which carries no `dem_chunks` at all), so it is safe to
/// call unconditionally. An empty H_sparse with dem_chunks present is still
/// treated as chunk form.
///
/// @return The closed DEM the flat fields were derived from, so a caller that
///         also wants its per-fault priors does not have to expand a second
///         time. Empty when the configuration was already flat.
/// @throws std::runtime_error if `num_rounds` is missing, or if the phases
///         cannot be expanded to that many rounds.
__attribute__((visibility("default")))
std::optional<cudaq::qec::detector_error_model>
expand_dem_chunks(decoder_config &config);

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
