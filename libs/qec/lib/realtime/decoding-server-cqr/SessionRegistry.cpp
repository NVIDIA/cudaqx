/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "SessionRegistry.h"
#include "cuda-qx/core/tensor.h"
#include "cudaq/qec/logger.h"
#include "cudaq/qec/realtime/decoding_config.h"

#include <fstream>
#include <iterator>
#include <stdexcept>

namespace cudaq::qec::decoder_server {

using cudaq::qec::decoding::config::multi_decoder_config;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a dense H tensor from the flat -1-terminated H_sparse vector.
static cudaqx::tensor<uint8_t>
make_H_tensor(const std::vector<int64_t> &H_sparse, size_t syndrome_size,
              size_t block_size) {
  cudaqx::tensor<uint8_t> H({syndrome_size, block_size});
  std::fill(H.data(), H.data() + syndrome_size * block_size, uint8_t{0});
  size_t row = 0;
  for (int64_t v : H_sparse) {
    if (v == -1) {
      ++row;
      continue;
    }
    if (row < syndrome_size && static_cast<size_t>(v) < block_size)
      H.data()[row * block_size + static_cast<size_t>(v)] = 1;
  }
  return H;
}

/// Build the default single-VP pass-through syndrome mapping table.
/// mapping_id=0 → VP 0 → empty index list (pass-through)
///
/// An empty index list signals RoundAccumulator to copy bits directly without
/// scatter.  This is correct for the nominal per-round enqueue pattern where
/// the caller sends exactly the syndromes for one round and does not need
/// index remapping.  An identity-sized index list would force every enqueue
/// to provide exactly syndrome_size bits, which breaks per-round batching.
static SyndromeMappingTable
make_default_mapping_table(size_t /*syndrome_size*/) {
  SyndromeMappingTable table;
  table[0] = {{}}; // syndrome_mapping_id=0, VP 0, pass-through
  return table;
}

// ---------------------------------------------------------------------------
// SessionRegistry
// ---------------------------------------------------------------------------

void SessionRegistry::load_from_config(const std::string &yaml_path) {
  std::ifstream f(yaml_path);
  if (!f.is_open())
    throw std::runtime_error("Cannot open config file: " + yaml_path);

  std::string yaml_str((std::istreambuf_iterator<char>(f)),
                       std::istreambuf_iterator<char>());
  load_from_config(multi_decoder_config::from_yaml_str(yaml_str), yaml_path);
}

void SessionRegistry::load_from_config(const multi_decoder_config &config,
                                       const std::string &source_name) {
  for (const auto &dc : config.decoders) {
    const uint64_t id = static_cast<uint64_t>(dc.id);
    if (sessions_.count(id))
      throw std::runtime_error("Duplicate decoder id " + std::to_string(dc.id) +
                               " in " + source_name);

    // All decoders in one server instance must share the same transport type
    // because there is one receive loop per unique transceiver.
    if (sessions_.empty()) {
      transport_ = dc.transport;
    } else if (dc.transport != transport_) {
      throw std::runtime_error(
          "Mixed transport types in " + source_name +
          ": all decoder entries must declare the same transport");
    }

    CUDA_QEC_INFO("SessionRegistry: creating decoder id={} type={}", dc.id,
                  dc.type);

    // Build H tensor.
    auto H = make_H_tensor(dc.H_sparse, static_cast<size_t>(dc.syndrome_size),
                           static_cast<size_t>(dc.block_size));

    // Gather decoder-specific params from decoder_custom_args.
    cudaqx::heterogeneous_map params = std::visit(
        [](const auto &c) -> cudaqx::heterogeneous_map {
          using T = std::decay_t<decltype(c)>;
          if constexpr (std::is_same_v<T, std::monostate>)
            return {};
          else
            return c.to_heterogeneous_map();
        },
        dc.decoder_custom_args);

    // Create decoder session.
    auto mapping_table =
        make_default_mapping_table(static_cast<size_t>(dc.syndrome_size));

    auto session = DecoderSession::create(
        dc.type, cudaq::qec::decoder_init{cudaq::qec::sparse_binary_matrix(H)},
        params, std::move(mapping_table));

    // Configure O and D sparse matrices on the decoder.
    if (!dc.O_sparse.empty())
      session->dec->set_O_sparse(dc.O_sparse);
    if (!dc.D_sparse.empty())
      session->dec->set_D_sparse(dc.D_sparse);

    // [For follow-up] dc.transport (cpu_roce / gpu_roce) is parsed from YAML
    // but not yet used to select a transceiver here. Transport binding requires
    // CpuRoceTransceiverAdapter / GpuRoceTransceiverAdapter (gated on
    // CUDAQ_REALTIME headers); the split-transport DecoderServer constructor
    // is already in place to accept the resulting dispatch map.
    session->start_worker();
    sessions_.emplace(id, std::move(session));
  }

  CUDA_QEC_INFO("SessionRegistry: loaded {} decoder session(s)",
                sessions_.size());
}

DecoderSession &SessionRegistry::get(uint64_t decoder_id) {
  auto it = sessions_.find(decoder_id);
  if (it == sessions_.end())
    throw std::out_of_range("Unknown decoder_id: " +
                            std::to_string(decoder_id));
  return *it->second;
}

const DecoderSession &SessionRegistry::get(uint64_t decoder_id) const {
  auto it = sessions_.find(decoder_id);
  if (it == sessions_.end())
    throw std::out_of_range("Unknown decoder_id: " +
                            std::to_string(decoder_id));
  return *it->second;
}

} // namespace cudaq::qec::decoder_server
