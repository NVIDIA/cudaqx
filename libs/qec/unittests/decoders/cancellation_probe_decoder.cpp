/*******************************************************************************
 * Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/qec/decoder.h"

namespace cudaq::qec {

/// @brief Test-only decoder that reports std::nullopt once `tok` reports a
/// requested stop, and a converged result otherwise, so composite decoders
/// can be checked for forwarding the token they were given unchanged.
class cancellation_probe_decoder : public decoder {
public:
  cancellation_probe_decoder(const cudaq::qec::sparse_binary_matrix &H,
                             const cudaqx::heterogeneous_map &params)
      : decoder(H) {}

  using decoder::decode;
  decoder_result decode(const std::vector<float_t> &syndrome) override {
    // A default-constructed token never requests a stop, so this always has
    // a value.
    return decode(syndrome, cancellation_token{}).value();
  }

  std::optional<decoder_result>
  decode(const std::vector<float_t> &syndrome,
        cancellation_token tok) override {
    if (tok.stop_requested())
      return std::nullopt;
    decoder_result result;
    result.converged = true;
    result.result.assign(block_size, 0.0);
    return result;
  }

  using decoder::decode_batch;
  std::vector<std::optional<decoder_result>>
  decode_batch(const std::vector<std::vector<float_t>> &syndromes,
               cancellation_token tok) override {
    std::vector<std::optional<decoder_result>> results;
    for (const auto &s : syndromes)
      results.push_back(decode(s, tok));
    return results;
  }

  CUDAQ_EXTENSION_CUSTOM_CREATOR_FUNCTION(
      cancellation_probe_decoder,
      static std::unique_ptr<decoder> create(
          const cudaq::qec::decoder_init &init,
          const cudaqx::heterogeneous_map &params) {
        return cudaq::qec::make_pcm_decoder<cancellation_probe_decoder>(init,
                                                                        params);
      })
};

CUDAQ_EXT_PT_REGISTER_TYPE(cancellation_probe_decoder)

} // namespace cudaq::qec
