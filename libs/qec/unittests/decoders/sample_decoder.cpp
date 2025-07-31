/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/qec/decoder.h"
#include <vector>

using namespace cudaqx;

namespace cudaq::qec {

/// @brief This is a sample (dummy) decoder that demonstrates how to build a
/// bare bones custom decoder based on the `cudaqx::qec::decoder` interface.
class sample_decoder : public decoder {
public:
  sample_decoder(const cudaqx::tensor<uint8_t> &H,
                 const cudaqx::heterogeneous_map &params)
      : decoder(H) {
    // Decoder-specific constructor arguments can be placed in `params`.
  }

  virtual decoder_result decode(const std::vector<float_t> &syndrome) {
    // This is a simple decoder that simply results
    decoder_result result;
    result.converged = true;
    result.result = std::vector<float_t>(block_size, 0.0f);
    return result;
  }

  virtual ~sample_decoder() {}

  CUDAQ_ADD_EXTENSION_CUSTOM_CREATOR_FUNCTION(
      sample_decoder, static std::unique_ptr<decoder> create(
                          const cudaqx::tensor<uint8_t> &H,
                          const cudaqx::heterogeneous_map &params) {
        return std::make_unique<sample_decoder>(H, params);
      })
};

CUDAQ_REGISTER_EXTENSION(sample_decoder)

} // namespace cudaq::qec
