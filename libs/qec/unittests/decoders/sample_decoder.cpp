/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/qec/decoder.h"
#include <stdexcept>
#include <vector>

using namespace cudaqx;

namespace cudaq::qec {

/// @brief This is a sample (dummy) decoder that demonstrates how to build a
/// bare bones custom decoder based on the `cudaq::qec::decoder` interface.
class sample_decoder : public decoder {
public:
  sample_decoder(cudaq::qec::decoder_inputs inputs,
                 decoder_output requested_output,
                 const cudaqx::heterogeneous_map &params)
      : decoder(std::move(inputs), requested_output) {
    // This decoder computes an error frame. Producing observables requires an
    // observable mapping to project through; reject at construction rather
    // than on the first decode.
    if (requested_output == decoder_output::observables &&
        !get_inputs().has_observable_model())
      throw std::invalid_argument(
          "sample_decoder was constructed for observable output but its model "
          "supplies no observable mapping");
  }

  decoder_result decode(const std::vector<float_t> &syndrome) override {
    decoder_result result;
    result.converged = true;
    result.result = std::vector<float_t>(block_size, 0.0f);

    // Whether the frame is projected is fixed at construction, so the decision
    // is read from immutable instance state rather than negotiated per call.
    if (get_output() == decoder_output::observables) {
      std::vector<float_t> observables(get_num_observables(), 0.0);
      project_errors_to_observables(result.result.data(), observables.data(),
                                    observables.size());
      result.result = std::move(observables);
    }
    return result;
  }

  virtual ~sample_decoder() {}

  CUDAQ_EXTENSION_CUSTOM_CREATOR_FUNCTION(
      sample_decoder, static std::unique_ptr<decoder> create(
                          cudaq::qec::decoder_inputs inputs,
                          std::optional<decoder_output> output,
                          const cudaqx::heterogeneous_map &params) {
        return std::make_unique<sample_decoder>(
            std::move(inputs), output.value_or(decoder_output::errors), params);
      })
};

CUDAQ_EXT_PT_REGISTER_TYPE(sample_decoder)

} // namespace cudaq::qec
