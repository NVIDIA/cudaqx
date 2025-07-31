/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/qec/decoder.h"
#include <cassert>
#include <map>
#include <vector>

namespace cudaq::qec {

/// @brief This is a simple LUT (LookUp Table) decoder that demonstrates how to
/// build a simple decoder that can correctly decode errors during a single bit
/// flip in the block.
class single_error_lut_example : public decoder {
private:
  std::map<std::string, std::size_t> single_qubit_err_signatures;

public:
  single_error_lut_example(const cudaqx::tensor<uint8_t> &H,
                           const cudaqx::heterogeneous_map &params)
      : decoder(H) {
    // Decoder-specific constructor arguments can be placed in `params`.

    // Build a lookup table for an error on each possible qubit

    // For each qubit with a possible error, calculate an error signature.
    for (std::size_t qErr = 0; qErr < block_size; qErr++) {
      std::string err_sig(syndrome_size, '0');
      for (std::size_t r = 0; r < syndrome_size; r++) {
        bool syndrome = 0;
        // Toggle syndrome on every "1" entry in the row.
        // Except if there is an error on this qubit (c == qErr).
        for (std::size_t c = 0; c < block_size; c++)
          syndrome ^= (c != qErr) && H.at({r, c});
        err_sig[r] = syndrome ? '1' : '0';
      }
      // printf("Adding err_sig=%s for qErr=%lu\n", err_sig.c_str(), qErr);
      single_qubit_err_signatures.insert({err_sig, qErr});
    }
  }

  virtual decoder_result decode(const std::vector<float_t> &syndrome) {
    // This is a simple decoder that simply results
    decoder_result result{false, std::vector<float_t>(block_size, 0.0)};

    // Convert syndrome to a string
    std::string syndrome_str(syndrome.size(), '0');
    assert(syndrome_str.length() == syndrome_size);
    bool anyErrors = false;
    for (std::size_t i = 0; i < syndrome_size; i++) {
      if (syndrome[i] >= 0.5) {
        syndrome_str[i] = '1';
        anyErrors = true;
      }
    }

    if (!anyErrors) {
      result.converged = true;
      return result;
    }

    auto it = single_qubit_err_signatures.find(syndrome_str);
    if (it != single_qubit_err_signatures.end()) {
      assert(it->second < block_size);
      result.converged = true;
      result.result[it->second] = 1.0;
    } else {
      // Leave result.converged set to false.
    }

    return result;
  }

  virtual ~single_error_lut_example() {}

  CUDAQ_ADD_EXTENSION_CUSTOM_CREATOR_FUNCTION(
      single_error_lut_example, static std::unique_ptr<decoder> create(
                                    const cudaqx::tensor<uint8_t> &H,
                                    const cudaqx::heterogeneous_map &params) {
        return std::make_unique<single_error_lut_example>(H, params);
      })
};

CUDAQ_REGISTER_EXTENSION(single_error_lut_example)

} // namespace cudaq::qec
