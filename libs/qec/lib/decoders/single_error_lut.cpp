/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
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
class single_error_lut : public decoder {
private:
  std::map<std::string, std::size_t> single_qubit_err_signatures;

  // List of available result types for this decoder
  const std::vector<std::string> available_result_types = {
      "error_probability", // Probability of the detected error (bool)
      "syndrome_weight",   // Number of non-zero syndrome measurements (bool)
      "decoding_time",     // Time taken to perform the decoding (bool)
      "num_repetitions"    // Number of repetitions to perform (int > 0)
  };

  bool has_opt_results = false;
  bool error_probability = false;
  bool syndrome_weight = false;
  bool decoding_time = false;
  int num_repetitions = 0;

public:
  single_error_lut(const cudaqx::tensor<uint8_t> &H,
                   const cudaqx::heterogeneous_map &params)
      : decoder(H) {
    // Decoder-specific constructor arguments can be placed in `params`.
    // Check if opt_results was requested
    if (params.contains("opt_results")) {
      try {
        auto requested_results =
            params.get<cudaqx::heterogeneous_map>("opt_results");

        // Validate requested result types
        auto invalid_types = validate_config_parameters(requested_results,
                                                        available_result_types);

        if (!invalid_types.empty()) {
          std::string error_msg = "Requested result types not available in "
                                  "single_error_lut decoder: ";
          for (size_t i = 0; i < invalid_types.size(); ++i) {
            error_msg += invalid_types[i];
            if (i < invalid_types.size() - 1) {
              error_msg += ", ";
            }
          }
          throw std::runtime_error(error_msg);
        } else {
          has_opt_results = true;
          error_probability = requested_results.get<bool>("error_probability",
                                                          error_probability);
          syndrome_weight =
              requested_results.get<bool>("syndrome_weight", syndrome_weight);
          decoding_time =
              requested_results.get<bool>("decoding_time", decoding_time);
          num_repetitions =
              requested_results.get<int>("num_repetitions", num_repetitions);
        }
      } catch (const std::runtime_error &e) {
        throw; // Re-throw if it's our error
      } catch (...) {
        throw std::runtime_error("opt_results must be a heterogeneous_map");
      }
    }

    // Build a lookup table for an error on each possible qubit

    // For each qubit with a possible error, calculate an error signature.
    for (std::size_t qErr = 0; qErr < block_size; qErr++) {
      std::string err_sig(syndrome_size, '0');
      for (std::size_t r = 0; r < syndrome_size; r++) {
        err_sig[r] = H.at({r, qErr}) ? '1' : '0';
      }
      // printf("Adding err_sig=%s for qErr=%lu\n", err_sig.c_str(), qErr);
      single_qubit_err_signatures.insert({err_sig, qErr});
    }
  }

  virtual decoder_result decode(const std::vector<float_t> &syndrome) {
    // This is a simple decoder with trivial results
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

    // Add opt_results if requested
    /*
     * Example opt_results map:
     * {
     *   "error_probability": true,    // Include error probability in results
     *   "syndrome_weight": true,      // Include syndrome weight in results
     *   "decoding_time": false,       // Don't include decoding time
     *   "num_repetitions": 5          // Include num_repetitions=5 in results
     * }
     */
    if (has_opt_results) {
      result.opt_results =
          cudaqx::heterogeneous_map(); // Initialize the optional map
      // Values are for demonstration purposes only.
      if (error_probability) {
        result.opt_results->insert("error_probability", 1.0);
      }
      if (syndrome_weight) {
        result.opt_results->insert("syndrome_weight", 1);
      }
      if (decoding_time) {
        result.opt_results->insert("decoding_time", 0.0);
      }
      if (num_repetitions > 0) {
        result.opt_results->insert("num_repetitions", num_repetitions);
      }
    }

    return result;
  }

  virtual ~single_error_lut() {}

  CUDAQ_ADD_EXTENSION_CUSTOM_CREATOR_FUNCTION(
      single_error_lut, static std::unique_ptr<decoder> create(
                            const cudaqx::tensor<uint8_t> &H,
                            const cudaqx::heterogeneous_map &params) {
        return std::make_unique<single_error_lut>(H, params);
      })
};

CUDAQ_REGISTER_EXTENSION(single_error_lut)

} // namespace cudaq::qec
