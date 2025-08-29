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
/// build a simple decoder that can decode errors with a small number of errors
/// in the block.
class multi_error_lut : public decoder {
private:
  std::map<std::string, std::vector<std::size_t>> error_signatures;

  // Input parameters
  int lut_error_depth = 1;

  // List of available result types for this decoder
  const std::vector<std::string> available_result_types = {
      "error_probability", // Probability of the detected error (bool)
      "syndrome_weight",   // Number of non-zero syndrome measurements (bool)
      "decoding_time",     // Time taken to perform the decoding (bool)
      "num_repetitions"    // Number of repetitions to perform (int > 0)
  };

  // Output parameters
  bool has_opt_results = false;
  bool error_probability = false;
  bool syndrome_weight = false;
  bool decoding_time = false;
  int num_repetitions = 0;

public:
  multi_error_lut(const cudaqx::tensor<uint8_t> &H,
                  const cudaqx::heterogeneous_map &params)
      : decoder(H) {
    if (params.contains("lut_error_depth")) {
      lut_error_depth = params.get<int>("lut_error_depth");
      if (lut_error_depth < 1) {
        throw std::runtime_error("lut_error_depth must be >= 1");
      }
      if (lut_error_depth > block_size) {
        throw std::runtime_error("lut_error_depth must be <= block_size");
      }
    }
    // Binomial coefficient to check if lut_error_depth is too large
    auto binom = [](int n, int k) {
      return 1 / ((n + 1) * std::beta(n - k + 1, k + 1));
    };
    if (binom(block_size, lut_error_depth) > 1e9) {
      throw std::runtime_error(
          "lut_error_depth is too large for multi_error_lut decoder");
    }
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
    std::vector<std::vector<std::size_t>> H_e2d(block_size);
    for (std::size_t c = 0; c < block_size; c++)
      for (std::size_t r = 0; r < syndrome_size; r++)
        if (H.at({r, c}) != 0)
          H_e2d[c].push_back(r);

    auto toggleSynForError = [&H_e2d](std::string &err_sig, std::size_t qErr) {
      for (std::size_t r : H_e2d[qErr])
        err_sig[r] = err_sig[r] == '1' ? '0' : '1';
    };

    // For each qubit with a possible error, calculate an error signature.
    if (lut_error_depth >= 1) {
      std::string err_sig(syndrome_size, '0');
      for (std::size_t qErr = 0; qErr < block_size; qErr++) {
        toggleSynForError(err_sig, qErr);
        // printf("Adding err_sig=%s for qErr=%lu\n", err_sig.c_str(), qErr);
        error_signatures.insert({err_sig, {qErr}});
        toggleSynForError(err_sig, qErr);
      }
    }
    if (lut_error_depth >= 2) {
      std::string err_sig(syndrome_size, '0');
      for (std::size_t qErr1 = 0; qErr1 < block_size; qErr1++) {
        toggleSynForError(err_sig, qErr1);
        for (std::size_t qErr2 = qErr1 + 1; qErr2 < block_size; qErr2++) {
          toggleSynForError(err_sig, qErr2);
          error_signatures.insert({err_sig, {qErr1, qErr2}});
          toggleSynForError(err_sig, qErr2);
        }
        toggleSynForError(err_sig, qErr1);
      }
    }
    if (lut_error_depth >= 3) {
      std::string err_sig(syndrome_size, '0');
      for (std::size_t qErr1 = 0; qErr1 < block_size; qErr1++) {
        toggleSynForError(err_sig, qErr1);
        for (std::size_t qErr2 = qErr1 + 1; qErr2 < block_size; qErr2++) {
          toggleSynForError(err_sig, qErr2);
          for (std::size_t qErr3 = qErr2 + 1; qErr3 < block_size; qErr3++) {
            toggleSynForError(err_sig, qErr3);
            error_signatures.insert({err_sig, {qErr1, qErr2, qErr3}});
            toggleSynForError(err_sig, qErr3);
          }
          toggleSynForError(err_sig, qErr2);
        }
        toggleSynForError(err_sig, qErr1);
      }
    }
    if (lut_error_depth >= 4) {
      throw std::runtime_error("lut_error_depth >= 4 is not supported");
    }
  }

  virtual decoder_result decode(const std::vector<float_t> &syndrome) {
    // This is a simple decoder with trivial results
    auto t0 = std::chrono::high_resolution_clock::now();
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

    auto it = error_signatures.find(syndrome_str);
    if (it != error_signatures.end()) {
      result.converged = true;
      for (auto qErr : it->second)
        result.result[qErr] = 1.0 - result.result[qErr];
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
        auto t1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = t1 - t0;
        result.opt_results->insert("decoding_time", duration.count());
      }
      if (num_repetitions > 0) {
        result.opt_results->insert("num_repetitions", num_repetitions);
      }
    }

    return result;
  }

  virtual ~multi_error_lut() {}

  CUDAQ_EXTENSION_CUSTOM_CREATOR_FUNCTION(
      multi_error_lut, static std::unique_ptr<decoder> create(
                           const cudaqx::tensor<uint8_t> &H,
                           const cudaqx::heterogeneous_map &params) {
        return std::make_unique<multi_error_lut>(H, params);
      })
};

CUDAQ_REGISTER_TYPE(multi_error_lut)

class single_error_lut : public multi_error_lut {
public:
  single_error_lut(const cudaqx::tensor<uint8_t> &H,
                   const cudaqx::heterogeneous_map &params)
      : multi_error_lut(H, params) {}

  virtual ~single_error_lut() {}

  CUDAQ_EXTENSION_CUSTOM_CREATOR_FUNCTION(
      single_error_lut, static std::unique_ptr<decoder> create(
                            const cudaqx::tensor<uint8_t> &H,
                            const cudaqx::heterogeneous_map &params) {
        return std::make_unique<single_error_lut>(H, params);
      })
};

CUDAQ_REGISTER_TYPE(single_error_lut)

} // namespace cudaq::qec
