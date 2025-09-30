/*******************************************************************************
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/Logger.h"
#include "cudaq/qec/decoder.h"
#include "cudaq/qec/pcm_utils.h"
#include <cassert>
#include <vector>

namespace cudaq::qec {

/// @brief This is a sliding window decoder that receives syndromes on a
/// round-by-round basis, and decodes them according window-specific parameters
/// provided in the decoder.
class sliding_window : public decoder {
private:
  // Input parameters.
  std::size_t window_size = 1;
  std::size_t step_size = 1;
  std::size_t num_syndromes_per_round = 0;
  bool straddle_start_round = false;
  bool straddle_end_round = true;
  std::vector<cudaq::qec::float_t> error_rate_vec;
  std::string inner_decoder_name;
  cudaqx::heterogeneous_map inner_decoder_params;

  // Derived parameters.
  std::size_t num_windows = 0;
  std::size_t num_rounds = 0;
  std::size_t num_syndromes_per_window = 0;
  std::size_t num_rounds_since_last_decode = 0;
  std::vector<std::unique_ptr<decoder>> inner_decoders;
  std::vector<std::size_t> first_columns;
  cudaqx::tensor<std::uint8_t> full_pcm;
  cudaqx::tensor<std::uint8_t> full_pcm_T;

  // Constants
  static constexpr int NUM_WINDOW_PROC_TIMES = 10;

  // State data
  std::vector<std::vector<cudaq::qec::float_t>>
      rolling_window; // [batch_size, num_syndromes_per_window]
  std::size_t rw_filled = 0;
  std::size_t num_windows_decoded = 0;
  std::vector<std::vector<bool>> syndrome_mods; // [batch_size, syndrome_size]
  std::vector<decoder_result> rw_results;       // [batch_size]
  std::vector<double> window_proc_times;
  std::array<double, NUM_WINDOW_PROC_TIMES> window_proc_times_arr = {};

public:
  sliding_window(const cudaqx::tensor<uint8_t> &H,
                 const cudaqx::heterogeneous_map &params)
      : decoder(H), full_pcm(H) {
    full_pcm_T = full_pcm.transpose();
    // Fetch parameters from the params map.
    window_size = params.get<std::size_t>("window_size", window_size);
    step_size = params.get<std::size_t>("step_size", step_size);
    num_syndromes_per_round = params.get<std::size_t>("num_syndromes_per_round",
                                                      num_syndromes_per_round);
    straddle_start_round =
        params.get<bool>("straddle_start_round", straddle_start_round);
    straddle_end_round =
        params.get<bool>("straddle_end_round", straddle_end_round);
    error_rate_vec = params.get<std::vector<cudaq::qec::float_t>>(
        "error_rate_vec", error_rate_vec);
    inner_decoder_name =
        params.get<std::string>("inner_decoder_name", inner_decoder_name);
    inner_decoder_params = params.get<cudaqx::heterogeneous_map>(
        "inner_decoder_params", inner_decoder_params);

    // Perform error checking on the inputs.
    if (num_syndromes_per_round == 0) {
      throw std::invalid_argument("sliding_window constructor: "
                                  "num_syndromes_per_round must be non-zero");
    }
    if (H.shape()[0] % num_syndromes_per_round != 0) {
      throw std::invalid_argument(
          "sliding_window constructor: Number of rows in H must be divisible "
          "by num_syndromes_per_round");
    }
    if (inner_decoder_name.empty()) {
      throw std::invalid_argument(
          "sliding_window constructor: inner_decoder_name must be non-empty");
    }
    if (inner_decoder_params.empty()) {
      CUDAQ_WARN("sliding_window constructor: inner_decoder_params is empty. "
                 "Is that intentional?");
    }
    if (step_size == 0) {
      throw std::invalid_argument(
          "sliding_window constructor: step_size must be non-zero");
    }
    if (error_rate_vec.empty()) {
      throw std::invalid_argument(
          "sliding_window constructor: error_rate_vec must be non-empty");
    }

    // Enforce that H is already sorted.
    if (!cudaq::qec::pcm_is_sorted(H, num_syndromes_per_round)) {
      throw std::invalid_argument("sliding_window constructor: PCM must be "
                                  "sorted. See cudaq::qec::simplify_pcm.");
    }

    num_rounds = H.shape()[0] / num_syndromes_per_round;
    num_windows = (num_rounds - window_size) / step_size + 1;
    num_syndromes_per_window = num_syndromes_per_round * window_size;

    // Create the inner decoders.
    for (std::size_t w = 0; w < num_windows; ++w) {
      std::size_t start_round = w * step_size;
      std::size_t end_round = start_round + window_size - 1;
      auto [H_round, first_column, last_column] =
          cudaq::qec::get_pcm_for_rounds(
              H, num_syndromes_per_round, start_round, end_round,
              straddle_start_round, straddle_end_round);
      first_columns.push_back(first_column);

      // Slice the error vector to only include the current window.
      auto inner_decoder_params_mod = inner_decoder_params;
      std::vector<cudaq::qec::float_t> error_vec_mod(
          error_rate_vec.begin() + first_column,
          error_rate_vec.begin() + last_column + 1);
      inner_decoder_params_mod.insert("error_rate_vec", error_vec_mod);

      CUDAQ_INFO("Creating a decoder for rounds {}-{} (dims {} x {}) "
                 "first_column = {}, last_column = {}",
                 start_round, end_round, H_round.shape()[0], H_round.shape()[1],
                 first_column, last_column);
      auto inner_decoder =
          decoder::get(inner_decoder_name, H_round, inner_decoder_params_mod);
      inner_decoders.push_back(std::move(inner_decoder));
    }
  }

  virtual decoder_result decode(const std::vector<float_t> &syndrome) override {
    if (syndrome.size() == this->syndrome_size) {
      auto t0 = std::chrono::high_resolution_clock::now();
      CUDAQ_DBG("Decoding whole block");
      // Decode the whole thing, iterating over windows manually.
      decoder_result result;
      std::vector<float_t> syndrome_round(num_syndromes_per_round);
      for (std::size_t r = 0; r < num_rounds; ++r) {
        std::copy(syndrome.begin() + r * num_syndromes_per_round,
                  syndrome.begin() + (r + 1) * num_syndromes_per_round,
                  syndrome_round.begin());
        result = decode(syndrome_round);
        // Note: result will be empty until the final loop iteration.
      }
      auto t1 = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> diff = t1 - t0;
      CUDAQ_INFO("Whole block time: {:.3f} ms", diff.count() * 1000);
      return result;
    }
    // Else we're receiving a single round.
    if (rw_filled == 0) {
      // Initialize the syndrome mods and rw_results.
      auto t0 = std::chrono::high_resolution_clock::now();
      window_proc_times_arr.fill(0.0);
      syndrome_mods.resize(1);
      syndrome_mods[0].clear();
      syndrome_mods[0].resize(this->syndrome_size);
      rw_results.clear();
      rw_results.resize(1);
      rw_results[0].converged = true; // Gets set to false if we fail to decode
      rw_results[0].result.resize(this->block_size);
      rolling_window.resize(1);
      rolling_window[0].clear();
      rolling_window[0].resize(num_syndromes_per_window);
      window_proc_times.resize(num_windows);
      std::fill(window_proc_times.begin(), window_proc_times.end(), 0.0);
      rw_filled = 0;
      num_rounds_since_last_decode = 0;
      CUDAQ_DBG("Initializing window");
      auto t1 = std::chrono::high_resolution_clock::now();
      window_proc_times_arr[0] =
          std::chrono::duration<double>(t1 - t0).count() * 1000;
    }
    if (this->rw_filled == num_syndromes_per_window) {
      auto t0 = std::chrono::high_resolution_clock::now();
      CUDAQ_DBG("Window is full, sliding the window");
      std::copy(this->rolling_window[0].begin() + num_syndromes_per_round,
                this->rolling_window[0].end(), this->rolling_window[0].begin());
      std::copy(syndrome.begin(), syndrome.end(),
                this->rolling_window[0].end() - num_syndromes_per_round);
      auto t1 = std::chrono::high_resolution_clock::now();
      window_proc_times_arr[1] +=
          std::chrono::duration<double>(t1 - t0).count() * 1000;
    } else {
      // Just copy the data to the end of the rolling window.
      auto t0 = std::chrono::high_resolution_clock::now();
      CUDAQ_DBG("Copying data to the end of the rolling window");
      std::copy(syndrome.begin(), syndrome.end(),
                this->rolling_window[0].begin() + this->rw_filled);
      this->rw_filled += num_syndromes_per_round;
      auto t1 = std::chrono::high_resolution_clock::now();
      window_proc_times_arr[2] +=
          std::chrono::duration<double>(t1 - t0).count() * 1000;
    }
    num_rounds_since_last_decode++;
    if (rw_filled == num_syndromes_per_window &&
        num_rounds_since_last_decode >= step_size) {
      CUDAQ_DBG("Decoding window {}/{}", num_windows_decoded + 1, num_windows);
      decode_window();
      num_rounds_since_last_decode = 0;

      num_windows_decoded++;
      if (num_windows_decoded == num_windows) {
        num_windows_decoded = 0;
        rw_filled = 0;
        // for (std::size_t w = 0; w < num_windows; ++w) {
        //   CUDAQ_DBG("Window {} time: {} ms", w, window_proc_times[w]);
        // }
        CUDAQ_DBG("Returning decoder_result");
        return std::move(this->rw_results[0]);
      }
    }
    CUDAQ_DBG("Returning empty decoder_result");
    return decoder_result(); // empty return value
  }

  virtual std::vector<decoder_result>
  decode_batch(const std::vector<std::vector<float_t>> &syndromes) override {
    if (syndromes[0].size() == this->syndrome_size) {
      CUDAQ_DBG("Decoding whole block");
      // Decode the whole thing, iterating over windows manually.
      std::vector<decoder_result> results;
      std::vector<std::vector<float_t>> syndromes_round(syndromes.size());
      for (std::size_t r = 0; r < num_rounds; ++r) {
        for (std::size_t s = 0; s < syndromes.size(); ++s) {
          syndromes_round[s].resize(num_syndromes_per_round);
          std::copy(syndromes[s].begin() + r * num_syndromes_per_round,
                    syndromes[s].begin() + (r + 1) * num_syndromes_per_round,
                    syndromes_round[s].begin());
        }
        results = decode_batch(syndromes_round);
      }
      return results;
    }
    // Else we're receiving a single round.
    if (rw_filled == 0) {
      // Initialize the syndrome mods and rw_results.
      syndrome_mods.resize(syndromes.size());
      for (std::size_t s = 0; s < syndromes.size(); ++s) {
        syndrome_mods[s].clear();
        syndrome_mods[s].resize(this->syndrome_size);
      }
      rw_results.clear();
      rw_results.resize(syndromes.size());
      for (std::size_t s = 0; s < syndromes.size(); ++s) {
        rw_results[s].converged =
            true; // Gets set to false if we fail to decode
        rw_results[s].result.resize(this->block_size);
      }
      rolling_window.resize(syndromes.size());
      for (std::size_t s = 0; s < syndromes.size(); ++s) {
        rolling_window[s].clear();
        rolling_window[s].resize(num_syndromes_per_window);
      }
      window_proc_times.resize(num_windows);
      std::fill(window_proc_times.begin(), window_proc_times.end(), 0.0);
      rw_filled = 0;
      num_rounds_since_last_decode = 0;
      CUDAQ_DBG("Initializing window");
    }
    if (this->rw_filled == num_syndromes_per_window) {
      CUDAQ_DBG("Window is full, sliding the window");
      // The window is full. Slide existing data to the left and write the new
      // data at the end.
      for (std::size_t s = 0; s < syndromes.size(); ++s) {
        std::copy(this->rolling_window[s].begin() + num_syndromes_per_round,
                  this->rolling_window[s].end(),
                  this->rolling_window[s].begin());
        std::copy(syndromes[s].begin(), syndromes[s].end(),
                  this->rolling_window[s].end() - num_syndromes_per_round);
      }
      num_rounds_since_last_decode++;
    } else {
      // Just copy the data to the end of the rolling window.
      CUDAQ_DBG("Copying data to the end of the rolling window");
      for (std::size_t s = 0; s < syndromes.size(); ++s) {
        std::copy(syndromes[s].begin(), syndromes[s].end(),
                  this->rolling_window[s].begin() + this->rw_filled);
      }
      this->rw_filled += num_syndromes_per_round;
      num_rounds_since_last_decode++;
    }
    if (rw_filled == num_syndromes_per_window &&
        num_rounds_since_last_decode >= step_size) {
      CUDAQ_DBG("Decoding window {}/{}", num_windows_decoded + 1, num_windows);
      decode_window();
      num_rounds_since_last_decode = 0;
      num_windows_decoded++;
      if (num_windows_decoded == num_windows) {
        num_windows_decoded = 0;
        rw_filled = 0;
        // Dump the per window processing times.
        // for (std::size_t w = 0; w < num_windows; ++w) {
        //   CUDAQ_DBG("Window {} time: {} ms", w, window_proc_times[w]);
        // }
        CUDAQ_DBG("Returning decoder_result");
        return std::move(this->rw_results);
      }
    }
    CUDAQ_DBG("Returning empty decoder_result");
    return std::vector<decoder_result>(); // empty return value
  }

  void decode_window() {
    auto t0 = std::chrono::high_resolution_clock::now();
    const auto &w = this->num_windows_decoded;
    std::size_t syndrome_start = w * step_size * num_syndromes_per_round;
    std::size_t syndrome_end = syndrome_start + num_syndromes_per_window - 1;
    std::size_t syndrome_start_next_window =
        (w + 1) * step_size * num_syndromes_per_round;
    std::size_t syndrome_end_next_window =
        syndrome_start_next_window + num_syndromes_per_round - 1;
    auto t3 = std::chrono::high_resolution_clock::now();
    if (w > 0) {
      // Modify the syndrome slice to account for the previous windows.
      for (std::size_t s = 0; s < this->rolling_window.size(); ++s) {
        for (std::size_t r = 0; r < num_syndromes_per_window; ++r) {
          auto &slice_val = this->rolling_window[s].at(r);
          slice_val =
              static_cast<double>(static_cast<std::uint8_t>(slice_val) ^
                                  syndrome_mods[s].at(r + syndrome_start));
        }
      }
    }
    auto t4 = std::chrono::high_resolution_clock::now();
    CUDAQ_DBG("Window {}: syndrome_start = {}, syndrome_end = {}, length1 = "
              "{}, length2 = {}",
              w, syndrome_start, syndrome_end, this->rolling_window[0].size(),
              syndrome_end - syndrome_start + 1);
    std::vector<decoder_result> inner_results;
    if (this->rolling_window.size() == 1) {
      inner_results.push_back(
          inner_decoders[w]->decode(this->rolling_window[0]));
    } else {
      inner_results = inner_decoders[w]->decode_batch(this->rolling_window);
    }
    if (!inner_results[0].converged) {
      CUDAQ_DBG("Window {}: inner decoder failed to converge", w);
    }
    auto t5 = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<uint8_t>> window_results(
        this->rolling_window.size());
    for (std::size_t s = 0; s < this->rolling_window.size(); ++s) {
      this->rw_results[s].converged &= inner_results[s].converged;
      cudaq::qec::convert_vec_soft_to_hard(inner_results[s].result,
                                           window_results[s]);
    }
    // Commit to everything up to the first column of the next window.
    auto t6 = std::chrono::high_resolution_clock::now();
    if (w < num_windows - 1) {
      // Prepare for the next window.
      auto next_window_first_column = first_columns[w + 1];
      auto this_window_first_column = first_columns[w];
      auto num_to_commit = next_window_first_column - this_window_first_column;
      CUDAQ_DBG("  Committing {} bits from window {}", num_to_commit, w);
      for (std::size_t s = 0; s < this->rolling_window.size(); ++s) {
        for (std::size_t c = 0; c < num_to_commit; ++c) {
          rw_results[s].result[c + this_window_first_column] =
              window_results[s][c];
        }
      }
      // We are committing to some errors that would affect the next round's
      // syndrome measurements. Therefore, we need to modify some of the
      // syndrome measurements for the next round to "back out" the errors
      // that we already know about (or more specifically, the errors we think
      // we've already accounted for).
      for (std::size_t s = 0; s < this->rolling_window.size(); ++s) {
        for (std::size_t c = 0; c < num_to_commit; ++c) {
          if (rw_results[s].result[c + this_window_first_column]) {
            // This bit is a 1, so we need to modify the syndrome measurements
            // for the next window to account for this already-accounted-for
            // error. We do this by flipping the bit in the syndrome
            // measurements if the corresponding entry in the PCM is a 1.
            auto *pcm_col = &full_pcm_T.at({c + this_window_first_column, 0});
            for (auto r = syndrome_start_next_window;
                 r <= syndrome_end_next_window; ++r) {
              syndrome_mods[s][r] =
                  syndrome_mods[s][r] ^ static_cast<bool>(pcm_col[r]);
            }
          }
        }
      }
    } else {
      // This is the last window. Append ALL of window_result to
      // decoded_result.
      auto this_window_first_column = first_columns[w];
      auto num_to_commit = window_results[0].size();
      CUDAQ_DBG("  Committing {} bits from window {}", num_to_commit, w);
      for (std::size_t s = 0; s < this->rolling_window.size(); ++s) {
        for (std::size_t c = 0; c < num_to_commit; ++c) {
          rw_results[s].result[c + this_window_first_column] =
              window_results[s][c];
        }
      }
    }
    auto t7 = std::chrono::high_resolution_clock::now();
    window_proc_times.at(w) +=
        std::chrono::duration<double>(t7 - t0).count() * 1000;
    window_proc_times_arr[3] =
        std::chrono::duration<double>(t3 - t0).count() * 1000;
    window_proc_times_arr[4] =
        std::chrono::duration<double>(t4 - t3).count() * 1000;
    window_proc_times_arr[5] =
        std::chrono::duration<double>(t5 - t4).count() * 1000;
    window_proc_times_arr[6] =
        std::chrono::duration<double>(t6 - t5).count() * 1000;
    window_proc_times_arr[7] =
        std::chrono::duration<double>(t7 - t6).count() * 1000;
    CUDAQ_INFO("Window {} time: {:.3f} ms (0:{:.3f}ms 1:{:.3f}ms 2:{:.3f}ms "
               "3:{:.3f}ms 4:{:.3f}ms 5:{:.3f}ms 6:{:.3f}ms 7:{:.3f}ms)",
               w, window_proc_times[w], window_proc_times_arr[0],
               window_proc_times_arr[1], window_proc_times_arr[2],
               window_proc_times_arr[3], window_proc_times_arr[4],
               window_proc_times_arr[5], window_proc_times_arr[6],
               window_proc_times_arr[7]);
  }

  virtual ~sliding_window() {}

  CUDAQ_EXTENSION_CUSTOM_CREATOR_FUNCTION(
      sliding_window, static std::unique_ptr<decoder> create(
                          const cudaqx::tensor<uint8_t> &H,
                          const cudaqx::heterogeneous_map &params) {
        return std::make_unique<sliding_window>(H, params);
      })
};

CUDAQ_REGISTER_TYPE(sliding_window)

} // namespace cudaq::qec
