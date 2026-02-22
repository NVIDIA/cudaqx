/*******************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "sliding_window.h"
#include "common/FmtCore.h"
#include "cudaq/qec/pcm_utils.h"
#include "cudaq/runtime/logger/logger.h"
#include <cassert>
#include <vector>

namespace cudaq::qec {

void sliding_window::validate_inputs() {
  uint32_t num_rows = H.num_rows();
  uint32_t num_cols = H.num_cols();
  if (window_size < 1 || window_size > num_rounds) {
    throw std::invalid_argument(
        fmt::format("sliding_window constructor: window_size ({}) must "
                    "be between 1 and num_rounds ({})",
                    window_size, num_rounds));
  }
  if (step_size < 1 || step_size > window_size) {
    throw std::invalid_argument(
        fmt::format("sliding_window constructor: step_size ({}) must "
                    "be between 1 and window_size ({})",
                    step_size, window_size));
  }
  if ((num_rounds - window_size) % step_size != 0) {
    throw std::invalid_argument(
        fmt::format("sliding_window constructor: num_rounds - "
                    "window_size ({}) must be divisible by step_size ({})",
                    num_rounds - window_size, step_size));
  }
  if (num_syndromes_per_round == 0) {
    throw std::invalid_argument("sliding_window constructor: "
                                "num_syndromes_per_round must be non-zero");
  }
  if (num_rows % num_syndromes_per_round != 0) {
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
  if (error_rate_vec.empty()) {
    throw std::invalid_argument(
        "sliding_window constructor: error_rate_vec must be non-empty");
  }

  // Enforce that H is already sorted.
  if (!cudaq::qec::pcm_is_sorted(H.to_dense(), this->num_syndromes_per_round)) {
    throw std::invalid_argument("sliding_window constructor: PCM must be "
                                "sorted. See cudaq::qec::simplify_pcm.");
  }
}

/// Helper function to initialize the window.
/// @param num_syndromes The number of syndromes to initialize the window for.
/// This will be 1 for non-batched mode.
void sliding_window::initialize_window(std::size_t num_syndromes) {
  // Initialize the syndrome mods and rw_results.
  auto t0 = std::chrono::high_resolution_clock::now();
  window_proc_times_arr.fill(0.0);
  syndrome_mods.resize(num_syndromes);
  for (std::size_t s = 0; s < num_syndromes; ++s) {
    syndrome_mods[s].clear();
    syndrome_mods[s].resize(this->syndrome_size);
  }
  rw_results.clear();
  rw_results.resize(num_syndromes);
  for (std::size_t s = 0; s < num_syndromes; ++s) {
    rw_results[s].converged = true; // Gets set to false if we fail to decode
    rw_results[s].result.resize(this->block_size);
  }
  rolling_window.resize(num_syndromes);
  for (std::size_t s = 0; s < num_syndromes; ++s) {
    rolling_window[s].clear();
    rolling_window[s].resize(num_syndromes_per_window);
  }
  window_proc_times.resize(num_windows);
  std::fill(window_proc_times.begin(), window_proc_times.end(), 0.0);
  rw_next_write_index = 0;
  rw_next_read_index = 0;
  rw_filled = 0;
  num_rounds_since_last_decode = 0;
  CUDAQ_DBG("Initializing window");
  auto t1 = std::chrono::high_resolution_clock::now();
  window_proc_times_arr[WindowProcTimes::INITIALIZE_WINDOW] =
      std::chrono::duration<double>(t1 - t0).count() * 1000;
}

/// Helper function to add a single syndrome to the rolling window (circular
/// buffer).
void sliding_window::add_syndrome_to_rolling_window(
    const std::vector<float_t> &syndrome, std::size_t syndrome_index,
    bool update_next_write_index) {
  // This assumes that the syndrome size evenly divides into the rolling
  // window (of length num_syndromes_per_window), so verify that here.
  if (num_syndromes_per_window % syndrome.size() != 0) {
    throw std::invalid_argument(
        fmt::format("add_syndrome_to_rolling_window: syndrome "
                    "size ({}) must evenly divide into the rolling "
                    "window size ({})",
                    syndrome.size(), num_syndromes_per_window));
  }
  std::copy(syndrome.begin(), syndrome.end(),
            rolling_window[syndrome_index].begin() + rw_next_write_index);
  if (update_next_write_index) {
    rw_next_write_index += syndrome.size();
    if (rw_next_write_index >= num_syndromes_per_window)
      rw_next_write_index = 0;
  }
}

/// Helper function to add a batch of syndromes to the rolling window
/// (circular buffer).
void sliding_window::add_syndromes_to_rolling_window(
    const std::vector<std::vector<float_t>> &syndromes) {
  // Set update_next_write_index to false in the loop because we will update
  // it once at the end.
  for (std::size_t s = 0; s < syndromes.size(); ++s) {
    add_syndrome_to_rolling_window(syndromes[s], s,
                                   /*update_next_write_index=*/false);
    if (syndromes[s].size() != syndromes[0].size()) {
      throw std::invalid_argument(
          fmt::format("add_syndromes_to_rolling_window: syndrome "
                      "size ({}) must be the same as the first syndrome "
                      "size ({})",
                      syndromes[s].size(), syndromes[0].size()));
    }
  }
  rw_next_write_index += syndromes[0].size();
  if (rw_next_write_index >= num_syndromes_per_window)
    rw_next_write_index = 0;
}

/// Helper function to get a single syndrome from the rolling window
/// (unwrapping a circular buffer).
std::vector<float_t>
sliding_window::get_syndrome_from_rolling_window(std::size_t syndrome_index) {
  std::vector<float_t> syndrome(num_syndromes_per_window);
  // Copy from rw_next_read_index to the end of the buffer.
  std::copy(rolling_window[syndrome_index].begin() + rw_next_read_index,
            rolling_window[syndrome_index].end(), syndrome.begin());
  // Copy from the beginning of the rolling window to rw_next_read_index.
  std::copy(rolling_window[syndrome_index].begin(),
            rolling_window[syndrome_index].begin() + rw_next_read_index,
            syndrome.end() - rw_next_read_index);
  return syndrome;
}

/// Helper function to get a batch of syndromes from the rolling window
/// (unwrapping a circular buffer).
std::vector<std::vector<float_t>>
sliding_window::get_syndromes_from_rolling_window() {
  std::vector<std::vector<float_t>> syndromes(rolling_window.size());
  for (std::size_t s = 0; s < rolling_window.size(); ++s) {
    syndromes[s] = get_syndrome_from_rolling_window(s);
  }
  return syndromes;
}

/// Helper function to update the read index for the rolling window.
void sliding_window::update_rw_next_read_index() {
  rw_next_read_index += step_size * num_syndromes_per_round;
  if (rw_next_read_index >= num_syndromes_per_window)
    rw_next_read_index -= num_syndromes_per_window;
}

sliding_window::sliding_window(const cudaq::qec::sparse_binary_matrix &H,
                               const cudaqx::heterogeneous_map &params)
    : decoder(H), full_pcm(H.to_dense()) {
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

  num_rounds = H.num_rows() / num_syndromes_per_round;
  num_windows = (num_rounds - window_size) / step_size + 1;
  num_syndromes_per_window = num_syndromes_per_round * window_size;

  validate_inputs();

  // FIXME - update downstream code to support sparse matrices.
  auto H_dense = H.to_dense();

  // Create the inner decoders.
  for (std::size_t w = 0; w < num_windows; ++w) {
    std::size_t start_round = w * step_size;
    std::size_t end_round = start_round + window_size - 1;
    auto [H_round, first_column, last_column] = cudaq::qec::get_pcm_for_rounds(
        H_dense, num_syndromes_per_round, start_round, end_round,
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

    if (last_column - first_column + 1 != H_round.shape()[1]) {
      throw std::invalid_argument(
          fmt::format("last_column - first_column + 1 ({}) must be equal to "
                      "the number of columns in H_round ({})",
                      last_column - first_column + 1, H_round.shape()[1]));
    }

    auto inner_decoder =
        decoder::get(inner_decoder_name, H_round, inner_decoder_params_mod);
    inner_decoders.push_back(std::move(inner_decoder));
  }
}

decoder_result sliding_window::decode(const std::vector<float_t> &syndrome) {
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
    initialize_window(/*num_syndromes=*/1);
  }
  if (this->rw_filled == num_syndromes_per_window) {
    auto t0 = std::chrono::high_resolution_clock::now();
    CUDAQ_DBG("Window is full, sliding the window by one round");
    add_syndrome_to_rolling_window(syndrome, 0);

    auto t1 = std::chrono::high_resolution_clock::now();
    window_proc_times_arr[WindowProcTimes::SLIDE_WINDOW] +=
        std::chrono::duration<double>(t1 - t0).count() * 1000;
  } else {
    // Just copy the data to the end of the rolling window.
    auto t0 = std::chrono::high_resolution_clock::now();
    CUDAQ_DBG("Copying data to the end of the rolling window");
    add_syndrome_to_rolling_window(syndrome, 0);
    this->rw_filled += num_syndromes_per_round;
    auto t1 = std::chrono::high_resolution_clock::now();
    window_proc_times_arr[WindowProcTimes::COPY_DATA] +=
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

std::vector<decoder_result> sliding_window::decode_batch(
    const std::vector<std::vector<float_t>> &syndromes) {
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
    initialize_window(syndromes.size());
  }
  if (this->rw_filled == num_syndromes_per_window) {
    CUDAQ_DBG("Window is full, sliding the window by one round");
    // The window is full. Slide existing data to the left and write the new
    // data at the end.
    add_syndromes_to_rolling_window(syndromes);
    num_rounds_since_last_decode++;
  } else {
    // Just copy the data to the end of the rolling window.
    CUDAQ_DBG("Copying data to the end of the rolling window");
    add_syndromes_to_rolling_window(syndromes);
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

/// This is an internal helper function that decodes a single window. Regular
/// users should use the regular `cudaq::qec::decoder::decode` or
/// `cudaq::qec::decoder::decode_batch` functions instead of trying to access
/// this function.
void sliding_window::decode_window() {
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
      std::size_t r2 = rw_next_read_index;
      for (std::size_t r = 0; r < num_syndromes_per_window; ++r) {
        auto &slice_val = this->rolling_window[s].at(r2);
        slice_val =
            static_cast<double>(static_cast<std::uint8_t>(slice_val) ^
                                syndrome_mods[s].at(r + syndrome_start));
        r2++;
        if (r2 >= num_syndromes_per_window)
          r2 = 0;
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
        inner_decoders[w]->decode(get_syndrome_from_rolling_window(0)));
  } else {
    inner_results =
        inner_decoders[w]->decode_batch(get_syndromes_from_rolling_window());
  }
  // We've grabbed data from the rolling window, so we need to update the
  // read index for the next call to decode_window.
  update_rw_next_read_index();
  if (!inner_results[0].converged) {
    CUDAQ_DBG("Window {}: inner decoder failed to converge", w);
  }
  auto t5 = std::chrono::high_resolution_clock::now();
  std::vector<std::vector<uint8_t>> window_results(this->rolling_window.size());
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
  window_proc_times_arr[WindowProcTimes::INDEX_CALCULATION] =
      std::chrono::duration<double>(t3 - t0).count() * 1000;
  window_proc_times_arr[WindowProcTimes::MODIFY_SYNDROME_SLICE] =
      std::chrono::duration<double>(t4 - t3).count() * 1000;
  window_proc_times_arr[WindowProcTimes::INNER_DECODE] =
      std::chrono::duration<double>(t5 - t4).count() * 1000;
  window_proc_times_arr[WindowProcTimes::CONVERT_TO_HARD] =
      std::chrono::duration<double>(t6 - t5).count() * 1000;
  window_proc_times_arr[WindowProcTimes::COMMIT_TO_RESULT] =
      std::chrono::duration<double>(t7 - t6).count() * 1000;
  CUDAQ_INFO("Window {} time: {:.3f} ms (0:{:.3f}ms 1:{:.3f}ms 2:{:.3f}ms "
             "3:{:.3f}ms 4:{:.3f}ms 5:{:.3f}ms 6:{:.3f}ms 7:{:.3f}ms)",
             w, window_proc_times[w], window_proc_times_arr[0],
             window_proc_times_arr[1], window_proc_times_arr[2],
             window_proc_times_arr[3], window_proc_times_arr[4],
             window_proc_times_arr[5], window_proc_times_arr[6],
             window_proc_times_arr[7]);
}

sliding_window::~sliding_window() {}

std::size_t sliding_window::get_num_syndromes_per_round() const {
  return num_syndromes_per_round;
}

CUDAQ_REGISTER_TYPE(sliding_window)

} // namespace cudaq::qec
