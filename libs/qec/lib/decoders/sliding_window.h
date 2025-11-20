/****************************************************************-*- C++ -*-****
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/qec/decoder.h"
#include <vector>

namespace cudaq::qec {

/// @brief A sliding window decoder that processes syndromes in overlapping
/// windows
///
/// This decoder divides the syndrome stream into overlapping windows and
/// decodes each window independently using an inner decoder. It's designed for
/// low-latency decoding of streaming syndrome data.
class sliding_window : public decoder {
private:
  // --- Input parameters ---

  /// The number of rounds of syndrome data in each window.
  std::size_t window_size = 1;
  /// The number of rounds to advance the window by each time.
  std::size_t step_size = 1;
  /// The number of syndromes per round.
  std::size_t num_syndromes_per_round = 0;
  /// When forming a window, should error mechanisms that span the start round
  /// and any preceding rounds be included?
  bool straddle_start_round = false;
  /// When forming a window, should error mechanisms that span the end round and
  /// any subsequent rounds be included?
  bool straddle_end_round = true;
  /// The vector of error rates for the error mechanisms.
  std::vector<cudaq::qec::float_t> error_rate_vec;
  /// The name of the inner decoder to use.
  std::string inner_decoder_name;
  /// The parameters to pass to the inner decoder.
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

  // Enum type for timing data.
  enum WindowProcTimes {
    INITIALIZE_WINDOW,     // 0
    SLIDE_WINDOW,          // 1
    COPY_DATA,             // 2
    INDEX_CALCULATION,     // 3
    MODIFY_SYNDROME_SLICE, // 4
    INNER_DECODE,          // 5
    CONVERT_TO_HARD,       // 6
    COMMIT_TO_RESULT,      // 7
    NUM_WINDOW_PROC_TIMES  // 8
  };

  // State data
  std::vector<std::vector<cudaq::qec::float_t>>
      rolling_window; // [batch_size, num_syndromes_per_window]
  // rolling window read and write indices (circular buffer)
  std::size_t rw_next_write_index = 0; // [0, num_syndromes_per_window)
  std::size_t rw_next_read_index = 0;  // [0, num_syndromes_per_window)
  std::size_t rw_filled = 0;
  std::size_t num_windows_decoded = 0;
  std::vector<std::vector<bool>> syndrome_mods; // [batch_size, syndrome_size]
  std::vector<decoder_result> rw_results;       // [batch_size]
  std::vector<double> window_proc_times;
  std::array<double, WindowProcTimes::NUM_WINDOW_PROC_TIMES>
      window_proc_times_arr = {};

  /// @brief Validate constructor inputs
  void validate_inputs();

  /// @brief Initialize the window
  /// @param num_syndromes The number of syndromes to initialize the window for
  void initialize_window(std::size_t num_syndromes);

  /// @brief Add a single syndrome to the rolling window (circular buffer)
  void add_syndrome_to_rolling_window(const std::vector<float_t> &syndrome,
                                      std::size_t syndrome_index,
                                      bool update_next_write_index = true);

  /// @brief Add a batch of syndromes to the rolling window (circular buffer)
  void add_syndromes_to_rolling_window(
      const std::vector<std::vector<float_t>> &syndromes);

  /// @brief Get a single syndrome from the rolling window (unwrapping circular
  /// buffer)
  std::vector<float_t>
  get_syndrome_from_rolling_window(std::size_t syndrome_index);

  /// @brief Get a batch of syndromes from the rolling window (unwrapping
  /// circular buffer)
  std::vector<std::vector<float_t>> get_syndromes_from_rolling_window();

  /// @brief Update the read index for the rolling window
  void update_rw_next_read_index();

  /// @brief Decode a single window (internal helper)
  void decode_window();

public:
  /// @brief Constructor
  /// @param H The full parity check matrix for all rounds
  /// @param params A heterogeneous map containing required parameters:
  ///   - window_size: Size of each decoding window (in rounds)
  ///   - step_size: Step size between consecutive windows (in rounds)
  ///   - num_rounds: Total number of rounds
  ///   - num_syndromes_per_round: Number of syndromes per round
  ///   - inner_decoder: Name of the inner decoder to use
  ///   - inner_decoder_params: Parameters for the inner decoder (optional)
  sliding_window(const cudaqx::tensor<uint8_t> &H,
                 const cudaqx::heterogeneous_map &params);

  /// @brief Decode a syndrome vector
  /// @param syndrome The syndrome measurements to decode
  /// @return The decoded error correction
  decoder_result decode(const std::vector<float_t> &syndrome) override;

  /// @brief Decode multiple syndromes in batch
  /// @param syndromes Multiple syndrome measurements to decode
  /// @return The decoded error corrections
  std::vector<decoder_result>
  decode_batch(const std::vector<std::vector<float_t>> &syndromes) override;

  /// @brief Get the number of syndromes per round
  /// @return The number of syndromes measured in each round
  std::size_t get_num_syndromes_per_round() const;

  /// @brief Destructor
  virtual ~sliding_window();

  // Plugin registration macros
  CUDAQ_EXTENSION_CUSTOM_CREATOR_FUNCTION(
      sliding_window, static std::unique_ptr<decoder> create(
                          const cudaqx::tensor<uint8_t> &H,
                          const cudaqx::heterogeneous_map &params) {
        return std::make_unique<sliding_window>(H, params);
      })
};

} // namespace cudaq::qec
