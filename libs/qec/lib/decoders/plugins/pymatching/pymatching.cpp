/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "pymatching/sparse_blossom/driver/mwpm_decoding.h"
#include "pymatching/sparse_blossom/driver/user_graph.h"
#include "cudaq/qec/decoder.h"
#include "cudaq/qec/pcm_utils.h"
#include <algorithm>
#include <cassert>
#include <map>
#include <vector>

// Enable this to debug decode times.
#define PERFORM_TIMING 0

namespace cudaq::qec {

/// @brief This is a wrapper around the PyMatching library that implements the
/// MWPM decoder.
class pymatching : public decoder {
private:
  pm::UserGraph user_graph;
  pm::Mwpm *mwpm = nullptr;

  // Input parameters
  std::vector<double> error_rate_vec;
  pm::MERGE_STRATEGY merge_strategy_enum = pm::MERGE_STRATEGY::DISALLOW;

  // Map of edge pairs to column indices. This does not seem particularly
  // efficient.
  std::map<std::pair<int64_t, int64_t>, size_t> edge2col_idx;

  bool decode_to_observables = false;

  // Helper function to make a canonical edge from two nodes.
  std::pair<int64_t, int64_t> make_canonical_edge(int64_t node1,
                                                  int64_t node2) {
    return std::make_pair(std::min(node1, node2), std::max(node1, node2));
  }

#if PERFORM_TIMING
  static constexpr size_t NUM_TIMING_STEPS = 4;
  std::array<double, NUM_TIMING_STEPS> decode_times;
#endif

public:
  pymatching(const cudaqx::tensor<uint8_t> &H,
             const cudaqx::heterogeneous_map &params)
      : decoder(H) {

    if (params.contains("error_rate_vec")) {
      error_rate_vec = params.get<std::vector<double>>("error_rate_vec");
      if (error_rate_vec.size() != block_size) {
        throw std::runtime_error("error_rate_vec must be of size block_size");
      }
      // Validate that the values in the error_rate_vec are between 0 and 0.5.
      // Values > 0.5 would have negative LLR, which is not supported by
      // PyMatching.
      for (auto error_rate : error_rate_vec) {
        if (error_rate <= 0.0 || error_rate > 0.5) {
          throw std::runtime_error(
              "error_rate_vec value is out of range (0, 0.5]");
        }
      }
    }

    if (params.contains("merge_strategy")) {
      std::string merge_strategy = params.get<std::string>("merge_strategy");
      if (merge_strategy == "disallow") {
        merge_strategy_enum = pm::MERGE_STRATEGY::DISALLOW;
      } else if (merge_strategy == "independent") {
        merge_strategy_enum = pm::MERGE_STRATEGY::INDEPENDENT;
      } else if (merge_strategy == "smallest_weight") {
        merge_strategy_enum = pm::MERGE_STRATEGY::SMALLEST_WEIGHT;
      } else if (merge_strategy == "keep_original") {
        merge_strategy_enum = pm::MERGE_STRATEGY::KEEP_ORIGINAL;
      } else if (merge_strategy == "replace") {
        merge_strategy_enum = pm::MERGE_STRATEGY::REPLACE;
      } else {
        throw std::runtime_error(
            "merge_strategy must be one of: disallow, independent, "
            "smallest_weight, keep_original, replace");
      }
    }

    std::vector<std::vector<size_t>> errs2observables(block_size);
    if (params.contains("O")) {
      auto O = params.get<cudaqx::tensor<uint8_t>>("O");
      if (O.rank() != 2) {
        throw std::runtime_error(
            "O must be a 2-dimensional tensor (num_observables x block_size)");
      }
      const size_t num_observables = O.shape()[0];
      if (O.shape()[1] != block_size) {
        throw std::runtime_error(
            "O must be of shape (num_observables, block_size); got second "
            "dimension " +
            std::to_string(O.shape()[1]) + ", block_size " +
            std::to_string(block_size));
      }
      std::vector<std::vector<uint32_t>> O_sparse;
      for (size_t i = 0; i < num_observables; i++) {
        O_sparse.emplace_back();
        auto *row = &O.at({i, 0});
        for (size_t j = 0; j < block_size; j++) {
          if (row[j] > 0) {
            O_sparse.back().push_back(static_cast<uint32_t>(j));
            errs2observables[j].push_back(static_cast<uint32_t>(i));
          }
        }
      }
      this->set_O_sparse(O_sparse);
      decode_to_observables = true;
    }

    user_graph = pm::UserGraph(H.shape()[0]);

    auto sparse = cudaq::qec::dense_to_sparse(H);
    std::size_t col_idx = 0;
    for (auto &col : sparse) {
      double weight = 1.0;
      if (col_idx < error_rate_vec.size()) {
        weight = -std::log(error_rate_vec[col_idx] /
                           (1.0 - error_rate_vec[col_idx]));
      }
      if (col.size() == 2) {
        edge2col_idx[make_canonical_edge(col[0], col[1])] = col_idx;
        user_graph.add_or_merge_edge(col[0], col[1],
                                     errs2observables.at(col_idx), weight, 0.0,
                                     merge_strategy_enum);
      } else if (col.size() == 1) {
        edge2col_idx[make_canonical_edge(col[0], -1)] = col_idx;
        user_graph.add_or_merge_boundary_edge(col[0],
                                              errs2observables.at(col_idx),
                                              weight, 0.0, merge_strategy_enum);
      } else {
        throw std::runtime_error(
            "Invalid column in H: " + std::to_string(col_idx) + " has " +
            std::to_string(col.size()) + " ones. Must have 1 or 2 ones.");
      }
      col_idx++;
    }
    this->mwpm = decode_to_observables
                     ? &user_graph.get_mwpm()
                     : &user_graph.get_mwpm_with_search_graph();
#if PERFORM_TIMING
    std::fill(decode_times.begin(), decode_times.end(), 0.0);
#endif
  }

  /// @brief Decode the syndrome using the MWPM decoder.
  /// @param syndrome The syndrome to decode.
  /// @return The decoder result.
  /// @throws std::runtime_error if no matching solution is found, or
  /// std::out_of_range if an edge is not found in the edge2col_idx map.
  virtual decoder_result decode(const std::vector<float_t> &syndrome) {
#if PERFORM_TIMING
    auto t0 = std::chrono::high_resolution_clock::now();
#endif
    decoder_result result{false, std::vector<float_t>()};
#if PERFORM_TIMING
    auto t1 = std::chrono::high_resolution_clock::now();
#endif

    std::vector<uint64_t> detection_events;
    detection_events.reserve(syndrome.size());
    for (size_t i = 0; i < syndrome.size(); i++)
      if (syndrome[i] > 0.5)
        detection_events.push_back(i);
#if PERFORM_TIMING
    auto t2 = std::chrono::high_resolution_clock::now();
#endif
    if (decode_to_observables) {
      if (mwpm->flooder.graph.num_observables < 64) {
        result.result.resize(mwpm->flooder.graph.num_observables);
        auto res = pm::decode_detection_events_for_up_to_64_observables(
            *mwpm, detection_events, /*edge_correlations=*/false);
        for (size_t i = 0; i < mwpm->flooder.graph.num_observables; i++) {
          result.result[i] =
              static_cast<float_t>(res.obs_mask & (1 << i) ? 1.0 : 0.0);
        }
      } else {
        result.result.resize(mwpm->flooder.graph.num_observables);
        assert(O_sparse.size() == mwpm.flooder.graph.num_observables);
        pm::total_weight_int weight = 0;
        std::vector<uint8_t> obs(mwpm->flooder.graph.num_observables, 0);
        obs.resize(mwpm->flooder.graph.num_observables);
        pm::decode_detection_events(*mwpm, detection_events, obs.data(), weight,
                                    /*edge_correlations=*/false);
        result.result.resize(mwpm->flooder.graph.num_observables);
        for (size_t i = 0; i < mwpm->flooder.graph.num_observables; i++) {
          result.result[i] = static_cast<float_t>(obs[i]);
        }
      }
    } else {
      std::vector<int64_t> edges;
      result.result.resize(block_size);
      pm::decode_detection_events_to_edges(*mwpm, detection_events, edges);
      // Loop over the edge pairs to reconstruct errors.
      assert(edges.size() % 2 == 0);
      for (size_t i = 0; i < edges.size(); i += 2) {
        auto edge = make_canonical_edge(edges.at(i), edges.at(i + 1));
        auto col_idx = edge2col_idx.at(edge);
        result.result[col_idx] = 1.0;
      }
    }
    // An exception is thrown if no matching solution is found, so we can just
    // set converged to true.
    result.converged = true;
#if PERFORM_TIMING
    auto t3 = std::chrono::high_resolution_clock::now();
    decode_times[0] +=
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() /
        1e6;
    decode_times[1] +=
        std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() /
        1e6;
    decode_times[2] +=
        std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count() /
        1e6;
    decode_times[3] +=
        std::chrono::duration_cast<std::chrono::microseconds>(t3 - t0).count() /
        1e6;
#endif
    return result;
  }

  virtual ~pymatching() {
#if PERFORM_TIMING
    for (int i = 0; i < NUM_TIMING_STEPS; i++) {
      std::cout << "Decode time[" << i << "]: " << decode_times[i] << " seconds"
                << std::endl;
    }
#endif
  }

  CUDAQ_EXTENSION_CUSTOM_CREATOR_FUNCTION(
      pymatching, static std::unique_ptr<decoder> create(
                      const cudaqx::tensor<uint8_t> &H,
                      const cudaqx::heterogeneous_map &params) {
        return std::make_unique<pymatching>(H, params);
      })
};

CUDAQ_REGISTER_TYPE(pymatching)

} // namespace cudaq::qec
