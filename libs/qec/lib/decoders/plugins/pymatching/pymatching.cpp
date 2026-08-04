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
#include "cudaq/qec/decoder_config_schema.h"
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
  // Error-output instances default to DISALLOW. Observable-output instances
  // default to INDEPENDENT to match upstream PyMatching's
  // from_detector_error_model behavior.
  pm::MERGE_STRATEGY merge_strategy_enum = pm::MERGE_STRATEGY::DISALLOW;
  bool merge_strategy_explicit = false;

  // Map of edge pairs to column indices. This does not seem particularly
  // efficient.
  std::map<std::pair<int64_t, int64_t>, size_t> edge2col_idx;
  std::map<std::pair<int64_t, int64_t>, double> edge2weight;

  bool decode_to_observables = false;
  std::vector<uint64_t> detection_events;
  std::vector<int64_t> matched_edges;
  std::vector<uint8_t> observable_bits;

  // Helper function to make a canonical edge from two nodes.
  std::pair<int64_t, int64_t> make_canonical_edge(int64_t node1,
                                                  int64_t node2) {
    return std::make_pair(std::min(node1, node2), std::max(node1, node2));
  }

  void record_error_column(const std::pair<int64_t, int64_t> &edge,
                           std::size_t column, double weight) {
    auto [column_it, inserted] = edge2col_idx.try_emplace(edge, column);
    if (inserted) {
      edge2weight.emplace(edge, weight);
      return;
    }

    const bool replace = merge_strategy_enum == pm::MERGE_STRATEGY::REPLACE;
    const bool smaller =
        merge_strategy_enum == pm::MERGE_STRATEGY::SMALLEST_WEIGHT &&
        weight < edge2weight.at(edge);
    if (replace || smaller) {
      column_it->second = column;
      edge2weight.at(edge) = weight;
    }
  }

#if PERFORM_TIMING
  static constexpr size_t NUM_TIMING_STEPS = 4;
  std::array<double, NUM_TIMING_STEPS> decode_times;
#endif

public:
  pymatching(cudaq::qec::decoder_inputs inputs, decoder_output requested_output,
             const cudaqx::heterogeneous_map &params)
      : decoder(std::move(inputs), requested_output) {
    const auto &H = get_inputs().detector_error_matrix();
    error_rate_vec = get_inputs().error_rates();
    decode_to_observables = requested_output == decoder_output::observables;

    if (!error_rate_vec.empty()) {
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
      merge_strategy_explicit = true;
    }

    std::vector<std::vector<size_t>> errs2observables(block_size);
    if (decode_to_observables) {
      const auto &O = get_inputs().observable_flips_matrix();
      if (O.num_cols() != block_size)
        throw std::runtime_error(
            "Observable matrix column count must equal block_size");
      const auto O_sparse = O.to_nested_csr();
      for (std::size_t observable = 0; observable < O_sparse.size();
           ++observable)
        for (auto error : O_sparse[observable])
          errs2observables[error].push_back(observable);
      if (!merge_strategy_explicit)
        merge_strategy_enum = pm::MERGE_STRATEGY::INDEPENDENT;
    }

    user_graph =
        decode_to_observables
            ? pm::UserGraph(H.num_rows(), get_inputs().num_observables())
            : pm::UserGraph(H.num_rows());

    H.validate_sorted_unique_indices("pymatching");

    // PyMatching dispatches on per-column nnz in {1, 2}. We deliberately do
    // NOT canonicalize H as a whole here: that would couple this wrapper to the
    // column-level semantics of a shared utility and risk silently reordering
    // or dropping the caller's columns. Instead we iterate the caller's columns
    // in their original order. This guarantees by construction that result
    // index `col` always maps back to the caller's column `col`.
    std::vector<std::vector<std::uint32_t>> H_e2d = H.to_nested_csc();
    for (std::size_t col = 0; col < block_size; col++) {
      double weight = 1.0;
      if (col < error_rate_vec.size()) {
        weight = -std::log(error_rate_vec[col] / (1.0 - error_rate_vec[col]));
      }

      const auto &col_rows = H_e2d[col];
      if (col_rows.size() == 2) {
        if (!decode_to_observables)
          record_error_column(make_canonical_edge(col_rows[0], col_rows[1]),
                              col, weight);
        user_graph.add_or_merge_edge(col_rows[0], col_rows[1],
                                     errs2observables.at(col), weight, 0.0,
                                     merge_strategy_enum);
      } else if (col_rows.size() == 1) {
        if (!decode_to_observables)
          record_error_column(make_canonical_edge(col_rows[0], -1), col,
                              weight);
        user_graph.add_or_merge_boundary_edge(col_rows[0],
                                              errs2observables.at(col), weight,
                                              0.0, merge_strategy_enum);
      } else {
        throw std::runtime_error("Invalid column in H: " + std::to_string(col) +
                                 " has " + std::to_string(col_rows.size()) +
                                 " ones. Must have 1 or 2 ones.");
      }
    }
    this->mwpm = decode_to_observables
                     ? &user_graph.get_mwpm()
                     : &user_graph.get_mwpm_with_search_graph();
    detection_events.reserve(syndrome_size);
    matched_edges.reserve(block_size * 2);
    observable_bits.resize(get_inputs().num_observables());
#if PERFORM_TIMING
    std::fill(decode_times.begin(), decode_times.end(), 0.0);
#endif
  }

  /// @brief Decode the syndrome using the MWPM decoder.
  /// @param syndrome The syndrome to decode.
  /// @return The decoder result.
  /// @throws std::runtime_error if no matching solution is found, or
  /// std::out_of_range if an edge is not found in the edge2col_idx map.
  decoder_result decode(const std::vector<float_t> &syndrome) override {
    decoder_result result;
    const auto result_size =
        decode_to_observables ? get_inputs().num_observables() : block_size;
    result.result.resize(result_size, float_t{0});
    auto *output = result.result.data();
#if PERFORM_TIMING
    auto t0 = std::chrono::high_resolution_clock::now();
#endif
#if PERFORM_TIMING
    auto t1 = std::chrono::high_resolution_clock::now();
#endif

    detection_events.clear();
    for (size_t i = 0; i < syndrome.size(); i++)
      if (cudaq::qec::convert_soft_to_hard(syndrome[i]))
        detection_events.push_back(i);
#if PERFORM_TIMING
    auto t2 = std::chrono::high_resolution_clock::now();
#endif
    if (decode_to_observables) {
      if (mwpm->flooder.graph.num_observables < 64) {
        auto res = pm::decode_detection_events_for_up_to_64_observables(
            *mwpm, detection_events, /*edge_correlations=*/false);
        for (size_t i = 0; i < mwpm->flooder.graph.num_observables; i++) {
          output[i] =
              static_cast<float_t>(res.obs_mask & (1ULL << i) ? 1.0 : 0.0);
        }
      } else {
        pm::total_weight_int weight = 0;
        std::fill(observable_bits.begin(), observable_bits.end(), uint8_t{0});
        pm::decode_detection_events(*mwpm, detection_events,
                                    observable_bits.data(), weight,
                                    /*edge_correlations=*/false);
        for (size_t i = 0; i < mwpm->flooder.graph.num_observables; i++) {
          output[i] = static_cast<float_t>(observable_bits[i]);
        }
      }
    } else {
      matched_edges.clear();
      pm::decode_detection_events_to_edges(*mwpm, detection_events,
                                           matched_edges);
      // Loop over the edge pairs to reconstruct errors.
      assert(matched_edges.size() % 2 == 0);
      for (size_t i = 0; i < matched_edges.size(); i += 2) {
        auto edge =
            make_canonical_edge(matched_edges.at(i), matched_edges.at(i + 1));
        auto col_idx = edge2col_idx.at(edge);
        output[col_idx] = 1.0;
      }
    }
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
    result.converged = true;
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
                      cudaq::qec::decoder_inputs inputs,
                      std::optional<decoder_output> output,
                      const cudaqx::heterogeneous_map &params) {
        return std::make_unique<pymatching>(
            std::move(inputs), output.value_or(decoder_output::errors), params);
      })
};

CUDAQ_EXT_PT_REGISTER_TYPE(pymatching)

// Parameter schema for the realtime decoding YAML (`decoder_custom_args` for
// `type: pymatching`, and the trt_decoder `global_decoder_params` section when
// `global_decoder: pymatching`). Registered here so the schema ships with the
// decoder itself.
namespace {
struct pymatching_schema_registrar {
  pymatching_schema_registrar() {
    using k = cudaq::qec::decoding::config::param_kind;
    cudaq::qec::decoding::config::register_decoder_schema(
        {"pymatching",
         {
             {"merge_strategy", k::string},
         }});
  }
};
pymatching_schema_registrar register_pymatching_schema;
} // namespace

} // namespace cudaq::qec
