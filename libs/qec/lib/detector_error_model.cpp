/*******************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/qec/detector_error_model.h"
#include "cudaq/qec/logger.h"
#include "cudaq/qec/pcm_utils.h"

#include "stim.h"

#include <exception>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace cudaq::qec {

detector_error_model dem_from_stim_text(const std::string &dem_text,
                                        bool use_decomp_suggestions) {
  auto dem = [&dem_text]() {
    try {
      return stim::DetectorErrorModel(dem_text);
    } catch (const std::exception &e) {
      throw std::runtime_error(std::string("Stim DEM parse failed: ") +
                               e.what());
    }
  }();
  const std::size_t num_detectors =
      static_cast<std::size_t>(dem.count_detectors());
  const std::size_t num_observables =
      static_cast<std::size_t>(dem.count_observables());

  std::vector<std::vector<std::size_t>> detector_hits;
  std::vector<std::vector<std::size_t>> observable_hits;
  std::vector<double> error_rates;
  std::size_t instruction_index = 0;

  dem.iter_flatten_error_instructions([&](const stim::DemInstruction &inst) {
    if (inst.arg_data.empty())
      throw std::runtime_error(
          "Stim DEM error instruction missing probability argument (index " +
          std::to_string(instruction_index) + ")");
    const double prob = inst.arg_data[0];
    if (!(prob >= 0.0 && prob <= 1.0))
      throw std::runtime_error("Stim DEM error probability " +
                               std::to_string(prob) +
                               " out of range [0, 1] at instruction index " +
                               std::to_string(instruction_index));

    std::set<std::size_t> dets_parity;
    std::set<std::size_t> obs_parity;

    auto toggle = [](std::set<std::size_t> &s, std::size_t v) {
      if (!s.erase(v)) {
        s.insert(v);
      }
    };

    auto push_target = [&](const stim::DemTarget &target) {
      if (target.is_relative_detector_id()) {
        toggle(dets_parity, static_cast<std::size_t>(target.val()));
      } else if (target.is_observable_id()) {
        toggle(obs_parity, static_cast<std::size_t>(target.val()));
      } else {
        throw std::runtime_error(
            "Stim DEM error instruction (index " +
            std::to_string(instruction_index) +
            ") contains an unsupported target kind; only D* (detector) and "
            "L* (observable) targets are supported by the fallback parser");
      }
    };

    auto flush = [&]() {
      if (!dets_parity.empty() || !obs_parity.empty()) {
        detector_hits.push_back({dets_parity.begin(), dets_parity.end()});
        observable_hits.push_back({obs_parity.begin(), obs_parity.end()});
        error_rates.push_back(prob);
        dets_parity.clear();
        obs_parity.clear();
      }
    };

    for (const auto &target : inst.target_data) {
      if (target.is_separator()) {
        if (use_decomp_suggestions) {
          flush();
        }
        continue;
      }
      push_target(target);
    }
    flush();
    ++instruction_index;
  });

  const std::size_t num_cols = detector_hits.size();
  if (num_cols == 0)
    throw std::runtime_error(
        "Stim DEM contains no error mechanisms after flattening");
  detector_error_model result;
  result.detector_error_matrix =
      cudaqx::tensor<uint8_t>({num_detectors, num_cols});
  result.observables_flips_matrix =
      cudaqx::tensor<uint8_t>({num_observables, num_cols});
  result.error_rates = std::move(error_rates);

  for (std::size_t err = 0; err < num_cols; ++err) {
    for (auto det : detector_hits[err]) {
      if (det >= num_detectors)
        throw std::runtime_error(
            "Stim DEM detector id out of range while extracting H");
      result.detector_error_matrix.at({det, err}) ^= 1;
    }
    for (auto ob : observable_hits[err]) {
      if (ob >= num_observables)
        throw std::runtime_error(
            "Stim DEM observable id out of range while extracting O");
      result.observables_flips_matrix.at({ob, err}) ^= 1;
    }
  }

  return result;
}

std::size_t detector_error_model::num_detectors() const {
  auto shape = detector_error_matrix.shape();
  if (shape.size() == 2)
    return shape[0];
  return 0;
}

std::size_t detector_error_model::num_error_mechanisms() const {
  auto shape = detector_error_matrix.shape();
  if (shape.size() == 2)
    return shape[1];
  return 0;
}

std::size_t detector_error_model::num_observables() const {
  auto shape = observables_flips_matrix.shape();
  if (shape.size() == 2)
    return shape[0];
  return 0;
}

void detector_error_model::canonicalize_for_rounds(
    uint32_t num_syndromes_per_round, bool remove_zero_syndrome_errors) {
  auto row_indices = dense_to_sparse(detector_error_matrix);
  auto column_order =
      get_sorted_pcm_column_indices(row_indices, num_syndromes_per_round);
  const std::size_t num_obs = this->num_observables();
  const auto num_cols = column_order.size();
  const bool has_error_ids =
      error_ids.has_value() && error_ids->size() == error_rates.size();

  if (row_indices.size() > error_rates.size()) {
    throw std::runtime_error(
        "canonicalize_for_rounds: row_indices size (" +
        std::to_string(row_indices.size()) +
        ") is greater than the number of error rates (" +
        std::to_string(error_rates.size()) +
        "). This likely means either 'error_rates' was populated incorrectly "
        "or the detector_error_matrix  was computed incorrectly.");
  }

  // March through the columns in topological order and merge columns that share
  // the SAME full signature: identical detector rows AND identical observable
  // rows. Columns that differ in either are distinct error mechanisms and are
  // kept separate (merging on detectors alone would relabel observable-flip
  // probability mass). The merge key is therefore (detector rows, observable
  // rows); because the sort above only orders by detector rows, columns with
  // the same detectors but different observables can be interleaved, so we
  // group by key explicitly rather than relying on adjacency.
  using signature_t =
      std::pair<std::vector<std::uint32_t>, std::vector<std::uint32_t>>;
  std::map<signature_t, std::size_t> sig_to_out;
  std::vector<std::uint32_t> final_column_order;
  // For each retained output column, accumulate probability mass grouped by the
  // exclusive-set it belongs to. Within one exclusive set (same error id) the
  // alternatives are mutually exclusive, so their rates add. Across exclusive
  // sets the mechanisms are independent, so they are combined with the XOR rule
  // P(A xor B) = P(A) + P(B) - 2 P(A) P(B). When error ids are absent every
  // column is treated as its own independent mechanism (keyed by its original
  // column index), reproducing the all-XOR behavior.
  std::vector<std::map<std::size_t, double>> out_exclusive;

  // Track the first observable signature seen for each detector signature so we
  // can flag columns that share a syndrome but flip a different observable.
  // These are kept as distinct mechanisms (above), but they are worth
  // surfacing: they often indicate an ambiguous/degenerate decoding situation.
  // Cap the per-invocation warnings since short-distance codes can have many
  // such mechanisms, and emit a single summary for the remainder.
  constexpr std::size_t max_same_syndrome_diff_obs_warnings = 10;
  std::size_t num_same_syndrome_diff_obs = 0;
  std::map<std::vector<std::uint32_t>,
           std::pair<std::vector<std::uint32_t>, std::uint32_t>>
      first_obs_for_detector;

  for (std::size_t c = 0; c < num_cols; c++) {
    const auto column_index = column_order[c];
    const auto &curr_row_indices = row_indices[column_index];
    const double rate = error_rates[column_index];

    // Build the observable-flip signature for this column.
    std::vector<std::uint32_t> obs_indices;
    for (std::size_t r = 0; r < num_obs; r++)
      if (this->observables_flips_matrix.at({r, column_index}))
        obs_indices.push_back(static_cast<std::uint32_t>(r));

    // Skip columns that carry no information: zero probability, or no detector
    // signature AND no observable flip. A column with no detectors but a
    // nonzero observable flip is a genuine (undetectable) logical error and is
    // retained by default so the model's observable-flip mass is preserved.
    // Such a column has no syndrome for a round-based decoder to act on, so
    // callers that only consume the detector matrix for decoding can drop all
    // zero-syndrome columns via remove_zero_syndrome_errors.
    const bool zero_syndrome = curr_row_indices.empty();
    if (rate == 0.0 || (zero_syndrome && obs_indices.empty()) ||
        (remove_zero_syndrome_errors && zero_syndrome))
      continue;

    signature_t sig{curr_row_indices, obs_indices};
    auto [it, inserted] = sig_to_out.try_emplace(sig, out_exclusive.size());
    if (inserted) {
      out_exclusive.emplace_back();
      final_column_order.push_back(column_index);

      // A new full signature. If this detector syndrome was already seen with a
      // different observable signature, this is a "same syndrome, different
      // observable" mechanism; flag it (capped).
      auto [dit, first_seen] = first_obs_for_detector.try_emplace(
          curr_row_indices, obs_indices, column_index);
      if (!first_seen && dit->second.first != obs_indices) {
        if (num_same_syndrome_diff_obs < max_same_syndrome_diff_obs_warnings)
          CUDA_QEC_WARN(
              "detector_error_model::canonicalize_for_rounds: identical "
              "syndromes exist in detector_error_matrix but have different "
              "observables in observables_flips_matrix; keeping column {} as a "
              "distinct error mechanism (previous column {})",
              column_index, dit->second.second);
        num_same_syndrome_diff_obs++;
      }
    }
    const std::size_t exclusive_key =
        has_error_ids ? error_ids->at(column_index) : column_index;
    out_exclusive[it->second][exclusive_key] += rate;
  }

  // Emit a single summary if we suppressed any per-column warnings above.
  if (num_same_syndrome_diff_obs > max_same_syndrome_diff_obs_warnings)
    CUDA_QEC_WARN(
        "detector_error_model::canonicalize_for_rounds: found {} columns with "
        "identical syndromes but different observables; suppressed {} "
        "additional warnings (only the first {} were shown).",
        num_same_syndrome_diff_obs,
        num_same_syndrome_diff_obs - max_same_syndrome_diff_obs_warnings,
        max_same_syndrome_diff_obs_warnings);

  std::vector<double> new_weights;
  std::vector<std::size_t> new_error_ids;
  new_weights.reserve(out_exclusive.size());
  for (std::size_t i = 0; i < out_exclusive.size(); i++) {
    double weight = 0.0;
    for (const auto &[id, p] : out_exclusive[i])
      weight = weight + p - 2.0 * weight * p;
    new_weights.push_back(weight);
    // Assign each output column a fresh unique id. Canonicalization does not
    // preserve any cross-column exclusivity structure: if a source mechanism's
    // mutually-exclusive outcomes land in different signature columns, that
    // relationship is no longer recoverable from the ids, so we do not pretend
    // it is. Unique ids simply mark every canonicalized column as an
    // independent mechanism, which is the relation the merged rates were
    // composed under.
    if (has_error_ids)
      new_error_ids.push_back(i);
  }

  std::swap(this->error_rates, new_weights);
  if (has_error_ids)
    std::swap(*this->error_ids, new_error_ids);

  // These two data structures should have the same number of columns.
  // (number of canonicalized error mechanisms)
  // Create the reordered, reduced Detector Error Matrix.
  this->detector_error_matrix = cudaq::qec::reorder_pcm_columns(
      this->detector_error_matrix, final_column_order);

  // Create the reordered, reduced Observables Flips Matrix.
  this->observables_flips_matrix = cudaq::qec::reorder_pcm_columns(
      this->observables_flips_matrix, final_column_order);
}

} // namespace cudaq::qec
