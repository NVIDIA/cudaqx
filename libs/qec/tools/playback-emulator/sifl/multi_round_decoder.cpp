/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file multi_round_decoder.cpp
/// @brief Dispatches a SIFL shot's variable-length stabilizer-round history
/// to one of `max_rounds` monolithic sub-decoders (e.g. pymatching), chosen
/// by how many rounds preceded the shot's data-qubit readout. No chunking,
/// seams, or stitching: each sub-decoder's H/O/D/error_rates come from a full
/// DEM analysed independently per round count by dem_templates.py, straight
/// from Stim. This decoder just loads the num_rounds monolithic matrices from
/// disk; it holds no notion of a chunk at all.
///
/// The shot boundary is detected by WIDTH, not a wire tag: every
/// enqueue_syndrome() of `round_width` bits is another stabilizer round; a
/// call of `terminal_width` bits is the data-qubit readout that ends the
/// shot -- its bits are still fed to the sub-decoder (D_sparse's last rows
/// depend on them, same as any real memory-experiment detector).
///
/// Decode runs synchronously, inline, on the readout call -- required for
/// a polling stream's poll-before-send design (see emulator.cpp) to observe
/// completion; a background-thread decode finishing on a later call would be
/// invisible to a client that stops polling once it gives up.

#include "cudaq/qec/decoder.h"
#include "cudaq/qec/decoder_config_schema.h"
#include "cudaq/qec/pcm_utils.h"

#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace cudaq::qec {

namespace {

/// A shot needs at least this many stabilizer rounds before its data-qubit
/// readout; kept in step with dem_templates.py's MIN_ROUNDS. Each round
/// count gets an independently built full DEM, so 1 is a valid shot -- and
/// a shot that keeps up with its cadence can genuinely close out after just
/// 1 round (see sifl_demo.py's schedule comment).
constexpr std::uint64_t kMinRounds = 1;

std::ifstream open_or_throw(const std::string &path) {
  std::ifstream f(path);
  if (!f)
    throw std::runtime_error("multi_round_decoder: cannot open " + path);
  return f;
}

/// Flat, -1-terminated sparse rows -- the format pcm_from_sparse_vec() uses.
std::vector<std::int64_t> read_flat_sparse(const std::string &path) {
  auto f = open_or_throw(path);
  std::vector<std::int64_t> vec;
  std::int64_t v;
  while (f >> v)
    vec.push_back(v);
  return vec;
}

std::vector<double> read_floats(const std::string &path) {
  auto f = open_or_throw(path);
  std::vector<double> vec;
  double v;
  while (f >> v)
    vec.push_back(v);
  return vec;
}

/// Split the -1-terminated flat form into rows.
std::vector<std::vector<std::uint32_t>>
rows_from_flat(const std::vector<std::int64_t> &flat) {
  std::vector<std::vector<std::uint32_t>> rows(1);
  for (auto v : flat) {
    if (v == -1)
      rows.emplace_back();
    else
      rows.back().push_back(static_cast<std::uint32_t>(v));
  }
  rows.pop_back();
  return rows;
}

} // namespace

class multi_round_decoder : public decoder {
public:
  multi_round_decoder(const cudaq::qec::sparse_binary_matrix &H,
                      const cudaqx::heterogeneous_map &params)
      : decoder(H) {
    const auto dir = params.get<std::string>("template_dir");
    round_width_ = params.get<std::uint64_t>("round_width");
    terminal_width_ = params.get<std::uint64_t>("terminal_width");
    max_rounds_ = params.get<std::uint64_t>("max_rounds");
    const auto delegate_type =
        params.get<std::string>("delegate_type", "pymatching");
    const auto n_obs = params.get<std::uint64_t>("num_obs", 1u);
    const auto delegate_params =
        params.get<cudaqx::heterogeneous_map>("delegate_params", {});

    sub_decoders_.reserve(max_rounds_ - kMinRounds + 1);
    for (std::uint64_t r = kMinRounds; r <= max_rounds_; ++r) {
      const auto prefix = dir + "/r" + std::to_string(r);
      auto rates = read_floats(prefix + ".rates");
      // Row count isn't in the flat sparse format; count its -1 terminators.
      const auto H_flat = read_flat_sparse(prefix + ".H");
      const auto num_detectors =
          static_cast<std::size_t>(std::count(H_flat.begin(), H_flat.end(), -1));
      auto H_r = pcm_from_sparse_vec(H_flat, num_detectors, rates.size());

      cudaqx::heterogeneous_map sub_params = delegate_params;
      sub_params.insert(
          "O", pcm_from_sparse_vec(read_flat_sparse(prefix + ".O"), n_obs,
                                   rates.size()));
      sub_params.insert("error_rate_vec", rates);
      auto sub = cudaq::qec::get_decoder(
          delegate_type, cudaq::qec::sparse_binary_matrix(H_r), sub_params);
      sub->set_D_sparse(rows_from_flat(read_flat_sparse(prefix + ".D")));
      sub_decoders_.push_back(std::move(sub));
    }
    corrections_.assign(n_obs, 0);
  }

  bool enqueue_syndrome(const uint8_t *syndrome, std::size_t len) override {
    if (len == terminal_width_) {
      if (rounds_seen_ < kMinRounds || rounds_seen_ > max_rounds_)
        throw std::runtime_error(
            "multi_round_decoder: shot ended with " +
            std::to_string(rounds_seen_) +
            " stabilizer rounds, outside the configured [" +
            std::to_string(kMinRounds) + ", " + std::to_string(max_rounds_) +
            "] range");
      buffer_.insert(buffer_.end(), syndrome, syndrome + len);
      auto &sub = *sub_decoders_[rounds_seen_ - kMinRounds];
      if (!sub.enqueue_syndrome(buffer_.data(), buffer_.size()))
        throw std::runtime_error(
            "multi_round_decoder: sub-decoder for " +
            std::to_string(rounds_seen_) + " rounds didn't fire on a full shot");
      std::copy_n(sub.get_obs_corrections(), corrections_.size(),
                 corrections_.begin());
      buffer_.clear();
      rounds_seen_ = 0;
      return true;
    }
    if (len != round_width_)
      throw std::runtime_error(
          "multi_round_decoder: enqueue width " + std::to_string(len) +
          " matches neither round_width (" + std::to_string(round_width_) +
          ") nor terminal_width (" + std::to_string(terminal_width_) + ")");
    buffer_.insert(buffer_.end(), syndrome, syndrome + len);
    if (++rounds_seen_ > max_rounds_)
      throw std::runtime_error(
          "multi_round_decoder: shot exceeded max_rounds (" +
          std::to_string(max_rounds_) + ") without a data-qubit readout; saw " +
          std::to_string(rounds_seen_));
    return false;
  }

  void reset_decoder() override {
    buffer_.clear();
    rounds_seen_ = 0;
    decoder::reset_decoder();
    std::fill(corrections_.begin(), corrections_.end(), 0u);
  }

  const uint8_t *get_obs_corrections() const override {
    return corrections_.data();
  }
  void clear_corrections() override {
    std::fill(corrections_.begin(), corrections_.end(), 0u);
  }
  decoder_result decode(const std::vector<float_t> &) override {
    return {true, std::vector<float_t>(block_size, 0.0f)};
  }

  CUDAQ_EXTENSION_CUSTOM_CREATOR_FUNCTION(
      multi_round_decoder,
      static std::unique_ptr<decoder>
      create(const cudaq::qec::decoder_init &init,
             const cudaqx::heterogeneous_map &params) {
        return cudaq::qec::make_pcm_decoder<multi_round_decoder>(init, params);
      })

private:
  std::uint64_t round_width_ = 0, terminal_width_ = 0, max_rounds_ = 0;
  std::vector<std::unique_ptr<decoder>> sub_decoders_;
  std::vector<std::uint8_t> buffer_;
  std::uint64_t rounds_seen_ = 0;
  std::vector<uint8_t> corrections_;
};

CUDAQ_EXT_PT_REGISTER_TYPE(multi_round_decoder)

namespace {
struct schema_reg {
  schema_reg() {
    using k = cudaq::qec::decoding::config::param_kind;
    cudaq::qec::decoding::config::register_decoder_schema(
        {"multi_round_delegate_params",
         {{"use_sparsity", k::boolean},
          {"max_iterations", k::int32},
          {"use_osd", k::boolean},
          {"osd_method", k::int32},
          {"osd_order", k::int32}}});
    cudaq::qec::decoding::config::register_decoder_schema(
        {"multi_round_decoder",
         {{"template_dir", k::string},
          {"round_width", k::uint64},
          {"terminal_width", k::uint64},
          {"max_rounds", k::uint64},
          {"delegate_type", k::string},
          {"num_obs", k::uint64},
          {"delegate_params", k::subschema, false,
           "multi_round_delegate_params"}}});
  }
} reg;
} // namespace

} // namespace cudaq::qec
