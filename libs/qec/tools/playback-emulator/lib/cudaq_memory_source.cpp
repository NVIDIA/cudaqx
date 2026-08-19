/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/qec/playback/cudaq_memory_source.h"

#include "cudaq.h"
#include "cuda-qx/core/tensor.h"
#include "cudaq/qec/code.h"
#include "device/memory_circuit.h"

#include <stdexcept>

namespace cudaq::qec::playback {

struct cudaq_memory_source::impl {
  const cudaq::qec::code &qec_code;
  cudaq::qec::operation state_prep;
  std::size_t max_rounds;
  cudaq::noise_model noise;
  std::uint64_t base_seed;

  std::uint64_t generation = 0; // bumped on every shot boundary
  std::size_t current_round = 0; // next_round() calls made so far this shot

  std::size_t num_cols = 0; // numAncx + numAncz: width of a raw round
  std::size_t num_data = 0; // width of the raw data-qubit readout

  // round_bits[r - 1]: raw ancilla measurements for round r.
  std::vector<std::vector<std::uint8_t>> round_bits;
  // data_bits[r - 1]: raw data-qubit measurements for a shot that reads out
  // right after round r.
  std::vector<std::vector<std::uint8_t>> data_bits;

  impl(const cudaq::qec::code &c, cudaq::qec::operation prep,
       std::size_t maxRounds, cudaq::noise_model n, std::uint64_t seed)
      : qec_code(c), state_prep(prep), max_rounds(maxRounds),
        noise(std::move(n)), base_seed(seed) {
    if (max_rounds == 0)
      throw std::runtime_error(
          "cudaq_memory_source: max_rounds must be >= 1.");
    rebuild_cache();
  }

  // Reruns the memory_circuit kernel for every 1 <= r <= max_rounds under
  // the same seed. memory_circuit allocates all its qubits up front, before
  // the round loop, so numRounds never changes *when* a qubit is allocated
  // during the rounds these runs share -- that keeps the stim backend's
  // per-instruction RNG draws bit-identical across the runs for any shared
  // instruction prefix. So run r's own raw measurement record --
  // numCols-wide ancilla chunks for rounds 1..r, then a numData-wide data
  // chunk -- only contributes one new piece of information beyond what run
  // r-1 already had: round r's own ancilla measurements, plus this run's
  // own data-qubit measurements (what a data readout would have been, had
  // the shot ended at round r).
  void rebuild_cache() {
    round_bits.assign(max_rounds, {});
    data_bits.assign(max_rounds, {});

    if (!qec_code.contains_operation(cudaq::qec::operation::stabilizer_round))
      throw std::runtime_error(
          "cudaq_memory_source: code has no stabilizer_round operation.");
    if (!qec_code.contains_operation(state_prep))
      throw std::runtime_error(
          "cudaq_memory_source: code does not support the requested state "
          "prep.");

    auto &prep =
        qec_code.get_operation<cudaq::qec::code::one_qubit_encoding>(
            state_prep);
    auto &stabRound =
        qec_code.get_operation<cudaq::qec::code::stabilizer_round>(
            cudaq::qec::operation::stabilizer_round);

    const bool is_z_prep = state_prep == cudaq::qec::operation::prep0 ||
                            state_prep == cudaq::qec::operation::prep1;

    auto sched_x = qec_code.get_stabilizer_schedule_x();
    auto sched_z = qec_code.get_stabilizer_schedule_z();
    std::vector<std::size_t> xVec(sched_x.data(),
                                   sched_x.data() + sched_x.size());
    std::vector<std::size_t> zVec(sched_z.data(),
                                   sched_z.data() + sched_z.size());
    auto logical_obs =
        is_z_prep ? qec_code.get_observables_z() : qec_code.get_observables_x();
    const std::size_t num_obs = logical_obs.shape()[0];
    std::vector<std::size_t> obs_flat(logical_obs.data(),
                                       logical_obs.data() + logical_obs.size());

    num_data = qec_code.get_num_data_qubits();
    const std::size_t numAncx = qec_code.get_num_ancilla_x_qubits();
    const std::size_t numAncz = qec_code.get_num_ancilla_z_qubits();
    num_cols = numAncx + numAncz;

    for (std::size_t r = 1; r <= max_rounds; ++r) {
      cudaq::set_random_seed(
          static_cast<std::size_t>(base_seed + generation));

      cudaq::sample_options opts{
          .shots = 1, .noise = noise, .explicit_measurements = true};
      auto result =
          cudaq::sample(opts, cudaq::qec::memory_circuit, stabRound, prep,
                        num_data, numAncx, numAncz, r, xVec, zVec, obs_flat,
                        num_obs, !is_z_prep);

      // mzTable[0, meas_idx]: raw measurement layout is r*num_cols ancilla
      // bits (num_cols per round), then num_data data-qubit bits.
      cudaqx::tensor<std::uint8_t> mzTable(result.sequential_data());

      auto &round = round_bits[r - 1];
      round.resize(num_cols);
      const std::size_t round_start = (r - 1) * num_cols;
      for (std::size_t i = 0; i < num_cols; ++i)
        round[i] = mzTable.at({0, round_start + i});

      auto &shot_data = data_bits[r - 1];
      shot_data.resize(num_data);
      const std::size_t data_start = num_cols * r;
      for (std::size_t i = 0; i < num_data; ++i)
        shot_data[i] = mzTable.at({0, data_start + i});
    }
  }

  std::vector<std::uint8_t> next_round() {
    if (current_round >= max_rounds)
      return {};
    return round_bits[current_round++];
  }

  std::vector<std::uint8_t> read_data() {
    if (current_round == 0)
      throw std::runtime_error(
          "cudaq_memory_source::read_data() called before any "
          "next_round(); memory_circuit always performs at least one "
          "stabilizer round before a data readout.");
    auto bits = data_bits[current_round - 1];
    reset();
    return bits;
  }

  void reset() {
    ++generation;
    current_round = 0;
    rebuild_cache();
  }
};

cudaq_memory_source::cudaq_memory_source(const code &code,
                                          operation statePrep,
                                          std::size_t max_rounds,
                                          cudaq::noise_model noise,
                                          std::uint64_t seed)
    : impl_(std::make_unique<impl>(code, statePrep, max_rounds,
                                    std::move(noise), seed)) {}

cudaq_memory_source::~cudaq_memory_source() = default;

std::vector<std::uint8_t> cudaq_memory_source::next_round() {
  return impl_->next_round();
}

std::vector<std::uint8_t> cudaq_memory_source::read_data() {
  return impl_->read_data();
}

void cudaq_memory_source::reset() { impl_->reset(); }

std::size_t cudaq_memory_source::max_rounds() const {
  return impl_->max_rounds;
}

std::uint32_t cudaq_memory_source::round_width() const {
  return static_cast<std::uint32_t>(impl_->num_cols);
}

std::uint32_t cudaq_memory_source::data_width() const {
  return static_cast<std::uint32_t>(impl_->num_data);
}

} // namespace cudaq::qec::playback
