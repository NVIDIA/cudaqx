/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/qec/playback/syndrome_source.h"

#include "stim.h"

#include <algorithm>
#include <memory>
#include <random>
#include <stdexcept>

namespace cudaq::qec::playback {

// ─── static_source ──────────────────────────────────────────────────────────

static_source::static_source(std::vector<std::vector<std::uint8_t>> rounds)
    : rounds_(std::move(rounds)) {}

std::vector<std::uint8_t> static_source::next_round() {
  if (next_ >= rounds_.size())
    return {};
  return rounds_[next_++];
}

void static_source::reset() { next_ = 0; }

// ─── stim_memory_source ─────────────────────────────────────────────────────
//
// Drives a live stim::FrameSimulator, advanced by exactly one round 
// per next_round() call, or by the
// circuit's terminal segment per
// read_data() call. Both happen synchronously on the calling thread.

constexpr std::size_t kSimdWidth = stim::MAX_BITWORD_WIDTH;

struct stim_memory_source::impl {
  stim::Circuit prefix, round_body, terminal_body;
  std::uint32_t prefix_width, round_width, terminal_width, num_qubits;
  std::uint64_t base_seed;

  std::uint64_t generation = 0; // bumped on every shot boundary
  bool prefix_played = false;   // cleared by rebuild_sim(); see next_round()
  std::unique_ptr<stim::FrameSimulator<kSimdWidth>> sim;

  // Splits the circuit at its (first) REPEAT block into a prefix, the repeating 
  // stabilizer-round body, and whatever terminal (data-qubit readout) segment follows, if any.
  impl(std::string stim_circuit_text, std::uint64_t seed) : base_seed(seed) {
    stim::Circuit full(stim_circuit_text);
    std::size_t i = 0;
    for (; i < full.operations.size(); ++i)
      if (full.operations[i].gate_type == stim::GateType::REPEAT)
        break;
    if (i == full.operations.size())
      throw std::runtime_error(
          "stim_memory_source: circuit has no REPEAT block; a "
          "syndrome-extraction round can only be derived from a repeating "
          "block.");
    for (std::size_t j = 0; j < i; ++j)
      prefix.safe_append(full.operations[j]);
    round_body = full.operations[i].repeat_block_body(full);
    for (std::size_t j = i + 1; j < full.operations.size(); ++j)
      terminal_body.safe_append(full.operations[j]);

    prefix_width = prefix.count_measurements();
    round_width = round_body.count_measurements();
    terminal_width = terminal_body.count_measurements();
    num_qubits = full.count_qubits();
    rebuild_sim();
  }

  void rebuild_sim() {
    stim::CircuitStats stats;
    stats.num_qubits = num_qubits;
    stats.max_lookback = std::max({prefix_width, round_width, terminal_width});
    sim = std::make_unique<stim::FrameSimulator<kSimdWidth>>(
        stats, stim::FrameSimulatorMode::STREAM_MEASUREMENTS_TO_DISK,
        /*batch_size=*/1, std::mt19937_64(base_seed + generation));
    sim->reset_all();
    prefix_played = false;
  }

  std::vector<std::uint8_t> read_lookback(std::uint32_t width) {
    std::vector<std::uint8_t> bits(width);
    for (std::uint32_t i = 0; i < width; ++i)
      bits[i] = sim->m_record.lookback(width - i)[0] ? 1 : 0;
    sim->m_record.mark_all_as_written();
    return bits;
  }

  // The first call after a shot boundary also plays the prefix, folding it
  // into the round rather than exposing it as an extra-wide result: keeps
  // running round_body until a full round_width of fresh bits is available.
  std::vector<std::uint8_t> next_round() {
    std::vector<std::uint8_t> bits;
    if (!prefix_played) {
      prefix_played = true;
      sim->safe_do_circuit(prefix);
      bits = read_lookback(prefix_width);
    }
    while (bits.size() < round_width) {
      sim->safe_do_circuit(round_body);
      auto rest = read_lookback(round_width);
      bits.insert(bits.end(), rest.begin(), rest.end());
    }
    if (bits.size() != round_width)
      throw std::runtime_error(
          "stim_memory_source: prefix width is not a multiple of "
          "round_width; a uniform round_width() can't be formed.");
    return bits;
  }

  // Runs the terminal segment against the simulator's current state then starts a fresh shot.
  std::vector<std::uint8_t> read_data() {
    sim->safe_do_circuit(terminal_body);
    auto bits = read_lookback(terminal_width);
    ++generation;
    rebuild_sim();
    return bits;
  }

  void reset() {
    ++generation;
    rebuild_sim();
  }
};

stim_memory_source::stim_memory_source(std::string stim_circuit_text,
                                        std::uint64_t seed)
    : impl_(std::make_unique<impl>(std::move(stim_circuit_text), seed)) {}

stim_memory_source::~stim_memory_source() = default;

std::vector<std::uint8_t> stim_memory_source::next_round() {
  return impl_->next_round();
}

std::vector<std::uint8_t> stim_memory_source::read_data() {
  return impl_->read_data();
}

void stim_memory_source::reset() { impl_->reset(); }

std::uint32_t stim_memory_source::round_width() const {
  return impl_->round_width;
}

std::uint32_t stim_memory_source::data_width() const {
  return impl_->terminal_width;
}

} // namespace cudaq::qec::playback
