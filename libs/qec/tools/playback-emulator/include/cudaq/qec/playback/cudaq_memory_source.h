/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

/// @file cudaq_memory_source.h
/// @brief `cudaq_memory_source`: a syndrome_source backed by CUDA-Q's own
/// `memory_circuit` kernel (run under the `stim` target), for tests and
/// tools that want syndrome streams tied to the actual kernel rather than a
/// hand-supplied or independently-derived Stim circuit (see
/// `stim_memory_source` for that).

#include "cudaq/qec/playback/syndrome_source.h"

#include <cstdint>
#include <memory>

namespace cudaq {
class noise_model;
}

namespace cudaq::qec {
class code;
enum class operation;
} // namespace cudaq::qec

namespace cudaq::qec::playback {

/// Emulates streaming raw measurements out of CUDA-Q's `memory_circuit`
/// kernel under the `stim` target -- ancilla measurement outcomes per round
/// and data-qubit measurement outcomes at readout, the same granularity
/// `stim_memory_source` streams, not the XOR-combined detector values
/// `sample_memory_circuit` returns.
///
/// A single kernel launch can't be driven round-by-round: the Stim NVQIR
/// backend has no amplitude-based state to snapshot (state injection is
/// unconditionally rejected for it), and CUDA-Q has no hook to extract
/// results mid-execution. So instead of one persistent simulator advanced
/// incrementally (`stim_memory_source`'s approach), this runs the
/// `memory_circuit` kernel once per candidate round count `r`, for every
/// `1 <= r <= max_rounds`, all under the same seed -- at construction, and
/// again on every `reset()`/shot boundary. `memory_circuit` allocates every
/// qubit up front, before its round loop, so `numRounds` never changes
/// *when* a qubit is allocated during the shared rounds; that keeps the
/// stim backend's per-instruction RNG draws bit-identical across those
/// separate launches for any prefix the runs share. So each run for round
/// count `r` only contributes two genuinely new pieces of information: its
/// own round-`r` raw ancilla measurements, and its own raw data-qubit
/// measurements (what a data readout would have been, had the shot ended at
/// round `r`). Both are cached; `next_round()` and `read_data()` just serve
/// slices of the cache built this way -- no circuit executes on those calls
/// themselves.
///
/// The assembled output -- every next_round() block plus whichever
/// read_data() call ends the shot -- IS bit-identical to what a single real
/// memory_circuit shot of that length would have produced at the same seed:
/// the seed fixes the whole noise trajectory, not just the first run's. The
/// cost is purely computational, not a fidelity gap: producing those
/// `max_rounds` round blocks and `max_rounds` possible data-readouts takes
/// `max_rounds` separate kernel launches (paid at construction/reset, not
/// spread across next_round() calls), each re-simulating its rounds from
/// scratch rather than resuming -- O(max_rounds^2) circuit-instruction work
/// for O(max_rounds) useful outputs, where a true incremental simulator
/// would do O(max_rounds).
class cudaq_memory_source : public syndrome_source {
public:
  /// `code` must outlive this object; `statePrep` and `noise` are copied.
  /// Throws if `max_rounds == 0` or `code` doesn't support `statePrep`
  /// (matching `sample_memory_circuit`'s own validation).
  cudaq_memory_source(const code &code, operation statePrep,
                      std::size_t max_rounds, cudaq::noise_model noise,
                      std::uint64_t seed);
  ~cudaq_memory_source() override;

  cudaq_memory_source(const cudaq_memory_source &) = delete;
  cudaq_memory_source &operator=(const cudaq_memory_source &) = delete;

  /// Always round_width() raw ancilla measurement bits, one call per
  /// stabilizer-extraction round, up to `max_rounds` calls per shot; an
  /// empty vector once that's exhausted.
  std::vector<std::uint8_t> next_round() override;

  /// Raw data-qubit measurement bits (data_width() of them) for a shot
  /// ending after however many rounds have actually been consumed via
  /// `next_round()` so far, then starts a fresh shot. Throws if called
  /// before any `next_round()` call (`memory_circuit` always performs at
  /// least one round before readout).
  std::vector<std::uint8_t> read_data() override;

  /// Starts a fresh shot (new seed generation), discarding any unconsumed
  /// rounds from the current one.
  void reset() override;

  bool is_streamed() const override { return true; }

  /// The `max_rounds` given at construction: how many `next_round()` calls
  /// a shot can serve before exhausting.
  std::size_t max_rounds() const;

  /// Width, in bits, of one round's raw ancilla measurements
  /// (numAncx + numAncz for the underlying code).
  std::uint32_t round_width() const;

  /// Width, in bits, of the raw data-qubit readout (numData for the
  /// underlying code).
  std::uint32_t data_width() const;

private:
  struct impl;
  std::unique_ptr<impl> impl_;
};

} // namespace cudaq::qec::playback
