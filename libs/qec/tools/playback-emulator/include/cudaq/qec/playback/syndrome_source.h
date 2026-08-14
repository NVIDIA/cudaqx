/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

/// @file syndrome_source.h
/// @brief `syndrome_source` and its two 
/// implementations: `static_source` (replay pre-supplied rounds) and
/// `stim_memory_source` (JIT round generation from a Stim memory circuit,
/// produced strictly on demand so the persistent simulator's state never 
/// advances further than what has actually been consumed. 

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace cudaq::qec::playback {

/// Yields one round of bits per `next_round()` call, and an empty round to
/// signal exhaustion.
class syndrome_source {
public:
  virtual ~syndrome_source() = default;

  /// Empty vector => exhausted.
  virtual std::vector<std::uint8_t> next_round() = 0;

  /// The terminal data-qubit readout that ends a shot (`enqueue_data`'s
  /// source call. 
  /// Default just pulls the next produced item, same as `next_round()`
  /// A source with a genuine two-phase circuit (stabilizer
  /// rounds vs. a distinct terminal segment against a live simulator, e.g.
  /// `stim_memory_source`) overrides this to run that segment instead.
  virtual std::vector<std::uint8_t> read_data() { return next_round(); }

  /// Rewind for a new shot.
  virtual void reset() {}

  /// True for a source that generates data on demand rather than replaying
  /// pre-supplied rounds (e.g. stim_memory_source's persistent simulator) --
  /// required for `stream_until`. A source that isn't streamed runs dry mid-runaway
  /// and reports SOURCE_EXHAUSTED where the experiment needs continued
  /// growth.
  virtual bool is_streamed() const { return false; }
};

/// Replay pre-supplied rounds. The reference source: any test that needs an
/// exactly reproducible input, an oracle comparison, or a clean timing
/// measurement should use it. Round widths may differ between elements.
class static_source : public syndrome_source {
public:
  explicit static_source(std::vector<std::vector<std::uint8_t>> rounds);

  std::vector<std::uint8_t> next_round() override;
  void reset() override;

private:
  std::vector<std::vector<std::uint8_t>> rounds_;
  std::size_t next_ = 0;
};

/// Just-in-time round generation from a Stim memory circuit. Each
/// `next_round()` call generates the next round of measurement outcomes at
/// the time of the call -- not at construction -- by advancing a persistent
/// Pauli-frame simulator by exactly one round, always returning exactly
/// `round_width()` bits. On the first call after construction or a shot
/// boundary, the circuit's prefix (if any) is folded in as warm-up
/// simulation rather than exposed as an extra-wide result: it's played
/// first, and if it doesn't already amount to a full round, round_body is
/// run until one is. `read_data()` runs the circuit's terminal segment
/// (everything after the `REPEAT` block) against that same simulator's
/// current state -- i.e. exactly however many rounds have actually been
/// consumed via `next_round()` so far -- and then starts a fresh simulator
/// for the next shot.
///
/// A `next_round()` or `read_data()` call does its Stim work synchronously, on the calling
/// thread. 
class stim_memory_source : public syndrome_source {
public:
  /// `stim_circuit_text` is a Stim circuit whose body is exactly one
  /// `REPEAT N { ... }` syndrome-extraction block, optionally preceded by a
  /// prefix and optionally followed by a terminal data-qubit readout segment
  /// A circuit with nothing after the `REPEAT` block is a valid
  /// stabilizer-only source whose `read_data()` just returns zero bits
  /// (and still starts a fresh simulator for the next shot). 
  stim_memory_source(std::string stim_circuit_text, std::uint64_t seed);
  ~stim_memory_source() override;

  stim_memory_source(const stim_memory_source &) = delete;
  stim_memory_source &operator=(const stim_memory_source &) = delete;

  /// Always round_width() bits, including the first call after construction
  /// or a shot boundary (see the class doc comment for how a non-trivial
  /// prefix is folded in).
  std::vector<std::uint8_t> next_round() override;
  std::vector<std::uint8_t> read_data() override;
  void reset() override;
  bool is_streamed() const override { return true; }

  /// Width, in bits, of one syndrome-extraction round (not the terminal
  /// data readout, which may differ).
  std::uint32_t round_width() const;

  /// Width, in bits, of the terminal data-qubit readout. Zero if the
  /// circuit has no segment after its `REPEAT` block.
  std::uint32_t data_width() const;

private:
  struct impl;
  std::unique_ptr<impl> impl_;
};

} // namespace cudaq::qec::playback
