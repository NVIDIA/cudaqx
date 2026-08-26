/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// Tests syndrome sources. static_source (exact replay) pins down replay,
/// exhaustion, and rewind. stim_memory_source's tests centre on round width,
/// seed determinism, and reset/generation logic, cross-checked against a
/// whole-circuit run of Stim's built-ins; cudaq_memory_source is validated
/// against direct runs of the `memory_circuit` kernel.

#include "syndrome_source.h"

#include "cudaq.h"
#include "cuda-qx/core/tensor.h"
#include "cudaq/qec/code.h"
#include "cudaq/qec/noise_model.h"
#include "device/memory_circuit.h"
#include "stim.h"
#include "stim/gen/gen_color_code.h"
#include "stim/gen/gen_rep_code.h"
#include "stim/gen/gen_surface_code.h"
#include "stim/simulators/frame_simulator_util.h"

#include <chrono>
#include <gtest/gtest.h>
#include <span>

using namespace cudaq::qec::playback;

using rounds_t = std::vector<std::vector<uint8_t>>;

// ─── static_source ──────────────────────────────────────────────────────────

TEST(StaticSource, ReplaysEveryRoundInOrderThenStaysExhausted) {
  const std::vector<uint8_t> wide(1000, 1);
  const rounds_t cases[] = {
      {},                            // nothing to replay: exhausted at once
      {{1, 0, 1}},                   // exactly one round
      {{0, 1}, {1, 1, 0}, {0}},      // several, of differing widths
      {{1}, wide, {}},               // one bit, then 1000, then none
  };
  for (const auto &rounds : cases) {
    SCOPED_TRACE("rounds=" + std::to_string(rounds.size()));
    static_source src(rounds);
    EXPECT_FALSE(src.is_streamed());
    for (std::size_t i = 0; i < rounds.size(); ++i)
      EXPECT_EQ(src.next_round(), rounds[i]) << "round " << i;
    // Past the end it stays empty forever: no wraparound, no throw, no
    // corruption however many times it is asked.
    for (int i = 0; i < 10; ++i)
      EXPECT_TRUE(src.next_round().empty()) << "call " << i << " past the end";
  }
}

TEST(StaticSource, AZeroWidthRoundIsNotMistakenForExhaustion) {
  // An empty return is ambiguous by content alone -- it is both "this round
  // has no bits" and "there are no more rounds". Only a later call can tell
  // them apart, so a zero-width round sandwiched between two real ones must
  // not stop the replay.
  static_source src({{1}, {}, {1}});
  EXPECT_EQ(src.next_round(), (std::vector<uint8_t>{1}));
  EXPECT_TRUE(src.next_round().empty());                  // the zero-width round
  EXPECT_EQ(src.next_round(), (std::vector<uint8_t>{1})); // proves it kept going
  EXPECT_TRUE(src.next_round().empty());                  // now genuinely done
}

TEST(StaticSource, ResetRewindsToRoundZeroFromAnywhereAndIsIdempotent) {
  static_source src({{1}, {2}, {3}, {4}});
  const auto expect_full_replay = [&] {
    for (uint8_t v : {1, 2, 3, 4})
      EXPECT_EQ(src.next_round(), (std::vector<uint8_t>{v}));
  };

  src.reset(); // before anything was consumed
  src.reset(); // and again: repeated resets are idempotent
  expect_full_replay();

  src.next_round(); // partially consumed, nowhere near the end
  src.next_round();
  src.reset(); // must rewind all the way, not to "wherever we were"
  expect_full_replay();

  ASSERT_TRUE(src.next_round().empty()); // fully exhausted
  src.reset();
  expect_full_replay();
}

TEST(StaticSource, PassesThroughNonBinaryValuesUnchanged) {
  // static_source is a pure replay buffer: next_round() returns a copy of
  // whatever it was handed, with no check that values are really 0/1 --
  // consistent with what replay means, though a caller cannot lean on it to
  // catch a malformed fixture.
  static_source src({{0, 1, 2, 255}});
  EXPECT_EQ(src.next_round(), (std::vector<uint8_t>{0, 1, 2, 255}));
}

// ─── stim_memory_source ─────────────────────────────────────────────────────

namespace {

// One measurement per round, in a REPEAT block wide enough to be
// practically unbounded for a test.
const char *kSingleM = R"CIRCUIT(
R 0
REPEAT 1000000 {
  H 0
  M 0
}
)CIRCUIT";

// Several separate M instructions in one round: the width is the sum across
// all of them (1 + 3), not just the last one seen.
const char *kMultiM = R"CIRCUIT(
R 0 1 2 3
REPEAT 1000000 {
  H 0 1 2 3
  M 0
  M 1 2 3
}
)CIRCUIT";

// A single M instruction spanning many qubits.
const char *kManyQubitSingleM = R"CIRCUIT(
R 0 1 2 3 4
REPEAT 1000000 {
  H 0 1 2 3 4
  M 0 1 2 3 4
}
)CIRCUIT";

// Dozens of qubits split across several M instructions, so "many qubits" and
// "many instructions" are exercised together.
const char *kWide = R"CIRCUIT(
R 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39
REPEAT 1000000 {
  H 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39
  M 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
  M 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39
}
)CIRCUIT";

} // namespace

TEST(StimMemorySource, RoundWidthIsEveryMeasurementInTheRepeatBlockSummed) {
  const std::pair<const char *, std::size_t> cases[] = {
      {kSingleM, 1}, {kMultiM, 4}, {kManyQubitSingleM, 5}, {kWide, 40}};
  for (const auto &[circuit, width] : cases) {
    SCOPED_TRACE("width=" + std::to_string(width));
    stim_memory_source src(circuit, /*seed=*/1);
    EXPECT_EQ(src.round_width(), width);
    // Drawn well past any plausible internal batch boundary: every round is
    // exactly round_width() wide and holds only real bits.
    for (int i = 0; i < 300; ++i) {
      auto round = src.next_round();
      ASSERT_EQ(round.size(), width) << "round " << i;
      for (auto b : round)
        ASSERT_LE(b, 1u) << "round " << i;
    }
  }
}

TEST(StimMemorySource, IsUnboundedAndOneSeedAlwaysGivesOneStream) {
  stim_memory_source a(kSingleM, /*seed=*/42);
  EXPECT_TRUE(a.is_streamed());

  stim_memory_source same_seed(kSingleM, /*seed=*/42);
  for (int i = 0; i < 100; ++i)
    ASSERT_EQ(a.next_round(), same_seed.next_round()) << "round " << i;

  // ...and a different seed must not silently produce the same stream, which
  // is what a seed that never reached the simulator would look like.
  stim_memory_source b(kSingleM, /*seed=*/1), c(kSingleM, /*seed=*/2);
  bool any_different = false;
  for (int i = 0; i < 200 && !any_different; ++i)
    any_different = b.next_round() != c.next_round();
  EXPECT_TRUE(any_different);
}

TEST(StimMemorySource, ResetReseedsWithoutWedgingOrStallingTheSource) {
  // reset() bumps a generation counter and rebuilds the simulator. Covers
  // the shapes that could wedge it: resetting before any draw, resetting
  // repeatedly with none in between, and alternating tightly with draws.
  // A fresh shot re-seeds rather than rewinds, so bits need not repeat.
  const auto start = std::chrono::steady_clock::now();
  stim_memory_source src(kMultiM, /*seed=*/11);

  src.reset(); // before any next_round() at all
  ASSERT_EQ(src.next_round().size(), src.round_width());

  for (int i = 0; i < 100; ++i)
    src.reset(); // back to back, nothing drawn between
  ASSERT_EQ(src.next_round().size(), src.round_width());

  for (int iter = 0; iter < 50; ++iter) {
    for (int i = 0; i < 3; ++i)
      ASSERT_EQ(src.next_round().size(), src.round_width())
          << "iter=" << iter << " i=" << i;
    src.reset();
  }
  EXPECT_LT(std::chrono::steady_clock::now() - start, std::chrono::seconds(5));
}

TEST(StimMemorySource, DestructionAfterUseDoesNotHangOrCrash) {
  for (int i = 0; i < 20; ++i) {
    stim_memory_source src(kSingleM, /*seed=*/static_cast<uint64_t>(i));
    src.next_round();
    src.next_round();
    // src goes out of scope here -- generation is synchronous (no producer
    // thread to join), but the destructor must still be well-formed.
  }
}

TEST(StimMemorySource, RejectsACircuitWithNoRepeatBlock) {
  EXPECT_THROW(stim_memory_source("H 0\nM 0\n", /*seed=*/1),
              std::runtime_error);
}

// ─── stim_memory_source vs. a single whole-circuit run ─────────────────────
//
// Cross-checks stim_memory_source's round-by-round generation against a
// single whole-circuit `do_circuit()` call: splitting into prefix/round/
// terminal must produce EXACTLY the same bits. Covers all six of Stim's
// generated-circuit families; color_code:memory_xyz's zero-measurement
// prefix guards against it being skipped and desyncing the RNG.

namespace {

constexpr std::size_t kSimdWidth = stim::MAX_BITWORD_WIDTH;

std::string to_bit_string(const stim::simd_bits<kSimdWidth> &bits,
                          std::size_t count) {
  std::string out;
  out.reserve(count);
  for (std::size_t i = 0; i < count; ++i)
    out += char('0' + (bits[i] ? 1 : 0));
  return out;
}

// One shot through stim's own batch sampler -- the entry point Python's
// `compile_sampler().sample()` reaches, rather than a hand-built
// FrameSimulator that could drift. `reference_sample` picks what comes
// back: all-zero yields the raw noise frame, a real sample yields absolutes.
std::string sample_whole_circuit_once(
    const stim::Circuit &circuit, std::uint64_t seed,
    const stim::simd_bits<kSimdWidth> &reference_sample) {
  // stim_memory_source seeds its own simulator with mt19937_64(seed), so the
  // reference has to be seeded identically or the two consume different
  // random streams and nothing below means anything.
  std::mt19937_64 rng(seed);
  const std::size_t n = circuit.count_measurements();
  auto table = stim::sample_batch_measurements<kSimdWidth>(
      circuit, reference_sample, /*num_samples=*/1, rng, /*transposed=*/false);

  std::string bits;
  bits.reserve(n);
  for (std::size_t i = 0; i < n; ++i)
    bits += char('0' + (table[i][0] ? 1 : 0));
  return bits;
}

// The all-zero reference: sample_batch_measurements then returns the frame
// simulator's output untouched.
stim::simd_bits<kSimdWidth> zero_reference(const stim::Circuit &circuit) {
  return stim::simd_bits<kSimdWidth>(circuit.count_measurements());
}

// Drives stim_memory_source round-by-round until every stabilizer-round bit
// the reference circuit has is covered, then one read_data() call for the
// terminal segment. How many next_round() calls that takes varies by family
// (prefix folds differently), so this counts by total bits, not round count.
std::string sample_via_stim_memory_source(const std::string &circuit_text, std::uint64_t seed,
                                          std::size_t total_measurements) {
  stim_memory_source source(circuit_text, seed);
  const std::size_t stabilizer_bits_needed = total_measurements - source.data_width();
  std::string bits;
  while (bits.size() < stabilizer_bits_needed)
    for (auto b : source.next_round())
      bits += char('0' + b);
  for (auto b : source.read_data())
    bits += char('0' + b);
  return bits;
}

/// Every one of Stim's built-in generated-circuit families. Each has a
/// differently-shaped prefix/round/terminal split, which is what makes them
/// worth running one by one.
constexpr std::pair<const char *, const char *> kGeneratedTasks[] = {
    {"repetition_code", "memory"},      {"surface_code", "rotated_memory_x"},
    {"surface_code", "rotated_memory_z"}, {"surface_code", "unrotated_memory_x"},
    {"surface_code", "unrotated_memory_z"}, {"color_code", "memory_xyz"},
};

/// One of Stim's built-in generated circuits, by code family name.
stim::GeneratedCircuit generate(const std::string &code,
                                const stim::CircuitGenParameters &params) {
  if (code == "surface_code")
    return stim::generate_surface_code_circuit(params);
  if (code == "repetition_code")
    return stim::generate_rep_code_circuit(params);
  if (code == "color_code")
    return stim::generate_color_code_circuit(params);
  throw std::invalid_argument("unknown code family: " + code);
}

/// One (distance, rounds) point to generate a circuit at.
struct shape {
  std::uint32_t distance;
  std::uint32_t rounds;
};

/// Distance changes round width; rounds changes REPEAT replay count and how
/// much sits outside it -- both move the prefix/round/terminal split, so
/// both are swept. Rounds start at 3 because stim inlines the body for 1 or
/// 2 (see NoRepeatBlockIsRejected).
constexpr shape kSweep[] = {
    {2, 3},  // the smallest distance stim will generate
    {3, 3}, {3, 5}, {3, 12}, // one distance, several round counts
    {5, 3}, {5, 8},
    {7, 4},  // wide rounds, to catch a width the split gets wrong only when big
};

/// color_code:memory_xyz takes odd distances only, and emits a *second*
/// REPEAT block from 4 rounds up -- which is outside what stim_memory_source
/// documents it accepts (see TwoRepeatBlocksStillProduceTheRightBits).
constexpr shape kColorSweep[] = {{3, 2}, {3, 3}, {5, 2}, {5, 3}, {7, 3}};

void check_task(const std::string &code, const std::string &task,
                shape s) {
  constexpr std::uint64_t kSeed = 424242;
  SCOPED_TRACE("distance=" + std::to_string(s.distance) + " rounds=" +
               std::to_string(s.rounds));

  stim::CircuitGenParameters params(s.rounds, s.distance, task);
  params.after_clifford_depolarization = 0.001;
  params.before_measure_flip_probability = 0.001;

  const stim::GeneratedCircuit gen = generate(code, params);

  const std::string reference = sample_whole_circuit_once(
      gen.circuit, kSeed, zero_reference(gen.circuit));
  const std::string via_source = sample_via_stim_memory_source(
      gen.circuit.str(), kSeed, gen.circuit.count_measurements());

  EXPECT_EQ(via_source, reference) << "mismatch for " << code << ":" << task;
  EXPECT_EQ(via_source.size(), gen.circuit.count_measurements())
      << "round-by-round generation didn't cover every measurement for "
      << code << ":" << task;
}

} // namespace

TEST(StimMemorySourceVsFullCircuit, EveryGeneratedCodeFamilyMatchesOneWholeCircuitRun) {
  // All six of Stim's built-in generated-circuit families, across a range of
  // distances and round counts. color_code:memory_xyz is the sharpest case:
  // its prefix is a bare `R` with zero measurements, which next_round() must
  // still execute rather than skip.
  int cases = 0;
  for (const auto &[code, task] : kGeneratedTasks) {
    SCOPED_TRACE(std::string(code) + ":" + task);
    const bool is_color = std::string(code) == "color_code";
    for (const auto &s : is_color ? std::span<const shape>(kColorSweep)
                                  : std::span<const shape>(kSweep)) {
      check_task(code, task, s);
      ++cases;
    }
  }
  // The two sweeps are deliberately different sizes; assert the total so a
  // future edit cannot quietly shrink the matrix to nothing.
  EXPECT_EQ(cases, 5 * std::size(kSweep) + std::size(kColorSweep));
}

TEST(StimMemorySourceVsFullCircuit, AGeneratedCircuitWithTooFewRoundsHasNoRepeatBlock) {
  // Not a contrived input: stim inlines the round body rather than emitting
  // a REPEAT block whenever a generated circuit asks for 1 or 2 rounds, so
  // the documented precondition is one an ordinary caller can trip. It has
  // to be a constructor error, because there is no round to derive.
  for (std::uint32_t rounds : {1u, 2u}) {
    SCOPED_TRACE(rounds);
    stim::CircuitGenParameters params(rounds, /*distance=*/3, "memory");
    const auto gen = generate("repetition_code", params);
    EXPECT_THROW(stim_memory_source(gen.circuit.str(), /*seed=*/1),
                 std::runtime_error);
  }
}

TEST(StimMemorySourceVsFullCircuit, MoreThanOneRepeatBlockIsRejected) {
  // Only the first REPEAT can be the round; a later one is copied by
  // safe_append() without its block body, so its measurements would
  // disappear (here, 7 in the circuit but only 4 ever produced). The
  // constructor refuses this rather than returning a short shot.
  const std::string text = R"CIRCUIT(
R 0 1
REPEAT 2 {
  X_ERROR(0.1) 0
  M 0
}
REPEAT 3 {
  X_ERROR(0.1) 1
  M 1
}
M 0 1
)CIRCUIT";
  ASSERT_EQ(stim::Circuit(text).count_measurements(), 7u);
  EXPECT_THROW(stim_memory_source(text, /*seed=*/1), std::runtime_error);

  // Reachable without hand-writing anything: stim emits a second REPEAT for
  // color_code:memory_xyz from four rounds up.
  stim::CircuitGenParameters params(/*rounds=*/8, /*distance=*/3, "memory_xyz");
  const auto gen = generate("color_code", params);
  EXPECT_THROW(stim_memory_source(gen.circuit.str(), /*seed=*/1),
               std::runtime_error);
}

TEST(StimMemorySourceVsFullCircuit, AGeneratedMemoryCircuitsReferenceSampleIsAllZero) {
  // Why the check above can compare noise frames as measurement outcomes: a
  // memory circuit's stabilizers have noiseless result 0, so its reference
  // sample is all zero and `frame == outcome` for every bit. If that stopped
  // holding, the test above would silently check something weaker.
  constexpr std::uint32_t kRounds = 3, kDistance = 3;
  for (const auto &[code, task] : kGeneratedTasks) {
    SCOPED_TRACE(std::string(code) + ":" + task);
    stim::CircuitGenParameters params(kRounds, kDistance, task);
    const stim::GeneratedCircuit gen = generate(code, params);
    const auto reference =
        stim::TableauSimulator<kSimdWidth>::reference_sample_circuit(
            gen.circuit);
    EXPECT_EQ(to_bit_string(reference, gen.circuit.count_measurements())
                  .find('1'),
              std::string::npos);
  }
}

TEST(StimMemorySourceVsFullCircuit, TheFrameSourcePlusAReferenceSampleIsStimsMeasurements) {
  // The general statement on a circuit whose noiseless outcome is *not* all
  // zero: stim_memory_source reports the frame relative to that noiseless
  // run, so XORing back stim's own reference sample must reproduce exactly
  // what stim's own sampler returns. That relationship is what makes the
  // output a *syndrome*: a 1 means disagreement with the noiseless trajectory.
  constexpr std::uint64_t kSeed = 424242;
  constexpr int kRounds = 4;
  // stim_memory_source needs a REPEAT count it will never reach; the
  // reference circuit uses exactly the rounds this test samples.
  const char *kBody = "  X_ERROR(0.05) 0 1\n  M 0 1\n";
  const std::string streamed_text =
      std::string("R 0 1\nX 0\nREPEAT 1000000 {\n") + kBody + "}\n";
  const std::string reference_text = std::string("R 0 1\nX 0\nREPEAT ") +
                                     std::to_string(kRounds) + " {\n" + kBody +
                                     "}\n";

  const stim::Circuit full(reference_text);
  const std::size_t n = full.count_measurements();
  ASSERT_EQ(n, static_cast<std::size_t>(kRounds) * 2);

  const auto reference_sample =
      stim::TableauSimulator<kSimdWidth>::reference_sample_circuit(full);
  const std::string reference_bits = to_bit_string(reference_sample, n);
  // Non-trivial in both directions, or the XOR below would prove nothing.
  ASSERT_NE(reference_bits.find('1'), std::string::npos);
  ASSERT_NE(reference_bits.find('0'), std::string::npos);

  const std::string absolute =
      sample_whole_circuit_once(full, kSeed, reference_sample);

  stim_memory_source source(streamed_text, kSeed);
  std::string frame;
  for (int i = 0; i < kRounds; ++i)
    for (auto b : source.next_round())
      frame += char('0' + b);
  ASSERT_EQ(frame.size(), n);
  ASSERT_EQ(absolute.size(), n);

  for (std::size_t i = 0; i < n; ++i)
    EXPECT_EQ(absolute[i] - '0', (frame[i] - '0') ^ (reference_bits[i] - '0'))
        << "measurement " << i;
}

// Narrower regression coverage for the zero-measurement-prefix case: a
// hand-written circuit whose prefix is a bare reset (no measurements)
// followed by a REPEAT body, isolated from any Stim-generated circuit's
// incidental structure. Uses a small fixed REPEAT count (matching the
// rounds actually sampled) so sample_whole_circuit_once stays cheap.
TEST(StimMemorySourceVsFullCircuit, ZeroMeasurementPrefixIsStillExecuted) {
  constexpr std::uint64_t kSeed = 7;
  constexpr int kRounds = 3;
  const std::string streamed_text = R"CIRCUIT(
R 0
REPEAT 1000000 {
  X_ERROR(0.5) 0
  M 0
}
)CIRCUIT";
  const std::string reference_text = R"CIRCUIT(
R 0
REPEAT 3 {
  X_ERROR(0.5) 0
  M 0
}
)CIRCUIT";

  stim::Circuit full(reference_text);
  const std::string reference =
      sample_whole_circuit_once(full, kSeed, zero_reference(full));

  stim_memory_source source(streamed_text, kSeed);
  std::string via_source;
  for (int i = 0; i < kRounds; ++i)
    for (auto b : source.next_round())
      via_source += char('0' + b);

  EXPECT_EQ(via_source, reference);
}

// ─── cudaq_memory_source ────────────────────────────────────────────────────
//
// Validated against direct runs of `memory_circuit` itself (raw ancilla/
// data-qubit bits, not sample_memory_circuit's XOR-combined detectors).
// cudaq_memory_source re-launches the kernel once per round count under one
// seed, relying on shared-prefix RNG draws staying bit-identical; these
// tests confirm that holds against a direct kernel run for every r.

namespace {

cudaq::noise_model make_noise() {
  cudaq::noise_model noise;
  noise.add_all_qubit_channel("x", cudaq::qec::two_qubit_bitflip(0.05),
                               /*num_controls=*/1);
  return noise;
}

// Runs memory_circuit directly (bypassing sample_memory_circuit's
// detector-XOR step) and returns the raw measurement row: numRounds chunks
// of (numAncx + numAncz) ancilla bits, then numData data-qubit bits -- the
// same layout `cudaq_memory_source` reads internally.
std::vector<std::uint8_t> raw_measurements(const cudaq::qec::code &code,
                                            cudaq::qec::operation prep_op,
                                            std::size_t numRounds,
                                            cudaq::noise_model &noise,
                                            std::uint64_t seed) {
  auto &prep =
      code.get_operation<cudaq::qec::code::one_qubit_encoding>(prep_op);
  auto &stabRound = code.get_operation<cudaq::qec::code::stabilizer_round>(
      cudaq::qec::operation::stabilizer_round);
  const bool is_z_prep = prep_op == cudaq::qec::operation::prep0 ||
                          prep_op == cudaq::qec::operation::prep1;

  auto sched_x = code.get_stabilizer_schedule_x();
  auto sched_z = code.get_stabilizer_schedule_z();
  std::vector<std::size_t> xVec(sched_x.data(), sched_x.data() + sched_x.size());
  std::vector<std::size_t> zVec(sched_z.data(), sched_z.data() + sched_z.size());
  auto logical_obs =
      is_z_prep ? code.get_observables_z() : code.get_observables_x();
  const std::size_t num_obs = logical_obs.shape()[0];
  std::vector<std::size_t> obs_flat(logical_obs.data(),
                                     logical_obs.data() + logical_obs.size());

  const std::size_t numData = code.get_num_data_qubits();
  const std::size_t numAncx = code.get_num_ancilla_x_qubits();
  const std::size_t numAncz = code.get_num_ancilla_z_qubits();

  cudaq::set_random_seed(static_cast<std::size_t>(seed));
  cudaq::sample_options opts{
      .shots = 1, .noise = noise, .explicit_measurements = true};
  auto result =
      cudaq::sample(opts, cudaq::qec::memory_circuit, stabRound, prep, numData,
                    numAncx, numAncz, numRounds, xVec, zVec, obs_flat, num_obs,
                    !is_z_prep);
  cudaqx::tensor<std::uint8_t> mzTable(result.sequential_data());

  const std::size_t width = mzTable.shape()[1];
  std::vector<std::uint8_t> out(width);
  for (std::size_t i = 0; i < width; ++i)
    out[i] = mzTable.at({0, i});
  return out;
}

} // namespace

// For every 1 <= r <= max_rounds, the source's next_round() output for
// round r must equal round r's raw ancilla bits from an independent, direct
// kernel run with numRounds=r at the same seed, and its cached data bits
// for round r must equal that same run's raw data-qubit measurements.
TEST(CudaqMemorySource, MatchesDirectKernelRunPerRound) {
  constexpr std::uint64_t kSeed = 99;
  constexpr std::size_t kMaxRounds = 5;

  auto code = cudaq::qec::get_code("repetition",
                                    cudaqx::heterogeneous_map{{"distance", 3}});
  auto noise = make_noise();

  for (auto prep_op : {cudaq::qec::operation::prep0, cudaq::qec::operation::prep1}) {
    SCOPED_TRACE(prep_op == cudaq::qec::operation::prep0 ? "prep0" : "prep1");
    for (std::size_t r = 1; r <= kMaxRounds; ++r) {
      auto reference = raw_measurements(*code, prep_op, r, noise, kSeed);

      cudaq_memory_source local(*code, prep_op, r, noise, kSeed);
      ASSERT_EQ(local.round_width() * r + local.data_width(), reference.size());

      std::vector<std::uint8_t> streamed;
      for (std::size_t k = 0; k < r; ++k) {
        auto bits = local.next_round();
        ASSERT_EQ(bits.size(), local.round_width())
            << "round " << k + 1 << " of " << r;
        streamed.insert(streamed.end(), bits.begin(), bits.end());
      }
      const std::size_t ancilla_width = r * local.round_width();
      for (std::size_t i = 0; i < ancilla_width; ++i)
        EXPECT_EQ(streamed[i], reference[i])
            << "ancilla bit mismatch at r=" << r << ", index " << i;

      auto data_bits = local.read_data();
      ASSERT_EQ(data_bits.size(), local.data_width());
      for (std::size_t i = 0; i < data_bits.size(); ++i)
        EXPECT_EQ(data_bits[i], reference[ancilla_width + i])
            << "data bit mismatch at r=" << r << ", index " << i;
    }
  }
}

// Exercises the streaming interface end-to-end: a variable number of
// next_round() calls then one read_data() call, matching stim_memory_source's
// shape. Confirms calling read_data() early (a shorter shot) reproduces the
// same-seed direct kernel run for that shorter round count.
TEST(CudaqMemorySource, ReadDataAfterVariableRoundCountMatchesDirectCall) {
  constexpr std::uint64_t kSeed = 7;
  constexpr std::size_t kMaxRounds = 6;

  auto code = cudaq::qec::get_code("repetition",
                                    cudaqx::heterogeneous_map{{"distance", 3}});
  auto noise = make_noise();

  for (std::size_t k = 1; k <= kMaxRounds; ++k) {
    cudaq_memory_source source(*code, cudaq::qec::operation::prep0,
                                kMaxRounds, noise, kSeed);
    for (std::size_t i = 0; i < k; ++i)
      ASSERT_FALSE(source.next_round().empty());
    auto data = source.read_data();

    auto reference =
        raw_measurements(*code, cudaq::qec::operation::prep0, k, noise, kSeed);
    const std::size_t ancilla_width = k * source.round_width();
    ASSERT_EQ(data.size(), reference.size() - ancilla_width);
    for (std::size_t i = 0; i < data.size(); ++i)
      EXPECT_EQ(data[i], reference[ancilla_width + i]) << "k=" << k;
  }
}

TEST(CudaqMemorySource, ARoundBudgetIsEnforcedAndDataNeedsARoundFirst) {
  auto code = cudaq::qec::get_code("repetition",
                                    cudaqx::heterogeneous_map{{"distance", 3}});
  auto noise = make_noise();
  cudaq_memory_source source(*code, cudaq::qec::operation::prep0, 3, noise, 1);

  // The terminal readout only means anything after at least one round.
  EXPECT_THROW(source.read_data(), std::runtime_error);

  // Bounded, unlike stim_memory_source: it cannot serve a `stream ...
  // until=` that outlasts max_rounds, so is_streamed() must say so.
  EXPECT_FALSE(source.is_streamed());

  EXPECT_EQ(source.max_rounds(), 3u);
  for (int i = 0; i < 3; ++i)
    EXPECT_FALSE(source.next_round().empty()) << "round " << i;
  // Bounded, unlike stim_memory_source: past the budget it stays empty.
  EXPECT_TRUE(source.next_round().empty());
  EXPECT_TRUE(source.next_round().empty());
}

// reset() starts a fresh shot (new seed generation): re-running the same
// number of rounds afterward should reproduce a direct kernel run at the
// *bumped* seed, not the original one.
TEST(CudaqMemorySource, ResetAdvancesToANewSeedGeneration) {
  constexpr std::uint64_t kSeed = 55;
  auto code = cudaq::qec::get_code("repetition",
                                    cudaqx::heterogeneous_map{{"distance", 3}});
  auto noise = make_noise();
  cudaq_memory_source source(*code, cudaq::qec::operation::prep0, 3, noise,
                              kSeed);
  source.reset();

  std::vector<std::uint8_t> streamed;
  for (int i = 0; i < 3; ++i) {
    auto bits = source.next_round();
    streamed.insert(streamed.end(), bits.begin(), bits.end());
  }

  auto reference = raw_measurements(*code, cudaq::qec::operation::prep0, 3,
                                     noise, kSeed + 1);
  ASSERT_LE(streamed.size(), reference.size());
  for (std::size_t i = 0; i < streamed.size(); ++i)
    EXPECT_EQ(streamed[i], reference[i]);
}
