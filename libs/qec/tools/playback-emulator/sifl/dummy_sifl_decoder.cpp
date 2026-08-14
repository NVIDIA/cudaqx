/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file dummy_sifl_decoder.cpp
/// @brief dummy_sifl_decoder — async-sleep decoder for SIFL runaway demos.
///
/// State machine for enqueue_syndrome():
///
///   IDLE → first complete shot arrives (bits_since_decode_ >= bits_per_shot):
///            launch background thread sleeping
///              bits_since_decode_ × us_per_bit µs,
///            return false (decode in flight, not done yet).
///
///   RUNNING → background still sleeping: accumulate bits, return false.
///
///   DONE → background finished (done_ flag set): next enqueue_syndrome call
///            joins the thread, starts the NEXT decode proportional to ALL
///            bits accumulated since the last completion, returns TRUE so
///            DecodingSession transitions to result_ready.
///
/// The trigger (IDLE → RUNNING) fires at the data-qubit readout boundary:
/// after receiving  bits_per_shot  bits the decoder knows it has a complete
/// shot and launches its sleep.  If the sleep takes longer than one RTT, more
/// syndrome rounds arrive before the sleep ends — the SIFL feedback loop.
///
/// bits_per_shot = init_rounds × syndrome_width + data_width
///   e.g. d=3 rep code, r=2: 2×2+3 = 7
///
/// To prevent DecodingSession::on_enqueue from overflowing its counter, supply
///   H_sparse: [999999, -1]  (block_size = 1 000 000)
///   D_sparse: [999999, -1]  (num_msyn_per_decode = 1 000 000)

#include "cudaq/qec/decoder.h"
#include "cudaq/qec/decoder_config_schema.h"

#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

namespace cudaq::qec {

class dummy_sifl_decoder : public decoder {
public:
  dummy_sifl_decoder(const cudaq::qec::sparse_binary_matrix &H,
                     const cudaqx::heterogeneous_map &params)
      : decoder(H) {
    us_per_bit_   = params.get<double>("us_per_bit", 1.0);
    bits_per_shot_ = params.get<std::uint64_t>("bits_per_shot", 0u);
    const auto n_obs = params.get<std::uint64_t>("num_obs", 1u);
    corrections_.assign(n_obs, 0);
    set_O_sparse(std::vector<std::vector<uint32_t>>(n_obs));
  }

  ~dummy_sifl_decoder() override {
    if (thread_.joinable()) thread_.join();
  }

  bool enqueue_syndrome(const uint8_t *, std::size_t len) override {
    // ── DONE path: previous decode finished ──────────────────────────────
    // The first enqueue that arrives after the background sleep ends starts
    // the NEXT decode (proportional to all bits accumulated) and signals the
    // session that this decode's result is ready.
    if (done_.load(std::memory_order_acquire)) {
      if (thread_.joinable()) thread_.join();
      done_.store(false, std::memory_order_relaxed);
      const auto sleep_bits = bits_since_decode_;
      bits_since_decode_ = len;
      shot_bits_ = len;
      launch_decode(sleep_bits);
      return true; // → DecodingSession sets shot_state = result_ready
    }

    // ── Accumulate ────────────────────────────────────────────────────────
    bits_since_decode_ += len;
    shot_bits_ += len;

    // ── IDLE → RUNNING: first complete shot ───────────────────────────────
    // When bits_per_shot_ == 0 the old behaviour applies: fire on the first
    // enqueue (kept for backward compatibility).
    if (!running_) {
      const bool shot_done = (bits_per_shot_ == 0) ||
                             (shot_bits_ >= bits_per_shot_);
      if (shot_done) {
        shot_bits_ = 0;
        launch_decode(bits_since_decode_);
      }
    }
    return false;
  }

  void reset_decoder() override {
    if (thread_.joinable()) thread_.join();
    running_ = false;
    done_.store(false, std::memory_order_relaxed);
    bits_since_decode_ = 0;
    shot_bits_ = 0;
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
      dummy_sifl_decoder,
      static std::unique_ptr<decoder>
      create(const cudaq::qec::decoder_init &init,
             const cudaqx::heterogeneous_map &params) {
        return cudaq::qec::make_pcm_decoder<dummy_sifl_decoder>(init, params);
      })

private:
  void launch_decode(std::uint64_t sleep_bits) {
    running_ = true;
    thread_ = std::thread([this, sleep_bits] {
      std::this_thread::sleep_for(std::chrono::microseconds(
          static_cast<std::uint64_t>(sleep_bits * us_per_bit_ + 0.5)));
      done_.store(true, std::memory_order_release);
    });
  }

  double               us_per_bit_;
  std::uint64_t        bits_per_shot_{0};  // 0 = fire on first enqueue
  std::vector<uint8_t> corrections_;
  bool                 running_{false};
  std::atomic<bool>    done_{false};
  std::thread          thread_;
  std::uint64_t        bits_since_decode_{0};
  std::uint64_t        shot_bits_{0};      // resets when shot boundary fires
};

CUDAQ_EXT_PT_REGISTER_TYPE(dummy_sifl_decoder)

namespace {
struct schema_reg {
  schema_reg() {
    using k = cudaq::qec::decoding::config::param_kind;
    cudaq::qec::decoding::config::register_decoder_schema(
        {"dummy_sifl_decoder",
         {{"us_per_bit", k::f64},
          {"num_obs", k::uint64},
          {"bits_per_shot", k::uint64}}});
  }
} reg;
} // namespace

} // namespace cudaq::qec
