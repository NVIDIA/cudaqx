/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

/// @file cancellation.h
/// @brief `cudaq::qec::cancellation_token` / `cudaq::qec::cancellation_source`:
/// a small, header-only, cooperative-cancellation pair.
///
/// A `cancellation_source` owns a shared cancellation state; any number of
/// `cancellation_token` copies observe it. A decoder that supports cooperative
/// cancellation receives a token (see `decoder::decode(syndrome,
/// cancellation_token)` in `decoder.h`), polls `stop_requested()`, and returns
/// `std::nullopt` once it answers true. That is the whole contract on the
/// decoder side.
///
/// Which stop level a decoder reacts to is chosen by whoever hands it the
/// token, not by the decoder: an unmasked token reports both soft and hard
/// stops, while a `hard_only()` view reports hard stops only. A soft stop thus
/// means "stop the nodes handed an unmasked token"; a hard stop means "stop
/// everything".
///
/// Levels only ever escalate (`none -> soft -> hard`); a hard request after a
/// soft one upgrades it, a soft request after a hard one is a no-op.

#include <atomic>
#include <cstdint>
#include <memory>

namespace cudaq::qec {

/// @brief Requested stop level, ordered by severity.
enum class cancellation_level : std::uint8_t {
  none = 0, ///< No stop requested.
  soft = 1, ///< Stop the nodes that were handed an unmasked token.
  hard = 2  ///< Stop every node, including `hard_only()` views.
};

class cancellation_source;

/// @brief Read-only view of a `cancellation_source`'s shared cancellation
/// state.
class cancellation_token {
public:
  cancellation_token() noexcept = default;

  /// @brief False if the token was default-constructed
  bool stop_possible() const noexcept { return static_cast<bool>(state_); }

  /// @brief A view on the same stop state whose `stop_requested()` only
  /// answers true for a stop of at least `lvl`.
  cancellation_token at_least(cancellation_level lvl) const noexcept {
    cancellation_token masked = *this;
    if (lvl > masked.min_level_)
      masked.min_level_ = lvl;
    return masked;
  }

  /// @brief `at_least(cancellation_level::hard)`: a view whose
  /// `stop_requested()` ignores every level below a hard stop.
  cancellation_token hard_only() const noexcept {
    return at_least(cancellation_level::hard);
  }

  /// @brief The lowest level `stop_requested()` answers to; `none` for an
  /// unmasked token.
  cancellation_level min_level() const noexcept { return min_level_; }

  /// @brief True if a stop of at least `min_level()` has been requested
  bool stop_requested() const noexcept {
    cancellation_level lvl = level();
    return lvl != cancellation_level::none && lvl >= min_level_;
  }

private:
  friend class cancellation_source;
  using state_type = std::atomic<cancellation_level>;

  // The raw level is deliberately not exposed: a decoder only ever asks
  // `stop_requested()`, and the caller picks the level via the mask.
  cancellation_level level() const noexcept {
    if (!state_)
      return cancellation_level::none;
    return state_->load(std::memory_order_acquire);
  }

  explicit cancellation_token(std::shared_ptr<state_type> state) noexcept
      : state_(std::move(state)) {}

  std::shared_ptr<state_type> state_;
  cancellation_level min_level_ = cancellation_level::none;
};

/// @brief Owns the cancellation state observed by the `cancellation_token`s
/// obtained from it. The state outlives the source for as long as any token
/// referencing it exists
class cancellation_source {
public:
  cancellation_source()
      : state_(std::make_shared<cancellation_token::state_type>(
            cancellation_level::none)) {}

  /// @brief Request at least `lvl`. Never lowers the current level.
  /// @returns true if this call raised the level, otherwise false.
  bool request(cancellation_level lvl) noexcept {
    auto cur = state_->load(std::memory_order_relaxed);
    while (cur < lvl) {
      if (state_->compare_exchange_weak(cur, lvl, std::memory_order_acq_rel,
                                        std::memory_order_relaxed))
        return true;
    }
    return false;
  }

  /// @brief Request a soft stop (see `cancellation_level::soft`).
  bool request_soft_stop() noexcept {
    return request(cancellation_level::soft);
  }

  /// @brief Request a hard stop (see `cancellation_level::hard`).
  bool request_hard_stop() noexcept {
    return request(cancellation_level::hard);
  }

  cancellation_level level() const noexcept {
    return state_->load(std::memory_order_acquire);
  }

  cancellation_token get_token() const noexcept {
    return cancellation_token(state_);
  }

private:
  std::shared_ptr<cancellation_token::state_type> state_;
};

} // namespace cudaq::qec
