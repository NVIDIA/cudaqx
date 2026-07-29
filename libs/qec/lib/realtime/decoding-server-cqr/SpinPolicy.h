/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

/// Bounded spin-then-block policy for the decoding server's cross-thread
/// waits (session work queue, blocking-RPC response).
///
/// A sleeping waiter costs a futex wake on the producer plus a scheduler
/// switch-in on the consumer (~3-4 us on Grace-class hosts, with deep-idle
/// outliers near 100 us).  Spinning on an atomic for a bounded window before
/// blocking removes that cost whenever the producer answers within the
/// budget (~0.3-0.5 us per handoff), while an idle thread still parks after
/// one budget window.  The blocking primitive stays armed on every path, so
/// semantics are identical to pure blocking; the spin only skips the sleep.
///
/// Budget selection (read once, at first use):
///   QEC_DECODING_SERVER_SPIN_US   unset/"" -> kDefaultSpinBudgetUs;
///                                 0  -> always block (no spin);
///                                 N  -> spin N us, then block;
///                                 -1 -> spin forever, never block -- the
///                                       dedicated-core realtime posture,
///                                       best paired with QEC_PIN_* pinning.

#include <chrono>
#include <cstdint>
#include <cstdlib>

namespace cudaq::qec::decoding_server {

/// Default spin budget (µs): long enough to cover any realistic decode
/// cadence -- request gaps within a shot are well under this even on the
/// software udp wire -- while an idle thread burns at most one such window
/// per idle transition before sleeping.
inline constexpr long kDefaultSpinBudgetUs = 200;

/// Effective spin budget in nanoseconds: -1 = infinite, 0 = disabled.
inline int64_t spin_budget_ns() {
  static const int64_t v = [] {
    const char *e = std::getenv("QEC_DECODING_SERVER_SPIN_US");
    const long long us = (e && e[0]) ? std::atoll(e) : kDefaultSpinBudgetUs;
    return us < 0 ? int64_t{-1} : static_cast<int64_t>(us) * 1000;
  }();
  return v;
}

inline void cpu_relax() {
#if defined(__aarch64__)
  asm volatile("yield" ::: "memory");
#elif defined(__x86_64__)
  asm volatile("pause" ::: "memory");
#endif
}

/// Spin until \p pred holds or the budget expires; returns pred's last value
/// so callers fall through to their blocking wait on false.  The caller's
/// blocking primitive must still be armed by every completer -- the spin is
/// an optimization of the wait, never a replacement for the wakeup.
template <typename Pred>
inline bool spin_until(Pred pred) {
  const int64_t budget = spin_budget_ns();
  if (budget == 0)
    return pred();
  const auto now_ns = [] {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
  };
  const int64_t deadline = budget < 0 ? 0 : now_ns() + budget;
  while (!pred()) {
    if (budget > 0 && now_ns() >= deadline)
      return false;
    cpu_relax();
  }
  return true;
}

} // namespace cudaq::qec::decoding_server
