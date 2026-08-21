/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file emulator.cpp
/// @brief plan() and run(): pre-serialize every static frame before t0,
/// validate frame sizes before t0, then run ONE timing thread that does
/// nothing between its deadlines but wait and dispatch. 

#include "emulator.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <ctime>
#include <functional>
#include <mutex>
#include <optional>
#include <sstream>
#include <thread>
#include <unordered_map>

namespace cudaq::qec::playback {

using cudaq::realtime::RPC_MAGIC_REQUEST;
using cudaq::realtime::RPCHeader;
namespace wire = cudaq::qec::decoding::rpc;

namespace {

// -- Timing core: sleep off the bulk of a wait, then spin
// the last `kSpinSlackNs` to the clock-read floor.

std::uint64_t now_ns() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return static_cast<std::uint64_t>(ts.tv_sec) * 1'000'000'000ull +
         static_cast<std::uint64_t>(ts.tv_nsec);
}

constexpr std::uint64_t kMaxNapNs = 1'000'000; // cap on one nanosleep step

void sleep_until(std::uint64_t target_ns) {
  struct timespec ts;
  ts.tv_sec = static_cast<time_t>(target_ns / 1'000'000'000ull);
  ts.tv_nsec = static_cast<long>(target_ns % 1'000'000'000ull);
  clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, nullptr);
}

/// How much of a wait `wait_until` spins rather than sleeps: a little above
/// typical clock_nanosleep overshoot. Fixed rather than measured or
/// configurable -- a calibration samples whatever the host happens to be
/// doing at startup, and on a loaded or virtualized box one bad preemption
/// pushes it into the milliseconds, which then gets spent spinning at every
/// deadline for the rest of the run.
constexpr std::uint64_t kSpinSlackNs = 50'000;

/// Saturating add. A deadline far enough out to overflow the clock is
/// clamped to "never" 
inline std::uint64_t add_sat(std::uint64_t a, std::uint64_t b) {
  std::uint64_t sum = 0;
  return __builtin_add_overflow(a, b, &sum) ? ~std::uint64_t(0) : sum;
}

/// Block until CLOCK_MONOTONIC reaches `deadline_ns` (an absolute
/// CLOCK_MONOTONIC timestamp). Sleeps in bounded naps down to
/// `deadline_ns - kSpinSlackNs`, then spins a plain clock read to the deadline
void wait_until(std::uint64_t deadline_ns) {
  const std::uint64_t spin_from =
      deadline_ns > kSpinSlackNs ? deadline_ns - kSpinSlackNs : 0;
  for (std::uint64_t t = now_ns(); t < spin_from; t = now_ns())
    sleep_until(t + std::min(spin_from - t, kMaxNapNs));
  while (now_ns() < deadline_ns) {
  }
}

// -- Frame construction / parsing helpers. --

void append_bytes(std::vector<std::uint8_t> &buf, const void *p, std::size_t n) {
  const std::size_t off = buf.size();
  buf.resize(off + n);
  std::memcpy(buf.data() + off, p, n);
}

/// Pack one-byte-per-bit (0x00/0x01) values from `bits` into LSB-first bytes,
/// appended to `buf`.
void append_packed_bits(std::vector<std::uint8_t> &buf, const std::uint8_t *bits,
                        std::size_t n) {
  const std::size_t off = buf.size();
  buf.resize(off + wire::bit_packed_bytes(n), 0);
  for (std::size_t i = 0; i < n; ++i)
    if (bits[i])
      buf[off + i / 8] |= static_cast<std::uint8_t>(1u << (i % 8));
}

/// Unpack LSB-first bit-packed bytes into one-byte-per-bit values, appended
/// to `out`.
void append_unpacked_bits(std::vector<std::uint8_t> &out, const std::uint8_t *packed,
                          std::size_t n_bits) {
  for (std::size_t i = 0; i < n_bits; ++i)
    out.push_back((packed[i / 8] >> (i % 8)) & 1u);
}

/// Build one wire frame: RPCHeader + a fixed payload struct, plus optional
/// trailing bit-packed syndrome bytes. The single point
/// where every operation's frame shape is assembled.
std::vector<std::uint8_t> build_frame(std::uint32_t function_id, std::uint32_t request_id,
                                      const void *payload, std::size_t payload_len,
                                      const std::uint8_t *bits = nullptr,
                                      std::size_t n_bits = 0) {
  const std::size_t trailing = bits ? wire::bit_packed_bytes(n_bits) : 0;
  RPCHeader h{};
  h.magic = RPC_MAGIC_REQUEST;
  h.function_id = function_id;
  h.arg_len = static_cast<std::uint32_t>(payload_len + trailing);
  h.request_id = request_id;
  h.ptp_timestamp = 0;

  std::vector<std::uint8_t> buf;
  append_bytes(buf, &h, sizeof(h));
  append_bytes(buf, payload, payload_len);
  if (bits)
    append_packed_bits(buf, bits, n_bits);
  return buf;
}

/// Overwrites a frame's request_id in place, after it was built with a
/// placeholder.
void set_request_id(std::uint8_t *frame_bytes, std::uint32_t rid) {
  reinterpret_cast<RPCHeader *>(frame_bytes)->request_id = rid;
}

std::vector<std::uint8_t> build_reset_frame(std::uint64_t decoder_id, std::uint32_t rid) {
  wire::ResetRequestPayload p{static_cast<std::int64_t>(decoder_id)};
  return build_frame(wire::kResetDecoderFunctionId, rid, &p, sizeof(p));
}

std::vector<std::uint8_t> build_enqueue_frame(std::uint64_t decoder_id, std::uint32_t rid,
                                              const std::uint8_t *bits, std::size_t n_bits) {
  wire::EnqueueRequestPayload p{static_cast<std::int64_t>(decoder_id),
                                /*counter=*/0,
                                /*syndrome_mapping_id=*/0,
                                static_cast<std::int64_t>(n_bits)};
  return build_frame(wire::kEnqueueSyndromesFunctionId, rid, &p, sizeof(p), bits, n_bits);
}

std::vector<std::uint8_t> build_get_corrections_frame(std::uint64_t decoder_id,
                                                       std::int64_t return_size, bool reset,
                                                       std::uint32_t rid) {
  wire::GetCorrectionsRequestPayload p{static_cast<std::int64_t>(decoder_id), return_size,
                                       static_cast<std::uint8_t>(reset ? 1 : 0)};
  return build_frame(wire::kGetCorrectionsFunctionId, rid, &p, sizeof(p));
}

/// The correction width a get_corrections requests.
/// Explicit return_size=N takes precedence; falls back to the expected-bits
/// width; 0 (request nothing) when neither is given.
std::uint32_t return_size_for(const event &e) {
  return std::max(e.return_size, e.expected_count);
}

/// A session that returns OK but hands back fewer bytes than `return_size`
/// bits requires is indistinguishable from "the correction happens to be
/// all-zero" once those bytes are naively unpacked from a zero-initialized
/// buffer 
RpcStatus reject_truncated_reply(RpcStatus status, std::size_t reply_len,
                                 std::uint32_t return_size) {
  if (status == RpcStatus::OK && reply_len < wire::bit_packed_bytes(return_size))
    return RpcStatus::INTERNAL_ERROR;
  return status;
}

bool mismatches_expected(const schedule &sched, const event &e, const std::uint8_t *bits,
                         std::size_t n) {
  if (e.expected_count == 0)
    return false; // nothing to compare against
  return n != e.expected_count ||
        !std::equal(bits, bits + n, sched.expected_arena.begin() + e.expected_offset);
}

/// Appends one line to `result.warnings`. 
void warn(run_result &result, std::mutex &logs_mu, const std::string &message) {
  std::lock_guard<std::mutex> lock(logs_mu);
  result.warnings.push_back(message);
}

/// Describes an event the way a warning should name it.
std::string event_label(std::size_t event_index, const char *op_name,
                        std::uint64_t decoder_id) {
  return "event " + std::to_string(event_index) + " (" + op_name +
         ", decoder_id=" + std::to_string(decoder_id) + ")";
}

/// Records the warning and flips `aborted` so the timing thread stops
/// dispatching further events.
void abort_on_hard_error(RpcStatus status, std::size_t event_index, const char *op_name,
                         std::uint64_t decoder_id, run_result &result,
                         std::atomic<bool> &aborted, std::mutex &logs_mu) {
  if (status == RpcStatus::OK || status == RpcStatus::NOT_READY)
    return;
  aborted.store(true, std::memory_order_relaxed);
  warn(result, logs_mu,
       event_label(event_index, op_name, decoder_id) + " returned status " +
           std::to_string(static_cast<int>(status)) + "; aborting the run");
}

/// Reject a schedule whose `until=` can never come up before it is asked for
void check_signal_order(const schedule &sched) {
  std::vector<bool> raised(sched.signal_names.size(), false);
  for (std::size_t i = 0; i < sched.events.size(); ++i) {
    const auto &e = sched.events[i];
    if (e.op == operation::stream && e.until_signal_id != kNoSignal &&
        !raised[e.until_signal_id])
      throw std::invalid_argument(
          "event " + std::to_string(i) + " streams until signal '" +
          sched.signal_names[e.until_signal_id] +
          "', which no earlier 'signal=' event raises");
    if (e.signal_id != kNoSignal)
      raised[e.signal_id] = true;
  }
}

} // namespace

std::shared_ptr<run_plan> plan(const schedule &sched_in, const std::unordered_map<std::uint64_t, session *> &router,
             const std::unordered_map<std::uint32_t, syndrome_source *> &sources,
             const run_params &params) {
  auto impl = std::make_shared<run_plan>();
  impl->sched = sched_in;
  impl->router = router;
  impl->sources = sources;
  impl->params = params;
  auto &sched = impl->sched;

  // -- Capability validation
  for (auto &e : sched.events) {
    if (!router.contains(e.decoder_id))
      throw std::invalid_argument("no session routes decoder_id=" +
                                  std::to_string(e.decoder_id));

    // Literal `source=0b<bits>` events carry their own round and name no
    // source, so only the ones that do have anything to look up.
    if (e.source_id != kNoSource &&
        (e.op == operation::stream || e.op == operation::enqueue_data) &&
        !sources.contains(e.source_id))
      throw std::invalid_argument("no syndrome_source registered for source_id=" +
                                  std::to_string(e.source_id));
  }

  // -- Draw and serialize everything whose bytes are known before t0, in
  // file order, so a source is consumed in exactly the order the schedule
  // sends it. A source stops being pre-drawable the moment an event on it
  // decides its own round count at run time (a `stream ... until=`): every
  // later event on that source has to draw at dispatch time too, or it would
  // take rounds belonging to the stream in front of it.
  impl->event_plans.resize(sched.events.size());
  std::unordered_map<std::uint32_t, bool> predrawable;

  for (std::size_t i = 0; i < sched.events.size(); ++i) {
    auto &e = sched.events[i];
    auto &ep = impl->event_plans[i];
    const auto max_frame_bytes = router.at(e.decoder_id)->max_frame_bytes;

    auto place = [&](std::vector<std::uint8_t> bytes, std::uint32_t bits_offset,
                     std::uint32_t bits_count) {
      if (max_frame_bytes != 0 && bytes.size() > max_frame_bytes)
        throw std::invalid_argument(
            "event " + std::to_string(i) + " (decoder_id=" +
            std::to_string(e.decoder_id) + ") builds a " +
            std::to_string(bytes.size()) +
            "-byte frame, exceeding the session's max_frame_bytes=" +
            std::to_string(max_frame_bytes));
      round_plan rp;
      rp.frame_offset = static_cast<std::uint32_t>(impl->frame_arena.size());
      rp.frame_len = static_cast<std::uint32_t>(bytes.size());
      rp.bits_offset = bits_offset;
      rp.bits_count = bits_count;
      impl->frame_arena.insert(impl->frame_arena.end(), bytes.begin(), bytes.end());
      ep.rounds.push_back(rp);
    };

    switch (e.op) {
    case operation::reset:
      place(build_reset_frame(e.decoder_id, /*rid=*/0), 0, 0);
      break;
    case operation::get_corrections:
      place(build_get_corrections_frame(e.decoder_id, return_size_for(e),
                                        /*reset=*/true, /*rid=*/0),
            0, 0);
      break;
    case operation::stream:
    case operation::enqueue_data: {
      if (e.source_id == kNoSource) {
        // Literal `source=0b...`: the same round, repeated. The parser has
        // already pinned the count, so every frame is known now.
        for (std::uint32_t r = 0; r < e.stream_min_rounds; ++r)
          place(build_enqueue_frame(e.decoder_id, /*rid=*/0,
                                    sched.syndrome_arena.data() + e.syndrome_offset,
                                    e.syndrome_count),
                e.syndrome_offset, e.syndrome_count);
        break;
      }

      auto &src = *sources.at(e.source_id);
      auto &ok = predrawable.try_emplace(e.source_id, true).first->second;
      // `enqueue_data` is always exactly one readout; a stream's count is
      // known only when nothing at run time can shorten it.
      const bool fixed =
          e.op == operation::enqueue_data || e.until_signal_id == kNoSignal;
      if (!fixed || src.is_streamed() || !ok) {
        ok = false; // and so is everything after it on this source
        break;
      }

      const std::uint32_t want =
          e.op == operation::enqueue_data ? 1u : e.stream_min_rounds;
      for (std::uint32_t r = 0; r < want; ++r) {
        auto round = e.op == operation::enqueue_data ? src.read_data()
                                                     : src.next_round();
        if (round.empty())
          break; // dry: dispatch reports SOURCE_EXHAUSTED for what is missing
        const auto off = static_cast<std::uint32_t>(sched.syndrome_arena.size());
        const auto len = static_cast<std::uint32_t>(round.size());
        sched.syndrome_arena.insert(sched.syndrome_arena.end(), round.begin(),
                                    round.end());
        if (r == 0) {
          e.syndrome_offset = off;
          e.syndrome_count = len;
        }
        place(build_enqueue_frame(e.decoder_id, /*rid=*/0,
                                  sched.syndrome_arena.data() + off, len),
              off, len);
      }
      break;
    }
    }
  }

  // -- One session per decoder_id, unconditionally. 
  {
    std::unordered_map<session *, std::uint64_t> owner;
    for (const auto &e : sched.events) {
      auto [it, inserted] = owner.emplace(router.at(e.decoder_id), e.decoder_id);
      if (!inserted && it->second != e.decoder_id)
        throw std::invalid_argument(
            "decoder_id=" + std::to_string(e.decoder_id) + " and decoder_id=" +
            std::to_string(it->second) +
            " share one session instance -- each decoder_id must have its "
            "own");
    }
  }

  check_signal_order(sched);

  return impl;
}

namespace {

/// Collects the answers to one decoder's `get_corrections ... signal=`
/// reads, off the timing thread that issued them. One thread per session,
/// because a decoder answers its own requests in the order they arrived, so
/// awaiting them in submission order never waits on the wrong one.
class reader_thread {
public:
  struct pending {
    std::uint32_t request_id = 0;
    std::uint32_t event_index = 0;
    std::uint32_t return_size = 0;
  };

  /// `collect` fills in the record and raises the signal; it runs on this
  /// thread, one reply at a time. It may only ever `await` on the session it
  /// is handed 
  reader_thread(session &s, std::function<void(session &, const pending &)> collect)
      : session_(s), collect_(std::move(collect)), thread_([this] { loop(); }) {}

  ~reader_thread() { close(); }

  void submit(pending p) {
    {
      std::lock_guard<std::mutex> lock(mu_);
      queue_.push_back(p);
    }
    cv_.notify_one();
  }

  /// Drain what is outstanding and stop. Called before run() returns, so
  /// every record a reader owns is complete by the time anyone reads it.
  void close() {
    {
      std::lock_guard<std::mutex> lock(mu_);
      if (closed_)
        return;
      closed_ = true;
    }
    cv_.notify_all();
    if (thread_.joinable())
      thread_.join();
  }

private:
  void loop() {
    for (;;) {
      pending p;
      {
        std::unique_lock<std::mutex> lock(mu_);
        cv_.wait(lock, [&] { return closed_ || !queue_.empty(); });
        if (queue_.empty())
          return;
        p = queue_.front();
        queue_.pop_front();
      }
      collect_(session_, p);
    }
  }

  session &session_;
  std::function<void(session &, const pending &)> collect_;
  std::mutex mu_;
  std::condition_variable cv_;
  std::deque<pending> queue_;
  bool closed_ = false;
  std::thread thread_; // last, so loop() only ever sees initialized members
};

/// Everything the timing thread and the reader threads share, in one place
/// Built on run()'s stack and fully populated before any reader starts, so `readers`
/// is never inserted into again and concurrent lookups are safe.
struct run_state {
  // One global request_id space
  std::atomic<std::uint32_t> next_request_id{1};
  std::atomic<bool> aborted{false};
  /// Guards the shared syndrome/correction logs, which the timing thread and
  /// any reader thread both append to. 
  std::mutex logs_mu;

  /// One flag per signal name in the schedule, raised by a reader thread
  /// when a `signal=` read's answer lands and read at a stream's round
  /// boundaries. Plain atomics suffice -- nothing ever needs to sleep until
  /// a raise, because a waiting stream always has rounds to send meanwhile.
  std::vector<std::unique_ptr<std::atomic<bool>>> signals;

  /// One reader per decoder that has at least one `signal=` read.
  std::unordered_map<std::uint64_t, std::unique_ptr<reader_thread>> readers;

  reader_thread &reader(std::uint64_t decoder_id) {
    return *readers.at(decoder_id);
  }

  bool signal_raised(std::uint32_t id) const {
    return signals[id]->load(std::memory_order_acquire);
  }
  void raise_signal(std::uint32_t id) {
    signals[id]->store(true, std::memory_order_release);
  }
};

/// Take the next request_id and note it against `rec`. Every RPC the run puts
/// on the wire goes through here, so `request_id_log` ends up holding all of
/// them in issue order and each record's slice of it is what that event sent.
std::uint32_t issue_request_id(run_state &st, run_result &result, record &rec) {
  const std::uint32_t rid =
      st.next_request_id.fetch_add(1, std::memory_order_relaxed);
  if (rec.request_id_count == 0)
    rec.request_id_offset =
        static_cast<std::uint32_t>(result.request_id_log.size());
  result.request_id_log.push_back(rid);
  ++rec.request_id_count;
  return rid;
}

/// Forces a real yield where the timing thread would otherwise check "has
/// that signal come up yet?" in a tight loop
constexpr std::uint64_t kMinYieldGapNs = 50'000;

/// One multi-round or just-in-time `stream` event: draw a round from the
/// source, send it, decide whether to send another. Returns once a terminal
/// status is reached; `rec` and `result` are updated in place.
void run_stream(run_plan &plan, run_state &st, const event &e,
                const event_plan &ep, std::uint64_t deadline_abs_ns, session &s,
                record &rec, run_result &result) {
  // Either every round was drawn and serialized by plan(), or none was and
  // this loop draws them as it goes -- plan() never leaves half a stream
  // pre-built, so there is no third case to reconcile here.
  const bool prebuilt = !ep.rounds.empty();
  syndrome_source *source = prebuilt ? nullptr : plan.sources.at(e.source_id);

  std::uint64_t next = deadline_abs_ns;
  std::uint32_t rounds = 0;
  std::optional<stream_terminate> local_give_up;
  for (;;) {
    if (st.aborted.load(std::memory_order_relaxed)) {
      local_give_up = stream_terminate::ERROR;
      break;
    }
    if (rounds >= e.stream_max_rounds) {
      local_give_up = stream_terminate::EXHAUSTED_ROUNDS;
      break;
    }

    // Pre-built rounds run out only when the source went dry while plan()
    // was drawing them, which is the same exhaustion a run-time draw sees.
    if (prebuilt && rounds >= ep.rounds.size()) {
      local_give_up = stream_terminate::SOURCE_EXHAUSTED;
      break;
    }
    std::vector<std::uint8_t> drawn;
    if (!prebuilt) {
      drawn = source->next_round();
      if (drawn.empty()) {
        local_give_up = stream_terminate::SOURCE_EXHAUSTED;
        break;
      }
      if (s.max_frame_bytes != 0 &&
          build_enqueue_frame(e.decoder_id, 0, drawn.data(), drawn.size()).size() >
              s.max_frame_bytes) {
        local_give_up = stream_terminate::ERROR;
        break;
      }
    }
    const round_plan *rp = prebuilt ? &ep.rounds[rounds] : nullptr;
    const std::uint8_t *bits =
        prebuilt ? plan.sched.syndrome_arena.data() + rp->bits_offset : drawn.data();
    const std::size_t n_bits = prebuilt ? rp->bits_count : drawn.size();

    if (e.stream_every_ticks > 0) {
      wait_until(next);
      next += e.stream_every_ticks * plan.sched.tick_ns;
    }

    const std::uint32_t rid_enqueue = issue_request_id(st, result, rec);
    std::vector<std::uint8_t> built;
    if (prebuilt) {
      set_request_id(plan.frame_arena.data() + rp->frame_offset, rid_enqueue);
      s.send_async({plan.frame_arena.data() + rp->frame_offset, rp->frame_len});
    } else {
      built = build_enqueue_frame(e.decoder_id, rid_enqueue, bits, n_bits);
      s.send_async({built.data(), built.size()});
    }
    {
      std::lock_guard<std::mutex> lock(st.logs_mu);
      if (rounds == 0)
        rec.syndrome_offset = static_cast<std::uint32_t>(result.syndrome_log.size());
      result.syndrome_log.insert(result.syndrome_log.end(), bits, bits + n_bits);
    }
    rec.syndrome_count += static_cast<std::uint32_t>(n_bits);
    ++rounds;

    if (rounds < e.stream_min_rounds)
      continue; // below the floor: nothing can stop this stream yet

    if (e.until_signal_id == kNoSignal || st.signal_raised(e.until_signal_id))
      break;

    // Unpaced (every==0): this is the loop's only yield point, so block
    // briefly rather than spinning -- otherwise this thread can starve the
    // reader that owes us the raise of a core outright. Paced (every>0):
    // wait_until() above already slept.
    if (e.stream_every_ticks == 0)
      wait_until(now_ns() + kMinYieldGapNs);
  }

  const stream_terminate term = local_give_up.value_or(stream_terminate::OK);
  rec.status = static_cast<std::int32_t>(term);
  rec.rounds_streamed = rounds;
}

/// Collect one submitted blocking request's bare acknowledgement
void collect_ack(run_result &result, run_state &st, const event &e,
                 std::uint32_t i, session &s, record &rec,
                 std::uint32_t request_id) {
  std::size_t reply_len = 0;
  const auto status = s.await(request_id, {}, reply_len);
  rec.status = static_cast<std::int32_t>(status);
  abort_on_hard_error(status, i, to_string(e.op), e.decoder_id, result,
                      st.aborted, st.logs_mu);
}

/// Collect one submitted read's answer into `rec` and the run's correction
/// log. 
void collect_corrections(run_plan &plan, run_result &result, run_state &st,
                         const event &e, std::uint32_t i, session &s,
                         record &rec, std::uint32_t request_id,
                         std::uint32_t return_size) {
  std::vector<std::uint8_t> reply(wire::bit_packed_bytes(return_size));
  std::size_t reply_len = 0;
  const auto raw_status = s.await(request_id, reply, reply_len);
  const RpcStatus status =
      reject_truncated_reply(raw_status, reply_len, return_size);
  rec.status = static_cast<std::int32_t>(status);
  if (status != RpcStatus::OK) {
    abort_on_hard_error(status, i, to_string(e.op), e.decoder_id, result,
                        st.aborted, st.logs_mu);
    return;
  }
  rec.read_completed = true;
  rec.correction_count = return_size;
  std::lock_guard<std::mutex> lock(st.logs_mu);
  rec.correction_offset =
      static_cast<std::uint32_t>(result.correction_log.size());
  append_unpacked_bits(result.correction_log, reply.data(), return_size);
  rec.correction_mismatch =
      mismatches_expected(plan.sched, e,
                          result.correction_log.data() + rec.correction_offset,
                          return_size);
}

} // namespace

/// One event, dispatched on run()'s single timing thread. `prev_return_ns`
/// is the previous event's actual completion, for resolving a relative-tick
/// deadline (0 until anything has dispatched).
void dispatch_event(run_plan &plan, run_result &result, run_state &st, std::uint64_t t0,
                    std::uint32_t i, session &s,
                    std::uint64_t &prev_return_ns) {
  auto &sched = plan.sched;
  const auto &e = sched.events[i];
  const auto &ep = plan.event_plans[i];
  auto &rec = result.records[i];
  const auto decoder_id = e.decoder_id;
  auto &aborted = st.aborted;
  auto &logs_mu = st.logs_mu;

  const std::uint64_t deadline_ns = e.trig == trigger::tick
                                        ? e.deadline_ns
                                        : add_sat(prev_return_ns, e.deadline_ns);
  rec.deadline_ns = deadline_ns;

  wait_until(add_sat(t0, deadline_ns));
  rec.call_ns = now_ns() - t0;
  rec.dispatched = true;

  // Set by a `signal=` read: its record is finished by a reader thread when
  // the answer lands, so this thread must not stamp it as done here.
  bool async_reply = false;

  switch (e.op) {
  case operation::reset: {
    const std::uint32_t rid = issue_request_id(st, result, rec);
    set_request_id(plan.frame_arena.data() + ep.rounds[0].frame_offset, rid);
    const frame f{plan.frame_arena.data() + ep.rounds[0].frame_offset,
                  ep.rounds[0].frame_len};
    const std::uint32_t t = s.submit(f);
    if (e.signal_id != kNoSignal) {
      st.reader(decoder_id).submit({t, i, /*return_size=*/0});
      async_reply = true;
      break;
    }
    collect_ack(result, st, e, i, s, rec, t);
    break;
  }
  case operation::stream:
  case operation::enqueue_data: {
    // A stream of more than one round, or one drawing from a live source,
    // has no pre-built frame and runs its own paced loop instead.
    if (e.op == operation::stream && ep.rounds.size() != 1) {
      run_stream(plan, st, e, ep, t0 + deadline_ns, s, rec, result);
      break;
    }
    const auto ok_status = e.op == operation::stream
                               ? static_cast<std::int32_t>(stream_terminate::OK)
                               : static_cast<std::int32_t>(RpcStatus::OK);
    auto give_up_exhausted = [&] {
      rec.status = static_cast<std::int32_t>(stream_terminate::SOURCE_EXHAUSTED);
      warn(result, logs_mu,
           event_label(i, to_string(e.op), decoder_id) +
               " sent nothing: its syndrome source is exhausted");
    };
    if (!ep.rounds.empty()) {
      // Drawn and serialized by plan(): stamp an id and send.
      const auto &rp = ep.rounds[0];
      if (e.op == operation::stream)
        rec.rounds_streamed = 1;
      const std::uint32_t rid = issue_request_id(st, result, rec);
      set_request_id(plan.frame_arena.data() + rp.frame_offset, rid);
      s.send_async({plan.frame_arena.data() + rp.frame_offset, rp.frame_len});
      rec.status = ok_status;
      rec.syndrome_count = rp.bits_count;
      {
        std::lock_guard<std::mutex> lock(logs_mu);
        rec.syndrome_offset = static_cast<std::uint32_t>(result.syndrome_log.size());
        result.syndrome_log.insert(result.syndrome_log.end(),
                                  sched.syndrome_arena.begin() + rp.bits_offset,
                                  sched.syndrome_arena.begin() + rp.bits_offset +
                                      rp.bits_count);
      }
    } else {
      // JIT source: resolve now and send.
      auto &src = *plan.sources.at(e.source_id);
      std::vector<std::uint8_t> bits =
          e.op == operation::enqueue_data ? src.read_data() : src.next_round();
      if (bits.empty()) {
        give_up_exhausted();
        break;
      }
      if (e.op == operation::stream)
        rec.rounds_streamed = 1;
      // Built with a placeholder id and stamped only once it is going out, so
      // a frame rejected for size does not burn an id the record would then
      // claim to have sent.
      auto f_bytes = build_enqueue_frame(decoder_id, 0, bits.data(), bits.size());
      if (s.max_frame_bytes != 0 && f_bytes.size() > s.max_frame_bytes) {
        rec.status = static_cast<std::int32_t>(RpcStatus::INTERNAL_ERROR);
        abort_on_hard_error(RpcStatus::INTERNAL_ERROR, i, to_string(e.op), decoder_id,
                            result, aborted, logs_mu);
        break;
      }
      set_request_id(f_bytes.data(), issue_request_id(st, result, rec));
      s.send_async({f_bytes.data(), f_bytes.size()});
      rec.status = ok_status;
      rec.syndrome_count = static_cast<std::uint32_t>(bits.size());
      {
        std::lock_guard<std::mutex> lock(logs_mu);
        rec.syndrome_offset = static_cast<std::uint32_t>(result.syndrome_log.size());
        result.syndrome_log.insert(result.syndrome_log.end(), bits.begin(), bits.end());
      }
    }
    break;
  }
  case operation::get_corrections: {
    const std::uint32_t return_size = return_size_for(e);
    const std::uint32_t rid = issue_request_id(st, result, rec);
    set_request_id(plan.frame_arena.data() + ep.rounds[0].frame_offset, rid);
    const frame f{plan.frame_arena.data() + ep.rounds[0].frame_offset,
                  ep.rounds[0].frame_len};
    const std::uint32_t t = s.submit(f);
    if (e.signal_id != kNoSignal) {
      st.reader(decoder_id).submit({t, i, return_size});
      async_reply = true;
      break;
    }
    collect_corrections(plan, result, st, e, i, s, rec, t, return_size);
    break;
  }
  }

  const std::uint64_t done_ns = now_ns() - t0;
  if (!async_reply)
    rec.return_ns = done_ns;
  prev_return_ns = done_ns; // the timeline is free either way
}

run_result run(std::shared_ptr<run_plan> p) {
  auto &plan = *p;
  auto &sched = plan.sched;

  run_result result;
  result.records.resize(sched.events.size());
  // Every event gets an identified slot up front, dispatched or not 
  // record::dispatched is how a caller tells "ran" from
  // "pre-empted by the abort."
  for (std::size_t i = 0; i < sched.events.size(); ++i) {
    auto &rec = result.records[i];
    rec.event_index = static_cast<std::uint32_t>(i);
    rec.decoder_id = sched.events[i].decoder_id;
    rec.op = sched.events[i].op;
  }
  result.syndrome_log.reserve(sched.syndrome_arena.size());
  result.correction_log.reserve(sched.expected_arena.size());

  const std::uint64_t t0 = now_ns() + plan.params.lead_in_ns;
  wait_until(t0);

  // Everything the timing thread and the readers share. Populated in full
  // here, before any reader exists.
  run_state st;
  st.signals.reserve(sched.signal_names.size());
  for (std::size_t i = 0; i < sched.signal_names.size(); ++i)
    st.signals.push_back(std::make_unique<std::atomic<bool>>(false));

  // One reader per decoder that has an unblocking request, so those
  // requests have somewhere to be waited for other than the timing thread
  // that issued them. Op-agnostic: `signal=` means the same thing wherever
  // the parser accepts it.
  for (const auto &e : sched.events) {
    if (e.signal_id == kNoSignal || st.readers.count(e.decoder_id))
      continue;
    st.readers.emplace(
        e.decoder_id,
        std::make_unique<reader_thread>(
            *plan.router.at(e.decoder_id),
            [&plan, &result, &st, t0](session &s, const reader_thread::pending &pd) {
              const auto &ev = plan.sched.events[pd.event_index];
              auto &rec = result.records[pd.event_index];
              if (ev.op == operation::get_corrections)
                collect_corrections(plan, result, st, ev, pd.event_index, s,
                                    rec, pd.request_id, pd.return_size);
              else
                collect_ack(result, st, ev, pd.event_index, s, rec,
                            pd.request_id);
              rec.return_ns = now_ns() - t0;
              // Last: everything the signal promises is in place before a
              // stream can see it come up.
              st.raise_signal(ev.signal_id);
            }));
  }

  // The whole dispatch model: one thread, schedule order. 
  std::uint64_t prev_return_ns = 0;
  for (std::size_t i = 0;
       i < sched.events.size() && !st.aborted.load(std::memory_order_relaxed); ++i)
    dispatch_event(plan, result, st, t0, static_cast<std::uint32_t>(i),
                   *plan.router.at(sched.events[i].decoder_id), prev_return_ns);

  // Dispatch is done, so no more reads can be issued; drain the ones still
  // in flight before anybody looks at their records.
  for (auto &[id, reader] : st.readers)
    reader->close();

  result.t0_ns = t0;
  result.tick_ns = sched.tick_ns;
  return result;
}

// -- Downstream analysis

namespace {

// syndrome_log/correction_log are one byte (0x00/0x01) per BIT 
// Pack four bits into one hex digit, MSB-first, zero-padding the
// final partial nibble.
std::string hex_encode_bits(const std::uint8_t *bits, std::size_t count) {
  static const char kHexChars[] = "0123456789abcdef";
  std::string out;
  if (count == 0)
    return out;
  const std::size_t ndigits = (count + 3) / 4;
  out.reserve(ndigits);
  for (std::size_t d = 0; d < ndigits; ++d) {
    std::uint8_t nibble = 0;
    for (std::size_t b = 0; b < 4; ++b) {
      const std::size_t bit_index = d * 4 + b;
      const std::uint8_t bit = bit_index < count ? bits[bit_index] : 0;
      nibble = static_cast<std::uint8_t>((nibble << 1) | (bit & 1));
    }
    out.push_back(kHexChars[nibble]);
  }
  return out;
}

/// Bounds-checks (offset, count) against `arena` and returns how many bits
/// are actually safe to read 
std::pair<const std::uint8_t *, std::size_t>
safe_bit_span(const std::vector<std::uint8_t> &arena, std::uint32_t offset,
             std::uint32_t count) {
  if (arena.empty() || offset >= arena.size())
    return {nullptr, 0};
  return {arena.data() + offset, std::min<std::size_t>(count, arena.size() - offset)};
}

/// One record's request_ids as a single space-separated cell: variable-length
/// like the hex columns, and free of commas so it needs no CSV quoting.
/// Bounds-checked the same way, so a record pointing past the log renders
/// empty rather than reading off the end.
std::string join_request_ids(const std::vector<std::uint32_t> &log,
                             std::uint32_t offset, std::uint32_t count) {
  if (log.empty() || offset >= log.size())
    return {};
  const std::size_t n = std::min<std::size_t>(count, log.size() - offset);
  std::string out;
  for (std::size_t i = 0; i < n; ++i) {
    if (i != 0)
      out.push_back(' ');
    out += std::to_string(log[offset + i]);
  }
  return out;
}

} // namespace

void write_csv(const run_result &result, std::ostream &out) {
  out << "event_index,decoder_id,op,deadline_ns,call_ns,return_ns,"
         "lateness_ns,latency_ns,status,rounds_streamed,read_completed,"
         "syndrome_hex,correction_hex,correction_mismatch,request_ids,"
         "dispatched\n";
  for (const auto &r : result.records) {
    const auto [syndrome_bits, syndrome_n] =
        safe_bit_span(result.syndrome_log, r.syndrome_offset, r.syndrome_count);
    const auto [correction_bits, correction_n] = safe_bit_span(
        result.correction_log, r.correction_offset, r.correction_count);
    out << r.event_index << ',' << r.decoder_id << ',' << to_string(r.op) << ','
        << r.deadline_ns << ',' << r.call_ns << ',' << r.return_ns << ','
        << static_cast<std::int64_t>(r.call_ns) - static_cast<std::int64_t>(r.deadline_ns)
        << ','
        << static_cast<std::int64_t>(r.return_ns) - static_cast<std::int64_t>(r.call_ns)
        << ',' << r.status << ',' << r.rounds_streamed << ',' << (r.read_completed ? 1 : 0)
        << ',' << hex_encode_bits(syndrome_bits, syndrome_n) << ','
        << hex_encode_bits(correction_bits, correction_n) << ','
        << (r.correction_mismatch ? 1 : 0) << ','
        << join_request_ids(result.request_id_log, r.request_id_offset,
                            r.request_id_count)
        << ','
        << (r.dispatched ? 1 : 0) << '\n';
  }
}

std::string write_csv(const run_result &result) {
  std::ostringstream oss;
  write_csv(result, oss);
  return oss.str();
}

} // namespace cudaq::qec::playback
