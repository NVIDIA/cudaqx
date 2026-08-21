/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file emulator.cpp
/// @brief plan() and run(): pre-serialize every static frame before t0,
/// validate every frame size it can before t0, then run one timing thread that does
/// nothing between its deadlines but wait and dispatch. 

#include "emulator.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <ctime>
#include <functional>
#include <mutex>
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

/// The sendable bytes of a round plan()'s already serialized.
frame frame_of(const run_plan &plan, const round_plan &rp) {
  return {plan.frame_arena.data() + rp.frame_offset, rp.frame_len};
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
                                                       std::int64_t return_size,
                                                       std::uint32_t rid) {
  // reset=1 always: a playback read consumes the shot it reports on.
  wire::GetCorrectionsRequestPayload p{static_cast<std::int64_t>(decoder_id), return_size,
                                       /*reset=*/1};
  return build_frame(wire::kGetCorrectionsFunctionId, rid, &p, sizeof(p));
}

/// The width a returning RPC requests
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

  check_signal_order(sched);

  // -- Every event must have somewhere to send to and something to draw from
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
      ep.push_back(rp);
    };

    switch (e.op) {
    case operation::reset:
      place(build_reset_frame(e.decoder_id, /*rid=*/0), 0, 0);
      break;
    case operation::get_corrections:
      place(build_get_corrections_frame(e.decoder_id, return_size_for(e), /*rid=*/0),
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

  // -- No session may serve two decoder_ids
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
/// Built on run()'s stack. 
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
  std::vector<std::atomic<bool>> signals;

  /// One reader per decoder that has at least one `signal=` read.
  std::unordered_map<std::uint64_t, std::unique_ptr<reader_thread>> readers;

  reader_thread &reader(std::uint64_t decoder_id) {
    return *readers.at(decoder_id);
  }

  bool signal_raised(std::uint32_t id) const {
    return signals[id].load(std::memory_order_acquire);
  }
  void raise_signal(std::uint32_t id) {
    signals[id].store(true, std::memory_order_release);
  }
};

/// The four things every step of a run needs
struct run_ctx {
  run_plan &plan;
  run_result &result;
  run_state &st;
  std::uint64_t t0;

  const event &ev(std::uint32_t i) const { return plan.sched.events[i]; }
  record &rec(std::uint32_t i) const { return result.records[i]; }
};

/// Appends one line to `result.warnings`.
void warn(const run_ctx &c, const std::string &message) {
  std::lock_guard<std::mutex> lock(c.st.logs_mu);
  c.result.warnings.push_back(message);
}

/// Describes an event the way a warning should name it.
std::string event_label(const run_ctx &c, std::uint32_t i) {
  return "event " + std::to_string(i) + " (" + to_string(c.ev(i).op) +
         ", decoder_id=" + std::to_string(c.ev(i).decoder_id) + ")";
}

/// Records the warning and flips `aborted` so the timing thread stops
/// dispatching further events.
void abort_on_hard_error(const run_ctx &c, RpcStatus status, std::uint32_t i) {
  if (status == RpcStatus::OK || status == RpcStatus::NOT_READY)
    return;
  c.st.aborted.store(true, std::memory_order_relaxed);
  warn(c, event_label(c, i) + " returned status " +
              std::to_string(static_cast<int>(status)) + "; aborting the run");
}

/// Take the next request_id and note it against `rec`. Every RPC the run puts
/// on the wire goes through here, so `request_id_log` ends up holding all of
/// them in issue order and each record's slice of it is what that event sent.
std::uint32_t issue_request_id(const run_ctx &c, record &rec) {
  const std::uint32_t rid =
      c.st.next_request_id.fetch_add(1, std::memory_order_relaxed);
  if (rec.request_id_count == 0)
    rec.request_id_offset =
        static_cast<std::uint32_t>(c.result.request_id_log.size());
  c.result.request_id_log.push_back(rid);
  ++rec.request_id_count;
  return rid;
}

/// Record one outgoing round's bits against `rec` and append them to the
/// run's shared syndrome log. `first` marks the round that owns
/// `rec.syndrome_offset`; every round adds to `rec.syndrome_count`.
void log_syndromes(const run_ctx &c, record &rec, const std::uint8_t *bits,
                   std::size_t n, bool first) {
  std::lock_guard<std::mutex> lock(c.st.logs_mu);
  if (first)
    rec.syndrome_offset =
        static_cast<std::uint32_t>(c.result.syndrome_log.size());
  c.result.syndrome_log.insert(c.result.syndrome_log.end(), bits, bits + n);
  rec.syndrome_count += static_cast<std::uint32_t>(n);
}

/// One `stream` or `enqueue_data` event: draw a round from the source, send
/// it, decide whether to send another. Returns once a terminal status is
/// reached; `rec` and `result` are updated in place.
///
/// The two ops share this loop because they are the same wire operation.
/// `enqueue_data` differs in exactly one respect -- it pulls the source's
/// terminal data-qubit readout instead of its next stabilizer round -- and
/// the parser leaves its round bounds at 1/1 with no `until=`, so the loop
/// below sends one round and stops without needing to know which op it is.
void run_stream(const run_ctx &c, std::uint32_t i,
                std::uint64_t deadline_abs_ns, session &s) {
  auto &plan = c.plan;
  auto &st = c.st;
  const event &e = c.ev(i);
  const event_plan &ep = c.plan.event_plans[i];
  record &rec = c.rec(i);
  // Either every round was drawn and serialized by plan(), or none was and
  // this loop draws them as it goes -- plan() never leaves half a stream
  // pre-built, so there is no third case to reconcile here.
  const bool prebuilt = !ep.empty();
  syndrome_source *source = prebuilt ? nullptr : plan.sources.at(e.source_id);

  std::uint64_t next = deadline_abs_ns;
  std::uint32_t rounds = 0;
  stream_terminate term = stream_terminate::OK;
  for (;;) {
    if (st.aborted.load(std::memory_order_relaxed)) {
      term = stream_terminate::ERROR;
      break;
    }
    if (rounds >= e.stream_max_rounds) {
      term = stream_terminate::EXHAUSTED_ROUNDS;
      break;
    }

    // Pre-built rounds run out only when the source went dry while plan()
    // was drawing them, which is the same exhaustion a run-time draw sees.
    if (prebuilt && rounds >= ep.size()) {
      term = stream_terminate::SOURCE_EXHAUSTED;
      break;
    }
    std::vector<std::uint8_t> drawn;
    if (!prebuilt) {
      drawn = e.op == operation::enqueue_data ? source->read_data()
                                              : source->next_round();
      if (drawn.empty()) {
        term = stream_terminate::SOURCE_EXHAUSTED;
        break;
      }
      if (s.max_frame_bytes != 0 &&
          build_enqueue_frame(e.decoder_id, 0, drawn.data(), drawn.size()).size() >
              s.max_frame_bytes) {
        term = stream_terminate::ERROR;
        break;
      }
    }
    if (e.stream_every_ticks > 0) {
      wait_until(next);
      next += e.stream_every_ticks * plan.sched.tick_ns;
    }

    // Where this round's bits live and how its frame is produced are the
    // only things the two modes disagree on.
    const std::uint32_t rid = issue_request_id(c, rec);
    const std::uint8_t *bits;
    std::size_t n_bits;
    std::vector<std::uint8_t> built;
    if (prebuilt) {
      const round_plan &rp = ep[rounds];
      bits = plan.sched.syndrome_arena.data() + rp.bits_offset;
      n_bits = rp.bits_count;
      set_request_id(plan.frame_arena.data() + rp.frame_offset, rid);
      s.send_async(frame_of(plan, rp));
    } else {
      bits = drawn.data();
      n_bits = drawn.size();
      built = build_enqueue_frame(e.decoder_id, rid, bits, n_bits);
      s.send_async({built.data(), built.size()});
    }
    log_syndromes(c, rec, bits, n_bits, /*first=*/rounds == 0);
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
      std::this_thread::sleep_for(std::chrono::nanoseconds(kSpinSlackNs));
  }

  rec.status = static_cast<std::int32_t>(term);
  rec.rounds_streamed = rounds;
}

/// Collect one submitted request's bare acknowledgement
void collect_ack(const run_ctx &c, std::uint32_t i, session &s,
                 std::uint32_t request_id) {
  record &rec = c.rec(i);
  std::size_t reply_len = 0;
  const auto status = s.await(request_id, {}, reply_len);
  rec.status = static_cast<std::int32_t>(status);
  abort_on_hard_error(c, status, i);
}

/// Collect one submitted read's answer into `rec` and the run's correction
/// log. 
void collect_corrections(const run_ctx &c, std::uint32_t i, session &s,
                         std::uint32_t request_id) {
  const event &e = c.ev(i);
  record &rec = c.rec(i);
  const std::uint32_t return_size = return_size_for(e);
  std::vector<std::uint8_t> reply(wire::bit_packed_bytes(return_size));
  std::size_t reply_len = 0;
  const auto raw_status = s.await(request_id, reply, reply_len);
  const RpcStatus status =
      reject_truncated_reply(raw_status, reply_len, return_size);
  rec.status = static_cast<std::int32_t>(status);
  if (status != RpcStatus::OK) {
    abort_on_hard_error(c, status, i);
    return;
  }
  rec.read_completed = true;
  rec.correction_count = return_size;
  std::lock_guard<std::mutex> lock(c.st.logs_mu);
  rec.correction_offset =
      static_cast<std::uint32_t>(c.result.correction_log.size());
  append_unpacked_bits(c.result.correction_log, reply.data(), return_size);
  rec.correction_mismatch =
      mismatches_expected(c.plan.sched, e,
                          c.result.correction_log.data() + rec.correction_offset,
                          return_size);
}

/// Collect one submitted request's answer. 
void collect_reply(const run_ctx &c, std::uint32_t i, session &s,
                   std::uint32_t request_id) {
  if (c.ev(i).op == operation::get_corrections)
    collect_corrections(c, i, s, request_id);
  else
    collect_ack(c, i, s, request_id);
}

} // namespace

/// One event, dispatched on run()'s single timing thread. `prev_return_ns`
/// is the previous event's actual completion, for resolving a relative-tick
/// deadline (0 until anything has dispatched).
void dispatch_event(const run_ctx &c, std::uint32_t i, session &s,
                    std::uint64_t &prev_return_ns) {
  auto &plan = c.plan;
  auto &st = c.st;
  auto &sched = plan.sched;
  const auto &e = c.ev(i);
  const auto &ep = plan.event_plans[i];
  auto &rec = c.rec(i);
  const auto decoder_id = e.decoder_id;
  const std::uint64_t t0 = c.t0;

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
  case operation::reset:
  case operation::get_corrections: {
    const auto &rp = ep[0];
    set_request_id(plan.frame_arena.data() + rp.frame_offset,
                   issue_request_id(c, rec));
    const std::uint32_t req = s.submit(frame_of(plan, rp));
    if (e.signal_id != kNoSignal) {
      st.reader(decoder_id).submit({req, i});
      async_reply = true;
      break;
    }
    collect_reply(c, i, s, req);
    break;
  }
  case operation::stream:
  case operation::enqueue_data:
    // Same wire operation, same loop -- see run_stream.
    run_stream(c, i, add_sat(t0, deadline_ns), s);
    break;
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
  // here. 
  run_state st;
  st.signals = std::vector<std::atomic<bool>>(sched.signal_names.size());

  const run_ctx c{plan, result, st, t0};

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
            [c](session &s, const reader_thread::pending &pd) {
              collect_reply(c, pd.event_index, s, pd.request_id);
              c.rec(pd.event_index).return_ns = now_ns() - c.t0;
              // Last: everything the signal promises is in place before a
              // stream can see it come up.
              c.st.raise_signal(c.ev(pd.event_index).signal_id);
            }));
  }

  // The whole dispatch model: one thread, schedule order. 
  std::uint64_t prev_return_ns = 0;
  for (std::size_t i = 0;
       i < sched.events.size() && !st.aborted.load(std::memory_order_relaxed); ++i)
    dispatch_event(c, static_cast<std::uint32_t>(i),
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
