/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file emulator.cpp
/// @brief plan() and run(): pre-serialize every
/// static frame before t0, validate capabilities before t0, then run a
/// single timing thread that does nothing between deadlines but wait and
/// dispatch. 

#include "cudaq/qec/playback/emulator.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <ctime>
#include <functional>
#include <future>
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
// the last `slack_ns` to the clock-read floor.

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

/// Sample clock_nanosleep overshoot a handful of times and return a
/// comfortable multiple of the worst observed value, clamped to a sane
/// range, for use as `wait_until`'s slack. Run once at startup, before t0.
std::uint64_t calibrate_spin_slack_ns() {
  constexpr int kSamples = 8;
  constexpr std::uint64_t kSampleGapNs = 200'000; // 200 us
  std::uint64_t worst = 0;
  for (int i = 0; i < kSamples; ++i) {
    const std::uint64_t target = now_ns() + kSampleGapNs;
    sleep_until(target);
    const std::uint64_t actual = now_ns();
    worst = std::max(worst, actual > target ? actual - target : 0);
  }
  // A comfortable multiple of the worst observed overshoot (below
  // worst-case, the sleep lands past the deadline and the spin has nothing
  // left to correct), clamped: a floor because a near-zero sample can't be
  // trusted, a ceiling because one bad preemption during calibration
  // shouldn't blow up the spin budget for the whole run (spinning past a
  // point wastes a core with no accuracy benefit).
  constexpr std::uint64_t kFloorNs = 5'000;
  constexpr std::uint64_t kMaxSlackNs = 2'000'000;
  return std::clamp(worst * 4, kFloorNs, kMaxSlackNs);
}

/// Block until CLOCK_MONOTONIC reaches `deadline_ns` (an absolute
/// CLOCK_MONOTONIC timestamp). Sleeps in bounded naps down to
/// `deadline_ns - slack_ns`, then spins a plain clock read to the deadline
void wait_until(std::uint64_t deadline_ns, std::uint64_t slack_ns) {
  const std::uint64_t spin_from =
      deadline_ns > slack_ns ? deadline_ns - slack_ns : 0;
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
/// trailing bit-packed syndrome bytes (enqueue only). The single point
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
/// placeholder -- RPCHeader is always a frame's leading bytes, at whatever
/// offset it lives at (the frame arena for a pre-built frame, or a
/// freshly-built buffer). Used so request_id can be assigned fresh right
/// before a frame is actually sent (run()'s single global counter) without
/// re-serializing the whole frame on the hot path.
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

/// The return_size a get_corrections/stream_until call requests: the width
/// of the operand's expected-bits string, when given. 0 (request nothing) when omitted
std::uint32_t return_size_for(const event &e) { return e.expected_count; }

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

/// Records the warning and flips `aborted` for the caller to stop dispatching further events.
void abort_on_hard_error(RpcStatus status, std::size_t event_index, const char *op_name,
                         std::uint64_t decoder_id, run_result &result, bool &aborted) {
  if (status == RpcStatus::OK || status == RpcStatus::NOT_READY)
    return;
  aborted = true;
  result.warnings.push_back("event " + std::to_string(event_index) + " (" + op_name +
                            ", decoder_id=" + std::to_string(decoder_id) +
                            ") returned status " + std::to_string(static_cast<int>(status)) +
                            "; aborting the run");
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
    session *s = nullptr;
    try {
      s = router.at(e.decoder_id);
    } catch (const std::out_of_range &) {
      throw std::invalid_argument("no session routes decoder_id=" +
                             std::to_string(e.decoder_id));
    }
    const auto caps = s->caps();

    if (e.op == operation::stream_until && !caps.reports_not_ready)
      throw std::invalid_argument(
          "stream_until on decoder_id=" + std::to_string(e.decoder_id) +
          " requires a session with reports_not_ready capability, but the "
          "routed session does not have it");

    if ((e.op == operation::enqueue && e.source_id != kNoSource) ||
        e.op == operation::enqueue_data || e.op == operation::stream_until) {
      if (!sources.contains(e.source_id))
        throw std::invalid_argument("no syndrome_source registered for source_id=" +
                               std::to_string(e.source_id));
    }
  }

  // -- Pre-draw source-backed rounds where the count AND the content are
  // known ahead of t0.
  for (auto &e : sched.events) {
    if ((e.op != operation::enqueue && e.op != operation::enqueue_data) ||
        e.source_id == kNoSource)
      continue;
    auto &src = *sources.at(e.source_id);
    if (src.is_streamed())
      continue;
    auto round = e.op == operation::enqueue_data ? src.read_data() : src.next_round();
    e.syndrome_offset = static_cast<std::uint32_t>(sched.syndrome_arena.size());
    e.syndrome_count = static_cast<std::uint32_t>(round.size());
    sched.syndrome_arena.insert(sched.syndrome_arena.end(), round.begin(), round.end());
  }

  // -- Build every frame whose bytes are known now.
  impl->event_plans.resize(sched.events.size());

  for (std::size_t i = 0; i < sched.events.size(); ++i) {
    const auto &e = sched.events[i];
    auto &ep = impl->event_plans[i];
    const auto max_frame_bytes = router.at(e.decoder_id)->caps().max_frame_bytes;

    auto place_frame = [&](std::vector<std::uint8_t> bytes) {
      if (max_frame_bytes != 0 && bytes.size() > max_frame_bytes)
        throw std::invalid_argument(
            "event " + std::to_string(i) + " (decoder_id=" + std::to_string(e.decoder_id) +
            ") builds a " + std::to_string(bytes.size()) +
            "-byte frame, exceeding the session's max_frame_bytes=" +
            std::to_string(max_frame_bytes));
      ep.frame_offset = static_cast<std::uint32_t>(impl->frame_arena.size());
      ep.frame_len = static_cast<std::uint32_t>(bytes.size());
      impl->frame_arena.insert(impl->frame_arena.end(), bytes.begin(), bytes.end());
      ep.has_frame = true;
    };

    switch (e.op) {
    case operation::reset:
      place_frame(build_reset_frame(e.decoder_id, /*rid=*/0));
      break;
    case operation::enqueue:
    case operation::enqueue_data: {
      const bool preloaded =
          e.source_id == kNoSource || !sources.at(e.source_id)->is_streamed();
      if (preloaded)
        place_frame(build_enqueue_frame(e.decoder_id, /*rid=*/0,
                                        sched.syndrome_arena.data() + e.syndrome_offset,
                                        e.syndrome_count));
      // else: JIT source -- bits aren't known until this event's actual
      // turn in run()'s dispatch loop; leave the frame unbuilt
      // (ep.has_frame stays false) and resolve it there.
      break;
    }
    case operation::get_corrections:
      place_frame(build_get_corrections_frame(e.decoder_id, return_size_for(e),
                                              /*reset=*/true, /*rid=*/0));
      break;
    case operation::stream_until:
      break; // frames built round-by-round at run time; nothing to do here.
    }
  }

  return impl;
}

namespace {

/// Outcome of one get_corrections issued by stream_until's listener.
struct listener_reply {
  RpcStatus status = RpcStatus::INTERNAL_ERROR;
  std::vector<std::uint8_t> bits; // valid only when status == OK
};

/// Runs stream_until's blocking get_corrections calls off the main thread.
/// Created once per run() and reused by every stream_until event
class listener_worker {
public:
  listener_worker() : thread_([this] { loop(); }) {}
  ~listener_worker() {
    {
      std::lock_guard<std::mutex> lock(mu_);
      stop_ = true;
    }
    cv_.notify_one();
    thread_.join();
  }
  listener_worker(const listener_worker &) = delete;
  listener_worker &operator=(const listener_worker &) = delete;

  /// Hands `job` to the background thread and returns a future for its
  /// result. Returns once that thread has picked the job up, so the caller
  /// can count on the request being in flight rather than still waiting to
  /// be scheduled.
  std::future<listener_reply> submit(std::function<listener_reply()> job) {
    std::promise<listener_reply> promise;
    auto future = promise.get_future();
    std::unique_lock<std::mutex> lock(mu_);
    job_ = std::move(job);
    promise_ = std::move(promise);
    cv_.notify_one();
    taken_.wait(lock, [this] { return !job_; });
    return future;
  }

private:
  void loop() {
    std::unique_lock<std::mutex> lock(mu_);
    for (;;) {
      cv_.wait(lock, [this] { return job_ || stop_; });
      if (!job_)
        return;
      auto job = std::move(job_);
      auto promise = std::move(promise_);
      job_ = nullptr;
      taken_.notify_one();
      lock.unlock();
      promise.set_value(job());
      lock.lock();
    }
  }

  std::mutex mu_;
  std::condition_variable cv_;    // a job was submitted, or stop_ was set
  std::condition_variable taken_; // the submitted job was picked up
  std::function<listener_reply()> job_;
  std::promise<listener_reply> promise_;
  bool stop_ = false;
  std::thread thread_; // last, so loop() only ever sees initialized members
};

/// Forces a real yield wherever a thread would otherwise check "is the other
/// side done yet?" in a tight loop: (1) the shortest gap between two of the
/// listener's own get_corrections calls, and (2) how long the main thread's
/// round-generation loop blocks on the listener's future before checking
/// again. Neither use changes correctness -- a ready result or an actual
/// reply is noticed the instant it happens, not delayed until the next tick
/// of this gap -- it only bounds how long a thread can go without giving up
/// its timeslice. Without it, a session with no real wait to speak of
/// (in-process, null) can spin one thread flat out and starve the other of
/// a core entirely on a busy host: a correctness problem (the rounds/polls
/// it should be making never happen), not just wasted CPU.
constexpr std::uint64_t kMinYieldGapNs = 50'000;

/// One stream_until event. Returns once a terminal status is reached; `rec`
/// and `result` are updated in place. 
/// 
/// `next_request_id` is run()'s single global counter, shared across every event 
/// -- each RPC sent here increments it by 1. The listener claims ids from it too, 
/// which is safe precisely because this thread never touches it between submitting 
/// the job and collecting its result.
///
/// The server dispatches every request for a decoder -- enqueue and
/// get_corrections alike -- inline on one thread, strictly in arrival order
/// (see DecodingSession's "Inline HOST_CALL path"), so a get_corrections
/// queued behind a slow synchronous decode cannot be answered until that
/// decode finishes. `worker` therefore issues blocking get_corrections
/// calls whose reply lands when the server becomes free to answer, while this
/// thread independently keeps generating rounds at the schedule's cadence until
/// that reply arrives or a local bound (max_rounds, source exhaustion, an
/// oversized frame) is hit.
///
/// A local bound sets `abort_listener`, which the listener checks between
/// polls, so it can be safely joined
///
/// Rounds are cached and flushed in one go only once that reply is in
/// hand: DecodingSession::enqueue_core resets shot_state unconditionally,
/// so sending a round before the outcome is known could clobber a
/// correction that became ready in the meantime.
void run_stream_until(run_plan &plan, const event &e, std::uint64_t t0, session &s,
                      record &rec, run_result &result, std::uint64_t slack,
                      std::uint32_t &next_request_id, listener_worker &worker) {
  auto &source = *plan.sources.at(e.source_id);
  const std::uint32_t return_size = return_size_for(e);
  const auto caps = s.caps();

  std::atomic<bool> abort_listener{false};
  const std::uint32_t rid_gc = next_request_id++;

  // Submit the listener job to the background thread and await response
  auto future = worker.submit([&s, &e, &abort_listener, &next_request_id, return_size,
                               rid_gc] {
    std::vector<std::uint8_t> bits(wire::bit_packed_bytes(return_size));
    for (std::uint32_t rid = rid_gc;; rid = next_request_id++) {
      auto gc_frame =
          build_get_corrections_frame(e.decoder_id, return_size, /*reset=*/true, rid);
      std::size_t reply_len = 0;
      const std::uint64_t sent_ns = now_ns();
      const auto raw_status =
          s.send_sync({gc_frame.data(), gc_frame.size()}, bits, reply_len);
      const auto status = reject_truncated_reply(raw_status, reply_len, return_size);
      if (status == RpcStatus::OK) {
        bits.resize(reply_len);
        return listener_reply{status, std::move(bits)};
      }
      if (status != RpcStatus::NOT_READY)
        return listener_reply{status, {}}; // a genuine wire error ends the stream
      if (abort_listener.load(std::memory_order_acquire))
        return listener_reply{status, {}}; // the main thread is done; let its reason win
      if (now_ns() - sent_ns < kMinYieldGapNs)
        std::this_thread::sleep_for(std::chrono::nanoseconds(kMinYieldGapNs));
    }
  });

  // Always generate one round before first checking the listener
  std::uint64_t next = t0 + e.deadline_ns;
  std::uint32_t rounds = 0;
  std::vector<std::vector<std::uint8_t>> cached;
  std::optional<stream_terminate> local_give_up;
  do {
    if (cached.size() >= e.stream_max_rounds) {
      local_give_up = stream_terminate::EXHAUSTED_ROUNDS;
      break;
    }

    auto bits = source.next_round();
    if (bits.empty()) {
      local_give_up = stream_terminate::SOURCE_EXHAUSTED;
      break;
    }

    // Ensure that the frame size is within the session's limits
    if (caps.max_frame_bytes != 0 &&
        build_enqueue_frame(e.decoder_id, 0, bits.data(), bits.size()).size() >
            caps.max_frame_bytes) {
      local_give_up = stream_terminate::ERROR;
      break;
    }

    if (e.stream_every_ticks > 0) {
      wait_until(next, slack);
      next += e.stream_every_ticks * plan.sched.tick_ns;
    }

    cached.push_back(std::move(bits));
    // Unpaced (every==0): this is the loop's only yield point, so sleep 
    // allow wait_for to yield the thread, otherwise listener thread may starve.
    // Paced (every>0): wait_until() above slept, which yields the
    // thread -- no need to block here too.
  } while (future.wait_for(std::chrono::nanoseconds(
               e.stream_every_ticks > 0 ? 0 : kMinYieldGapNs)) !=
          std::future_status::ready);

  abort_listener.store(true, std::memory_order_release);
  const listener_reply reply = future.get();

  // Submit all cached syndromes
  for (auto &bits : cached) {
    const std::uint32_t rid_enqueue = next_request_id++;
    auto enqueue_frame = build_enqueue_frame(e.decoder_id, rid_enqueue, bits.data(), bits.size());
    s.send_async({enqueue_frame.data(), enqueue_frame.size()});
    if (rounds == 0) {
      rec.first_request_id = rid_enqueue;
      rec.syndrome_offset = static_cast<std::uint32_t>(result.syndrome_log.size());
    }
    result.syndrome_log.insert(result.syndrome_log.end(), bits.begin(), bits.end());
    rec.syndrome_count += static_cast<std::uint32_t>(bits.size());
    ++rounds;
  }
  if (rounds == 0)
    rec.first_request_id = rid_gc; // nothing was ever cached; the poll was this event's only RPC

  stream_terminate term;
  if (reply.status == RpcStatus::OK) { // everything went well
    term = stream_terminate::READY;
    // Load corrections into the correction log
    rec.read_completed = true;
    rec.correction_offset = static_cast<std::uint32_t>(result.correction_log.size());
    rec.correction_count = return_size;
    append_unpacked_bits(result.correction_log, reply.bits.data(), return_size);
    rec.correction_mismatch = mismatches_expected(
        plan.sched, e, result.correction_log.data() + rec.correction_offset, return_size);
  } else if (reply.status == RpcStatus::NOT_READY) { // the listener had an error
    term = *local_give_up;
  } else { // the main thread had an error
    term = stream_terminate::ERROR;
  }

  rec.status = static_cast<std::int32_t>(term);
  rec.rounds_streamed = rounds;
}

} // namespace

run_result run(std::shared_ptr<run_plan> p) {
  auto &plan = *p;
  auto &sched = plan.sched;

  run_result result;
  result.records.resize(sched.events.size());
  result.syndrome_log.reserve(sched.syndrome_arena.size());
  result.correction_log.reserve(sched.expected_arena.size());

  std::uint64_t slack = plan.params.spin_slack_ns;
  if (slack == 0)
    slack = calibrate_spin_slack_ns();

  for (auto &[id, s] : plan.router)
    s->warm_up();

  const std::uint64_t t0 = now_ns() + plan.params.lead_in_ns;
  wait_until(t0, slack);

  // Single global counter, incremented once per RPC actually sent to a session
  std::uint32_t next_request_id = 1;

  // Spawned listener thread (if needed)
  std::optional<listener_worker> listener;
  if (std::any_of(sched.events.begin(), sched.events.end(),
                  [](const event &e) { return e.op == operation::stream_until; }))
    listener.emplace();

  bool aborted = false;
  for (std::size_t i = 0; i < sched.events.size() && !aborted; ++i) {
    const auto &e = sched.events[i];
    const auto &ep = plan.event_plans[i];
    auto &rec = result.records[i];
    rec.event_index = static_cast<std::uint32_t>(i);
    rec.decoder_id = e.decoder_id;
    rec.op = e.op;
    rec.deadline_ns = e.deadline_ns;

    wait_until(t0 + e.deadline_ns, slack);
    rec.call_ns = now_ns() - t0;

    session &s = *plan.router.at(e.decoder_id);

    switch (e.op) {
    case operation::reset: {
      const std::uint32_t rid = next_request_id++;
      set_request_id(plan.frame_arena.data() + ep.frame_offset, rid);
      const frame f{plan.frame_arena.data() + ep.frame_offset, ep.frame_len};
      std::size_t reply_len = 0;
      const auto status = s.send_sync(f, {}, reply_len);
      rec.status = static_cast<std::int32_t>(status);
      rec.first_request_id = rid;
      abort_on_hard_error(status, i, "reset", e.decoder_id, result, aborted);
      break;
    }
    case operation::enqueue:
    case operation::enqueue_data: {
      if (ep.has_frame) {
        // Pre-built at plan() time (raw literal bits, or a preloadable
        // source): memcpy + send.
        const std::uint32_t rid = next_request_id++;
        set_request_id(plan.frame_arena.data() + ep.frame_offset, rid);
        rec.first_request_id = rid;
        const frame f{plan.frame_arena.data() + ep.frame_offset, ep.frame_len};
        s.send_async(f);
        rec.status = static_cast<std::int32_t>(RpcStatus::OK);
        rec.syndrome_offset = static_cast<std::uint32_t>(result.syndrome_log.size());
        rec.syndrome_count = e.syndrome_count;
        result.syndrome_log.insert(result.syndrome_log.end(),
                                  sched.syndrome_arena.begin() + e.syndrome_offset,
                                  sched.syndrome_arena.begin() + e.syndrome_offset +
                                      e.syndrome_count);
      } else {
        // JIT source: resolve now and send
        auto &src = *plan.sources.at(e.source_id);
        auto bits = e.op == operation::enqueue_data ? src.read_data() : src.next_round();
        const auto caps = s.caps();
        const std::uint32_t rid = next_request_id++;
        rec.first_request_id = rid;
        auto f_bytes = build_enqueue_frame(e.decoder_id, rid, bits.data(), bits.size());
        if (caps.max_frame_bytes != 0 && f_bytes.size() > caps.max_frame_bytes) {
          rec.status = static_cast<std::int32_t>(RpcStatus::INTERNAL_ERROR);
          abort_on_hard_error(RpcStatus::INTERNAL_ERROR, i, to_string(e.op), e.decoder_id,
                              result, aborted);
          break;
        }
        s.send_async({f_bytes.data(), f_bytes.size()});
        rec.status = static_cast<std::int32_t>(RpcStatus::OK);
        rec.syndrome_offset = static_cast<std::uint32_t>(result.syndrome_log.size());
        rec.syndrome_count = static_cast<std::uint32_t>(bits.size());
        result.syndrome_log.insert(result.syndrome_log.end(), bits.begin(), bits.end());
      }
      break;
    }
    case operation::get_corrections: {
      const std::uint32_t return_size = return_size_for(e);
      std::vector<std::uint8_t> reply(wire::bit_packed_bytes(return_size));
      std::size_t reply_len = 0;
      const std::uint64_t retry_deadline =
          plan.params.dispatch.retry_not_ready
              ? now_ns() + plan.params.dispatch.not_ready_deadline_ns
              : 0;
      RpcStatus status;
      std::uint32_t first_rid = 0;
      bool first_attempt = true;
      while (true) {
        // Each retry is its own RPC send, so it gets its own fresh
        // request_id -- reusing one across retries would let a stale reply
        // to an earlier attempt be mistaken for the current one.
        const std::uint32_t rid = next_request_id++;
        if (first_attempt) {
          first_rid = rid;
          first_attempt = false;
        }
        set_request_id(plan.frame_arena.data() + ep.frame_offset, rid);
        const frame f{plan.frame_arena.data() + ep.frame_offset, ep.frame_len};
        const auto raw_status = s.send_sync(f, reply, reply_len);
        status = reject_truncated_reply(raw_status, reply_len, return_size);
        if (status != RpcStatus::NOT_READY || !plan.params.dispatch.retry_not_ready ||
            now_ns() >= retry_deadline)
          break;
      }
      rec.status = static_cast<std::int32_t>(status);
      rec.first_request_id = first_rid;
      if (status == RpcStatus::OK) {
        rec.read_completed = true;
        rec.correction_offset = static_cast<std::uint32_t>(result.correction_log.size());
        rec.correction_count = return_size;
        append_unpacked_bits(result.correction_log, reply.data(), return_size);
        rec.correction_mismatch = mismatches_expected(
            sched, e, result.correction_log.data() + rec.correction_offset, return_size);
      } else {
        abort_on_hard_error(status, i, "get_corrections", e.decoder_id, result, aborted);
      }
      break;
    }
    case operation::stream_until:
      run_stream_until(plan, e, t0, s, rec, result, slack, next_request_id, *listener);
      break;
    }

    rec.return_ns = now_ns() - t0;
    if (aborted)
      result.records.resize(i + 1);
  }

  result.meta.t0_ns = t0;
  result.meta.tick_ns = sched.tick_ns;
  result.meta.spin_slack_ns = slack;
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

} // namespace

void write_csv(const run_result &result, std::ostream &out) {
  out << "event_index,decoder_id,op,deadline_ns,call_ns,return_ns,"
         "lateness_ns,latency_ns,status,rounds_streamed,read_completed,"
         "syndrome_hex,correction_hex,correction_mismatch,first_request_id\n";
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
        << (r.correction_mismatch ? 1 : 0) << ',' << r.first_request_id << '\n';
  }
}

std::string write_csv(const run_result &result) {
  std::ostringstream oss;
  write_csv(result, oss);
  return oss.str();
}

} // namespace cudaq::qec::playback
