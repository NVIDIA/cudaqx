/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file py_playback_emulator.cpp
/// @brief Python binding for the playback emulator. `parse()`/`plan()` and
/// the session/schedule/capabilities/run_params machinery that ties them
/// together are internal, C++-only plumbing (see emulator.h) -- the Python
/// surface is just `run()`: a schedule string plus flags selecting the
/// session backend, run against the given syndrome sources, returning the
/// same `run_result` the CLI tool gets.

#include "py_playback_emulator.h"

#include "session.h"
#include "syndrome_source.h"
#include "emulator.h"

#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

namespace cudaq::qec::playback {

namespace {

/// Parses, plans, and runs `schedule_text` in one call -- the session
/// backend is picked by which one of `decoders` / `udp_endpoints` /
/// `null_decoder_ids` is given (exactly one must be). `sources` maps a
/// schedule's source_id to the syndrome_source instance it reads from.
run_result run_schedule(
    const std::string &schedule_text, std::uint64_t tick_ns,
    const std::unordered_map<std::uint32_t, syndrome_source *> &sources,
    const std::optional<cudaq::qec::decoding::config::multi_decoder_config>
        &decoders,
    const std::optional<std::unordered_map<std::uint64_t, std::string>>
        &udp_endpoints,
    std::uint32_t udp_timeout_ms,
    const std::optional<std::vector<std::uint64_t>> &null_decoder_ids,
    std::uint64_t lead_in_ns) {
  if (int(decoders.has_value()) + int(udp_endpoints.has_value()) +
          int(null_decoder_ids.has_value()) !=
      1)
    throw std::invalid_argument(
        "run: specify exactly one of decoders=, udp_endpoints=, or "
        "null_decoder_ids= to select the session backend");

  std::vector<std::unique_ptr<session>> owned_sessions;
  std::unordered_map<std::uint64_t, session *> router;
  auto adopt = [&](std::vector<std::pair<std::uint64_t, std::unique_ptr<session>>> sessions) {
    for (auto &[id, s] : sessions) {
      router[id] = s.get();
      owned_sessions.push_back(std::move(s));
    }
  };

  if (decoders) {
    adopt(make_inproc_sessions(*decoders));
  } else if (udp_endpoints) {
    adopt(make_udp_sessions(*udp_endpoints, udp_timeout_ms));
  } else {
    // One session per decoder_id, same as the other backends -- each
    // decoder dispatches on its own thread, so sharing one instance
    // across decoder_ids is never safe.
    for (auto id : *null_decoder_ids) {
      auto null_sess = make_null_session();
      router[id] = null_sess.get();
      owned_sessions.push_back(std::move(null_sess));
    }
  }

  std::vector<std::uint64_t> known_decoder_ids;
  known_decoder_ids.reserve(router.size());
  for (auto &[id, _] : router)
    known_decoder_ids.push_back(id);

  run_params params;
  params.lead_in_ns = lead_in_ns;

  auto sched = parse(schedule_text, known_decoder_ids, tick_ns);
  auto run_plan_ = plan(sched, router, sources, params);

  return run(std::move(run_plan_));
}

} // namespace

void bindPlaybackEmulator(nb::module_ &mod) {
  auto m = mod.def_submodule("playback",
                             "Playback emulator: replay a pre-recorded RPC "
                             "schedule against decoders on a precise, "
                             "hardware-independent timing loop.");

  nb::enum_<operation>(m, "operation")
      .value("reset", operation::reset)
      .value("stream", operation::stream)
      .value("enqueue_data", operation::enqueue_data)
      .value("get_corrections", operation::get_corrections);

  nb::class_<record>(m, "record")
      .def_ro("event_index", &record::event_index)
      .def_ro("decoder_id", &record::decoder_id)
      .def_ro("op", &record::op)
      .def_ro("dispatched", &record::dispatched,
             "True once the dispatch loop actually reached this event; "
             "false if a hard error aborted the run first, in which case "
             "every other field is left default.")
      .def_ro("deadline_ns", &record::deadline_ns)
      .def_ro("call_ns", &record::call_ns)
      .def_ro("return_ns", &record::return_ns)
      .def_prop_ro("status",
                   [](const record &r) {
                     // The two status spaces are disjoint by value
                     // (RpcStatus 0..7, stream_terminate 100..103), so the
                     // value picks the enum -- not the op, since a dry
                     // source gives even an enqueue a SOURCE_EXHAUSTED.
                     if (r.status == kNoStatus)
                       return "NOT_DISPATCHED";
                     return r.status >= 100
                                ? to_string(static_cast<stream_terminate>(r.status))
                                : to_string(static_cast<RpcStatus>(r.status));
                   },
                   "status as a human-readable string: a stream_terminate "
                   "name for stream, an RpcStatus name for every other op, "
                   "or NOT_DISPATCHED for an event an abort pre-empted.")
      .def_ro("rounds_streamed", &record::rounds_streamed)
      .def_ro("read_completed", &record::read_completed)
      .def_ro("syndrome_offset", &record::syndrome_offset)
      .def_ro("syndrome_count", &record::syndrome_count)
      .def_ro("correction_offset", &record::correction_offset)
      .def_ro("correction_count", &record::correction_count)
      .def_ro("correction_mismatch", &record::correction_mismatch)
      .def_ro("request_id_offset", &record::request_id_offset)
      .def_ro("request_id_count", &record::request_id_count,
             "How many RPCs this event sent: one per round for a stream, one "
             "for every other op, and zero if it sent nothing. Together with "
             "request_id_offset this slices run_result.request_id_log; "
             "run_result.request_ids() does the slicing for you.");

  nb::class_<run_result>(m, "run_result")
      .def_ro("records", &run_result::records)
      .def_ro("syndrome_log", &run_result::syndrome_log)
      .def_ro("correction_log", &run_result::correction_log)
      .def_ro("request_id_log", &run_result::request_id_log)
      .def("request_ids",
          [](const run_result &r, std::size_t event_index) {
            const auto &rec = r.records.at(event_index);
            const auto first = std::min<std::size_t>(rec.request_id_offset,
                                                     r.request_id_log.size());
            const auto last = std::min<std::size_t>(
                first + rec.request_id_count, r.request_id_log.size());
            return std::vector<std::uint32_t>(r.request_id_log.begin() + first,
                                              r.request_id_log.begin() + last);
          },
          nb::arg("event_index"),
          "That event's slice of request_id_log: every request_id it put on "
          "the wire, in send order, for correlating against a server log.")
      .def_ro("warnings", &run_result::warnings)
      .def_ro("t0_ns", &run_result::t0_ns)
      .def_ro("tick_ns", &run_result::tick_ns)
      .def("write_csv", [](const run_result &r) { return write_csv(r); },
          "Serialize this run's records to a CSV string.");

  nb::class_<syndrome_source>(m, "syndrome_source");

  nb::class_<static_source, syndrome_source>(m, "static_source")
      .def(nb::init<std::vector<std::vector<std::uint8_t>>>(), nb::arg("rounds"))
      .def("reset", &static_source::reset);

  nb::class_<stim_memory_source, syndrome_source>(m, "stim_memory_source")
      .def(nb::init<std::string, std::uint64_t>(), nb::arg("stim_circuit_text"),
          nb::arg("seed"))
      .def("reset", &stim_memory_source::reset);

  m.def(
      "run", &run_schedule, nb::arg("schedule"), nb::arg("tick_ns"),
      nb::arg("sources"), nb::arg("decoders") = nb::none(),
      nb::arg("udp_endpoints") = nb::none(), nb::arg("udp_timeout_ms") = 200,
      nb::arg("null_decoder_ids") = nb::none(),
      nb::arg("lead_in_ns") = 20'000'000,
      "Parse, plan, and run a line-oriented playback schedule. `sources` "
      "maps a schedule's source_id -> syndrome_source. Exactly one of "
      "`decoders` (in-process decoders from a multi_decoder_config), "
      "`udp_endpoints` ({decoder_id: \"host:port\"}), or "
      "`null_decoder_ids` (the discard-everything jitter floor, routed "
      "under the given decoder_ids) selects the session backend.");
}

} // namespace cudaq::qec::playback
