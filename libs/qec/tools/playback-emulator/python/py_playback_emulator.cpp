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

#include "cudaq/qec/code.h"
#include "cudaq/qec/experiments.h"
#include "cudaq/qec/playback/backends.h"
#include "cudaq/qec/playback/cudaq_memory_source.h"
#include "cudaq/qec/playback/emulator.h"
#include "cudaq/qec/playback/syndrome_source.h"

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
    std::uint64_t lead_in_ns, bool retry_not_ready,
    std::uint64_t not_ready_deadline_ns) {
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
    // decoder now dispatches on its own thread, so sharing one instance
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
  params.dispatch.retry_not_ready = retry_not_ready;
  params.dispatch.not_ready_deadline_ns = not_ready_deadline_ns;

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
      .value("enqueue", operation::enqueue)
      .value("enqueue_data", operation::enqueue_data)
      .value("get_corrections", operation::get_corrections)
      .value("stream_until", operation::stream_until);

  nb::enum_<stream_terminate>(m, "stream_terminate")
      .value("OK", stream_terminate::OK)
      .value("SOURCE_EXHAUSTED", stream_terminate::SOURCE_EXHAUSTED)
      .value("EXHAUSTED_ROUNDS", stream_terminate::EXHAUSTED_ROUNDS)
      .value("ERROR", stream_terminate::ERROR);

  nb::enum_<RpcStatus>(m, "RpcStatus")
      .value("OK", RpcStatus::OK)
      .value("INVALID_DECODER", RpcStatus::INVALID_DECODER)
      .value("BAD_REQUEST", RpcStatus::BAD_REQUEST)
      .value("INTERNAL_ERROR", RpcStatus::INTERNAL_ERROR)
      .value("NOT_READY", RpcStatus::NOT_READY)
      .value("BUSY", RpcStatus::BUSY)
      .value("SYNDROMES_DROPPED", RpcStatus::SYNDROMES_DROPPED);

  nb::class_<record>(m, "record")
      .def_ro("event_index", &record::event_index)
      .def_ro("decoder_id", &record::decoder_id)
      .def_ro("op", &record::op)
      .def_ro("dispatched", &record::dispatched,
             "True once this event's decoder thread actually reached and "
             "dispatched it; false if a hard error elsewhere aborted the "
             "run first, in which case every other field is left default.")
      .def_ro("deadline_ns", &record::deadline_ns)
      .def_ro("call_ns", &record::call_ns)
      .def_ro("return_ns", &record::return_ns)
      .def_prop_ro("status",
                   [](const record &r) {
                     // record::status is the wire RpcStatus for every op
                     // except stream_until, which has no wire RPC of its own
                     // and instead uses the disjoint stream_terminate range
                     // (see types.h's `record::status` comment).
                     return r.op == operation::stream_until
                                ? to_string(static_cast<stream_terminate>(r.status))
                                : to_string(static_cast<RpcStatus>(r.status));
                   },
                   "status as a human-readable string: an RpcStatus name for "
                   "every op except stream_until, whose status is a "
                   "stream_terminate name instead.")
      .def_ro("rounds_streamed", &record::rounds_streamed)
      .def_ro("read_completed", &record::read_completed)
      .def_ro("syndrome_offset", &record::syndrome_offset)
      .def_ro("syndrome_count", &record::syndrome_count)
      .def_ro("correction_offset", &record::correction_offset)
      .def_ro("correction_count", &record::correction_count)
      .def_ro("correction_mismatch", &record::correction_mismatch)
      .def_ro("first_request_id", &record::first_request_id);

  nb::class_<run_metadata>(m, "run_metadata")
      .def_ro("t0_ns", &run_metadata::t0_ns)
      .def_ro("tick_ns", &run_metadata::tick_ns)
      .def_ro("backend", &run_metadata::backend)
      .def_ro("spin_slack_ns", &run_metadata::spin_slack_ns);

  nb::class_<run_result>(m, "run_result")
      .def_ro("records", &run_result::records)
      .def_ro("syndrome_log", &run_result::syndrome_log)
      .def_ro("correction_log", &run_result::correction_log)
      .def_ro("warnings", &run_result::warnings)
      .def_ro("meta", &run_result::meta)
      .def("write_csv", [](const run_result &r) { return write_csv(r); },
          "Serialize this run's records to a CSV string.");

  nb::class_<syndrome_source>(m, "syndrome_source");

  nb::class_<static_source, syndrome_source>(m, "static_source")
      .def(nb::init<std::vector<std::vector<std::uint8_t>>>(), nb::arg("rounds"))
      .def("next_round", &static_source::next_round)
      .def("reset", &static_source::reset);

  nb::class_<stim_memory_source, syndrome_source>(m, "stim_memory_source")
      .def(nb::init<std::string, std::uint64_t>(), nb::arg("stim_circuit_text"),
          nb::arg("seed"))
      .def("next_round", &stim_memory_source::next_round)
      .def("read_data", &stim_memory_source::read_data)
      .def("reset", &stim_memory_source::reset)
      .def("is_streamed", &stim_memory_source::is_streamed)
      .def("round_width", &stim_memory_source::round_width)
      .def("data_width", &stim_memory_source::data_width);

  nb::class_<cudaq_memory_source, syndrome_source>(m, "cudaq_memory_source")
      .def(
          "__init__",
          [](cudaq_memory_source *self, const cudaq::qec::code &code,
             cudaq::qec::operation statePrep, std::size_t max_rounds,
             std::uint64_t seed,
             std::optional<cudaq::noise_model> noise) {
            new (self) cudaq_memory_source(
                code, statePrep, max_rounds,
                noise ? *noise : cudaq::noise_model(), seed);
          },
          nb::arg("code"), nb::arg("state_prep"), nb::arg("max_rounds"),
          nb::arg("seed"), nb::arg("noise") = nb::none(),
          "Streams syndrome rounds from CUDA-Q's own memory_circuit kernel "
          "(run under the stim target) rather than a Stim circuit -- see "
          "the C++ class doc comment for how it emulates round-by-round "
          "streaming despite the stim backend having no way to carry "
          "quantum state across separate kernel launches.")
      .def("next_round", &cudaq_memory_source::next_round)
      .def("read_data", &cudaq_memory_source::read_data)
      .def("reset", &cudaq_memory_source::reset)
      .def("is_streamed", &cudaq_memory_source::is_streamed)
      .def("max_rounds", &cudaq_memory_source::max_rounds)
      .def("round_width", &cudaq_memory_source::round_width)
      .def("data_width", &cudaq_memory_source::data_width);

  m.def(
      "run", &run_schedule, nb::arg("schedule"), nb::arg("tick_ns"),
      nb::arg("sources"), nb::arg("decoders") = nb::none(),
      nb::arg("udp_endpoints") = nb::none(), nb::arg("udp_timeout_ms") = 200,
      nb::arg("null_decoder_ids") = nb::none(),
      nb::arg("lead_in_ns") = 20'000'000, nb::arg("retry_not_ready") = false,
      nb::arg("not_ready_deadline_ns") = 5'000'000,
      "Parse, plan, and run a line-oriented playback schedule. `sources` "
      "maps a schedule's source_id -> syndrome_source. Exactly one of "
      "`decoders` (in-process decoders from a multi_decoder_config), "
      "`udp_endpoints` ({decoder_id: \"host:port\"}), or "
      "`null_decoder_ids` (the discard-everything jitter floor, routed "
      "under the given decoder_ids) selects the session backend.");
}

} // namespace cudaq::qec::playback
