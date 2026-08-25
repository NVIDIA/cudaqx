/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file playback_emulator_main.cpp
/// @brief CLI entry point for the playback emulator ("callable
/// both from the CLI and from a Python binding"). Hand-parsed `--flag=value`
/// arguments, matching this repo's convention (see
/// tools/decoding-server/decoding_server.cpp) rather than pulling in a CLI
/// argument-parsing dependency.

#include "session.h"
#include "emulator.h"
#include "syndrome_source.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>

using namespace cudaq::qec::playback;

namespace {

bool starts_with(const std::string &s, const char *prefix) {
  return s.rfind(prefix, 0) == 0;
}

std::string read_file(const std::string &path) {
  std::ifstream in(path);
  if (!in)
    throw std::runtime_error("cannot open file: " + path);
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

void print_usage() {
  std::cout <<
      "Usage: playback-emulator --config=<decoders.yaml> --schedule=<file> "
      "[options]\n"
      "\n"
      "Required:\n"
      "  --config=PATH       multi_decoder_config YAML (decoders section)\n"
      "  --schedule=PATH     playback schedule text, one\n"
      "                      '<trigger> <op> [key=value...]' per line\n"
      "\n"
      "Options:\n"
      "  --tick=DURATION     wall-clock width of one tick (default: 1us)\n"
      "  --backend=NAME      null | inproc | udp (default: inproc)\n"
      "  --udp-endpoint=ID:HOST:PORT   repeatable; required for --backend=udp\n"
      "  --source=ID:PATH    static_source for source_id ID, one 0/1 bit\n"
      "                      string per round, one round per line in PATH\n"
      "  --out=PATH          write the run's CSV here (default: stdout)\n"
      "  --help\n";
}

std::uint64_t parse_duration_ns(const std::string &s) {
  std::size_t split = s.size();
  while (split > 0 && !std::isdigit(static_cast<unsigned char>(s[split - 1])))
    --split;
  const std::uint64_t value = std::stoull(s.substr(0, split));
  const std::string unit = s.substr(split);
  if (unit.empty() || unit == "ns")
    return value;
  if (unit == "us")
    return value * 1'000;
  if (unit == "ms")
    return value * 1'000'000;
  if (unit == "s")
    return value * 1'000'000'000;
  throw std::runtime_error("unknown duration unit in '" + s + "'");
}

std::vector<std::vector<std::uint8_t>> read_rounds(const std::string &path) {
  std::vector<std::vector<std::uint8_t>> rounds;
  std::ifstream in(path);
  if (!in)
    throw std::runtime_error("cannot open source file: " + path);
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty())
      continue;
    std::vector<std::uint8_t> bits;
    bits.reserve(line.size());
    for (char c : line)
      if (c == '0' || c == '1')
        bits.push_back(static_cast<std::uint8_t>(c - '0'));
    if (!bits.empty())
      rounds.push_back(std::move(bits));
  }
  return rounds;
}

} // namespace

int main(int argc, char **argv) {
  std::string config_path, schedule_path, backend_name = "inproc";
  std::string out_path;
  std::uint64_t tick_ns = 1000;
  std::unordered_map<std::uint64_t, std::string> udp_endpoints;
  std::unordered_map<std::uint32_t, std::string> source_files;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--help") {
      print_usage();
      return 0;
    } else if (starts_with(a, "--config=")) {
      config_path = a.substr(9);
    } else if (starts_with(a, "--schedule=")) {
      schedule_path = a.substr(11);
    } else if (starts_with(a, "--tick=")) {
      tick_ns = parse_duration_ns(a.substr(7));
    } else if (starts_with(a, "--backend=")) {
      backend_name = a.substr(10);
    } else if (starts_with(a, "--out=")) {
      out_path = a.substr(6);
    } else if (starts_with(a, "--udp-endpoint=")) {
      const std::string v = a.substr(15);
      const auto first_colon = v.find(':');
      if (first_colon == std::string::npos)
        throw std::runtime_error("--udp-endpoint expects ID:HOST:PORT");
      const std::uint64_t id = std::stoull(v.substr(0, first_colon));
      udp_endpoints[id] = v.substr(first_colon + 1);
    } else if (starts_with(a, "--source=")) {
      const std::string v = a.substr(9);
      const auto colon = v.find(':');
      if (colon == std::string::npos)
        throw std::runtime_error("--source expects ID:PATH");
      const std::uint32_t id =
          static_cast<std::uint32_t>(std::stoul(v.substr(0, colon)));
      source_files[id] = v.substr(colon + 1);
    } else {
      std::cerr << "unrecognized argument: " << a << "\n";
      print_usage();
      return 1;
    }
  }

  if (config_path.empty() || schedule_path.empty()) {
    print_usage();
    return 1;
  }

  try {
    auto config =
        cudaq::qec::decoding::config::multi_decoder_config::from_yaml_str(
            read_file(config_path));

    std::vector<std::uint64_t> decoder_ids;
    for (const auto &d : config.decoders)
      decoder_ids.push_back(static_cast<std::uint64_t>(d.id));

    std::vector<std::pair<std::uint64_t, std::unique_ptr<session>>> owned_sessions;
    std::unordered_map<std::uint64_t, session *> router;

    if (backend_name == "null") {
      owned_sessions = make_null_sessions(decoder_ids);
    } else if (backend_name == "inproc") {
      owned_sessions = make_inproc_sessions(config);
    } else if (backend_name == "udp") {
      if (udp_endpoints.empty())
        throw std::runtime_error("--backend=udp requires at least one "
                                 "--udp-endpoint=ID:HOST:PORT");
      owned_sessions = make_udp_sessions(udp_endpoints);
    } else {
      throw std::runtime_error("unknown --backend='" + backend_name + "'");
    }
    route_sessions(owned_sessions, router);

    std::vector<std::unique_ptr<syndrome_source>> owned_sources;
    std::unordered_map<std::uint32_t, syndrome_source *> sources;
    for (const auto &[id, path] : source_files) {
      auto src = std::make_unique<static_source>(read_rounds(path));
      sources[id] = src.get();
      owned_sources.push_back(std::move(src));
    }

    auto sched = parse(read_file(schedule_path), decoder_ids, tick_ns);

    run_params params;
    auto p = plan(sched, router, sources, params);
    auto result = run(std::move(p));

    if (out_path.empty()) {
      write_csv(result, std::cout);
    } else {
      std::ofstream out(out_path);
      write_csv(result, out);
    }
  } catch (const std::exception &e) {
    std::cerr << "playback-emulator: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
