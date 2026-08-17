/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file parser.cpp
/// @brief parse(): Line-oriented. Unknown operation, malformed bit string,
/// or decoder_id absent from the config is a parse error 

#include "cudaq/qec/playback/emulator.h"

#include <algorithm>
#include <charconv>
#include <sstream>
#include <type_traits>
#include <unordered_set>

namespace cudaq::qec::playback {

namespace {

/// Splits on runs of whitespace; '#' starts a comment that runs to EOL.
std::vector<std::string> tokenize(const std::string &line) {
  std::string trimmed = line;
  if (auto hash = trimmed.find('#'); hash != std::string::npos)
    trimmed.resize(hash);
  std::vector<std::string> tokens;
  std::istringstream iss(trimmed);
  std::string tok;
  while (iss >> tok)
    tokens.push_back(tok);
  return tokens;
}

bool is_digits(const std::string &s) {
  return !s.empty() && std::all_of(s.begin(), s.end(), [](unsigned char c) {
           return std::isdigit(c);
         });
}

/// Parses `s` as a base-10 `T` (T is uint32_t or uint64_t), rejecting
/// anything non-digit or too wide for T 
template <typename T>
T parse_uint(const std::string &s, const char *what) {
  static_assert(std::is_same_v<T, std::uint32_t> || std::is_same_v<T, std::uint64_t>);
  if (!is_digits(s))
    throw std::invalid_argument(std::string(what) + " '" + s +
                                "' is not a valid integer");
  T v = 0;
  const auto end = s.data() + s.size();
  const auto res = std::from_chars(s.data(), end, v);
  if (res.ec != std::errc() || res.ptr != end)
    throw std::invalid_argument(std::string(what) + " '" + s + "' does not fit in " +
                                (sizeof(T) == 4 ? "32" : "64") + " bits");
  return v;
}

std::uint32_t parse_u32(const std::string &s, const char *what) {
  return parse_uint<std::uint32_t>(s, what);
}

std::uint64_t parse_u64(const std::string &s, const char *what) {
  return parse_uint<std::uint64_t>(s, what);
}

/// Bit-string token: only '0'/'1' characters. Appends one byte (0x00/0x01)
/// per character to `arena`
void append_bits(std::vector<std::uint8_t> &arena, const std::string &s,
                 const char *what) {
  for (char c : s) {
    if (c != '0' && c != '1')
      throw std::invalid_argument(std::string("malformed ") + what +
                                  " bit string '" + s +
                                  "': expected only '0'/'1'");
    arena.push_back(static_cast<std::uint8_t>(c - '0'));
  }
}

/// True if `s` is a "key=value" token with the given key.
bool key_is(const std::string &s, std::string_view key, std::string &value) {
  auto eq = s.find('=');
  if (eq == std::string::npos || s.compare(0, eq, key) != 0)
    return false;
  value = s.substr(eq + 1);
  return true;
}

} // namespace

schedule parse(std::string_view text,
               const std::vector<std::uint64_t> &known_decoder_ids,
               std::uint64_t tick_ns) {
  schedule sched;
  sched.tick_ns = tick_ns;

  std::unordered_set<std::uint64_t> known(known_decoder_ids.begin(),
                                          known_decoder_ids.end());
  for (auto id : known_decoder_ids)
    sched.decoders.push_back(id);

  std::uint64_t last_tick = 0;
  bool have_last_tick = false;

  std::istringstream stream{std::string(text)};
  std::string raw_line;
  std::size_t line_number = 0;
  while (std::getline(stream, raw_line)) {
    ++line_number;
    auto tokens = tokenize(raw_line);
    if (tokens.empty())
      continue; // blank or comment-only line

    try {
      if (tokens.size() < 3)
        throw std::invalid_argument(
            "expected '<tick> <decoder_id> <operation> [operands...]', got '" +
            raw_line + "'");

      const std::uint64_t tick = parse_u64(tokens[0], "tick");
      if (have_last_tick && tick < last_tick)
        throw std::invalid_argument(
            "ticks must be non-decreasing (saw " + std::to_string(tick) +
            " after " + std::to_string(last_tick) + ")");
      last_tick = tick;
      have_last_tick = true;

      std::uint64_t deadline_ns = 0;
      if (tick != 0 && __builtin_mul_overflow(tick, tick_ns, &deadline_ns))
        throw std::invalid_argument(
            "tick " + tokens[0] +
            " * tick_ns overflows a 64-bit nanosecond offset");

      const std::uint64_t decoder_id = parse_u64(tokens[1], "decoder_id");
      if (!known.contains(decoder_id))
        throw std::invalid_argument("decoder_id " + tokens[1] +
                                    " is not present in the decoder config");

      event e;
      e.deadline_ns = deadline_ns;
      e.decoder_id = decoder_id;

      const std::string &op_name = tokens[2];
      std::vector<std::string> operands(tokens.begin() + 3, tokens.end());

      if (op_name == "reset") {
        e.op = operation::reset;
        if (!operands.empty())
          throw std::invalid_argument("'reset' takes no operands");
      } else if (op_name == "enqueue") {
        e.op = operation::enqueue;
        if (operands.size() != 1)
          throw std::invalid_argument(
              "'enqueue' takes exactly one operand (bits or source=N)");
        std::string value;
        if (key_is(operands[0], "source", value)) {
          e.source_id = parse_u32(value, "source");
        } else {
          e.syndrome_offset = static_cast<std::uint32_t>(sched.syndrome_arena.size());
          append_bits(sched.syndrome_arena, operands[0], "enqueue");
          e.syndrome_count = static_cast<std::uint32_t>(operands[0].size());
        }
      } else if (op_name == "enqueue_data") {
        e.op = operation::enqueue_data;
        std::string value;
        if (operands.size() != 1 || !key_is(operands[0], "source", value))
          throw std::invalid_argument(
              "'enqueue_data' takes exactly one operand, source=N -- a shot "
              "boundary has nothing to read out without a source");
        e.source_id = parse_u32(value, "source");
      } else if (op_name == "get_corrections") {
        e.op = operation::get_corrections;
        if (operands.size() > 1)
          throw std::invalid_argument(
              "'get_corrections' takes at most one operand (expected bits)");
        if (!operands.empty()) {
          e.expected_offset = static_cast<std::uint32_t>(sched.expected_arena.size());
          append_bits(sched.expected_arena, operands[0], "get_corrections expected");
          e.expected_count = static_cast<std::uint32_t>(operands[0].size());
        }
      } else if (op_name == "stream_until") {
        e.op = operation::stream_until;
        bool have_source = false;
        // Paced at the schedule's tick unless an explicit every= says
        // otherwise 
        e.stream_every_ticks = 1;
        for (const auto &tok : operands) {
          std::string value;
          if (key_is(tok, "source", value)) {
            e.source_id = parse_u32(value, "source");
            have_source = true;
          } else if (key_is(tok, "every", value)) {
            e.stream_every_ticks = parse_u64(value, "every");
          } else if (key_is(tok, "max_rounds", value)) {
            e.stream_max_rounds = parse_u32(value, "max_rounds");
          } else if (std::all_of(tok.begin(), tok.end(), [](unsigned char c) {
                       return c == '0' || c == '1';
                     })) {
            e.expected_offset = static_cast<std::uint32_t>(sched.expected_arena.size());
            append_bits(sched.expected_arena, tok, "stream_until expected");
            e.expected_count = static_cast<std::uint32_t>(tok.size());
          } else {
            throw std::invalid_argument("unrecognized 'stream_until' operand '" +
                                        tok + "'");
          }
        }
        if (!have_source)
          throw std::invalid_argument("'stream_until' requires 'source=N'");
      } else {
        throw std::invalid_argument("unknown operation '" + op_name + "'");
      }

      sched.events.push_back(e);
    } catch (const std::invalid_argument &ex) {
      throw std::invalid_argument("playback schedule, line " +
                                  std::to_string(line_number) + ": " + ex.what());
    }
  }

  std::stable_sort(sched.events.begin(), sched.events.end(),
                   [](const event &a, const event &b) {
                     return a.deadline_ns < b.deadline_ns;
                   });

  return sched;
}

} // namespace cudaq::qec::playback
