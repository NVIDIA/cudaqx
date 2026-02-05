/*
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.
 * All rights reserved.
 *
 * This source code and the accompanying materials are made available under
 * the terms of the Apache License 2.0 which accompanies this distribution.
 */

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <hololink/core/data_channel.hpp>
#include <hololink/core/enumerator.hpp>
#include <hololink/core/hololink.hpp>
#include <hololink/core/metadata.hpp>
#include <hololink/core/timeout.hpp>

#include "cudaq/nvqlink/daemon/dispatcher/dispatch_kernel_launch.h"

namespace {

constexpr std::uint32_t PLAYER_ADDR = 0x5000'0000;
constexpr std::uint32_t RAM_ADDR = 0x5010'0000;
constexpr std::uint32_t PLAYER_TIMER_OFFSET = 0x0008;
constexpr std::uint32_t PLAYER_WINDOW_SIZE_OFFSET = 0x000C;
constexpr std::uint32_t PLAYER_WINDOW_NUMBER_OFFSET = 0x0010;
constexpr std::uint32_t PLAYER_ENABLE_OFFSET = 0x0004;
constexpr std::uint32_t RAM_NUM = 16;
constexpr std::uint32_t RAM_DEPTH = 512;

constexpr std::uint32_t PLAYER_ENABLE = 0x0000'0001;
constexpr std::uint32_t PLAYER_DISABLE = 0x0000'0000;

constexpr std::uint32_t DEFAULT_TIMER_SPACING_US = 10;
constexpr std::uint32_t RF_SOC_TIMER_SCALE = 322;
constexpr std::uint32_t OTHER_TIMER_SCALE = 201;

constexpr std::uint32_t MOCK_DECODE_FUNCTION_ID =
    cudaq::nvqlink::fnv1a_hash("mock_decode");

std::size_t align_up(std::size_t value, std::size_t alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

uint32_t load_le_u32(const uint8_t *bytes) {
  uint32_t value = 0;
  std::memcpy(&value, bytes, sizeof(value));
  return value;
}

uint64_t parse_scalar(const std::string &content,
                      const std::string &field_name) {
  std::size_t pos = content.find(field_name + ":");
  if (pos == std::string::npos)
    return 0;

  pos = content.find(':', pos);
  if (pos == std::string::npos)
    return 0;

  std::size_t end_pos = content.find_first_of("\n[", pos + 1);
  if (end_pos == std::string::npos)
    end_pos = content.length();

  std::string value_str = content.substr(pos + 1, end_pos - pos - 1);
  value_str.erase(0, value_str.find_first_not_of(" \t\n\r"));
  value_str.erase(value_str.find_last_not_of(" \t\n\r") + 1);

  try {
    return std::stoull(value_str);
  } catch (...) {
    return 0;
  }
}

struct SyndromeEntry {
  std::vector<uint8_t> measurements;
  uint8_t expected_correction;
};

std::vector<SyndromeEntry> load_syndromes(const std::string &path,
                                          std::size_t syndrome_size) {
  std::vector<SyndromeEntry> entries;
  std::ifstream file(path);
  if (!file.good())
    return entries;

  std::string line;
  std::vector<uint8_t> current_shot;
  std::vector<std::vector<uint8_t>> shots;
  std::vector<uint8_t> corrections;
  bool reading_shot = false;
  bool reading_corrections = false;

  while (std::getline(file, line)) {
    if (line.find("SHOT_START") == 0) {
      if (reading_shot && !current_shot.empty()) {
        shots.push_back(current_shot);
      }
      current_shot.clear();
      reading_shot = true;
      reading_corrections = false;
      continue;
    }
    if (line == "CORRECTIONS_START") {
      if (reading_shot && !current_shot.empty()) {
        shots.push_back(current_shot);
      }
      current_shot.clear();
      reading_shot = false;
      reading_corrections = true;
      continue;
    }
    if (line == "CORRECTIONS_END") {
      break;
    }
    if (line.find("NUM_DATA") == 0 || line.find("NUM_LOGICAL") == 0) {
      continue;
    } else if (reading_shot) {
      line.erase(0, line.find_first_not_of(" \t\n\r"));
      line.erase(line.find_last_not_of(" \t\n\r") + 1);
      if (line.empty())
        continue;
      try {
        int bit = std::stoi(line);
        current_shot.push_back(static_cast<uint8_t>(bit));
      } catch (...) {
      }
    } else if (reading_corrections) {
      line.erase(0, line.find_first_not_of(" \t\n\r"));
      line.erase(line.find_last_not_of(" \t\n\r") + 1);
      if (line.empty())
        continue;
      try {
        int bit = std::stoi(line);
        corrections.push_back(static_cast<uint8_t>(bit));
      } catch (...) {
      }
    }
  }

  if (reading_shot && !current_shot.empty()) {
    shots.push_back(current_shot);
  }

  for (std::size_t i = 0; i < shots.size(); ++i) {
    if (shots[i].size() < syndrome_size) {
      shots[i].resize(syndrome_size, 0);
    } else if (shots[i].size() > syndrome_size) {
      shots[i].resize(syndrome_size);
    }
    SyndromeEntry entry{};
    entry.measurements = std::move(shots[i]);
    entry.expected_correction = (i < corrections.size()) ? corrections[i] : 0;
    entries.push_back(std::move(entry));
  }

  return entries;
}

std::vector<uint8_t>
build_rpc_payload(const std::vector<uint8_t> &measurements) {
  std::vector<uint8_t> payload(sizeof(cudaq::nvqlink::RPCHeader) +
                               measurements.size());

  auto *header = reinterpret_cast<cudaq::nvqlink::RPCHeader *>(payload.data());
  header->magic = cudaq::nvqlink::RPC_MAGIC_REQUEST;
  header->function_id = MOCK_DECODE_FUNCTION_ID;
  header->arg_len = static_cast<std::uint32_t>(measurements.size());

  std::memcpy(payload.data() + sizeof(cudaq::nvqlink::RPCHeader),
              measurements.data(), measurements.size());
  return payload;
}

struct Options {
  std::string hololink_ip;
  std::string data_dir = "libs/qec/unittests/decoders/realtime/data";
  std::string config_path;
  std::string syndromes_path;
  std::optional<std::string> uuid;
  std::uint32_t total_sensors = 1;
  std::uint32_t total_dataplanes = 1;
  std::uint32_t sifs_per_sensor = 2;
  std::optional<std::size_t> window_number;
  std::optional<std::size_t> frame_size;
  std::optional<std::uint32_t> mtu;
  bool skip_reset = false;
  bool ptp_sync = false;
  std::optional<std::uint32_t> timer_value;
  std::uint32_t timer_spacing_us = DEFAULT_TIMER_SPACING_US;
  bool board_is_rfsoc = true;
};

void print_usage(const char *argv0) {
  std::cerr
      << "Usage: " << argv0 << " --hololink <ip> [options]\n"
      << "Options:\n"
      << "  --data-dir <path>           Base data directory\n"
      << "  --config <path>             Config YAML path\n"
      << "  --syndromes <path>          Syndromes text path\n"
      << "  --window-number <n>         Number of windows to program\n"
      << "  --frame-size <bytes>        Frame size per window (must be >= "
         "payload, "
         "multiple of 64)\n"
      << "  --timer-value <ticks>       Raw timer value (overrides "
         "board/spacing)\n"
      << "  --timer-spacing-us <us>     Spacing for timer calculation\n"
      << "  --board <RFSoC|Other>       Board type for timer calculation\n"
      << "  --mtu <bytes>               Suggest MTU in enumeration metadata\n"
      << "  --uuid <uuid>               Set explicit UUID enumeration "
         "strategy\n"
      << "  --total-sensors <n>         Enumeration strategy sensors\n"
      << "  --total-dataplanes <n>      Enumeration strategy dataplanes\n"
      << "  --sifs-per-sensor <n>       Enumeration strategy SIFs per sensor\n"
      << "  --skip-reset                Skip hololink reset\n"
      << "  --ptp-sync                  Wait for PTP synchronization\n";
}

Options parse_args(int argc, char **argv) {
  Options options;
  std::map<std::string, std::string> kv;
  std::vector<std::string> flags;

  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    if (arg.rfind("--", 0) != 0) {
      continue;
    }
    auto eq = arg.find('=');
    if (eq != std::string::npos) {
      kv[arg.substr(2, eq - 2)] = arg.substr(eq + 1);
      continue;
    }
    std::string key = arg.substr(2);
    if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
      kv[key] = argv[++i];
    } else {
      flags.push_back(key);
    }
  }

  if (kv.count("hololink"))
    options.hololink_ip = kv["hololink"];
  if (kv.count("data-dir"))
    options.data_dir = kv["data-dir"];
  if (kv.count("config"))
    options.config_path = kv["config"];
  if (kv.count("syndromes"))
    options.syndromes_path = kv["syndromes"];
  if (kv.count("window-number"))
    options.window_number = std::stoull(kv["window-number"]);
  if (kv.count("frame-size"))
    options.frame_size = std::stoull(kv["frame-size"]);
  if (kv.count("mtu"))
    options.mtu = static_cast<std::uint32_t>(std::stoul(kv["mtu"]));
  if (kv.count("uuid"))
    options.uuid = kv["uuid"];
  if (kv.count("total-sensors"))
    options.total_sensors =
        static_cast<std::uint32_t>(std::stoul(kv["total-sensors"]));
  if (kv.count("total-dataplanes"))
    options.total_dataplanes =
        static_cast<std::uint32_t>(std::stoul(kv["total-dataplanes"]));
  if (kv.count("sifs-per-sensor"))
    options.sifs_per_sensor =
        static_cast<std::uint32_t>(std::stoul(kv["sifs-per-sensor"]));
  if (kv.count("timer-value"))
    options.timer_value =
        static_cast<std::uint32_t>(std::stoul(kv["timer-value"]));
  if (kv.count("timer-spacing-us"))
    options.timer_spacing_us =
        static_cast<std::uint32_t>(std::stoul(kv["timer-spacing-us"]));
  if (kv.count("board")) {
    std::string board = kv["board"];
    options.board_is_rfsoc = (board == "RFSoC" || board == "rfsoc");
  }

  for (const auto &flag : flags) {
    if (flag == "skip-reset")
      options.skip_reset = true;
    if (flag == "ptp-sync")
      options.ptp_sync = true;
  }

  return options;
}

uint32_t compute_timer_value(const Options &options) {
  if (options.timer_value)
    return *options.timer_value;
  uint32_t scale =
      options.board_is_rfsoc ? RF_SOC_TIMER_SCALE : OTHER_TIMER_SCALE;
  return scale * options.timer_spacing_us;
}

void write_bram(hololink::Hololink &hololink,
                const std::vector<std::vector<uint8_t>> &windows,
                std::size_t bytes_per_window) {
  if (bytes_per_window % 64 != 0) {
    throw std::runtime_error("bytes_per_window must be a multiple of 64");
  }

  std::size_t cycles = bytes_per_window / 64;
  if (cycles == 0) {
    throw std::runtime_error("bytes_per_window is too small");
  }

  if (windows.size() * cycles > RAM_DEPTH) {
    std::ostringstream msg;
    msg << "Requested " << windows.size() << " windows with " << cycles
        << " cycles each exceeds RAM depth " << RAM_DEPTH
        << ". Reduce --window-number or --frame-size.";
    throw std::runtime_error(msg.str());
  }

  uint32_t w_sample_addr = 0;
  while ((1u << w_sample_addr) < RAM_DEPTH) {
    ++w_sample_addr;
  }
  if ((1u << w_sample_addr) != RAM_DEPTH) {
    throw std::runtime_error("RAM_DEPTH must be a power of two");
  }

  constexpr std::size_t kBatchWrites = 4096;
  hololink::Hololink::WriteData write_data;

  for (std::size_t w = 0; w < windows.size(); ++w) {
    const auto &window = windows[w];
    for (std::size_t s = 0; s < cycles; ++s) {
      for (std::size_t i = 0; i < RAM_NUM; ++i) {
        std::size_t word_index = s * RAM_NUM + i;
        std::size_t byte_offset = word_index * sizeof(uint32_t);
        uint32_t value = 0;
        if (byte_offset + sizeof(uint32_t) <= window.size()) {
          value = load_le_u32(window.data() + byte_offset);
        }

        uint32_t ram_addr = static_cast<uint32_t>(i << (w_sample_addr + 2));
        uint32_t sample_addr = static_cast<uint32_t>((s + (w * cycles)) * 0x4);
        uint32_t address = RAM_ADDR + ram_addr + sample_addr;

        write_data.queue_write_uint32(address, value);
        if (write_data.size() >= kBatchWrites) {
          if (!hololink.write_uint32(write_data)) {
            throw std::runtime_error("Failed to write BRAM batch");
          }
          write_data = hololink::Hololink::WriteData();
        }
      }
    }
  }

  if (write_data.size() > 0) {
    if (!hololink.write_uint32(write_data)) {
      throw std::runtime_error("Failed to write BRAM batch");
    }
  }
}

} // namespace

int main(int argc, char **argv) {
  Options options = parse_args(argc, argv);
  if (options.hololink_ip.empty()) {
    print_usage(argv[0]);
    return 1;
  }

  std::string config_path =
      options.config_path.empty()
          ? (options.data_dir + "/config_multi_err_lut.yml")
          : options.config_path;
  std::string syndromes_path =
      options.syndromes_path.empty()
          ? (options.data_dir + "/syndromes_multi_err_lut.txt")
          : options.syndromes_path;

  std::ifstream config_file(config_path);
  if (!config_file.good()) {
    std::cerr << "Could not open config file: " << config_path << "\n";
    return 1;
  }
  std::string config_content((std::istreambuf_iterator<char>(config_file)),
                             std::istreambuf_iterator<char>());

  std::size_t syndrome_size = parse_scalar(config_content, "syndrome_size");
  if (syndrome_size == 0) {
    std::cerr << "Invalid syndrome_size in config file\n";
    return 1;
  }

  auto syndromes = load_syndromes(syndromes_path, syndrome_size);
  if (syndromes.empty()) {
    std::cerr << "No syndrome data loaded from " << syndromes_path << "\n";
    return 1;
  }

  std::vector<std::vector<uint8_t>> windows;
  windows.reserve(syndromes.size());

  for (const auto &entry : syndromes) {
    windows.push_back(build_rpc_payload(entry.measurements));
  }

  std::size_t payload_size = windows.front().size();
  std::size_t bytes_per_window = align_up(payload_size, 64);
  if (options.frame_size) {
    bytes_per_window = *options.frame_size;
    if (bytes_per_window < payload_size || bytes_per_window % 64 != 0) {
      std::cerr << "--frame-size must be >= payload and a multiple of 64\n";
      return 1;
    }
  }

  for (auto &window : windows) {
    window.resize(bytes_per_window, 0);
  }

  std::size_t window_number =
      options.window_number ? *options.window_number : windows.size();
  if (window_number == 0 || window_number > windows.size()) {
    std::cerr << "Invalid --window-number; must be in [1, " << windows.size()
              << "]\n";
    return 1;
  }
  windows.resize(window_number);

  if (options.uuid) {
    hololink::Metadata additional_metadata;
    auto strategy = std::make_shared<hololink::BasicEnumerationStrategy>(
        additional_metadata, options.total_sensors, options.total_dataplanes,
        options.sifs_per_sensor);
    hololink::Enumerator::set_uuid_strategy(*options.uuid, strategy);
  }

  auto channel_metadata =
      hololink::Enumerator::find_channel(options.hololink_ip);
  hololink::DataChannel::use_sensor(channel_metadata, 0);
  if (options.mtu) {
    hololink::DataChannel::use_mtu(channel_metadata, *options.mtu);
  }
  hololink::DataChannel hololink_channel(channel_metadata);
  auto hololink = hololink_channel.hololink();

  hololink->start();
  if (!options.skip_reset) {
    hololink->reset();
  }
  if (options.ptp_sync) {
    auto timeout = std::make_shared<hololink::Timeout>(10.0f);
    hololink->ptp_synchronize(timeout);
  }

  if (!hololink->write_uint32(PLAYER_ADDR + PLAYER_ENABLE_OFFSET,
                              PLAYER_DISABLE)) {
    throw std::runtime_error("Failed to disable player");
  }

  hololink::Hololink::WriteData config_write;
  config_write.queue_write_uint32(PLAYER_ADDR + PLAYER_WINDOW_SIZE_OFFSET,
                                  static_cast<uint32_t>(bytes_per_window));
  config_write.queue_write_uint32(PLAYER_ADDR + PLAYER_WINDOW_NUMBER_OFFSET,
                                  static_cast<uint32_t>(window_number));
  config_write.queue_write_uint32(PLAYER_ADDR + PLAYER_TIMER_OFFSET,
                                  compute_timer_value(options));
  if (!hololink->write_uint32(config_write)) {
    throw std::runtime_error("Failed to configure player");
  }

  write_bram(*hololink, windows, bytes_per_window);

  if (!hololink->write_uint32(PLAYER_ADDR + PLAYER_ENABLE_OFFSET,
                              PLAYER_ENABLE)) {
    throw std::runtime_error("Failed to enable player");
  }

  std::cout << "Programmed " << window_number << " windows ("
            << bytes_per_window << " bytes each) into FPGA BRAM\n";
  std::cout << "Playback enabled on hololink " << options.hololink_ip << "\n";

  return 0;
}
