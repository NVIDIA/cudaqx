/*
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.
 * All rights reserved.
 *
 * This source code and the accompanying materials are made available under
 * the terms of the Apache License 2.0 which accompanies this distribution.
 */

#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <hololink/core/data_channel.hpp>
#include <hololink/core/enumerator.hpp>
#include <hololink/core/hololink.hpp>
#include <hololink/core/metadata.hpp>
#include <hololink/core/timeout.hpp>

#include "cudaq/nvqlink/daemon/dispatcher/dispatch_kernel_launch.h"

namespace {

// ============================================================================
// Playback BRAM Constants
// ============================================================================
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

constexpr std::uint32_t MOCK_DECODE_FUNCTION_ID =
    cudaq::nvqlink::fnv1a_hash("mock_decode");

// ============================================================================
// ILA Capture Block Constants — from spec_sif_tx.json
//
// The SIF TX capture block at 0x4000_0000 records the full 512-bit incoming
// data bus.  Each captured sample is 521 bits wide:
//   bits [511:0]   sif_tx_axis_tdata_0  — 512-bit raw payload
//   bit  512       sif_tx_axis_tvalid_0 — valid indicator
//   bit  513       sif_tx_axis_tlast_0  — last beat of ethernet packet
//   bits [520:514] sif_ila_wr_tcnt_0    — valid byte count in this beat
//
// Samples are captured only on clock cycles where valid data is present.
// The status register at base+0x84 holds the current sample write address,
// allowing us to poll for the expected number of captured frames rather than
// waiting for the full buffer to fill.
// ============================================================================
constexpr std::uint32_t ILA_BASE_ADDR = 0x4000'0000;
constexpr std::uint32_t ILA_CTRL_OFFSET = 0x0000;
constexpr std::uint32_t ILA_STATUS_OFFSET = 0x0080;
constexpr std::uint32_t ILA_SAMPLE_ADDR_OFFSET = 0x0084;
constexpr std::uint32_t ILA_W_DATA = 521;
constexpr std::uint32_t ILA_DEPTH = 8192;
constexpr std::uint32_t ILA_NUM_RAM = (ILA_W_DATA + 31) / 32; // 17
constexpr std::uint32_t ILA_W_ADDR = 13;                      // log2(8192)
constexpr std::uint32_t ILA_W_RAM = 5;                        // ceil(log2(17))

constexpr std::uint32_t ILA_CTRL_ENABLE = 0x0000'0001;
constexpr std::uint32_t ILA_CTRL_RESET = 0x0000'0002;
constexpr std::uint32_t ILA_CTRL_DISABLE = 0x0000'0000;
constexpr std::uint32_t ILA_STATUS_DONE = 0x2;

// Signal bit positions within the 521-bit captured word.
constexpr std::uint32_t ILA_TVALID_BIT = 512;
constexpr std::uint32_t ILA_TLAST_BIT = 513;
constexpr std::uint32_t ILA_WR_TCNT_LSB = 514;
constexpr std::uint32_t ILA_WR_TCNT_WIDTH = 7;

// ============================================================================
// Helper Functions
// ============================================================================

std::size_t align_up(std::size_t value, std::size_t alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

std::uint32_t load_le_u32(const std::uint8_t *bytes) {
  std::uint32_t value = 0;
  std::memcpy(&value, bytes, sizeof(value));
  return value;
}

std::uint64_t parse_scalar(const std::string &content,
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

// ============================================================================
// Syndrome Data
// ============================================================================

struct SyndromeEntry {
  std::vector<std::uint8_t> measurements;
  std::uint8_t expected_correction;
};

std::vector<SyndromeEntry> load_syndromes(const std::string &path,
                                          std::size_t syndrome_size) {
  std::vector<SyndromeEntry> entries;
  std::ifstream file(path);
  if (!file.good())
    return entries;

  std::string line;
  std::vector<std::uint8_t> current_shot;
  std::vector<std::vector<std::uint8_t>> shots;
  std::vector<std::uint8_t> corrections;
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
        current_shot.push_back(static_cast<std::uint8_t>(bit));
      } catch (...) {
      }
    } else if (reading_corrections) {
      line.erase(0, line.find_first_not_of(" \t\n\r"));
      line.erase(line.find_last_not_of(" \t\n\r") + 1);
      if (line.empty())
        continue;
      try {
        int bit = std::stoi(line);
        corrections.push_back(static_cast<std::uint8_t>(bit));
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

std::vector<std::uint8_t>
build_rpc_payload(const std::vector<std::uint8_t> &measurements) {
  std::vector<std::uint8_t> payload(sizeof(cudaq::nvqlink::RPCHeader) +
                                    measurements.size());

  auto *header = reinterpret_cast<cudaq::nvqlink::RPCHeader *>(payload.data());
  header->magic = cudaq::nvqlink::RPC_MAGIC_REQUEST;
  header->function_id = MOCK_DECODE_FUNCTION_ID;
  header->arg_len = static_cast<std::uint32_t>(measurements.size());

  std::memcpy(payload.data() + sizeof(cudaq::nvqlink::RPCHeader),
              measurements.data(), measurements.size());
  return payload;
}

// ============================================================================
// Command-Line Options
// ============================================================================

struct Options {
  std::string hololink_ip;
  std::string data_dir;
  std::optional<std::size_t> num_shots;
  bool verify = false;

  // When set, bypass BOOTP enumeration and connect directly to this UDP port.
  // Used with the FPGA emulator which doesn't support BOOTP.
  std::optional<std::uint16_t> control_port;
  std::uint32_t vp_address = 0x1000;   // VP register base (sensor 0 default)
  std::uint32_t hif_address = 0x0800;  // HIF register base

  // RDMA target configuration (from bridge tool output).
  // When all three are provided, the FPGA SIF registers are configured
  // so playback data is sent via RDMA to the bridge tool's GPU buffers.
  std::optional<std::uint32_t> qp_number;
  std::optional<std::uint32_t> rkey;
  std::optional<std::uint64_t> buffer_addr;
  std::optional<std::uint32_t> rdma_page_size; // ring buffer slot size (bytes)
  std::optional<std::uint32_t> rdma_num_pages; // number of ring buffer slots
};

void print_usage(const char *argv0) {
  std::cerr
      << "Usage: " << argv0
      << " --hololink <ip> --data-dir <path> [options]\n\n"
      << "Required:\n"
      << "  --hololink <ip>       FPGA or emulator IP address\n"
      << "  --data-dir <path>     Path to syndrome data directory\n\n"
      << "Optional:\n"
      << "  --num-shots <n>       Number of shots to play back (default: all)\n"
      << "  --verify              Capture and verify correction responses "
         "via ILA\n\n"
      << "Emulator mode (bypass BOOTP enumeration):\n"
      << "  --control-port <n>   UDP control port of the emulator\n"
      << "  --vp-address <n>     VP register base (default: 0x1000)\n"
      << "  --hif-address <n>    HIF register base (default: 0x0800)\n\n"
      << "RDMA target (from bridge tool output):\n"
      << "  --qp-number <n>      Destination QP number (hex or decimal)\n"
      << "  --rkey <n>            Remote key\n"
      << "  --buffer-addr <n>    GPU buffer address (hex or decimal)\n"
      << "  --page-size <n>      Ring buffer slot size in bytes (default: "
         "256)\n"
      << "  --num-pages <n>      Number of ring buffer slots (default: 64)\n"
      << "\n"
      << "When --qp-number, --rkey, and --buffer-addr are all provided, the\n"
      << "FPGA SIF registers are configured to send playback data via RDMA\n"
      << "to the bridge tool's GPU buffers.\n";
}

Options parse_args(int argc, char **argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    if (arg == "--hololink" && i + 1 < argc) {
      options.hololink_ip = argv[++i];
    } else if (arg == "--data-dir" && i + 1 < argc) {
      options.data_dir = argv[++i];
    } else if (arg == "--num-shots" && i + 1 < argc) {
      options.num_shots = std::stoull(argv[++i]);
    } else if (arg == "--verify") {
      options.verify = true;
    } else if (arg == "--qp-number" && i + 1 < argc) {
      options.qp_number =
          static_cast<std::uint32_t>(std::stoul(argv[++i], nullptr, 0));
    } else if (arg == "--rkey" && i + 1 < argc) {
      options.rkey =
          static_cast<std::uint32_t>(std::stoul(argv[++i], nullptr, 0));
    } else if (arg == "--buffer-addr" && i + 1 < argc) {
      options.buffer_addr = std::stoull(argv[++i], nullptr, 0);
    } else if (arg == "--page-size" && i + 1 < argc) {
      options.rdma_page_size =
          static_cast<std::uint32_t>(std::stoul(argv[++i], nullptr, 0));
    } else if (arg == "--num-pages" && i + 1 < argc) {
      options.rdma_num_pages =
          static_cast<std::uint32_t>(std::stoul(argv[++i], nullptr, 0));
    } else if (arg == "--control-port" && i + 1 < argc) {
      options.control_port =
          static_cast<std::uint16_t>(std::stoul(argv[++i], nullptr, 0));
    } else if (arg == "--vp-address" && i + 1 < argc) {
      options.vp_address =
          static_cast<std::uint32_t>(std::stoul(argv[++i], nullptr, 0));
    } else if (arg == "--hif-address" && i + 1 < argc) {
      options.hif_address =
          static_cast<std::uint32_t>(std::stoul(argv[++i], nullptr, 0));
    } else if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      std::exit(0);
    }
  }
  return options;
}

// ============================================================================
// BRAM Write
// ============================================================================

/// Compute the address-width parameter for the playback BRAM (log2 of depth).
std::uint32_t bram_w_sample_addr() {
  std::uint32_t w = 0;
  while ((1u << w) < RAM_DEPTH)
    ++w;
  return w;
}

void write_bram(hololink::Hololink &hololink,
                const std::vector<std::vector<std::uint8_t>> &windows,
                std::size_t bytes_per_window) {
  if (bytes_per_window % 64 != 0)
    throw std::runtime_error("bytes_per_window must be a multiple of 64");

  std::size_t cycles = bytes_per_window / 64;
  if (cycles == 0)
    throw std::runtime_error("bytes_per_window is too small");

  if (windows.size() * cycles > RAM_DEPTH) {
    std::ostringstream msg;
    msg << "Requested " << windows.size() << " windows with " << cycles
        << " cycles each exceeds RAM depth " << RAM_DEPTH;
    throw std::runtime_error(msg.str());
  }

  const std::uint32_t w_sample_addr = bram_w_sample_addr();

  constexpr std::size_t kBatchWrites = 4096;
  hololink::Hololink::WriteData write_data;

  for (std::size_t w = 0; w < windows.size(); ++w) {
    const auto &window = windows[w];
    for (std::size_t s = 0; s < cycles; ++s) {
      for (std::size_t i = 0; i < RAM_NUM; ++i) {
        std::size_t word_index = s * RAM_NUM + i;
        std::size_t byte_offset = word_index * sizeof(std::uint32_t);
        std::uint32_t value = 0;
        if (byte_offset + sizeof(std::uint32_t) <= window.size()) {
          value = load_le_u32(window.data() + byte_offset);
        }

        auto ram_addr = static_cast<std::uint32_t>(i << (w_sample_addr + 2));
        auto sample_addr = static_cast<std::uint32_t>((s + (w * cycles)) * 0x4);
        std::uint32_t address = RAM_ADDR + ram_addr + sample_addr;

        write_data.queue_write_uint32(address, value);
        if (write_data.size() >= kBatchWrites) {
          if (!hololink.write_uint32(write_data))
            throw std::runtime_error("Failed to write BRAM batch");
          write_data = hololink::Hololink::WriteData();
        }
      }
    }
  }

  if (write_data.size() > 0) {
    if (!hololink.write_uint32(write_data))
      throw std::runtime_error("Failed to write BRAM batch");
  }
}

// ============================================================================
// BRAM Readback Verification
// ============================================================================

/// Read back playback BRAM using block reads (one per RAM bank) and compare
/// against what was written.  Returns true if all words match.
bool verify_bram(hololink::Hololink &hololink,
                 const std::vector<std::vector<std::uint8_t>> &windows,
                 std::size_t bytes_per_window) {
  const std::size_t cycles = bytes_per_window / 64;
  const auto total_cycles = static_cast<std::uint32_t>(windows.size() * cycles);
  const std::uint32_t w_sample_addr = bram_w_sample_addr();
  auto timeout = std::shared_ptr<hololink::Timeout>();

  bool all_ok = true;
  std::size_t mismatches = 0;

  for (std::uint32_t i = 0; i < RAM_NUM; ++i) {
    std::uint32_t bank_base = RAM_ADDR + (i << (w_sample_addr + 2));
    auto [ok, readback] =
        hololink.read_uint32(bank_base, total_cycles, timeout);
    if (!ok) {
      std::cerr << "BRAM readback: failed to read bank " << i << "\n";
      return false;
    }

    for (std::size_t w = 0; w < windows.size(); ++w) {
      const auto &window = windows[w];
      for (std::size_t s = 0; s < cycles; ++s) {
        std::size_t word_index = s * RAM_NUM + i;
        std::size_t byte_offset = word_index * sizeof(std::uint32_t);
        std::uint32_t expected = 0;
        if (byte_offset + sizeof(std::uint32_t) <= window.size())
          expected = load_le_u32(window.data() + byte_offset);

        std::size_t sample_idx = w * cycles + s;
        std::uint32_t actual = readback[sample_idx];

        if (actual != expected) {
          if (mismatches < 10) {
            std::cerr << "  BRAM mismatch: bank=" << i
                      << " sample=" << sample_idx << " expected=0x" << std::hex
                      << expected << " got=0x" << actual << std::dec << "\n";
          }
          all_ok = false;
          ++mismatches;
        }
      }
    }
  }

  if (mismatches > 10) {
    std::cerr << "  ... and " << (mismatches - 10) << " more mismatches\n";
  }

  return all_ok;
}

// ============================================================================
// ILA Capture Functions
// ============================================================================

void ila_reset(hololink::Hololink &hl) {
  if (!hl.write_uint32(ILA_BASE_ADDR + ILA_CTRL_OFFSET, ILA_CTRL_RESET))
    throw std::runtime_error("ILA reset write failed");
  std::this_thread::sleep_for(std::chrono::seconds(1));
  if (!hl.write_uint32(ILA_BASE_ADDR + ILA_CTRL_OFFSET, ILA_CTRL_DISABLE))
    throw std::runtime_error("ILA disable-after-reset write failed");
}

void ila_enable(hololink::Hololink &hl) {
  if (!hl.write_uint32(ILA_BASE_ADDR + ILA_CTRL_OFFSET, ILA_CTRL_ENABLE))
    throw std::runtime_error("ILA enable write failed");
}

void ila_disable(hololink::Hololink &hl) {
  if (!hl.write_uint32(ILA_BASE_ADDR + ILA_CTRL_OFFSET, ILA_CTRL_DISABLE))
    throw std::runtime_error("ILA disable write failed");
}

/// Read the current ILA sample write address (number of samples captured).
std::uint32_t ila_sample_count(hololink::Hololink &hl) {
  return hl.read_uint32(ILA_BASE_ADDR + ILA_SAMPLE_ADDR_OFFSET);
}

/// Poll the ILA sample count register until at least @p expected samples have
/// been captured, or @p timeout_ms elapses.
bool ila_wait_for_samples(hololink::Hololink &hl,
                          std::uint32_t expected_samples, int timeout_ms) {
  int polls = std::max(1, timeout_ms / 100);
  for (int i = 0; i < polls; ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    std::uint32_t count = ila_sample_count(hl);
    if (count >= expected_samples)
      return true;
  }
  return false;
}

/// Read @p num_samples captured ILA samples using block reads (one per bank).
/// Returns vector of samples; each sample is ILA_NUM_RAM uint32 words ordered
/// LSW-first (word[0] = bits [31:0], word[1] = bits [63:32], ...).
std::vector<std::vector<std::uint32_t>> ila_dump(hololink::Hololink &hl,
                                                 std::uint32_t num_samples) {
  constexpr std::uint32_t ctrl_switch = 1u << (ILA_W_ADDR + 2 + ILA_W_RAM);
  auto timeout = std::shared_ptr<hololink::Timeout>();

  // Read each bank with a single block read.
  std::vector<std::vector<std::uint32_t>> bank_data(ILA_NUM_RAM);
  for (std::uint32_t y = 0; y < ILA_NUM_RAM; ++y) {
    std::uint32_t bank_base =
        ILA_BASE_ADDR + ctrl_switch + (y << (ILA_W_ADDR + 2));
    auto [ok, data] = hl.read_uint32(bank_base, num_samples, timeout);
    if (!ok)
      throw std::runtime_error("Failed to read ILA bank " + std::to_string(y));
    bank_data[y] = std::move(data);
  }

  // Transpose bank_data[bank][sample] → samples[sample][bank].
  std::vector<std::vector<std::uint32_t>> samples(
      num_samples, std::vector<std::uint32_t>(ILA_NUM_RAM));
  for (std::uint32_t i = 0; i < num_samples; ++i) {
    for (std::uint32_t y = 0; y < ILA_NUM_RAM; ++y) {
      samples[i][y] = bank_data[y][i];
    }
  }
  return samples;
}

// ============================================================================
// ILA Sample Field Extraction
// ============================================================================

/// Extract a single bit from a wide sample (array of uint32 words, LSW first).
bool extract_bit(const std::vector<std::uint32_t> &sample,
                 std::uint32_t bit_pos) {
  std::uint32_t word_idx = bit_pos / 32;
  std::uint32_t bit_idx = bit_pos % 32;
  return (sample.at(word_idx) >> bit_idx) & 1;
}

/// Extract a multi-bit field (up to 32 bits) from a wide sample.
std::uint32_t extract_field(const std::vector<std::uint32_t> &sample,
                            std::uint32_t lsb, std::uint32_t width) {
  std::uint32_t result = 0;
  for (std::uint32_t b = 0; b < width && b < 32; ++b) {
    std::uint32_t bit_pos = lsb + b;
    std::uint32_t word_idx = bit_pos / 32;
    std::uint32_t bit_idx = bit_pos % 32;
    if ((sample.at(word_idx) >> bit_idx) & 1)
      result |= (1u << b);
  }
  return result;
}

/// Extract raw bytes from the 512-bit data field (bits [511:0]) of a captured
/// ILA sample.  The data bus occupies words[0] through words[15].
std::vector<std::uint8_t>
extract_tdata_bytes(const std::vector<std::uint32_t> &sample,
                    std::size_t num_bytes) {
  std::vector<std::uint8_t> bytes(num_bytes, 0);
  for (std::size_t i = 0; i < num_bytes && i < 64; ++i) {
    std::uint32_t word_idx = static_cast<std::uint32_t>(i / 4);
    std::uint32_t byte_idx = static_cast<std::uint32_t>(i % 4);
    bytes[i] =
        static_cast<std::uint8_t>((sample[word_idx] >> (byte_idx * 8)) & 0xFF);
  }
  return bytes;
}

// ============================================================================
// Correction Verification
// ============================================================================

struct VerifyResult {
  std::size_t total_samples = 0;
  std::size_t responses_matched = 0;
  std::size_t header_errors = 0;
  std::size_t correction_errors = 0;
};

/// Scan captured ILA samples for RPC correction responses and compare each
/// against the expected values from the syndromes file.
///
/// Each captured 512-bit data word should contain a complete RPCResponse:
///   bytes [0:3]   RPCResponse.magic      = 0x43555153 (RPC_MAGIC_RESPONSE)
///   bytes [4:7]   RPCResponse.status     = 0 (success)
///   bytes [8:11]  RPCResponse.result_len = 1
///   byte  [12]    correction value
VerifyResult verify_captured_responses(
    const std::vector<std::vector<std::uint32_t>> &samples,
    const std::vector<SyndromeEntry> &syndromes, std::size_t num_expected) {
  VerifyResult result;
  result.total_samples = samples.size();
  std::size_t response_idx = 0;

  for (std::size_t i = 0; i < samples.size() && response_idx < num_expected;
       ++i) {
    const auto &sample = samples[i];

    bool tvalid = extract_bit(sample, ILA_TVALID_BIT);
    bool tlast = extract_bit(sample, ILA_TLAST_BIT);
    std::uint32_t wr_tcnt =
        extract_field(sample, ILA_WR_TCNT_LSB, ILA_WR_TCNT_WIDTH);

    if (!tvalid)
      continue;

    // Extract the raw payload bytes from the 512-bit data bus.
    constexpr std::size_t kResponseSize =
        sizeof(cudaq::nvqlink::RPCResponse) + 1; // header + 1 correction byte
    auto data_bytes = extract_tdata_bytes(sample, kResponseSize);

    cudaq::nvqlink::RPCResponse resp{};
    std::memcpy(&resp, data_bytes.data(), sizeof(resp));
    std::uint8_t correction_byte =
        data_bytes[sizeof(cudaq::nvqlink::RPCResponse)];

    std::cout << "  Sample " << i << " (tlast=" << tlast
              << " wr_tcnt=" << wr_tcnt << "):\n";
    std::cout << "    Raw first 16 bytes:";
    for (std::size_t b = 0; b < 16 && b < data_bytes.size(); ++b) {
      std::cout << " " << std::hex << std::setfill('0') << std::setw(2)
                << static_cast<int>(data_bytes[b]);
    }
    std::cout << std::dec << "\n";

    if (resp.magic != cudaq::nvqlink::RPC_MAGIC_RESPONSE) {
      std::cout << "    WARNING: magic=0x" << std::hex << resp.magic
                << " does not match RPC_MAGIC_RESPONSE (0x"
                << cudaq::nvqlink::RPC_MAGIC_RESPONSE << ")" << std::dec
                << " — skipping non-RPC frame\n";
      continue;
    }

    std::cout << "    RPCResponse: magic=0x" << std::hex << resp.magic
              << std::dec << " status=" << resp.status
              << " result_len=" << resp.result_len << "\n";

    bool header_ok = true;
    if (resp.status != 0) {
      std::cout << "    ERROR: status=" << resp.status << " (expected 0)\n";
      header_ok = false;
    }
    if (resp.result_len != 1) {
      std::cout << "    WARNING: result_len=" << resp.result_len
                << " (expected 1 for single correction byte)\n";
    }

    if (!header_ok) {
      result.header_errors++;
      response_idx++;
      continue;
    }

    std::uint8_t expected = (response_idx < syndromes.size())
                                ? syndromes[response_idx].expected_correction
                                : 0;
    std::cout << "    Correction: got=" << static_cast<int>(correction_byte)
              << " expected=" << static_cast<int>(expected);
    if (correction_byte == expected) {
      std::cout << " [PASS]\n";
      result.responses_matched++;
    } else {
      std::cout << " [FAIL]\n";
      result.correction_errors++;
    }

    response_idx++;
  }

  return result;
}

} // namespace

// ============================================================================
// Main
// ============================================================================

int main(int argc, char **argv) {
  Options options = parse_args(argc, argv);
  if (options.hololink_ip.empty() || options.data_dir.empty()) {
    print_usage(argv[0]);
    return 1;
  }

  // ------------------------------------------------------------------
  // Load configuration and syndrome data
  // ------------------------------------------------------------------
  std::string config_path = options.data_dir + "/config_multi_err_lut.yml";
  std::string syndromes_path =
      options.data_dir + "/syndromes_multi_err_lut.txt";

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

  std::size_t num_shots = options.num_shots
                              ? std::min(*options.num_shots, syndromes.size())
                              : syndromes.size();
  if (num_shots == 0) {
    std::cerr << "No shots to play back\n";
    return 1;
  }

  // ------------------------------------------------------------------
  // Build RPC payloads and pad to 64-byte alignment
  // ------------------------------------------------------------------
  std::vector<std::vector<std::uint8_t>> windows;
  windows.reserve(num_shots);
  for (std::size_t i = 0; i < num_shots; ++i)
    windows.push_back(build_rpc_payload(syndromes[i].measurements));

  std::size_t payload_size = windows.front().size();
  std::size_t bytes_per_window = align_up(payload_size, 64);

  for (auto &window : windows)
    window.resize(bytes_per_window, 0);

  std::size_t cycles_per_window = bytes_per_window / 64;
  if (num_shots * cycles_per_window > RAM_DEPTH) {
    std::cerr << "Data exceeds playback BRAM capacity: " << num_shots
              << " shots x " << cycles_per_window
              << " cycles = " << (num_shots * cycles_per_window) << " > "
              << RAM_DEPTH << " depth\n";
    return 1;
  }

  std::cout << "Loaded " << num_shots
            << " shots (syndrome_size=" << syndrome_size
            << ", payload=" << payload_size
            << " bytes, padded=" << bytes_per_window << " bytes, "
            << cycles_per_window << " cycles/shot)\n";

  // ------------------------------------------------------------------
  // Connect to Hololink (or emulator) and reset
  // ------------------------------------------------------------------
  hololink::Metadata channel_metadata;
  bool using_emulator = options.control_port.has_value();

  if (using_emulator) {
    // Direct connection to emulator — bypass BOOTP enumeration.
    // Construct synthetic metadata with the required fields.
    std::cout << "Using direct connection to emulator at "
              << options.hololink_ip << ":" << *options.control_port << "\n";
    channel_metadata["peer_ip"] = options.hololink_ip;
    channel_metadata["control_port"] =
        static_cast<std::int64_t>(*options.control_port);
    channel_metadata["serial_number"] = std::string("emulator");
    channel_metadata["sequence_number_checking"] =
        static_cast<std::int64_t>(0);
    channel_metadata["vp_mask"] = static_cast<std::int64_t>(0x1);
    channel_metadata["data_plane"] = static_cast<std::int64_t>(0);
    channel_metadata["sensor"] = static_cast<std::int64_t>(0);
    channel_metadata["sif_address"] = static_cast<std::int64_t>(0);
    channel_metadata["vp_address"] =
        static_cast<std::int64_t>(options.vp_address);
    channel_metadata["hif_address"] =
        static_cast<std::int64_t>(options.hif_address);
  } else {
    channel_metadata =
        hololink::Enumerator::find_channel(options.hololink_ip);
    hololink::DataChannel::use_sensor(channel_metadata, 0);
  }

  hololink::DataChannel hololink_channel(channel_metadata);
  auto hololink = hololink_channel.hololink();

  hololink->start();
  if (!using_emulator) {
    hololink->reset();
  }

  // ------------------------------------------------------------------
  // Configure FPGA SIF registers for RDMA target (if provided)
  // ------------------------------------------------------------------
  if (options.qp_number && options.rkey && options.buffer_addr) {
    std::uint32_t rdma_page_size = options.rdma_page_size.value_or(256);
    std::uint32_t rdma_num_pages = options.rdma_num_pages.value_or(64);

    std::cout << "Configuring FPGA SIF for RDMA target:\n"
              << "  QP number:    0x" << std::hex << *options.qp_number
              << std::dec << "\n"
              << "  RKEY:         " << *options.rkey << "\n"
              << "  Buffer addr:  0x" << std::hex << *options.buffer_addr
              << std::dec << "\n"
              << "  Page size:    " << rdma_page_size << " bytes\n"
              << "  Num pages:    " << rdma_num_pages << "\n"
              << "  Frame size:   " << bytes_per_window << " bytes\n";

    hololink_channel.authenticate(*options.qp_number, *options.rkey);
    hololink_channel.configure_roce(*options.buffer_addr, bytes_per_window,
                                    rdma_page_size, rdma_num_pages,
                                    /*local_data_port=*/0);

    std::cout << "FPGA SIF registers configured for RDMA\n";
  }

  // ------------------------------------------------------------------
  // Disable player, configure, and write BRAM
  // ------------------------------------------------------------------
  if (!hololink->write_uint32(PLAYER_ADDR + PLAYER_ENABLE_OFFSET,
                              PLAYER_DISABLE))
    throw std::runtime_error("Failed to disable player");

  hololink::Hololink::WriteData config_write;
  config_write.queue_write_uint32(PLAYER_ADDR + PLAYER_WINDOW_SIZE_OFFSET,
                                  static_cast<std::uint32_t>(bytes_per_window));
  config_write.queue_write_uint32(PLAYER_ADDR + PLAYER_WINDOW_NUMBER_OFFSET,
                                  static_cast<std::uint32_t>(num_shots));
  config_write.queue_write_uint32(PLAYER_ADDR + PLAYER_TIMER_OFFSET,
                                  RF_SOC_TIMER_SCALE *
                                      DEFAULT_TIMER_SPACING_US);
  if (!hololink->write_uint32(config_write))
    throw std::runtime_error("Failed to configure player");

  std::cout << "Writing " << num_shots << " windows to playback BRAM...\n";
  write_bram(*hololink, windows, bytes_per_window);

  // ------------------------------------------------------------------
  // BRAM readback verification (always-on)
  // ------------------------------------------------------------------
  std::cout << "Verifying playback BRAM contents...\n";
  if (!verify_bram(*hololink, windows, bytes_per_window)) {
    std::cerr << "BRAM readback verification FAILED\n";
    return 1;
  }
  std::cout << "BRAM readback verification PASSED\n";

  // ------------------------------------------------------------------
  // Arm ILA capture (before playback) if --verify
  // ------------------------------------------------------------------
  if (options.verify) {
    std::cout << "\n=== Arming ILA capture (SIF TX at 0x" << std::hex
              << ILA_BASE_ADDR << std::dec << ") ===\n";
    ila_disable(*hololink);
    ila_reset(*hololink);
    ila_enable(*hololink);
    std::cout << "ILA: armed for capture\n";
  }

  // ------------------------------------------------------------------
  // Enable playback
  // ------------------------------------------------------------------
  if (!hololink->write_uint32(PLAYER_ADDR + PLAYER_ENABLE_OFFSET,
                              PLAYER_ENABLE))
    throw std::runtime_error("Failed to enable player");

  std::cout << "Playback enabled: " << num_shots << " shots on hololink "
            << options.hololink_ip << "\n";

  // ------------------------------------------------------------------
  // ILA capture and correction verification
  // ------------------------------------------------------------------
  if (options.verify) {
    std::cout << "\n=== ILA Capture & Verification ===\n";
    std::cout << "Waiting for " << num_shots << " correction responses...\n";

    constexpr int kVerifyTimeoutMs = 5000;
    bool captured = ila_wait_for_samples(
        *hololink, static_cast<std::uint32_t>(num_shots), kVerifyTimeoutMs);

    std::uint32_t actual_samples = ila_sample_count(*hololink);
    ila_disable(*hololink);

    if (!captured) {
      std::cerr << "ILA: only captured " << actual_samples << " of "
                << num_shots << " expected samples (timeout "
                << kVerifyTimeoutMs << " ms)\n";
      if (actual_samples == 0)
        return 1;
      std::cout << "Proceeding with " << actual_samples
                << " captured samples\n";
    } else {
      std::cout << "ILA: captured " << actual_samples << " samples\n";
    }

    std::uint32_t samples_to_read = actual_samples;
    std::cout << "Reading " << samples_to_read << " samples (" << ILA_NUM_RAM
              << " words each)...\n";
    auto captured_samples = ila_dump(*hololink, samples_to_read);

    std::cout << "\nVerifying " << num_shots
              << " expected RPC responses...\n\n";
    auto vresult =
        verify_captured_responses(captured_samples, syndromes, num_shots);

    std::cout << "\n=== Verification Summary ===\n"
              << "  Captured samples:       " << vresult.total_samples << "\n"
              << "  RPC responses matched:  " << vresult.responses_matched
              << " / " << num_shots << "\n"
              << "  Header errors:          " << vresult.header_errors << "\n"
              << "  Correction mismatches:  " << vresult.correction_errors
              << "\n";

    if (vresult.responses_matched == num_shots) {
      std::cout << "  RESULT: PASS\n";
    } else {
      std::cout << "  RESULT: FAIL\n";
      return 1;
    }
  }

  return 0;
}
