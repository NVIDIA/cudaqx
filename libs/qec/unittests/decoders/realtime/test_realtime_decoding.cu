/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file test_realtime_decoding.cu
/// @brief End-to-end test for realtime decoding pipeline with mock decoder.
///
/// This test verifies the complete realtime decoding flow using the
/// cuda-quantum host API and the dispatch kernel linked from
/// libcudaq-realtime.so:
/// 1. Hololink-style ring buffer communication (cudaHostAllocMapped)
/// 2. Host API wires dispatcher and launches persistent kernel
/// 3. Mock decoder that returns pre-recorded expected corrections
/// 4. Data loaded from config_multi_err_lut.yml and syndromes_multi_err_lut.txt

#include <algorithm>
#include <cstdint>
#include <cuda_runtime.h>
#include <fstream>
#include <gtest/gtest.h>
#include <string>
#include <unistd.h>
#include <vector>

// cuda-quantum host API
#include "cudaq/nvqlink/daemon/dispatcher/cudaq_realtime.h"

// cuda-quantum RPC types/hash helper
#include "cudaq/nvqlink/daemon/dispatcher/dispatch_kernel_launch.h"

// cudaqx mock decoder
#include "cudaq/qec/realtime/mock_decode_handler.cuh"

// Helper macro for CUDA error checking
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    ASSERT_EQ(err, cudaSuccess) << "CUDA error: " << cudaGetErrorString(err);  \
  } while (0)

namespace {

//==============================================================================
// Function ID for mock decoder
//==============================================================================

// The dispatch kernel uses function_id to find the handler
constexpr std::uint32_t MOCK_DECODE_FUNCTION_ID =
    cudaq::nvqlink::fnv1a_hash("mock_decode");

//==============================================================================
// Hololink-Style Ring Buffer
//==============================================================================

/// @brief Allocate Hololink-style ring buffer with mapped memory.
bool allocate_ring_buffer(std::size_t num_slots, std::size_t slot_size,
                          volatile uint64_t **host_flags_out,
                          volatile uint64_t **device_flags_out,
                          uint8_t **host_data_out, uint8_t **device_data_out) {

  void *host_flags_ptr = nullptr;
  cudaError_t err = cudaHostAlloc(&host_flags_ptr, num_slots * sizeof(uint64_t),
                                  cudaHostAllocMapped);
  if (err != cudaSuccess)
    return false;

  void *device_flags_ptr = nullptr;
  err = cudaHostGetDevicePointer(&device_flags_ptr, host_flags_ptr, 0);
  if (err != cudaSuccess) {
    cudaFreeHost(host_flags_ptr);
    return false;
  }

  void *host_data_ptr = nullptr;
  err =
      cudaHostAlloc(&host_data_ptr, num_slots * slot_size, cudaHostAllocMapped);
  if (err != cudaSuccess) {
    cudaFreeHost(host_flags_ptr);
    return false;
  }

  void *device_data_ptr = nullptr;
  err = cudaHostGetDevicePointer(&device_data_ptr, host_data_ptr, 0);
  if (err != cudaSuccess) {
    cudaFreeHost(host_flags_ptr);
    cudaFreeHost(host_data_ptr);
    return false;
  }

  memset(host_flags_ptr, 0, num_slots * sizeof(uint64_t));

  *host_flags_out = static_cast<volatile uint64_t *>(host_flags_ptr);
  *device_flags_out = static_cast<volatile uint64_t *>(device_flags_ptr);
  *host_data_out = static_cast<uint8_t *>(host_data_ptr);
  *device_data_out = static_cast<uint8_t *>(device_data_ptr);

  return true;
}

void free_ring_buffer(volatile uint64_t *host_flags, uint8_t *host_data) {
  if (host_flags) {
    cudaFreeHost(const_cast<uint64_t *>(host_flags));
  }
  if (host_data) {
    cudaFreeHost(host_data);
  }
}

//==============================================================================
// Test Data Loading
//==============================================================================

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

//==============================================================================
// Kernel to initialize function table
//==============================================================================

/// @brief Initialize the device function table for dispatch.
__global__ void init_function_table(cudaq_function_entry_t *entries) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    entries[0].handler.device_fn_ptr =
        reinterpret_cast<void *>(&cudaq::qec::realtime::mock_decode_rpc);
    entries[0].function_id = MOCK_DECODE_FUNCTION_ID;
    entries[0].dispatch_mode = CUDAQ_DISPATCH_DEVICE_CALL;
    entries[0].reserved[0] = 0;
    entries[0].reserved[1] = 0;
    entries[0].reserved[2] = 0;

    // Schema: 1 bit-packed argument (128 bits = 16 bytes), 1 uint8 result
    entries[0].schema.num_args = 1;
    entries[0].schema.num_results = 1;
    entries[0].schema.reserved = 0;
    entries[0].schema.args[0].type_id = CUDAQ_TYPE_BIT_PACKED;
    entries[0].schema.args[0].reserved[0] = 0;
    entries[0].schema.args[0].reserved[1] = 0;
    entries[0].schema.args[0].reserved[2] = 0;
    entries[0].schema.args[0].size_bytes = 16;    // 128 bits
    entries[0].schema.args[0].num_elements = 128; // 128 bits
    entries[0].schema.results[0].type_id = CUDAQ_TYPE_UINT8;
    entries[0].schema.results[0].reserved[0] = 0;
    entries[0].schema.results[0].reserved[1] = 0;
    entries[0].schema.results[0].reserved[2] = 0;
    entries[0].schema.results[0].size_bytes = 1;
    entries[0].schema.results[0].num_elements = 1;
  }
}

//==============================================================================
// Host Launch Wrapper (C-compatible)
//==============================================================================

extern "C" void launch_dispatch_kernel_wrapper(
    volatile std::uint64_t *rx_flags, volatile std::uint64_t *tx_flags,
    std::uint8_t *rx_data, std::uint8_t *tx_data,
    std::size_t rx_stride_sz, std::size_t tx_stride_sz,
    cudaq_function_entry_t *function_table, std::size_t func_count,
    volatile int *shutdown_flag, std::uint64_t *stats, std::size_t num_slots,
    std::uint32_t num_blocks, std::uint32_t threads_per_block,
    cudaStream_t stream) {
  cudaq_launch_dispatch_kernel_regular(
      rx_flags, tx_flags, rx_data, tx_data,
      rx_stride_sz, tx_stride_sz,
      function_table, func_count, shutdown_flag, stats,
      num_slots, num_blocks, threads_per_block, stream);
}

// Helper function to check if a GPU is available
bool isGpuAvailable() {
  int deviceCount = 0;
  cudaError_t err = cudaGetDeviceCount(&deviceCount);
  return (err == cudaSuccess && deviceCount > 0);
}

} // namespace

//==============================================================================
// Test Fixture
//==============================================================================

class RealtimeDecodingTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Skip all tests if no GPU is available
    if (!isGpuAvailable()) {
      GTEST_SKIP() << "No GPU available, skipping realtime decoding tests";
    }

    // Enable host-mapped memory before any CUDA context creation.
    cudaError_t flags_err = cudaSetDeviceFlags(cudaDeviceMapHost);
    ASSERT_TRUE(flags_err == cudaSuccess ||
                flags_err == cudaErrorSetOnActiveProcess);

    data_dir_ = std::string(TEST_DATA_DIR);

    // Load config
    std::ifstream config_file(data_dir_ + "/config_multi_err_lut.yml");
    ASSERT_TRUE(config_file.good()) << "Could not open config file";
    std::string config_content((std::istreambuf_iterator<char>(config_file)),
                               std::istreambuf_iterator<char>());

    syndrome_size_ = parse_scalar(config_content, "syndrome_size");
    block_size_ = parse_scalar(config_content, "block_size");
    ASSERT_GT(syndrome_size_, 0u);
    ASSERT_GT(block_size_, 0u);

    // Load syndrome test data
    syndromes_ = load_syndromes(data_dir_ + "/syndromes_multi_err_lut.txt",
                                syndrome_size_);
    ASSERT_FALSE(syndromes_.empty()) << "No syndrome data loaded";

    // Build lookup table for mock decoder
    num_lookup_entries_ = syndromes_.size();
    lookup_measurements_.resize(num_lookup_entries_ * syndrome_size_);
    lookup_corrections_.resize(num_lookup_entries_);

    for (std::size_t i = 0; i < num_lookup_entries_; ++i) {
      memcpy(lookup_measurements_.data() + i * syndrome_size_,
             syndromes_[i].measurements.data(), syndrome_size_);
      lookup_corrections_[i] = syndromes_[i].expected_correction;
    }

    // Allocate device memory for lookup tables
    CUDA_CHECK(
        cudaMalloc(&d_lookup_measurements_, lookup_measurements_.size()));
    CUDA_CHECK(cudaMalloc(&d_lookup_corrections_, lookup_corrections_.size()));
    CUDA_CHECK(cudaMemcpy(d_lookup_measurements_, lookup_measurements_.data(),
                          lookup_measurements_.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_lookup_corrections_, lookup_corrections_.data(),
                          lookup_corrections_.size(), cudaMemcpyHostToDevice));

    // Set up mock decoder context
    ctx_.num_measurements = syndrome_size_;
    ctx_.num_observables = 1;
    ctx_.lookup_measurements = d_lookup_measurements_;
    ctx_.lookup_corrections = d_lookup_corrections_;
    ctx_.num_lookup_entries = num_lookup_entries_;

    // Allocate device context
    CUDA_CHECK(cudaMalloc(&d_ctx_,
                          sizeof(cudaq::qec::realtime::mock_decoder_context)));
    CUDA_CHECK(cudaMemcpy(d_ctx_, &ctx_, sizeof(ctx_), cudaMemcpyHostToDevice));

    // Set global context for RPC-style calls
    cudaq::qec::realtime::set_mock_decoder_context(d_ctx_);

    // Allocate ring buffers (with space for RPCHeader)
    slot_size_ = sizeof(cudaq::nvqlink::RPCHeader) +
                 std::max(syndrome_size_, static_cast<std::size_t>(256));
    ASSERT_TRUE(allocate_ring_buffer(num_slots_, slot_size_, &rx_flags_host_,
                                     &rx_flags_, &rx_data_host_, &rx_data_));
    ASSERT_TRUE(allocate_ring_buffer(num_slots_, slot_size_, &tx_flags_host_,
                                     &tx_flags_, &tx_data_host_, &tx_data_));

    // Allocate control variables
    void *tmp_shutdown = nullptr;
    CUDA_CHECK(cudaHostAlloc(&tmp_shutdown, sizeof(int), cudaHostAllocMapped));
    shutdown_flag_ = static_cast<volatile int *>(tmp_shutdown);
    void *tmp_d_shutdown = nullptr;
    CUDA_CHECK(cudaHostGetDevicePointer(&tmp_d_shutdown, tmp_shutdown, 0));
    d_shutdown_flag_ = static_cast<volatile int *>(tmp_d_shutdown);
    *shutdown_flag_ = 0;
    int zero = 0;
    CUDA_CHECK(cudaMemcpy(const_cast<int *>(d_shutdown_flag_), &zero,
                          sizeof(int), cudaMemcpyHostToDevice));

    // Allocate stats
    CUDA_CHECK(cudaMalloc(&d_stats_, sizeof(uint64_t)));
    CUDA_CHECK(cudaMemset(d_stats_, 0, sizeof(uint64_t)));

    // Set up function table for dispatch kernel
    setup_function_table();

    // Host API wiring
    ASSERT_EQ(cudaq_dispatch_manager_create(&manager_), CUDAQ_OK);
    cudaq_dispatcher_config_t config{};
    config.device_id = 0;
    config.num_blocks = 1;
    config.threads_per_block = 32;
    config.num_slots = static_cast<uint32_t>(num_slots_);
    config.slot_size = static_cast<uint32_t>(slot_size_);
    config.vp_id = 0;
    config.kernel_type = CUDAQ_KERNEL_REGULAR;
    config.dispatch_mode = CUDAQ_DISPATCH_DEVICE_CALL;

    ASSERT_EQ(cudaq_dispatcher_create(manager_, &config, &dispatcher_),
              CUDAQ_OK);

    cudaq_ringbuffer_t ringbuffer{};
    ringbuffer.rx_flags = rx_flags_;
    ringbuffer.tx_flags = tx_flags_;
    ringbuffer.rx_data = rx_data_;
    ringbuffer.tx_data = tx_data_;
    ringbuffer.rx_stride_sz = slot_size_;
    ringbuffer.tx_stride_sz = slot_size_;
    ASSERT_EQ(cudaq_dispatcher_set_ringbuffer(dispatcher_, &ringbuffer),
              CUDAQ_OK);

    cudaq_function_table_t table{};
    table.entries = d_function_entries_;
    table.count = func_count_;
    ASSERT_EQ(cudaq_dispatcher_set_function_table(dispatcher_, &table),
              CUDAQ_OK);

    ASSERT_EQ(
        cudaq_dispatcher_set_control(dispatcher_, d_shutdown_flag_, d_stats_),
        CUDAQ_OK);

    ASSERT_EQ(cudaq_dispatcher_set_launch_fn(dispatcher_,
                                             &launch_dispatch_kernel_wrapper),
              CUDAQ_OK);

    ASSERT_EQ(cudaq_dispatcher_start(dispatcher_), CUDAQ_OK);
  }

  void TearDown() override {
    if (shutdown_flag_) {
      *shutdown_flag_ = 1;
      __sync_synchronize();
    }
    if (dispatcher_) {
      cudaq_dispatcher_stop(dispatcher_);
      cudaq_dispatcher_destroy(dispatcher_);
      dispatcher_ = nullptr;
    }

    if (manager_) {
      cudaq_dispatch_manager_destroy(manager_);
      manager_ = nullptr;
    }

    free_ring_buffer(rx_flags_host_, rx_data_host_);
    free_ring_buffer(tx_flags_host_, tx_data_host_);

    if (shutdown_flag_)
      cudaFreeHost(const_cast<int *>(shutdown_flag_));
    if (d_lookup_measurements_)
      cudaFree(d_lookup_measurements_);
    if (d_lookup_corrections_)
      cudaFree(d_lookup_corrections_);
    if (d_ctx_)
      cudaFree(d_ctx_);
    if (d_stats_)
      cudaFree(d_stats_);
    if (d_function_entries_)
      cudaFree(d_function_entries_);
  }

  void setup_function_table() {
    CUDA_CHECK(
        cudaMalloc(&d_function_entries_, sizeof(cudaq_function_entry_t)));

    init_function_table<<<1, 1>>>(d_function_entries_);
    CUDA_CHECK(cudaDeviceSynchronize());

    func_count_ = 1;
  }

  /// @brief Write syndrome to RX buffer in RPC format.
  void write_rpc_request(std::size_t slot,
                         const std::vector<uint8_t> &measurements) {
    uint8_t *slot_data =
        const_cast<uint8_t *>(rx_data_host_) + slot * slot_size_;

    // Write RPCHeader
    cudaq::nvqlink::RPCHeader *header =
        reinterpret_cast<cudaq::nvqlink::RPCHeader *>(slot_data);
    header->magic = cudaq::nvqlink::RPC_MAGIC_REQUEST;
    header->function_id = MOCK_DECODE_FUNCTION_ID;
    header->arg_len = static_cast<std::uint32_t>(measurements.size());

    // Write measurement data after header
    memcpy(slot_data + sizeof(cudaq::nvqlink::RPCHeader), measurements.data(),
           measurements.size());
  }

  /// @brief Read response from TX buffer (symmetric: dispatch kernel writes to TX).
  bool read_rpc_response(std::size_t slot, uint8_t &correction,
                         std::int32_t *status_out = nullptr,
                         std::uint32_t *result_len_out = nullptr) {
    __sync_synchronize();
    const uint8_t *slot_data =
        const_cast<uint8_t *>(tx_data_host_) + slot * slot_size_;

    // Read RPCResponse
    const cudaq::nvqlink::RPCResponse *response =
        reinterpret_cast<const cudaq::nvqlink::RPCResponse *>(slot_data);

    if (response->magic != cudaq::nvqlink::RPC_MAGIC_RESPONSE) {
      return false;
    }
    if (status_out)
      *status_out = response->status;
    if (result_len_out)
      *result_len_out = response->result_len;

    if (response->status != 0) {
      return false;
    }

    // Read correction data after response header
    correction = *(slot_data + sizeof(cudaq::nvqlink::RPCResponse));
    return true;
  }

  std::string data_dir_;
  std::size_t syndrome_size_ = 0;
  std::size_t block_size_ = 0;
  std::vector<SyndromeEntry> syndromes_;

  std::vector<uint8_t> lookup_measurements_;
  std::vector<uint8_t> lookup_corrections_;
  std::size_t num_lookup_entries_ = 0;
  uint8_t *d_lookup_measurements_ = nullptr;
  uint8_t *d_lookup_corrections_ = nullptr;

  cudaq::qec::realtime::mock_decoder_context ctx_;
  cudaq::qec::realtime::mock_decoder_context *d_ctx_ = nullptr;

  static constexpr std::size_t num_slots_ = 4;
  std::size_t slot_size_ = 256;
  volatile uint64_t *rx_flags_host_ = nullptr;
  volatile uint64_t *tx_flags_host_ = nullptr;
  volatile uint64_t *rx_flags_ = nullptr;
  volatile uint64_t *tx_flags_ = nullptr;
  uint8_t *rx_data_host_ = nullptr;
  uint8_t *tx_data_host_ = nullptr;
  uint8_t *rx_data_ = nullptr;
  uint8_t *tx_data_ = nullptr;

  volatile int *shutdown_flag_ = nullptr;
  volatile int *d_shutdown_flag_ = nullptr;
  uint64_t *d_stats_ = nullptr;

  // Function table for dispatch kernel
  cudaq_function_entry_t *d_function_entries_ = nullptr;
  std::size_t func_count_ = 0;

  // Host API handles
  cudaq_dispatch_manager_t *manager_ = nullptr;
  cudaq_dispatcher_t *dispatcher_ = nullptr;
};

//==============================================================================
// Tests
//==============================================================================

/// @brief End-to-end test over the full syndromes file.
/// This verifies the integration between cudaqx and cuda-quantum.
TEST_F(RealtimeDecodingTest, DispatchKernelAllShots) {
  const std::size_t num_test_shots = syndromes_.size();
  std::size_t correct_count = 0;

  for (std::size_t i = 0; i < num_test_shots; ++i) {
    std::size_t slot = i % num_slots_;
    const auto &entry = syndromes_[i];

    // Wait for slot to be free
    int timeout = 50;
    while (rx_flags_host_[slot] != 0 && timeout-- > 0) {
      usleep(100);
    }
    ASSERT_GT(timeout, 0) << "Timeout waiting for RX slot " << slot;

    // Send request
    write_rpc_request(slot, entry.measurements);
    __sync_synchronize();
    const_cast<volatile uint64_t *>(rx_flags_host_)[slot] =
        reinterpret_cast<uint64_t>(rx_data_ + slot * slot_size_);

    // Wait for response
    timeout = 50;
    while (tx_flags_host_[slot] == 0 && timeout-- > 0) {
      usleep(100);
    }
    if (timeout <= 0) {
      std::cerr << "DispatchKernelAllShots timeout diagnostics:\n"
                << "  slot = " << slot << "\n"
                << "  rx_flags_host[slot] = " << rx_flags_host_[slot] << "\n"
                << "  tx_flags_host_[slot] = " << tx_flags_host_[slot] << "\n"
                << std::flush;
    }
    ASSERT_GT(timeout, 0) << "Timeout waiting for TX slot " << slot;

    // Check result
    uint8_t correction = 0;
    std::int32_t status = 0;
    std::uint32_t result_len = 0;
    if (read_rpc_response(slot, correction, &status, &result_len)) {
      if (correction == entry.expected_correction) {
        correct_count++;
      } else {
        std::cerr << "RPC mismatch slot " << slot << " status=" << status
                  << " result_len=" << result_len
                  << " expected=" << static_cast<int>(entry.expected_correction)
                  << " got=" << static_cast<int>(correction) << "\n";
      }
    } else {
      std::cerr << "RPC failure slot " << slot << " status=" << status
                << " result_len=" << result_len
                << " expected=" << static_cast<int>(entry.expected_correction)
                << " got=" << static_cast<int>(correction) << "\n";
    }

    // Clear TX flag
    const_cast<volatile uint64_t *>(tx_flags_host_)[slot] = 0;
  }

  double accuracy = 100.0 * correct_count / num_test_shots;
  std::cout << "Dispatch kernel mock decoder accuracy: " << correct_count << "/"
            << num_test_shots << " (" << accuracy << "%)" << std::endl;

  EXPECT_EQ(correct_count, num_test_shots)
      << "All shots should decode correctly";
}
