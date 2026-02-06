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

// Shared setup helpers (config parsing, syndrome loading, GPU context, launch wrapper)
#include "cudaq/qec/realtime/mock_decode_setup.h"

// Helper macro for CUDA error checking
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    ASSERT_EQ(err, cudaSuccess) << "CUDA error: " << cudaGetErrorString(err);  \
  } while (0)

namespace {

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
  // Convenience aliases for types from the shared header
  using SyndromeEntry = cudaq::qec::realtime::SyndromeEntry;

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

    syndrome_size_ = cudaq::qec::realtime::parse_scalar(config_content,
                                                         "syndrome_size");
    block_size_ = cudaq::qec::realtime::parse_scalar(config_content,
                                                      "block_size");
    ASSERT_GT(syndrome_size_, 0u);
    ASSERT_GT(block_size_, 0u);

    // Load syndrome test data
    syndromes_ = cudaq::qec::realtime::load_syndromes(
        data_dir_ + "/syndromes_multi_err_lut.txt", syndrome_size_);
    ASSERT_FALSE(syndromes_.empty()) << "No syndrome data loaded";

    // Set up mock decoder context on GPU using shared helper
    cudaError_t ctx_err =
        cudaq::qec::realtime::setup_mock_decoder_from_syndromes(
            syndromes_, syndrome_size_, gpu_resources_);
    ASSERT_EQ(ctx_err, cudaSuccess) << "Failed to set up mock decoder on GPU";

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

    // Set up function table using library helper
    CUDA_CHECK(
        cudaMalloc(&d_function_entries_, sizeof(cudaq_function_entry_t)));
    cudaq::qec::realtime::setup_mock_decode_function_table(
        d_function_entries_);

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
    table.count = 1;
    ASSERT_EQ(cudaq_dispatcher_set_function_table(dispatcher_, &table),
              CUDAQ_OK);

    ASSERT_EQ(
        cudaq_dispatcher_set_control(dispatcher_, d_shutdown_flag_, d_stats_),
        CUDAQ_OK);

    ASSERT_EQ(cudaq_dispatcher_set_launch_fn(
                  dispatcher_,
                  &cudaq::qec::realtime::mock_decode_launch_dispatch_kernel),
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
    if (d_stats_)
      cudaFree(d_stats_);
    if (d_function_entries_)
      cudaFree(d_function_entries_);

    gpu_resources_.cleanup();
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
    header->function_id =
        cudaq::qec::realtime::MOCK_DECODE_FUNCTION_ID;
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

  // GPU resources managed by shared helper
  cudaq::qec::realtime::MockDecoderGpuResources gpu_resources_;

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
