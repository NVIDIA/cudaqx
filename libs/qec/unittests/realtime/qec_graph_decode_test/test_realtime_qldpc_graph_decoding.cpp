/****************************************************************-*- C++ -*-****
 * Copyright (c) 2025-2026 NVIDIA Corporation & Affiliates.                    *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file test_realtime_qldpc_graph_decoding.cpp
/// @brief CI test for the CPU-launched CUDA graph relay BP decode path,
/// exercising the full libcudaq-realtime CUDAQ_DISPATCH_PATH_HOST dispatch.
///
/// Flow:
///   1. Loads a relay BP config YAML and syndrome data
///   2. Creates the decoder via the generic decoder::get() API
///   3. Calls capture_decode_graph() to get an opaque graph_resources*
///   4. Wires the libcudaq-realtime C API: manager -> dispatcher (HOST_LOOP)
///      -> ringbuffer -> function table (GRAPH_LAUNCH) -> mailbox -> start
///   5. For each syndrome: writes an RPC request into a ring buffer slot,
///      signals the slot, the host dispatcher launches the CUDA graph,
///      and the test polls for the RPCResponse and verifies corrections.

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <fstream>
#include <numeric>
#include <string>
#include <unistd.h>
#include <vector>

#include "cudaq/qec/decoder.h"
#include "cudaq/qec/realtime/decoding_config.h"
#include "cudaq/qec/realtime/graph_resources.h"
#include "cudaq/qec/realtime/sparse_to_csr.h"

#include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"
#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h"

#ifndef TEST_DATA_DIR
#define TEST_DATA_DIR "."
#endif

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    ASSERT_EQ(err, cudaSuccess) << "CUDA error: " << cudaGetErrorString(err);  \
  } while (0)

using namespace cudaq::qec;
using namespace cudaq::realtime;

//==============================================================================
// Syndrome file loader
//==============================================================================

struct SyndromeEntry {
  std::vector<uint8_t> measurements;
  uint8_t expected_correction;
};

static std::vector<SyndromeEntry> load_syndromes(const std::string &path,
                                                 std::size_t num_measurements) {
  std::ifstream file(path);
  if (!file.is_open())
    return {};

  std::vector<SyndromeEntry> entries;
  std::string line;
  bool in_corrections = false;
  std::size_t correction_idx = 0;

  while (std::getline(file, line)) {
    if (line.empty())
      continue;
    if (line.rfind("NUM_DATA", 0) == 0 || line.rfind("NUM_LOGICAL", 0) == 0)
      continue;
    if (line.rfind("CORRECTIONS_START", 0) == 0) {
      in_corrections = true;
      correction_idx = 0;
      continue;
    }
    if (line.rfind("CORRECTIONS_END", 0) == 0)
      break;

    if (line.rfind("SHOT_START", 0) == 0) {
      entries.emplace_back();
      entries.back().measurements.reserve(num_measurements);
      entries.back().expected_correction = 0;
      continue;
    }

    if (in_corrections) {
      if (correction_idx < entries.size())
        entries[correction_idx].expected_correction =
            static_cast<uint8_t>(std::stoi(line));
      correction_idx++;
    } else if (!entries.empty()) {
      entries.back().measurements.push_back(
          static_cast<uint8_t>(std::stoi(line)));
    }
  }
  return entries;
}

//==============================================================================
// Ring buffer helpers
//==============================================================================

static bool allocate_ring_buffer(std::size_t num_slots, std::size_t slot_size,
                                 volatile uint64_t **host_flags_out,
                                 volatile uint64_t **device_flags_out,
                                 uint8_t **host_data_out,
                                 uint8_t **device_data_out) {
  void *hf = nullptr;
  if (cudaHostAlloc(&hf, num_slots * sizeof(uint64_t), cudaHostAllocMapped) !=
      cudaSuccess)
    return false;
  void *df = nullptr;
  if (cudaHostGetDevicePointer(&df, hf, 0) != cudaSuccess) {
    cudaFreeHost(hf);
    return false;
  }
  void *hd = nullptr;
  if (cudaHostAlloc(&hd, num_slots * slot_size, cudaHostAllocMapped) !=
      cudaSuccess) {
    cudaFreeHost(hf);
    return false;
  }
  void *dd = nullptr;
  if (cudaHostGetDevicePointer(&dd, hd, 0) != cudaSuccess) {
    cudaFreeHost(hf);
    cudaFreeHost(hd);
    return false;
  }
  memset(hf, 0, num_slots * sizeof(uint64_t));
  *host_flags_out = static_cast<volatile uint64_t *>(hf);
  *device_flags_out = static_cast<volatile uint64_t *>(df);
  *host_data_out = static_cast<uint8_t *>(hd);
  *device_data_out = static_cast<uint8_t *>(dd);
  return true;
}

static void free_ring_buffer(volatile uint64_t *host_flags,
                             uint8_t *host_data) {
  if (host_flags)
    cudaFreeHost(const_cast<uint64_t *>(host_flags));
  if (host_data)
    cudaFreeHost(host_data);
}

//==============================================================================
// GTest fixture
//==============================================================================

class GraphDecodeTest : public ::testing::Test {
protected:
  std::unique_ptr<decoder> decoder_;
  realtime::graph_resources *graph_res_ = nullptr;

  std::vector<SyndromeEntry> syndromes_;
  std::size_t num_measurements_ = 0;
  std::size_t num_observables_ = 0;

  static constexpr std::size_t num_slots_ = 4;
  std::size_t slot_size_ = 0;

  volatile uint64_t *rx_flags_host_ = nullptr;
  volatile uint64_t *rx_flags_dev_ = nullptr;
  uint8_t *rx_data_host_ = nullptr;
  uint8_t *rx_data_dev_ = nullptr;
  volatile uint64_t *tx_flags_host_ = nullptr;
  volatile uint64_t *tx_flags_dev_ = nullptr;
  uint8_t *tx_data_host_ = nullptr;
  uint8_t *tx_data_dev_ = nullptr;

  cudaq_ringbuffer_t ringbuffer_{};
  cudaq_dispatch_manager_t *manager_ = nullptr;
  cudaq_dispatcher_t *dispatcher_ = nullptr;
  cudaq_function_entry_t host_table_[1]{};
  int shutdown_flag_ = 0;
  uint64_t stats_counter_ = 0;

  void SetUp() override {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0)
      GTEST_SKIP() << "No CUDA devices available";
    cudaError_t flags_err = cudaSetDeviceFlags(cudaDeviceMapHost);
    ASSERT_TRUE(flags_err == cudaSuccess ||
                flags_err == cudaErrorSetOnActiveProcess);

    // --- Load config via public API ---
    auto mdc = decoding::config::multi_decoder_config::from_yaml_str(
        read_file(std::string(TEST_DATA_DIR) + "/config_nv_qldpc_relay.yml"));
    ASSERT_EQ(mdc.decoders.size(), 1u);
    auto &dec = mdc.decoders[0];

    std::vector<uint32_t> h_row_ptr, h_col_idx;
    std::size_t h_rows =
        realtime::sparse_vec_to_csr(dec.H_sparse, h_row_ptr, h_col_idx);
    ASSERT_EQ(h_rows, dec.syndrome_size);

    std::size_t bs = dec.block_size;
    std::size_t ss = dec.syndrome_size;
    cudaqx::tensor<uint8_t> H_tensor({ss, bs});
    for (std::size_t r = 0; r < ss; ++r)
      for (uint32_t j = h_row_ptr[r]; j < h_row_ptr[r + 1]; ++j)
        H_tensor.at({r, static_cast<std::size_t>(h_col_idx[j])}) = 1;

    auto params = dec.decoder_custom_args_to_heterogeneous_map();
    decoder_ = decoder::get("nv-qldpc-decoder", H_tensor, params);
    ASSERT_NE(decoder_, nullptr);

    decoder_->set_D_sparse(dec.D_sparse);
    decoder_->set_O_sparse(dec.O_sparse);

    std::vector<uint32_t> d_rp, d_ci;
    realtime::sparse_vec_to_csr(dec.D_sparse, d_rp, d_ci);
    num_measurements_ = 0;
    for (auto c : d_ci)
      num_measurements_ = std::max(num_measurements_, (std::size_t)(c + 1));

    std::vector<uint32_t> o_rp, o_ci;
    num_observables_ = realtime::sparse_vec_to_csr(dec.O_sparse, o_rp, o_ci);

    printf("Config: block_size=%zu, syndrome_size=%zu, "
           "num_measurements=%zu, num_observables=%zu\n",
           bs, ss, num_measurements_, num_observables_);

    // --- Capture CUDA graph ---
    ASSERT_TRUE(decoder_->supports_graph_dispatch());
    void *raw = decoder_->capture_decode_graph();
    ASSERT_NE(raw, nullptr);
    graph_res_ = static_cast<realtime::graph_resources *>(raw);
    ASSERT_NE(graph_res_->graph_exec, nullptr);
    printf("Graph captured: function_id=0x%08X\n", graph_res_->function_id);

    // --- Load syndromes ---
    syndromes_ = load_syndromes(std::string(TEST_DATA_DIR) +
                                    "/syndromes_nv_qldpc_relay.txt",
                                num_measurements_);
    printf("Loaded %zu test syndromes\n", syndromes_.size());
    ASSERT_GT(syndromes_.size(), 0u);

    // --- Allocate ring buffers ---
    slot_size_ = std::max(sizeof(RPCHeader) + num_measurements_,
                          sizeof(RPCResponse) + num_observables_);
    slot_size_ = (slot_size_ + 255) & ~255u;
    printf("Buffer size: %zu bytes\n", slot_size_);

    ASSERT_TRUE(allocate_ring_buffer(num_slots_, slot_size_, &rx_flags_host_,
                                     &rx_flags_dev_, &rx_data_host_,
                                     &rx_data_dev_));
    ASSERT_TRUE(allocate_ring_buffer(num_slots_, slot_size_, &tx_flags_host_,
                                     &tx_flags_dev_, &tx_data_host_,
                                     &tx_data_dev_));

    // --- Wire the C API: HOST_LOOP dispatcher with separate RX/TX ---
    memset(&ringbuffer_, 0, sizeof(ringbuffer_));
    ringbuffer_.rx_flags = rx_flags_dev_;
    ringbuffer_.tx_flags = tx_flags_dev_;
    ringbuffer_.rx_data = rx_data_dev_;
    ringbuffer_.tx_data = tx_data_dev_;
    ringbuffer_.rx_stride_sz = slot_size_;
    ringbuffer_.tx_stride_sz = slot_size_;
    ringbuffer_.rx_flags_host = rx_flags_host_;
    ringbuffer_.tx_flags_host = tx_flags_host_;
    ringbuffer_.rx_data_host = rx_data_host_;
    ringbuffer_.tx_data_host = tx_data_host_;

    memset(host_table_, 0, sizeof(host_table_));
    host_table_[0].function_id = graph_res_->function_id;
    host_table_[0].dispatch_mode = CUDAQ_DISPATCH_GRAPH_LAUNCH;
    host_table_[0].handler.graph_exec = graph_res_->graph_exec;

    ASSERT_EQ(cudaq_dispatch_manager_create(&manager_), CUDAQ_OK);

    cudaq_dispatcher_config_t disp_config{};
    disp_config.device_id = 0;
    disp_config.num_slots = static_cast<uint32_t>(num_slots_);
    disp_config.slot_size = static_cast<uint32_t>(slot_size_);
    disp_config.dispatch_path = CUDAQ_DISPATCH_PATH_HOST;

    ASSERT_EQ(cudaq_dispatcher_create(manager_, &disp_config, &dispatcher_),
              CUDAQ_OK);
    ASSERT_EQ(cudaq_dispatcher_set_ringbuffer(dispatcher_, &ringbuffer_),
              CUDAQ_OK);

    cudaq_function_table_t table{};
    table.entries = host_table_;
    table.count = 1;
    ASSERT_EQ(cudaq_dispatcher_set_function_table(dispatcher_, &table),
              CUDAQ_OK);

    shutdown_flag_ = 0;
    stats_counter_ = 0;
    ASSERT_EQ(cudaq_dispatcher_set_control(dispatcher_, &shutdown_flag_,
                                           &stats_counter_),
              CUDAQ_OK);

    ASSERT_EQ(cudaq_dispatcher_set_mailbox(dispatcher_, graph_res_->h_mailbox),
              CUDAQ_OK);

    ASSERT_EQ(cudaq_dispatcher_start(dispatcher_), CUDAQ_OK);
  }

  void TearDown() override {
    if (dispatcher_) {
      shutdown_flag_ = 1;
      __sync_synchronize();
      cudaq_dispatcher_stop(dispatcher_);
      cudaq_dispatcher_destroy(dispatcher_);
      dispatcher_ = nullptr;
    }
    if (manager_) {
      cudaq_dispatch_manager_destroy(manager_);
      manager_ = nullptr;
    }
    if (decoder_)
      decoder_->release_decode_graph(graph_res_);
    free_ring_buffer(rx_flags_host_, rx_data_host_);
    free_ring_buffer(tx_flags_host_, tx_data_host_);
  }

  static std::string read_file(const std::string &path) {
    std::ifstream f(path);
    return std::string((std::istreambuf_iterator<char>(f)),
                       std::istreambuf_iterator<char>());
  }
};

//==============================================================================
// Test: Graph decode of all test syndromes via HOST_LOOP dispatch
//==============================================================================

TEST_F(GraphDecodeTest, DecodesAllSyndromes) {
  int success_count = 0;
  int error_count = 0;

  using clock_t = std::chrono::high_resolution_clock;
  std::vector<double> shot_durations_us;
  shot_durations_us.reserve(syndromes_.size());

  for (std::size_t shot = 0; shot < syndromes_.size(); ++shot) {
    uint32_t slot = static_cast<uint32_t>(shot % num_slots_);

    // Wait for slot to be available (both rx and tx flags clear)
    int timeout = 5000;
    while (!cudaq_host_ringbuffer_slot_available(&ringbuffer_, slot) &&
           timeout-- > 0)
      usleep(200);
    ASSERT_GT(timeout, 0) << "Timeout waiting for slot " << slot << " at shot "
                          << shot;

    // Clear stale data in both RX and TX slots
    memset(rx_data_host_ + slot * slot_size_, 0, slot_size_);
    memset(tx_data_host_ + slot * slot_size_, 0, slot_size_);

    // Write RPC request into the RX ring buffer slot
    ASSERT_EQ(cudaq_host_ringbuffer_write_rpc_request(
                  &ringbuffer_, slot, graph_res_->function_id,
                  syndromes_[shot].measurements.data(),
                  static_cast<uint32_t>(syndromes_[shot].measurements.size()),
                  static_cast<uint32_t>(shot), 0),
              CUDAQ_OK);

    auto t_start = clock_t::now();

    // Signal the slot (host dispatcher picks it up)
    cudaq_host_ringbuffer_signal_slot(&ringbuffer_, slot);

    // Poll for READY -- the graph kernel signals via tx_flag
    int cuda_err = 0;
    cudaq_tx_status_t st = CUDAQ_TX_EMPTY;
    for (int i = 0; i < 50000 && st != CUDAQ_TX_READY; ++i) {
      usleep(200);
      st = cudaq_host_ringbuffer_poll_tx_flag(&ringbuffer_, slot, &cuda_err);
    }
    ASSERT_EQ(st, CUDAQ_TX_READY)
        << "Expected READY from graph kernel at shot " << shot << " (got " << st
        << ", cuda_err=" << cuda_err << ")";

    CUDA_CHECK(cudaDeviceSynchronize());

    auto t_end = clock_t::now();
    double duration_us =
        std::chrono::duration<double, std::micro>(t_end - t_start).count();
    shot_durations_us.push_back(duration_us);

    // Read response from the TX buffer (separate from RX)
    __sync_synchronize();
    uint8_t *slot_data = tx_data_host_ + slot * slot_size_;
    auto *response = reinterpret_cast<const RPCResponse *>(slot_data);

    ASSERT_EQ(response->magic, RPC_MAGIC_RESPONSE)
        << "Bad response magic for shot " << shot;
    EXPECT_EQ(response->status, 0) << "Non-zero status for shot " << shot;
    EXPECT_EQ(response->result_len, static_cast<uint32_t>(num_observables_))
        << "Wrong result_len for shot " << shot;
    EXPECT_EQ(response->request_id, static_cast<uint32_t>(shot))
        << "request_id mismatch for shot " << shot;

    const uint8_t *corrections = slot_data + sizeof(RPCResponse);

    if (response->status == 0 &&
        response->result_len == static_cast<uint32_t>(num_observables_)) {
      success_count++;
    } else {
      error_count++;
    }

    printf("Shot %zu: status=%d, result_len=%u, time=%.1f us, corrections=[",
           shot, response->status, response->result_len, duration_us);
    for (uint32_t i = 0; i < response->result_len && i < 8; i++) {
      printf("%u", corrections[i]);
      if (i + 1 < response->result_len)
        printf(",");
    }
    printf("]\n");

    // Release the worker and clear the slot for reuse
    cudaq_host_release_worker(dispatcher_, 0);
    cudaq_host_ringbuffer_clear_slot(&ringbuffer_, slot);
  }

  printf("\nCompleted: %d/%zu shots successful, %d errors\n", success_count,
         syndromes_.size(), error_count);

  // Timing summary (skip shot 0 as warmup)
  if (shot_durations_us.size() > 1) {
    auto begin = shot_durations_us.begin() + 1;
    auto end = shot_durations_us.end();
    std::size_t n = std::distance(begin, end);
    double sum = std::accumulate(begin, end, 0.0);
    double avg = sum / n;
    double min_val = *std::min_element(begin, end);
    double max_val = *std::max_element(begin, end);
    printf("\n[GraphDecodeTiming] shots=%zu (excluding warmup shot 0)\n", n);
    printf("[GraphDecodeTiming] min=%.1f us  avg=%.1f us  max=%.1f us\n",
           min_val, avg, max_val);
  }

  EXPECT_EQ(success_count, static_cast<int>(syndromes_.size()));
  EXPECT_EQ(error_count, 0);
}
