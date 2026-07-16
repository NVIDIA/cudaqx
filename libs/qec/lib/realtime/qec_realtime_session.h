/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#ifdef CUDAQ_REALTIME_ROOT

#include "cudaq/qec/decoder.h"
#include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"
#include "cudaq/realtime/daemon/dispatcher/host_dispatcher.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <thread>
#include <vector>

namespace cudaq::qec::realtime {

class __attribute__((visibility("default"))) qec_realtime_session {
public:
  explicit qec_realtime_session(
      std::vector<std::unique_ptr<cudaq::qec::decoder>> &decoders);
  ~qec_realtime_session();

  void initialize();
  void finalize();
  bool initialized() const { return initialized_; }

  volatile std::uint64_t *rx_flags_host() { return rx_flags_.data(); }
  volatile std::uint64_t *tx_flags_host() { return tx_flags_.data(); }
  std::uint8_t *rx_data_host() { return rx_data_.data(); }
  std::uint8_t *tx_data_host() { return tx_data_.data(); }
  std::size_t num_slots() const { return num_slots_; }
  std::size_t slot_size() const { return slot_size_; }

private:
  void allocate_ring_buffer();
  void populate_function_table();
  void start_host_loop();
  void stop_host_loop();

  std::vector<std::unique_ptr<cudaq::qec::decoder>> &decoders_;

  bool initialized_ = false;
  std::size_t num_slots_ = 8;
  std::size_t slot_size_ = 0;

  std::vector<std::uint64_t> rx_flags_;
  std::vector<std::uint64_t> tx_flags_;
  std::vector<std::uint8_t> rx_data_;
  std::vector<std::uint8_t> tx_data_;
  int shutdown_flag_ = 0;
  std::uint64_t host_stats_counter_ = 0;

  std::vector<cudaq_function_entry_t> function_table_;
  cudaq_ringbuffer_t ringbuffer_{};
  cudaq_host_dispatch_loop_ctx_t host_ctx_{};
  std::thread host_loop_thread_;
};

} // namespace cudaq::qec::realtime

#endif // CUDAQ_REALTIME_ROOT
