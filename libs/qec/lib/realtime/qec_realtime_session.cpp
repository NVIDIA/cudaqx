/*******************************************************************************
 * Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#ifdef CUDAQ_REALTIME_ROOT

#include "qec_realtime_session.h"

#include "cudaq/qec/realtime/decoder_rpc_ids.h"
#include "cudaq/runtime/logger/logger.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>

namespace cudaq::qec::realtime {
namespace {

using Decoders = std::vector<std::unique_ptr<cudaq::qec::decoder>>;
Decoders *g_active_decoders = nullptr;

cudaq::qec::decoder *get_decoder_or_throw(std::int64_t decoder_id) {
  if (!g_active_decoders || decoder_id < 0 ||
      static_cast<std::size_t>(decoder_id) >= g_active_decoders->size() ||
      !(*g_active_decoders)[static_cast<std::size_t>(decoder_id)])
    throw std::runtime_error("invalid decoder_id " + std::to_string(decoder_id));
  return (*g_active_decoders)[static_cast<std::size_t>(decoder_id)].get();
}

void write_response(void *slot_host, std::int32_t status,
                    std::uint32_t result_len = 0) {
  auto *header = static_cast<cudaq::realtime::RPCHeader *>(slot_host);
  auto *response = static_cast<cudaq::realtime::RPCResponse *>(slot_host);
  const std::uint32_t request_id = header->request_id;
  const std::uint64_t ptp_timestamp = header->ptp_timestamp;
  response->status = status;
  response->result_len = result_len;
  response->request_id = request_id;
  response->ptp_timestamp = ptp_timestamp;
  __atomic_store_n(&response->magic, cudaq::realtime::RPC_MAGIC_RESPONSE,
                   __ATOMIC_RELEASE);
}

std::uint8_t *response_body(void *slot_host) {
  return static_cast<std::uint8_t *>(slot_host) +
         sizeof(cudaq::realtime::RPCResponse);
}

void enqueue_syndromes_host(void *slot_host, std::size_t slot_size) {
  namespace rpc = cudaq::qec::decoding::rpc;
  try {
    auto *header = static_cast<cudaq::realtime::RPCHeader *>(slot_host);
    if (header->arg_len < sizeof(rpc::EnqueueRequestPayload)) {
      write_response(slot_host, -1);
      return;
    }
    auto *body = reinterpret_cast<const rpc::EnqueueRequestPayload *>(
        static_cast<std::uint8_t *>(slot_host) +
        sizeof(cudaq::realtime::RPCHeader));
    if (body->num_syndromes < 0 || body->syndrome_mapping_id != 0) {
      write_response(slot_host, -4);
      return;
    }
    const auto num_syndromes =
        static_cast<std::uint64_t>(body->num_syndromes);
    const std::size_t expected_arg_len =
        rpc::align_to_8(sizeof(rpc::EnqueueRequestPayload) +
                        rpc::bit_packed_bytes(num_syndromes));
    if (header->arg_len != expected_arg_len ||
        sizeof(cudaq::realtime::RPCHeader) + expected_arg_len > slot_size) {
      write_response(slot_host, -4);
      return;
    }

    auto *decoder = get_decoder_or_throw(body->decoder_id);
    const std::uint8_t *bits =
        reinterpret_cast<const std::uint8_t *>(body + 1);
    std::vector<std::uint8_t> syndromes(num_syndromes, 0);
    for (std::uint64_t bit = 0; bit < num_syndromes; ++bit)
      syndromes[bit] = (bits[bit >> 3] >> (bit & 7)) & 0x1u;

    (void)decoder->enqueue_syndrome(syndromes.data(), syndromes.size());
    write_response(slot_host, 0);
  } catch (...) {
    write_response(slot_host, -2);
  }
}

void get_corrections_host(void *slot_host, std::size_t slot_size) {
  namespace rpc = cudaq::qec::decoding::rpc;
  try {
    auto *header = static_cast<cudaq::realtime::RPCHeader *>(slot_host);
    if (header->arg_len != sizeof(rpc::GetCorrectionsRequestPayload)) {
      write_response(slot_host, -1);
      return;
    }
    auto *body = reinterpret_cast<const rpc::GetCorrectionsRequestPayload *>(
        static_cast<std::uint8_t *>(slot_host) +
        sizeof(cudaq::realtime::RPCHeader));
    if (body->return_size < 0) {
      write_response(slot_host, -4);
      return;
    }

    auto *decoder = get_decoder_or_throw(body->decoder_id);
    const auto return_size = static_cast<std::uint64_t>(body->return_size);
    if (return_size > decoder->get_num_observables()) {
      write_response(slot_host, -4);
      return;
    }
    const std::size_t result_len =
        rpc::align_to_8(rpc::bit_packed_bytes(return_size));
    if (sizeof(cudaq::realtime::RPCResponse) + result_len > slot_size) {
      write_response(slot_host, -5);
      return;
    }

    std::uint8_t *out = response_body(slot_host);
    std::memset(out, 0, result_len);
    const std::uint8_t *corrections = decoder->get_obs_corrections();
    for (std::uint64_t i = 0; i < return_size; ++i) {
      if (corrections[i] & 0x1u)
        out[i >> 3] |= static_cast<std::uint8_t>(1u << (i & 7));
    }
    if (body->reset != 0)
      decoder->clear_corrections();
    write_response(slot_host, 0, static_cast<std::uint32_t>(result_len));
  } catch (...) {
    write_response(slot_host, -2);
  }
}

void reset_decoder_host(void *slot_host, std::size_t) {
  namespace rpc = cudaq::qec::decoding::rpc;
  try {
    auto *header = static_cast<cudaq::realtime::RPCHeader *>(slot_host);
    if (header->arg_len != sizeof(rpc::ResetRequestPayload)) {
      write_response(slot_host, -1);
      return;
    }
    auto *body = reinterpret_cast<const rpc::ResetRequestPayload *>(
        static_cast<std::uint8_t *>(slot_host) +
        sizeof(cudaq::realtime::RPCHeader));
    get_decoder_or_throw(body->decoder_id)->reset_decoder();
    write_response(slot_host, 0);
  } catch (...) {
    write_response(slot_host, -2);
  }
}

} // namespace

qec_realtime_session::qec_realtime_session(Decoders &decoders)
    : decoders_(decoders) {}

qec_realtime_session::~qec_realtime_session() { finalize(); }

void qec_realtime_session::initialize() {
  if (initialized_)
    return;
  try {
    allocate_ring_buffer();
    populate_function_table();
    g_active_decoders = &decoders_;
    start_host_loop();
    initialized_ = true;
  } catch (...) {
    finalize();
    throw;
  }
}

void qec_realtime_session::finalize() {
  const bool was_initialized = initialized_;
  initialized_ = false;
  stop_host_loop();
  if (g_active_decoders == &decoders_)
    g_active_decoders = nullptr;
  function_table_.clear();
  rx_flags_.clear();
  tx_flags_.clear();
  rx_data_.clear();
  tx_data_.clear();
  std::memset(&ringbuffer_, 0, sizeof(ringbuffer_));
  std::memset(&host_ctx_, 0, sizeof(host_ctx_));
  if (was_initialized)
    CUDAQ_INFO("qec_realtime_session: finalized HOST_CALL session");
}

void qec_realtime_session::allocate_ring_buffer() {
  namespace rpc = cudaq::qec::decoding::rpc;
  std::size_t max_syndromes = 0;
  std::size_t max_observables = 0;
  for (auto &decoder : decoders_) {
    if (!decoder)
      continue;
    max_syndromes =
        std::max<std::size_t>(max_syndromes, decoder->get_num_msyn_per_decode());
    max_observables =
        std::max<std::size_t>(max_observables, decoder->get_num_observables());
  }

  const std::size_t enqueue_bytes =
      sizeof(cudaq::realtime::RPCHeader) +
      rpc::align_to_8(sizeof(rpc::EnqueueRequestPayload) +
                      rpc::bit_packed_bytes(max_syndromes));
  const std::size_t get_request_bytes =
      sizeof(cudaq::realtime::RPCHeader) +
      sizeof(rpc::GetCorrectionsRequestPayload);
  const std::size_t get_response_bytes =
      sizeof(cudaq::realtime::RPCResponse) +
      rpc::align_to_8(rpc::bit_packed_bytes(max_observables));
  const std::size_t reset_bytes = sizeof(cudaq::realtime::RPCHeader) +
                                  sizeof(rpc::ResetRequestPayload);
  slot_size_ =
      std::max({enqueue_bytes, get_request_bytes, get_response_bytes,
                reset_bytes, std::size_t{64}});

  rx_flags_.assign(num_slots_, 0);
  tx_flags_.assign(num_slots_, 0);
  rx_data_.assign(num_slots_ * slot_size_, 0);
  tx_data_.assign(num_slots_ * slot_size_, 0);

  std::memset(&ringbuffer_, 0, sizeof(ringbuffer_));
  ringbuffer_.rx_flags_host = rx_flags_.data();
  ringbuffer_.tx_flags_host = tx_flags_.data();
  ringbuffer_.rx_data_host = rx_data_.data();
  ringbuffer_.tx_data_host = tx_data_.data();
  ringbuffer_.rx_stride_sz = slot_size_;
  ringbuffer_.tx_stride_sz = slot_size_;
}

void qec_realtime_session::populate_function_table() {
  namespace rpc = cudaq::qec::decoding::rpc;
  function_table_.assign(3, cudaq_function_entry_t{});

  function_table_[0].handler.host_fn = enqueue_syndromes_host;
  function_table_[0].function_id = rpc::kEnqueueSyndromesFunctionId;
  function_table_[0].dispatch_mode = CUDAQ_DISPATCH_HOST_CALL;

  function_table_[1].handler.host_fn = get_corrections_host;
  function_table_[1].function_id = rpc::kGetCorrectionsFunctionId;
  function_table_[1].dispatch_mode = CUDAQ_DISPATCH_HOST_CALL;

  function_table_[2].handler.host_fn = reset_decoder_host;
  function_table_[2].function_id = rpc::kResetDecoderFunctionId;
  function_table_[2].dispatch_mode = CUDAQ_DISPATCH_HOST_CALL;
}

void qec_realtime_session::start_host_loop() {
  shutdown_flag_ = 0;
  std::memset(&host_ctx_, 0, sizeof(host_ctx_));
  host_ctx_.ringbuffer = ringbuffer_;
  host_ctx_.config.num_slots = static_cast<std::uint32_t>(num_slots_);
  host_ctx_.config.slot_size = static_cast<std::uint32_t>(slot_size_);
  host_ctx_.config.dispatch_path = CUDAQ_DISPATCH_PATH_HOST;
  host_ctx_.config.dispatch_mode = CUDAQ_DISPATCH_HOST_CALL;
  host_ctx_.config.skip_tx_markers = 1;
  host_ctx_.function_table.entries = function_table_.data();
  host_ctx_.function_table.count =
      static_cast<std::uint32_t>(function_table_.size());
  host_ctx_.shutdown_flag = &shutdown_flag_;
  host_ctx_.stats_counter = &host_stats_counter_;
  host_ctx_.skip_stream_sweep = true;

  host_loop_thread_ =
      std::thread([this]() { cudaq_host_dispatcher_loop(&host_ctx_); });
}

void qec_realtime_session::stop_host_loop() {
  if (host_loop_thread_.joinable()) {
    __atomic_store_n(&shutdown_flag_, 1, __ATOMIC_RELEASE);
    __sync_synchronize();
    host_loop_thread_.join();
  }
}

} // namespace cudaq::qec::realtime

#endif // CUDAQ_REALTIME_ROOT
