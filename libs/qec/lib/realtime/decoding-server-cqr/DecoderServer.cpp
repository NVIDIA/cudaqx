/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "DecoderServer.h"

#include "cudaq/qec/logger.h"
#include "cudaq/qec/realtime/decoding_config.h"

#include <fstream>
#include <iterator>
#include <stdexcept>
#include <thread>
#include <vector>

namespace cudaq::qec::decoder_server {

using cudaq::qec::decoding::config::DecoderTransport;

// ---------------------------------------------------------------------------
// Constructors
// ---------------------------------------------------------------------------

std::unique_ptr<ITransceiver>
DecoderServer::make_transport(DecoderTransport transport_type) {
#ifdef CUDAQ_REALTIME_AVAILABLE
  // TODO: instantiate CpuRoceTransceiverAdapter / GpuRoceTransceiverAdapter
  // once the adapters are available via CUDAQ_REALTIME headers.
  (void)transport_type;
#endif
  throw std::runtime_error(
      "cpu_roce / gpu_roce require CUDAQ_REALTIME adapters (not yet "
      "available); for testing use DecoderServer(unique_ptr<ITransceiver>, "
      "config_yaml) with LoopbackTransceiver");
}

/// Read only the transport type from the YAML config without instantiating
/// any decoder sessions.
static DecoderTransport read_transport_from_yaml(const std::string &yaml_path) {
  std::ifstream f(yaml_path);
  if (!f.is_open())
    throw std::runtime_error("Cannot open config: " + yaml_path);
  std::string yaml_str((std::istreambuf_iterator<char>(f)), {});
  auto config =
      cudaq::qec::decoding::config::multi_decoder_config::from_yaml_str(
          yaml_str);
  if (config.decoders.empty())
    throw std::runtime_error("No decoders in config: " + yaml_path);
  auto transport = config.decoders.front().transport;
  // MVP limitation: all decoders in one server instance must share the same
  // transport type.  Heterogeneous deployments (e.g. decoder0=cpu_roce,
  // decoder1=gpu_roce) require per-session transceiver binding, deferred to
  // a follow-up once CpuRoce/GpuRoceTransceiverAdapter are available.
  for (const auto &dc : config.decoders)
    if (dc.transport != transport)
      throw std::runtime_error("Mixed transport types in " + yaml_path +
                               ": all decoder entries must declare the same "
                               "transport");
  return transport;
}

DecoderServer::DecoderServer(const std::string &config_yaml) {
  auto t = make_transport(read_transport_from_yaml(config_yaml));
  ITransceiver *raw = t.get();
  owned_transports_.push_back(std::move(t));
  function_transport_[kEnqueueSyndromesFunctionId] = raw;
  function_transport_[kGetCorrectionsFunctionId] = raw;
  function_transport_[kResetDecoderFunctionId] = raw;
  init(config_yaml);
}

DecoderServer::DecoderServer(std::unique_ptr<ITransceiver> transport,
                             const std::string &config_yaml) {
  ITransceiver *raw = transport.get();
  owned_transports_.push_back(std::move(transport));
  function_transport_[kEnqueueSyndromesFunctionId] = raw;
  function_transport_[kGetCorrectionsFunctionId] = raw;
  function_transport_[kResetDecoderFunctionId] = raw;
  init(config_yaml);
}

DecoderServer::DecoderServer(
    std::unique_ptr<ITransceiver> transport,
    const cudaq::qec::decoding::config::multi_decoder_config &config) {
  ITransceiver *raw = transport.get();
  owned_transports_.push_back(std::move(transport));
  function_transport_[kEnqueueSyndromesFunctionId] = raw;
  function_transport_[kGetCorrectionsFunctionId] = raw;
  function_transport_[kResetDecoderFunctionId] = raw;
  registry_.load_from_config(config, "configure_decoders()");
  register_handlers();
}

DecoderServer::DecoderServer(std::vector<std::unique_ptr<ITransceiver>> owned,
                             TransportMap function_transport,
                             const std::string &config_yaml)
    : owned_transports_(std::move(owned)),
      function_transport_(std::move(function_transport)) {
  init(config_yaml);
}

// ---------------------------------------------------------------------------
// init — load sessions and register RPC handlers
// ---------------------------------------------------------------------------

void DecoderServer::init(const std::string &config_yaml) {
  registry_.load_from_config(config_yaml);
  register_handlers();
}

void DecoderServer::register_handlers() {
  // enqueue_syndromes — worker thread sends a 24-byte ACK (result_len==0).
  dispatcher_.register_handler(
      kEnqueueSyndromesFunctionId,
      [this](RxFrame frame, ResponseWriter &writer) {
        if (frame.buf.size() < sizeof(RPCHeader) + sizeof(EnqueuePayload)) {
          writer.write_error(RpcStatus::BAD_REQUEST);
          return;
        }
        const auto *req = reinterpret_cast<const EnqueuePayload *>(
            frame.buf.data() + sizeof(RPCHeader));
        const auto *hdr = reinterpret_cast<const RPCHeader *>(frame.buf.data());

        auto &session = registry_.get(static_cast<uint64_t>(req->decoder_id));

        WorkItem item;
        item.function_id = kEnqueueSyndromesFunctionId;
        item.frame_buf = std::move(frame.buf);
        item.peer = frame.peer;
        item.request_id = hdr->request_id;
        item.ptp_timestamp = hdr->ptp_timestamp;
        item.vp_id = frame.vp_id;
        item.response_transport = writer.transport();

        if (!session.try_enqueue(std::move(item))) {
          session.latch_syndromes_dropped();
          writer.write_error(RpcStatus::SYNDROMES_DROPPED);
        }
      });

  // get_corrections — response sent by the worker thread.
  dispatcher_.register_handler(
      kGetCorrectionsFunctionId, [this](RxFrame frame, ResponseWriter &writer) {
        if (frame.buf.size() <
            sizeof(RPCHeader) + sizeof(GetCorrectionsPayload)) {
          writer.write_error(RpcStatus::BAD_REQUEST);
          return;
        }
        const auto *req = reinterpret_cast<const GetCorrectionsPayload *>(
            frame.buf.data() + sizeof(RPCHeader));
        const auto *hdr = reinterpret_cast<const RPCHeader *>(frame.buf.data());

        auto &session = registry_.get(static_cast<uint64_t>(req->decoder_id));

        WorkItem item;
        item.function_id = kGetCorrectionsFunctionId;
        item.frame_buf = std::move(frame.buf);
        item.peer = frame.peer;
        item.request_id = hdr->request_id;
        item.ptp_timestamp = hdr->ptp_timestamp;
        item.vp_id = frame.vp_id;
        item.response_transport = writer.transport();

        if (!session.try_enqueue(std::move(item)))
          writer.write_error(RpcStatus::BUSY);
      });

  // reset_decoder — response sent by the worker thread.
  dispatcher_.register_handler(
      kResetDecoderFunctionId, [this](RxFrame frame, ResponseWriter &writer) {
        if (frame.buf.size() < sizeof(RPCHeader) + sizeof(ResetPayload)) {
          writer.write_error(RpcStatus::BAD_REQUEST);
          return;
        }
        const auto *req = reinterpret_cast<const ResetPayload *>(
            frame.buf.data() + sizeof(RPCHeader));
        const auto *hdr = reinterpret_cast<const RPCHeader *>(frame.buf.data());

        auto &session = registry_.get(static_cast<uint64_t>(req->decoder_id));

        WorkItem item;
        item.function_id = kResetDecoderFunctionId;
        item.frame_buf = std::move(frame.buf);
        item.peer = frame.peer;
        item.request_id = hdr->request_id;
        item.ptp_timestamp = hdr->ptp_timestamp;
        item.vp_id = frame.vp_id;
        item.response_transport = writer.transport();

        if (!session.try_enqueue(std::move(item)))
          writer.write_error(RpcStatus::BUSY);
      });
} // register_handlers

// ---------------------------------------------------------------------------
// run / stop
// ---------------------------------------------------------------------------

void DecoderServer::run() {
  CUDA_QEC_INFO("DecoderServer: starting {} receiver thread(s)",
                owned_transports_.size());

  // All threads share dispatcher_ — routing is by function_id, not transport.
  std::vector<std::thread> recv_threads;
  recv_threads.reserve(owned_transports_.size());
  for (auto &transport : owned_transports_) {
    ITransceiver *t = transport.get();
    recv_threads.emplace_back([this, t] {
      while (!shutdown_.load(std::memory_order_acquire)) {
        auto frame = t->recv();
        if (frame.buf.empty())
          break; // shutdown sentinel; don't dispatch a spurious empty frame
        dispatcher_.dispatch(std::move(frame), *t);
      }
    });
  }

  for (auto &th : recv_threads)
    th.join();

  CUDA_QEC_INFO("DecoderServer: all receiver threads exited");
}

void DecoderServer::stop() {
  shutdown_.store(true, std::memory_order_release);
  // Unblock any receive loop parked in recv().
  for (auto &t : owned_transports_)
    t->shutdown();
}

} // namespace cudaq::qec::decoder_server
