/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "DecoderServer.h"

#include "cudaq/qec/logger.h"

#include <algorithm>
#include <stdexcept>
#include <thread>
#include <vector>

namespace cudaq::qec::decoder_server {

// ---------------------------------------------------------------------------
// Constructors
// ---------------------------------------------------------------------------

DecoderServer::DecoderServer(std::unique_ptr<ITransceiver> transport,
                             const std::string &config_yaml) {
  ITransceiver *raw = transport.get();
  owned_transports_.push_back(std::move(transport));
  dispatch_map_[kEnqueueSyndromesFunctionId] = raw;
  dispatch_map_[kGetCorrectionsFunctionId] = raw;
  dispatch_map_[kResetDecoderFunctionId] = raw;
  init(config_yaml);
}

DecoderServer::DecoderServer(std::vector<std::unique_ptr<ITransceiver>> owned,
                             TransportMap dispatch_map,
                             const std::string &config_yaml)
    : owned_transports_(std::move(owned)),
      dispatch_map_(std::move(dispatch_map)) {
  init(config_yaml);
}

// ---------------------------------------------------------------------------
// init — load sessions and register RPC handlers
// ---------------------------------------------------------------------------

void DecoderServer::init(const std::string &config_yaml) {
  registry_.load_from_config(config_yaml);

  // enqueue_syndromes — fire-and-forget; no synchronous response.
  dispatcher_.register_handler(
      kEnqueueSyndromesFunctionId,
      [this](const RxFrame &frame, ResponseWriter &writer) {
        if (frame.bytes.size() < sizeof(RPCHeader) + sizeof(EnqueuePayload)) {
          writer.write_error(RpcStatus::BAD_REQUEST);
          return;
        }
        const auto *req = reinterpret_cast<const EnqueuePayload *>(
            frame.bytes.data() + sizeof(RPCHeader));
        const auto *hdr =
            reinterpret_cast<const RPCHeader *>(frame.bytes.data());

        auto &session = registry_.get(static_cast<uint64_t>(req->decoder_id));

        WorkItem item;
        item.function_id = kEnqueueSyndromesFunctionId;
        item.payload.assign(frame.bytes.begin(), frame.bytes.end());
        item.peer = frame.peer;
        item.request_id = hdr->request_id;
        item.ptp_timestamp = hdr->ptp_timestamp;
        item.vp_id = frame.vp_id;
        item.response_transport = writer.transport();

        if (!session.try_enqueue(std::move(item)))
          session.latch_syndromes_dropped();
      });

  // get_corrections — response sent by the worker thread.
  dispatcher_.register_handler(
      kGetCorrectionsFunctionId,
      [this](const RxFrame &frame, ResponseWriter &writer) {
        if (frame.bytes.size() <
            sizeof(RPCHeader) + sizeof(GetCorrectionsPayload)) {
          writer.write_error(RpcStatus::BAD_REQUEST);
          return;
        }
        const auto *req = reinterpret_cast<const GetCorrectionsPayload *>(
            frame.bytes.data() + sizeof(RPCHeader));
        const auto *hdr =
            reinterpret_cast<const RPCHeader *>(frame.bytes.data());

        auto &session = registry_.get(static_cast<uint64_t>(req->decoder_id));

        WorkItem item;
        item.function_id = kGetCorrectionsFunctionId;
        item.payload.assign(frame.bytes.begin(), frame.bytes.end());
        item.peer = frame.peer;
        item.request_id = hdr->request_id;
        item.ptp_timestamp = 0;
        item.vp_id = frame.vp_id;
        item.response_transport = writer.transport();

        if (!session.try_enqueue(std::move(item)))
          writer.write_error(RpcStatus::BUSY);
      });

  // reset_decoder — response sent by the worker thread.
  dispatcher_.register_handler(
      kResetDecoderFunctionId,
      [this](const RxFrame &frame, ResponseWriter &writer) {
        if (frame.bytes.size() < sizeof(RPCHeader) + sizeof(ResetPayload)) {
          writer.write_error(RpcStatus::BAD_REQUEST);
          return;
        }
        const auto *req = reinterpret_cast<const ResetPayload *>(
            frame.bytes.data() + sizeof(RPCHeader));
        const auto *hdr =
            reinterpret_cast<const RPCHeader *>(frame.bytes.data());

        auto &session = registry_.get(static_cast<uint64_t>(req->decoder_id));

        WorkItem item;
        item.function_id = kResetDecoderFunctionId;
        item.payload.assign(frame.bytes.begin(), frame.bytes.end());
        item.peer = frame.peer;
        item.request_id = hdr->request_id;
        item.ptp_timestamp = 0;
        item.vp_id = frame.vp_id;
        item.response_transport = writer.transport();

        if (!session.try_enqueue(std::move(item)))
          writer.write_error(RpcStatus::BUSY);
      });
}

// ---------------------------------------------------------------------------
// run / stop
// ---------------------------------------------------------------------------

void DecoderServer::run() {
  std::vector<ITransceiver *> unique_transports;
  for (auto &[fid, t] : dispatch_map_) {
    if (std::find(unique_transports.begin(), unique_transports.end(), t) ==
        unique_transports.end())
      unique_transports.push_back(t);
  }

  CUDA_QEC_INFO("DecoderServer: starting {} receiver thread(s)",
                unique_transports.size());

  // All threads share dispatcher_ — routing is by function_id, not transport.
  std::vector<std::thread> recv_threads;
  recv_threads.reserve(unique_transports.size());
  for (ITransceiver *t : unique_transports) {
    recv_threads.emplace_back([this, t] {
      while (!shutdown_.load(std::memory_order_acquire)) {
        auto frame = t->recv();
        dispatcher_.dispatch(frame, *t);
        t->release(frame);
      }
    });
  }

  for (auto &th : recv_threads)
    th.join();

  CUDA_QEC_INFO("DecoderServer: all receiver threads exited");
}

void DecoderServer::stop() { shutdown_.store(true, std::memory_order_release); }

} // namespace cudaq::qec::decoder_server
