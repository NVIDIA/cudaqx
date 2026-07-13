/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "DecodingServer.h"
#include "CpuRoceTransceiver.h"
#include "GpuRoceTransceiver.h"

#include "cudaq/qec/logger.h"
#include "cudaq/qec/realtime/decoding_config.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <thread>
#include <vector>

namespace cudaq::qec::decoding_server {

using cudaq::qec::decoding::config::DecoderTransport;

namespace {

int resolve_gpu_roce_device_id(
    const cudaq::qec::decoding::config::decoder_config &decoder) {
  const char *raw_env_id = std::getenv("HOLOLINK_GPU_ID");
  const int env_id = raw_env_id ? std::stoi(raw_env_id) : 0;
  if (decoder.cuda_device_id.has_value() && raw_env_id &&
      decoder.cuda_device_id.value() != env_id)
    throw std::runtime_error("HOLOLINK_GPU_ID=" + std::to_string(env_id) +
                             " conflicts with decoder cuda_device_id=" +
                             std::to_string(decoder.cuda_device_id.value()));
  return decoder.cuda_device_id.value_or(env_id);
}

} // namespace

// ---------------------------------------------------------------------------
// Constructors
// ---------------------------------------------------------------------------

std::unique_ptr<ITransceiver>
DecodingServer::make_transport(DecoderTransport transport_type,
                               int cuda_device_id) {
  switch (transport_type) {
  case DecoderTransport::gpu_roce:
#ifdef CUDAQ_GPU_ROCE_AVAILABLE
  {
    auto config = GpuRoceConfig::from_env();
    if (cuda_device_id >= 0) {
      if (std::getenv("HOLOLINK_GPU_ID") && config.gpu_id != cuda_device_id)
        throw std::runtime_error(
            "HOLOLINK_GPU_ID=" + std::to_string(config.gpu_id) +
            " conflicts with decoder cuda_device_id=" +
            std::to_string(cuda_device_id));
      config.gpu_id = cuda_device_id;
    }
    return std::make_unique<GpuRoceTransceiver>(config);
  }
#else
    throw std::runtime_error(
        "gpu_roce transport requested but CUDAQ_GPU_ROCE_AVAILABLE is not set. "
        "Build with HOLOSCAN_SENSOR_BRIDGE_BUILD_DIR and DOCA libs.");
#endif

  case DecoderTransport::cpu_roce:
    // CpuRoceTransceiver constructor always throws (ibverbs pending).
    return std::make_unique<CpuRoceTransceiver>();
  }
  throw std::runtime_error("make_transport: unknown DecoderTransport value");
}

DecodingServer::DecodingServer(const std::string &config_yaml) {
  // Parse the YAML once: SessionRegistry validates the decoder entries
  // (including the uniform-transport rule — MVP limitation: heterogeneous
  // deployments require per-session transceiver binding, deferred to a
  // follow-up once CpuRoce/GpuRoceTransceiverAdapter are available) and
  // required_transport() then drives transceiver creation.
  std::ifstream f(config_yaml);
  if (!f.is_open())
    throw std::runtime_error("Cannot open config: " + config_yaml);
  std::string yaml_str((std::istreambuf_iterator<char>(f)), {});
  auto config =
      cudaq::qec::decoding::config::multi_decoder_config::from_yaml_str(
          yaml_str);
  if (config.decoders.empty())
    throw std::runtime_error("No decoders in config: " + config_yaml);

  if (config.decoders.front().transport == DecoderTransport::gpu_roce) {
    if (config.decoders.size() != 1)
      throw std::runtime_error(
          "GPU RoCE transport currently supports exactly one decoder session; "
          "found " +
          std::to_string(config.decoders.size()) +
          ". Multi-decoder GPU RoCE is deferred.");
    // GPU RoCE has one physical CUDA owner. If YAML omits placement, inherit
    // the transport's legacy environment setting (GPU 0 by default) before
    // decoder construction so graph capture and DOCA allocations cannot land
    // on different devices.
    config.decoders.front().cuda_device_id =
        resolve_gpu_roce_device_id(config.decoders.front());
  }

  registry_.load_from_config(config, config_yaml);
  register_handlers();

  int cuda_device_id = -1;
  if (registry_.required_transport() == DecoderTransport::gpu_roce) {
    const auto &sessions = registry_.sessions();
    if (sessions.size() != 1)
      throw std::runtime_error(
          "GPU RoCE transport currently supports exactly one decoder session; "
          "found " +
          std::to_string(sessions.size()) +
          ". Multi-decoder GPU RoCE is deferred.");
    cuda_device_id = sessions.begin()->second->dec->get_cuda_device_id();
  }

  auto t = make_transport(registry_.required_transport(), cuda_device_id);
  ITransceiver *raw = t.get();
  owned_transports_.push_back(std::move(t));
  function_transport_[kEnqueueSyndromesFunctionId] = raw;
  function_transport_[kGetCorrectionsFunctionId] = raw;
  function_transport_[kResetDecoderFunctionId] = raw;

#ifdef CUDAQ_GPU_ROCE_AVAILABLE
  // For the GPU RoCE path, wire the first (and only) session's decoder graph
  // to the Hololink ring buffer via the CUDAQ device-graph scheduler.
  // Multi-decoder GPU RoCE binding is deferred to a follow-up.
  if (auto *gpu_trx = dynamic_cast<GpuRoceTransceiver *>(raw)) {
    const auto &sessions = registry_.sessions();
    auto *session = sessions.begin()->second.get();
    if (!session->graph_resources)
      throw std::runtime_error(
          "GPU RoCE requires a decoder that supports graph dispatch "
          "(supports_graph_dispatch() must return true and "
          "capture_decode_graph() must succeed)");
    gpu_trx->launch_scheduler(session->graph_resources.get());
  }
#endif
}

DecodingServer::DecodingServer(std::unique_ptr<ITransceiver> transport,
                               const std::string &config_yaml) {
  ITransceiver *raw = transport.get();
  owned_transports_.push_back(std::move(transport));
  function_transport_[kEnqueueSyndromesFunctionId] = raw;
  function_transport_[kGetCorrectionsFunctionId] = raw;
  function_transport_[kResetDecoderFunctionId] = raw;
  init(config_yaml);
}

DecodingServer::DecodingServer(
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

DecodingServer::DecodingServer(std::vector<std::unique_ptr<ITransceiver>> owned,
                               TransportMap function_transport,
                               const std::string &config_yaml)
    : owned_transports_(std::move(owned)),
      function_transport_(std::move(function_transport)) {
  init(config_yaml);
}

DecodingServer::~DecodingServer() {
  stop();
  // Join session workers while owned_transports_ is still alive: queued
  // WorkItems reply via raw ITransceiver pointers.  Decoder/graph teardown
  // still happens in ~registry_, after the transports, per the member-order
  // comment in DecodingServer.h.
  registry_.stop_workers();
}

// ---------------------------------------------------------------------------
// init — load sessions and register RPC handlers
// ---------------------------------------------------------------------------

void DecodingServer::init(const std::string &config_yaml) {
  registry_.load_from_config(config_yaml);
  register_handlers();
}

void DecodingServer::register_handlers() {
  // enqueue_syndromes — fire-and-forget at the RPC level; the transport
  // layer ACKs delivery (ACCEPTED), and a queue-full drop is reported both
  // here and at the next get_corrections.
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
        item.release_fn = std::move(frame.release_fn);

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

void DecodingServer::run() {
  std::vector<ITransceiver *> unique_transports;
  for (auto &[fid, t] : function_transport_) {
    if (std::find(unique_transports.begin(), unique_transports.end(), t) ==
        unique_transports.end())
      unique_transports.push_back(t);
  }

  CUDA_QEC_INFO("DecodingServer: starting {} receiver thread(s)",
                unique_transports.size());

  // All threads share dispatcher_ — routing is by function_id, not transport.
  std::vector<std::thread> recv_threads;
  recv_threads.reserve(unique_transports.size());
  for (ITransceiver *t : unique_transports) {
    recv_threads.emplace_back([this, t] {
      while (!shutdown_.load(std::memory_order_acquire)) {
        RxFrame frame = t->recv();
        if (frame.buf.empty())
          continue; // shutdown sentinel; loop re-checks the flag
        dispatcher_.dispatch(std::move(frame), *t);
      }
    });
  }

  for (auto &th : recv_threads)
    th.join();

  CUDA_QEC_INFO("DecodingServer: all receiver threads exited");
}

void DecodingServer::stop() {
  shutdown_.store(true, std::memory_order_release);
  // Unblock any receive loop parked in recv().
  for (auto &t : owned_transports_)
    t->shutdown();
}

} // namespace cudaq::qec::decoding_server
