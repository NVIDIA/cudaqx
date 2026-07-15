/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "DecodingServer.h"
#include "CpuRoceTransceiver.h"

#include "cudaq/qec/logger.h"
#include "cudaq/qec/realtime/decoding_config.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <thread>
#include <vector>

// GPU RoCE support is an optional component (cudaq-qec-decoding-server-gpuroce)
// so this core library carries no DOCA / Hololink / CUDA-driver dependencies:
// those .so's require libcuda.so.1 at load time, which core consumers (unit
// tests, the CQR plugin) must not impose on driverless machines.  Binaries
// that want the gpu_roce transport link the component WHOLE_ARCHIVE, whose
// GpuRoceFactory.cpp provides the strong definition of this factory; anywhere
// else the weak reference is null and make_transport throws.
extern "C" __attribute__((weak)) cudaq::qec::decoding_server::ITransceiver *
cudaqx_qec_make_gpu_roce_transceiver(int pinned_cuda_device);

namespace cudaq::qec::decoding_server {

using cudaq::qec::decoding::config::DecoderTransport;

namespace {

int pinned_cuda_device_for_transport(const SessionRegistry &registry) {
  const auto &sessions = registry.sessions();
  return sessions.size() == 1
             ? sessions.begin()->second->dec->get_cuda_device_id()
             : -1;
}

void validate_transport_requirements(const SessionRegistry &registry,
                                     const std::string &source_name) {
  if (registry.required_transport() != DecoderTransport::gpu_roce)
    return;

  const auto &sessions = registry.sessions();
  if (sessions.size() != 1)
    throw std::runtime_error("GPU RoCE transport currently supports exactly "
                             "one decoder session in " +
                             source_name + "; found " +
                             std::to_string(sessions.size()) +
                             ". Multi-decoder GPU RoCE is deferred.");

  auto *session = sessions.begin()->second.get();
  if (!session->graph_resources)
    throw std::runtime_error(
        "GPU RoCE requires a decoder that supports graph dispatch in " +
        source_name +
        " (supports_graph_dispatch() must return true and "
        "capture_decode_graph() must succeed)");
}

} // namespace
// ---------------------------------------------------------------------------
// Constructors
// ---------------------------------------------------------------------------

/// Resolve the CUDA device a decode pipeline runs on from the decoder's
/// cuda_device_id pin; an unpinned decoder (-1) defaults to device 0. The
/// gpu_roce path relies on this to place its rings, dispatch scheduler, and
/// device-side graph fire on the one GPU the FPGA/NIC is affine to -- CUDA
/// graphs cannot split capture and launch across devices, so the decoder must
/// be pinned to that device.
int resolve_decode_device(int decoder_pin) {
  return decoder_pin >= 0 ? decoder_pin : 0;
}

std::unique_ptr<ITransceiver>
DecodingServer::make_transport(DecoderTransport transport_type,
                               int pinned_cuda_device) {
  switch (transport_type) {
  case DecoderTransport::gpu_roce:
    // gpu_roce lives in the cudaq-qec-decoding-server-gpuroce component,
    // reached through the weak factory.  The device is the decoder's
    // cuda_device_id pin, resolved inside the factory where GpuRoceConfig
    // lives; we just thread the pin to it.
    if (cudaqx_qec_make_gpu_roce_transceiver)
      return std::unique_ptr<ITransceiver>(
          cudaqx_qec_make_gpu_roce_transceiver(pinned_cuda_device));
    throw std::runtime_error(
        "gpu_roce transport requested but GPU RoCE support is not linked into "
        "this binary. Build with HOLOSCAN_SENSOR_BRIDGE_BUILD_DIR and DOCA "
        "libs, and link cudaq-qec-decoding-server-gpuroce (whole-archive).");

  case DecoderTransport::cpu_roce:
    // CpuRoceTransceiver constructor always throws (ibverbs pending).
    return std::make_unique<CpuRoceTransceiver>();
  }
  throw std::runtime_error("make_transport: unknown DecoderTransport value");
}

DecodingServer::DecodingServer(const std::string &config_path)
    : DecodingServer(
          yaml_string_tag_t{},
          [&config_path]() -> std::string {
            std::ifstream f(config_path);
            if (!f.is_open())
              throw std::runtime_error("Cannot open config: " + config_path);
            return {std::istreambuf_iterator<char>(f), {}};
          }(),
          config_path) {}

DecodingServer::DecodingServer(yaml_string_tag_t, const std::string &yaml_str,
                               const std::string &config_path) {
  try {
    auto config =
        cudaq::qec::decoding::config::multi_decoder_config::from_yaml_str(
            yaml_str);
    if (config.decoders.empty())
      throw std::runtime_error("No decoders in config: " + config_path);
    registry_.load_from_config(config, config_path);
    validate_transport_requirements(registry_, config_path);
    register_handlers();

    const auto transport_type = registry_.required_transport();
    // gpu_roce must run on the GPU the FPGA/NIC is affine to; when exactly
    // one session is booting, pass its decoder's cuda_device_id so the
    // factory can place the transport on that device.
    const int pinned_cuda_device = pinned_cuda_device_for_transport(registry_);
    auto t = make_transport(transport_type, pinned_cuda_device);
    transport_cuda_device_ = resolve_decode_device(pinned_cuda_device);
    ITransceiver *raw = t.get();
    owned_transports_.push_back(std::move(t));
    function_transport_[kEnqueueSyndromesFunctionId] = raw;
    function_transport_[kGetCorrectionsFunctionId] = raw;
    function_transport_[kResetDecoderFunctionId] = raw;

    // For the GPU RoCE path, wire the first session's decoder graph to the
    // Hololink ring buffer via the CUDAQ device-graph scheduler. The shared
    // validation above enforces that there is exactly one compatible session.
    if (transport_type == DecoderTransport::gpu_roce) {
      auto *session = registry_.sessions().begin()->second.get();
      if (!raw->launch_device_scheduler(session->graph_resources.get()))
        throw std::runtime_error(
            "gpu_roce transceiver did not provide a device scheduler");
    }
  } catch (...) {
    registry_.stop_workers();
    throw;
  }
}

std::unique_ptr<DecodingServer>
DecodingServer::from_yaml_str(const std::string &yaml_str,
                              const std::string &config_path) {
  // Use new directly: make_unique cannot reach the private constructor.
  return std::unique_ptr<DecodingServer>(
      new DecodingServer(yaml_string_tag_t{}, yaml_str, config_path));
}

void DecodingServer::reconfigure_from_yaml_str(const std::string &yaml_str,
                                               const std::string &config_path) {
  // GPU RoCE only: run() stays alive, but CPU handler lambdas are idle while
  // the device scheduler owns dispatch during registry replacement.
  auto config =
      cudaq::qec::decoding::config::multi_decoder_config::from_yaml_str(
          yaml_str);
  if (config.decoders.empty())
    throw std::runtime_error("No decoders in config: " + config_path);

  const auto transport_type = config.decoders.front().transport;
  for (const auto &decoder : config.decoders) {
    if (decoder.transport != transport_type)
      throw std::runtime_error("Mixed transport types in " + config_path +
                               ": all decoder entries must declare the same "
                               "transport");
  }
  if (transport_type != DecoderTransport::gpu_roce)
    throw std::runtime_error(
        "transport-preserving reconfigure currently requires gpu_roce");
  if (config.decoders.size() != 1)
    throw std::runtime_error(
        "GPU RoCE reconfigure currently requires exactly one decoder");
  const int new_cuda_device = resolve_decode_device(
      config.decoders.front().cuda_device_id.value_or(-1));
  if (transport_cuda_device_ >= 0 && new_cuda_device != transport_cuda_device_)
    throw std::runtime_error(
        "cannot change cuda_device_id during GPU RoCE live reconfigure");
  if (!registry_.sessions().empty() &&
      registry_.required_transport() != transport_type)
    throw std::runtime_error(
        "cannot change decoder transport during live reconfigure");
  if (owned_transports_.empty())
    throw std::runtime_error("cannot reconfigure without an owned transport");

  for (auto &transport : owned_transports_)
    transport->stop_device_scheduler();
  registry_.stop_workers();
  registry_ = SessionRegistry{};

  try {
    registry_.load_from_config(config, config_path);
    validate_transport_requirements(registry_, config_path);

    auto *session = registry_.sessions().begin()->second.get();
    bool launched = false;
    for (auto &transport : owned_transports_) {
      if (transport->launch_device_scheduler(session->graph_resources.get())) {
        launched = true;
        break;
      }
    }
    if (!launched)
      throw std::runtime_error(
          "gpu_roce transceiver did not provide a device scheduler");
    transport_cuda_device_ = new_cuda_device;
  } catch (...) {
    for (auto &transport : owned_transports_)
      transport->stop_device_scheduler();
    registry_.stop_workers();
    // Decoder-less awaiting-config state; the run() thread stays alive, and
    // RPCs fail until a valid reconfigure succeeds.
    registry_ = SessionRegistry{};
    throw;
  }
}

DecodingServer::DecodingServer(std::unique_ptr<ITransceiver> transport,
                               const std::string &config_yaml) {
  ITransceiver *raw = transport.get();
  owned_transports_.push_back(std::move(transport));
  function_transport_[kEnqueueSyndromesFunctionId] = raw;
  function_transport_[kGetCorrectionsFunctionId] = raw;
  function_transport_[kResetDecoderFunctionId] = raw;
  try {
    init(config_yaml);
  } catch (...) {
    registry_.stop_workers();
    throw;
  }
}

DecodingServer::DecodingServer(
    std::unique_ptr<ITransceiver> transport,
    const cudaq::qec::decoding::config::multi_decoder_config &config) {
  ITransceiver *raw = transport.get();
  owned_transports_.push_back(std::move(transport));
  function_transport_[kEnqueueSyndromesFunctionId] = raw;
  function_transport_[kGetCorrectionsFunctionId] = raw;
  function_transport_[kResetDecoderFunctionId] = raw;
  try {
    registry_.load_from_config(config, "configure_decoders()");
  } catch (...) {
    // Members destroy in reverse order (transports before registry); join any
    // already-started workers while the transports still exist.
    registry_.stop_workers();
    throw;
  }
  register_handlers();
}

DecodingServer::DecodingServer(std::vector<std::unique_ptr<ITransceiver>> owned,
                               TransportMap function_transport,
                               const std::string &config_yaml)
    : owned_transports_(std::move(owned)),
      function_transport_(std::move(function_transport)) {
  try {
    init(config_yaml);
  } catch (...) {
    registry_.stop_workers();
    throw;
  }
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

void DecodingServer::print_session_stats() const {
  for (const auto &[id, session] : registry_.sessions()) {
    std::cout << "QEC_DECODING_SERVER_DECODER_STATS id=" << id
              << " decodes=" << session->decode_count.load()
              << " enqueues=" << session->enqueue_count.load()
              << " corrections=" << session->get_corrections_count.load()
              << " resets=" << session->reset_count.load()
              << " errors=" << session->error_count.load() << std::endl;
  }
}

void DecodingServer::stop() {
  shutdown_.store(true, std::memory_order_release);
  // Unblock any receive loop parked in recv().
  for (auto &t : owned_transports_)
    t->shutdown();
}

} // namespace cudaq::qec::decoding_server
