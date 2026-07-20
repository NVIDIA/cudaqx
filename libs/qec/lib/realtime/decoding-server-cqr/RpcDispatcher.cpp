/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "RpcDispatcher.h"

#include "cudaq/qec/logger.h"

#include <cstring>
#include <stdexcept>

namespace cudaq::qec::decoding_server {

using cudaq::realtime::RPCHeader;
using cudaq::realtime::RPCResponse;

// ---------------------------------------------------------------------------
// ResponseWriter
// ---------------------------------------------------------------------------

void ResponseWriter::write_error(RpcStatus status) {
  std::vector<uint8_t> buf(sizeof(RPCResponse));
  auto *hdr = reinterpret_cast<RPCResponse *>(buf.data());
  hdr->magic = cudaq::realtime::RPC_MAGIC_RESPONSE;
  hdr->status = static_cast<int32_t>(status);
  hdr->result_len = 0;
  hdr->request_id = request_id_;
  hdr->ptp_timestamp = ptp_timestamp_;
  transport_.send(peer_, buf.data(), buf.size());
}

// ---------------------------------------------------------------------------
// RpcDispatcher
// ---------------------------------------------------------------------------

void RpcDispatcher::register_handler(uint32_t function_id, Handler h) {
  table_.emplace(function_id, std::move(h));
}

void RpcDispatcher::dispatch(RxFrame frame, ITransceiver &transport) {
  // Minimum frame: RPCHeader only.
  if (frame.buf.size() < sizeof(RPCHeader)) {
    CUDA_QEC_DBG("RpcDispatcher: frame too short ({} bytes)", frame.buf.size());
    // Cannot build a meaningful response without a valid request_id.
    return;
  }

  const auto *hdr = reinterpret_cast<const RPCHeader *>(frame.buf.data());

  if (hdr->magic != cudaq::realtime::RPC_MAGIC_REQUEST) {
    CUDA_QEC_DBG("RpcDispatcher: bad magic 0x{:08X}", hdr->magic);
    return;
  }

  ResponseWriter writer(transport, frame.peer, hdr->request_id,
                        hdr->ptp_timestamp);

  auto it = table_.find(hdr->function_id);
  if (it == table_.end()) {
    CUDA_QEC_DBG("RpcDispatcher: unknown function_id 0x{:08X}",
                 hdr->function_id);
    writer.write_error(RpcStatus::BAD_REQUEST);
    return;
  }

  try {
    it->second(std::move(frame), writer);
  } catch (const std::out_of_range &) {
    // SessionRegistry::get() throws std::out_of_range for unknown decoder_id.
    writer.write_error(RpcStatus::INVALID_DECODER);
  } catch (const std::invalid_argument &) {
    writer.write_error(RpcStatus::BAD_REQUEST);
  } catch (const std::exception &e) {
    cudaq::qec::error("RpcDispatcher: handler threw: {}", e.what());
    writer.write_error(RpcStatus::INTERNAL_ERROR);
  }
}

} // namespace cudaq::qec::decoding_server
