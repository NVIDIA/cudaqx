/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "ITransceiver.h"
#include "cudaq/qec/realtime/decoder_rpc_wire_format.h"

#include <cstdint>
#include <functional>
#include <unordered_map>

namespace cudaq::qec::decoding_server {

using cudaq::qec::decoding::rpc::RpcStatus;

/// Helper passed to RpcDispatcher handlers that writes synchronous error
/// responses (e.g. BAD_REQUEST from header validation, BUSY from try_enqueue).
/// Successful responses are sent asynchronously by the session worker thread.
class ResponseWriter {
public:
  ResponseWriter(ITransceiver &transport, const PeerId &peer,
                 uint32_t request_id, uint64_t ptp_timestamp)
      : transport_(transport), peer_(peer), request_id_(request_id),
        ptp_timestamp_(ptp_timestamp) {}

  void write_error(RpcStatus status);

  /// Expose the underlying transport so handlers can store it in WorkItem
  /// for async responses from the worker thread.
  ITransceiver *transport() const { return &transport_; }

private:
  ITransceiver &transport_;
  PeerId peer_;
  uint32_t request_id_;
  uint64_t ptp_timestamp_;
};

/// Routes incoming RxFrames to registered handlers by function_id.
///
/// Never calls decoder methods directly.  Validates the frame header and
/// dispatches to the appropriate handler; on error, writes the error response
/// via ResponseWriter before returning.
class RpcDispatcher {
public:
  /// Handler: enqueue a WorkItem or call writer.write_error() for sync errors.
  /// RxFrame is passed by value so the handler can move buf into WorkItem.
  using Handler = std::function<void(RxFrame, ResponseWriter &)>;

  void register_handler(uint32_t function_id, Handler h);

  /// Validate header, look up function_id, and invoke the registered handler.
  /// Writes BAD_REQUEST if the header is malformed or function_id is unknown.
  void dispatch(RxFrame frame, ITransceiver &transport);

private:
  std::unordered_map<uint32_t, Handler> table_;
};

} // namespace cudaq::qec::decoding_server
