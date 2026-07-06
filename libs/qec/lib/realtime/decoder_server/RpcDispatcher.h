/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "ITransceiver.h"
#include "RpcWireFormat.h"

#include <cstdint>
#include <functional>
#include <unordered_map>

namespace cudaq::qec::decoder_server {

/// Helper passed to RpcDispatcher handlers that writes synchronous error
/// responses (e.g. BAD_REQUEST from header validation, BUSY from try_enqueue).
/// Successful responses are sent asynchronously by the session worker thread.
class ResponseWriter {
public:
  ResponseWriter(ITransceiver &transport, const PeerId &peer,
                 uint64_t request_id)
      : transport_(transport), peer_(peer), request_id_(request_id) {}

  void write_error(RpcStatus status);

  /// Expose the underlying transport so handlers can store it in WorkItem
  /// for async responses from the worker thread.
  ITransceiver *transport() const { return &transport_; }

private:
  ITransceiver &transport_;
  PeerId peer_;
  uint64_t request_id_;
};

/// Routes incoming RxFrames to registered handlers by function_id.
///
/// Never calls decoder methods directly.  Validates the frame header and
/// dispatches to the appropriate handler; on error, writes the error response
/// via ResponseWriter before returning.
class RpcDispatcher {
public:
  /// Handler: enqueue a WorkItem or call writer.write_error() for sync errors.
  using Handler = std::function<void(const RxFrame &, ResponseWriter &)>;

  void register_handler(uint32_t function_id, Handler h);

  /// Validate header, look up function_id, and invoke the registered handler.
  /// Writes BAD_REQUEST if the header is malformed or function_id is unknown.
  void dispatch(const RxFrame &frame, ITransceiver &transport);

private:
  std::unordered_map<uint32_t, Handler> table_;
};

} // namespace cudaq::qec::decoder_server
