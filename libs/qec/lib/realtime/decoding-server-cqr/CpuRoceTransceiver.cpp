/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CpuRoceTransceiver.h"

#include <stdexcept>

namespace cudaq::qec::decoding_server {

CpuRoceTransceiver::CpuRoceTransceiver() {
  throw std::runtime_error(
      "CpuRoceTransceiver: ibverbs/SoftRoCE implementation pending. "
      "Use LoopbackTransceiver for development or GpuRoceTransceiver for "
      "production Hololink/DOCA deployments.");
}

RxFrame CpuRoceTransceiver::recv() {
  throw std::logic_error(
      "CpuRoceTransceiver::recv() called on failed instance");
}

void CpuRoceTransceiver::send(const PeerId & /*peer*/, const uint8_t * /*data*/,
                              size_t /*len*/) {
  throw std::logic_error(
      "CpuRoceTransceiver::send() called on failed instance");
}

void CpuRoceTransceiver::shutdown() {}

} // namespace cudaq::qec::decoding_server
