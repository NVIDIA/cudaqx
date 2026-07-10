/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "ITransceiver.h"

namespace cudaq::qec::decoding_server {

/// CPU RoCE transport skeleton (ibverbs).
///
/// This class exists as a build-time stub so the server can be compiled with
/// cpu_roce in the config without a link-time error.  The constructor throws at
/// runtime until a full ibverbs QP setup is provided.
///
/// Do NOT guard construction with CUDAQ_CPU_ROCE_AVAILABLE — the class is
/// always compiled so that make_transport() can give a clear runtime error
/// rather than a linker error or a missing-symbol crash.
class CpuRoceTransceiver final : public ITransceiver {
public:
  CpuRoceTransceiver();

  RxFrame recv() override;
  void send(const PeerId &peer, const uint8_t *data, size_t len) override;
  void shutdown() override;
};

} // namespace cudaq::qec::decoding_server
