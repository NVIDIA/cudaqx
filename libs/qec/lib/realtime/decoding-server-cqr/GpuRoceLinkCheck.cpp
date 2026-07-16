/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Link canary for the GPU RoCE component -- not meant to be executed
// (running it would require HOLOLINK_* env, a GPU driver, and RDMA-capable
// hardware).  Building it forces the linker to resolve GpuRoceTransceiver's
// full dependency chain (hololink, DOCA, CUDA driver stubs), so HSB API
// drift is caught at build time even on machines where nothing links the
// component into a runnable binary (driverless CI: the decoding_server
// tool's gpu_roce block is additionally gated on the proprietary cudevice
// archive, which CI does not provision).

namespace cudaq::qec::decoding_server {
struct ITransceiver;
}

extern "C" cudaq::qec::decoding_server::ITransceiver *
cudaqx_qec_make_gpu_roce_transceiver();

int main() {
  (void)cudaqx_qec_make_gpu_roce_transceiver();
  return 0;
}
