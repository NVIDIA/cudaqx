/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Strong definition of the GPU RoCE factory that DecodingServer.cpp declares
// weakly.  This translation unit lives in cudaq-qec-decoding-server-gpuroce
// (NOT the core library) so that only binaries linking that component carry
// the DOCA / Hololink / CUDA-driver dependencies.  Consumers must link the
// component WHOLE_ARCHIVE: the sole reference to this symbol is weak, which
// does not pull archive members on its own.

#include "DecodingServer.h" // reconcile_gpu_roce_device (core symbol)
#include "GpuRoceTransceiver.h"

extern "C" cudaq::qec::decoding_server::ITransceiver *
cudaqx_qec_make_gpu_roce_transceiver(int pinned_cuda_device) {
  using namespace cudaq::qec::decoding_server;
  // Reconcile the FPGA-affine GPU (HOLOLINK_GPU_ID) with the decoder's pin
  // here, inside the component, where GpuRoceConfig is visible.
  auto cfg = GpuRoceConfig::from_env();
  cfg.gpu_id = reconcile_gpu_roce_device(cfg.gpu_id_env, pinned_cuda_device);
  return new GpuRoceTransceiver(cfg);
}
