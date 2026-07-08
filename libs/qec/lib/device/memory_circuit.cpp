/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Offline copy of the `memory_circuit` device kernel shipped in libcudaq-qec.
// CUDAQ_QEC_ENABLE_REALTIME_DECODING is intentionally left undefined here so
// that this build carries no reference to the realtime decoding symbols.
#include "cudaq/qec/device/memory_circuit_kernel.h"
