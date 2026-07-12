/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// The translation unit that compiles the `memory_circuit` device kernels into
// libcudaq-qec. This TU must exist: it is what turns the header-defined kernels
// into linkable symbols + JIT kernel registrations for the sample/DEM entry
// points in experiments.cpp (and for applications, which only include the
// declarations in cudaq/qec/device/memory_circuit.h).
//
// This TU is compiled with CUDAQ_QEC_DISABLE_REALTIME_DECODING (set in
// lib/device/CMakeLists.txt), so its copy of the kernels is decoding-free and
// makes no reference to the realtime decoding API. The API implementations
// (enqueue_syndromes / get_corrections / reset_decoder) are NOT defined here:
// they come from a linked decoding shim (realtime/simulation,
// realtime/simulation-cqr, or realtime/quantinuum), or -- when no shim is
// linked -- from the no-op fallback archive (realtime/noop) that the
// cudaq-qec CMake target appends to the end of every consumer's link line.
// Housing default no-ops in libcudaq-qec itself instead made them win weak /
// first-in-DT_NEEDED symbol resolution over any real shim (libcudaq-qec is
// always linked first), silently shadowing it -- so a binary that meant to
// decode would no-op.
#include "cudaq/qec/device/memory_circuit.h"
