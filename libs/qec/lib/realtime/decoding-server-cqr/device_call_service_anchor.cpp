/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Out-of-line key functions for the CUDA-Q realtime device-call service base
// classes.
//
// libcudaq-device-call-runtime.so also defines these, but that library is the
// *caller* (QPU) side of the device_call ring: it implements
// cudaq_internal::device_call::RingBufferWrapper and friends, none of which a
// decoding server uses. The server only implements the service interface, so
// linking the caller-side runtime purely to pick up two destructors would drag
// a CUDA-Q install into an otherwise CUDA-Q-free server binary.
//
// Defining the key function of each class here emits its vtable and typeinfo
// into the CQR plugin instead. cudaq-realtime is a separate product from
// CUDA-Q, and this keeps the service side depending only on the former.

#include "cudaq/realtime/device_call_service.h"

namespace cudaq::realtime {

DeviceCallService::~DeviceCallService() = default;

DeviceCallServiceSession::~DeviceCallServiceSession() = default;

} // namespace cudaq::realtime
