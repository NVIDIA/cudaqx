/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "cudaq/qis/state.h"

namespace cudaq {

/// @brief prepare_state is an entry-point kernel that
/// simply prepares a know state provided as input. This is
/// useful for sampling or observation on a known state vector.
/// @param state
void prepare_state(cudaq::state &state);
} // namespace cudaq
