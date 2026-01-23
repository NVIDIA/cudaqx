/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

// Note: block_encoding.h is intentionally excluded from this umbrella header
// because it contains virtual functions with quantum types (cudaq::qview<>)
// that are incompatible with nvq++ compilation. Code that needs block encoding
// should include "operators/block_encoding.h" directly.
#include "operators/graph/clique.h"
#include "operators/graph/max_cut.h"
#include "operators/molecule.h"
#include "operators/operator_pool.h"
