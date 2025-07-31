/*******************************************************************************
 * Copyright (c) 2024 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cuda-qx/core/heterogeneous_map.h"
#include "cuda-qx/core/tensor.h"
#include "cudaq/utils/extension_point.h"

#include "cudaq/spin_op.h"

namespace cudaq::solvers {

/// @brief The `fermion_compiler` type serves as a base class defining
/// and interface for clients to map fermionic molecular operators to
/// `cudaq::spin_op` instances. The fermionic operator is represented
/// via its one body and two body electron overlap integrals.
class fermion_compiler : public cudaq::extension_point<fermion_compiler> {
public:
  /// @brief Given a fermionic representation of an operator
  /// generate an equivalent operator on spins.
  virtual cudaq::spin_op
  generate(const double constant, const cudaqx::tensor<> &hpq,
           const cudaqx::tensor<> &hpqrs,
           const cudaqx::heterogeneous_map &options = {}) = 0;
  virtual ~fermion_compiler() {}
};

} // namespace cudaq::solvers
