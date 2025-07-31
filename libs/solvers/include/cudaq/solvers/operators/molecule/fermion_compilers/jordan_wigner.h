/*******************************************************************************
 * Copyright (c) 2024 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "cudaq/solvers/operators/molecule/fermion_compiler.h"

namespace cudaq::solvers {
/// @brief Map fermionic operators to spin operators via the
/// Jordan-Wigner transformation.
class jordan_wigner : public fermion_compiler {
public:
  cudaq::spin_op generate(const double constant, const cudaqx::tensor<> &hpq,
                          const cudaqx::tensor<> &hpqrs,
                          const cudaqx::heterogeneous_map &options) override;

  CUDAQ_ADD_EXTENSION_CREATOR_FUNCTION(fermion_compiler, jordan_wigner)
};
CUDAQ_REGISTER_EXTENSION(jordan_wigner)
} // namespace cudaq::solvers
