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

/// @brief Helper function used by the Bravyi-Kitaev transformation.
cudaq::spin_op seeley_richard_love(std::size_t i, std::size_t j,
                                   std::complex<double> coef, int n_qubits);

/// @brief Map fermionic operators to spin operators via the
/// Bravyi-Kitaev transformation.
class bravyi_kitaev : public fermion_compiler {
public:
  cudaq::spin_op generate(const double constant, const cudaqx::tensor<> &hpq,
                          const cudaqx::tensor<> &hpqrs,
                          const cudaqx::heterogeneous_map &options) override;

  CUDAQ_ADD_EXTENSION_CREATOR_FUNCTION(fermion_compiler, bravyi_kitaev)
};
CUDAQ_REGISTER_EXTENSION(bravyi_kitaev)
} // namespace cudaq::solvers
