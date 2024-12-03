/*******************************************************************************
 * Copyright (c) 2024 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/solvers/operators/operator_pool.h"

INSTANTIATE_REGISTRY_NO_ARGS(cudaq::solvers::operator_pool)

namespace cudaq::solvers {
std::vector<cudaq::spin_op>
get_operator_pool(const std::string &name, const heterogeneous_map &options) {
  return cudaq::solvers::operator_pool::get(name)->generate(options);
}
} // namespace cudaq::solvers