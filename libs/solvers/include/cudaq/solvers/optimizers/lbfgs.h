/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/solvers/optimizer.h"

using namespace cudaqx;

namespace cudaq::optim {

/// @brief The limited-memory Broyden-Fletcher-Goldfarb-Shanno
/// gradient based black-box function optimizer.
class lbfgs : public optimizer {
public:
  using optimizer::optimize;

  /// @brief Return true indicating this optimizer requires an
  /// optimization functor that produces gradients.
  bool requiresGradients() const override { return true; }

  /// @brief Optimize the provided function according to the
  /// LBFGS algorithm.
  optimization_result
  optimize(std::size_t dim, const optimizable_function &opt_function,
           const cudaqx::heterogeneous_map &options) override;

  CUDAQ_ADD_EXTENSION_CREATOR_FUNCTION(optimizer, lbfgs)
};
CUDAQ_REGISTER_EXTENSION(lbfgs)
} // namespace cudaq::optim
