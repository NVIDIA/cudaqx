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

/// @brief The COBYLA derivative-free black-box function optimizer from the
/// [PRIMA](https://www.libprima.net) library.
class cobyla : public optimizer {
public:
  using optimizer::optimize;

  /// @brief Return true indicating this optimizer requires an
  /// optimization functor that produces gradients.
  bool requiresGradients() const override { return false; }

  /// @brief Optimize the provided function according to the
  /// cobyla algorithm.
  optimization_result optimize(std::size_t dim,
                               const optimizable_function &opt_function,
                               const heterogeneous_map &options) override;

  CUDAQ_EXTENSION_CREATOR_FUNCTION(optimizer, cobyla);
};

CUDAQ_REGISTER_TYPE(cobyla)

} // namespace cudaq::optim
