/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/solvers/observe_gradient.h"

namespace cudaq {

class parameter_shift : public observe_gradient {
protected:
  std::size_t
  getRequiredNumExpectationComputations(const std::vector<double> &x) override;

public:
  double shiftScalar = 0.5;
  using observe_gradient::observe_gradient;

  void calculateGradient(const std::vector<double> &x, std::vector<double> &dx,
                         double exp_h) override;
  CUDAQ_ADD_EXTENSION_CUSTOM_CREATOR_FUNCTION(
      parameter_shift,
      static std::unique_ptr<observe_gradient> create(
          const ParameterizedKernel &functor, const spin_op &op) {
        return std::make_unique<parameter_shift>(functor, op);
      })
};
CUDAQ_REGISTER_EXTENSION(parameter_shift)
} // namespace cudaq
