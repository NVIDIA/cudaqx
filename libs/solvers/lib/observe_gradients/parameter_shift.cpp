
/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/solvers/observe_gradients/parameter_shift.h"

namespace cudaq {
std::size_t parameter_shift::getRequiredNumExpectationComputations(
    const std::vector<double> &x) {
  return 2 * x.size();
}

void parameter_shift::calculateGradient(const std::vector<double> &x,
                                        std::vector<double> &dx, double exp_h) {
  auto tmpX = x;
  for (std::size_t i = 0; i < x.size(); i++) {
    // increase value to x_i + (shiftScalar * pi)
    tmpX[i] += shiftScalar * M_PI;
    auto px = expectation(tmpX);
    // decrease value to x_i - (shiftScalar * pi)
    tmpX[i] -= 2 * shiftScalar * M_PI;
    auto mx = expectation(tmpX);
    dx[i] = (px - mx) / 2.;
  }
}
} // namespace cudaq
