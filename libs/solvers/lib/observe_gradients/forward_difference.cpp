
/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/solvers/observe_gradients/forward_difference.h"

namespace cudaq {
std::size_t forward_difference::getRequiredNumExpectationComputations(
    const std::vector<double> &x) {
  return x.size();
}
void forward_difference::calculateGradient(const std::vector<double> &x,
                                           std::vector<double> &dx,
                                           double exp_h) {
  auto tmpX = x;
  for (std::size_t i = 0; i < x.size(); i++) {
    // increase value to x_i + dx_i
    tmpX[i] += step;
    auto px = expectation(tmpX);
    dx[i] = (px - exp_h) / step;
  }
}
} // namespace cudaq
