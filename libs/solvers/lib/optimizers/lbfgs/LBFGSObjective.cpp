/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "LBFGSObjective.h"

namespace cudaq::optim {
optimization_result LBFGSObjective::run(int N) {
  double value;
  std::vector<double> parameters = initialParameters;

  /*
      Start the L-BFGS optimization; this will invoke the callback functions
      evaluate() and progress() when necessary.
   */
  int ret =
      lbfgs(N, parameters.data(), &value, _evaluate, _progress, this, NULL);

  return std::make_tuple(value, parameters);
}
} // namespace cudaq::optim
