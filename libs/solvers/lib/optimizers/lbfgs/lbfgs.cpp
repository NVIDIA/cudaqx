/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/solvers/optimizers/lbfgs.h"
#include "LBFGSObjective.h"

namespace cudaq::optim {
optimization_result lbfgs::optimize(std::size_t dim,
                                    const optimizable_function &opt_function,
                                    const cudaqx::heterogeneous_map &options) {
  history.clear();
  cudaq::optim::LBFGSObjective f(
      opt_function, options.get("initial_parameters", std::vector<double>(dim)),
      options.get("tol", 1e-12),
      options.get("max_iterations", std::numeric_limits<std::size_t>::max()),
      history, options.get("verbose", false));
  return f.run(dim);
}

} // namespace cudaq::optim
