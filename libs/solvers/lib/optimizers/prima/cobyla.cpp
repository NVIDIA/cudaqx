/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <cmath>
#include <limits>
#include <stdint.h>
#include <stdio.h>

#include "cudaq/solvers/optimizers/cobyla.h"

#include "prima/prima.h"

namespace cudaq::optim {

struct PrimaContainer {
  const optimizable_function &function;
  std::size_t dim = 0;
};

static void evaluator(const double x[], double *retVal, double constr[],
                      const void *data) {
  const auto *container = reinterpret_cast<const PrimaContainer *>(data);
  std::vector<double> parameters(x, x + container->dim), dummy;
  *retVal = container->function(parameters, dummy);
}

optimization_result cobyla::optimize(std::size_t dim,
                                     const optimizable_function &opt_function,
                                     const heterogeneous_map &options) {
  history.clear();

  // Create a container type for our objective function
  // and pass as the forwarded void* data pointer to prima_cobyla
  PrimaContainer container{opt_function, dim};
  void *data = reinterpret_cast<void *>(&container);

  // Get the initial parameters
  auto initParams = options.get<std::vector<double>>(
      "initial_parameters", std::vector<double>(dim, 0.));

  // Set the upper and lower bounds
  std::vector<double> xupper(dim, M_PI), xlower(dim, -M_PI);

  // Set the max iterations / max function calls
  const int maxfun = options.get<int>("max_iterations", dim * 200);

  // Default values needed for coobyla
  double value = 0.0;
  double cstrv = 0.0;
  const double rhobeg = 1.0;
  const double rhoend = 1e-4;
  const double ftarget = -INFINITY;
  const int iprint = PRIMA_MSG_NONE;
  int nf = 0;

  // Run the optimization
  int rc = prima_cobyla(0, &evaluator, data, dim, initParams.data(), &value,
                        &cstrv, nullptr, 0, nullptr, nullptr, 0, nullptr,
                        nullptr, xlower.data(), xupper.data(), &nf, rhobeg,
                        rhoend, ftarget, maxfun, iprint);

  return std::make_tuple(value, initParams);
}

} // namespace cudaq::optim
