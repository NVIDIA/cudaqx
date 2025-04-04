/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once
#include <cmath>
#include <limits>
#include <stdio.h>

#include "cudaq/solvers/optimizer.h"
#include <lbfgs.h>

namespace cudaq::optim {

/// @brief
class LBFGSObjective {
protected:
  /// @brief The function to optimize
  const optimizable_function &function;

  /// @brief verbose printout
  bool verbose;

  /// @brief The initial parameters
  std::vector<double> initialParameters;

  /// @brief Vector reference to record history of the
  /// optimization
  std::vector<optimization_result> &history;

  /// @brief The function tolerance.
  double functionTolerance;

  /// @brief The function value from the last iteration
  // double lastFunctionVal = std::numeric_limits<double>::max();
  std::size_t maxIterations;

public:
  LBFGSObjective(const optimizable_function &func,
                 const std::vector<double> &init, double functionTol,
                 std::size_t maxIter,
                 std::vector<optimization_result> &in_history, bool verbose)
      : function(func), initialParameters(init), functionTolerance(functionTol),
        maxIterations(maxIter), history(in_history), verbose(verbose) {}

  /// @brief Run the optimization
  optimization_result run(int N);

protected:
  /// @brief Required hook into liblbfgs lbfgs() evaluation. Will delegate to
  /// non-static method on this class
  static lbfgsfloatval_t _evaluate(void *instance, const lbfgsfloatval_t *x,
                                   lbfgsfloatval_t *g, const int n,
                                   const lbfgsfloatval_t step) {
    return reinterpret_cast<LBFGSObjective *>(instance)->evaluate(x, g, n,
                                                                  step);
  }

  lbfgsfloatval_t evaluate(const lbfgsfloatval_t *x, lbfgsfloatval_t *g,
                           const int n, const lbfgsfloatval_t step) {
    std::vector<double> params(x, x + n), grad(g, g + n);
    // evaluate the function
    auto val = function(params, grad);
    // set the grad pointer
    for (int i = 0; i < n; i++)
      g[i] = grad[i];
    return val;
  }

  /// @brief Required hook into liblbfgs lbfgs() evaluation. Will delegate to
  /// non-static method on this class
  static int _progress(void *instance, const lbfgsfloatval_t *x,
                       const lbfgsfloatval_t *g, const lbfgsfloatval_t fx,
                       const lbfgsfloatval_t xnorm, const lbfgsfloatval_t gnorm,
                       const lbfgsfloatval_t step, int n, int k, int ls) {
    return reinterpret_cast<LBFGSObjective *>(instance)->progress(
        x, g, fx, xnorm, gnorm, step, n, k, ls);
  }

  /// @brief Check the progress of the optimization.
  int progress(const lbfgsfloatval_t *x, const lbfgsfloatval_t *g,
               const lbfgsfloatval_t fx, const lbfgsfloatval_t xnorm,
               const lbfgsfloatval_t gnorm, const lbfgsfloatval_t step, int n,
               int k, int ls) {
    if (verbose) {
      printf("Iteration %d:\n", k);
      printf("  fx = %f\n  ", fx);
      for (int i = 0; i < n; i++)
        if (i > 3)
          printf("...");
        else
          printf("x[%d] = %lf, ", i, x[i]);

      printf("\n  xnorm = %f, gnorm = %f, step = %f\n", xnorm, gnorm, step);
      printf("\n");
    }

    // append to the history
    history.push_back(std::make_tuple(fx, std::vector<double>(x, x + n)));

    if (k >= maxIterations) {
      return 1;
    }

    // If this is the first evaluation, then we don't have a lastFunctionVal,
    // if its not, then get the second to last element of the history
    auto lastFunctionVal = k == 1 ? std::numeric_limits<double>::max()
                                  : std::get<0>(history.rbegin()[1]);

    if (std::fabs(fx - lastFunctionVal) < functionTolerance)
      return 1;

    return 0;
  }
};
} // namespace cudaq::optim
