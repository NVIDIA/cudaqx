/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "solvers/py_optim.h"
#include "solvers/py_solvers.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

NB_MODULE(_pycudaqx_solvers_the_suffix_matters_cudaq_solvers, mod) {
  mod.doc() = "Python bindings for the CUDA-Q Solver Libraries.";
  // Ensure cudaq is loaded so its nanobind-registered types (spin_op,
  // sum_op<spin_handler>, observe_result, etc.) are available before any
  // solvers binding tries to return or consume them.
  nanobind::module_::import_("cudaq");
  cudaq::optim::bindOptim(mod);
  cudaq::solvers::bindSolvers(mod);
  // Suppress nanobind's reference-leak warnings for the same reason as the QEC
  // module: this module imports cudaq, so cudaq_runtime types are visible to
  // nanobind's leak checker here. Those types are kept alive by objects in the
  // calling script's global scope, which Python only clears after extension
  // modules are torn down. This is a cross-module cleanup ordering false
  // positive, not a genuine reference-counting bug. See the identical comment
  // in libs/qec/python/bindings/cudaqx_qec.cpp for the full explanation.
  nanobind::set_leak_warnings(false);
}
