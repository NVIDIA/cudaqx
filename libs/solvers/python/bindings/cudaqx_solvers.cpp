/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "solvers/py_block_encoding.h"
#include "solvers/py_optim.h"
#include "solvers/py_solvers.h"

#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

PYBIND11_MODULE(_pycudaqx_solvers_the_suffix_matters_cudaq_solvers, mod) {
  mod.doc() = "Python bindings for the CUDA-Q Solver Libraries.";
  cudaq::optim::bindOptim(mod);
  cudaq::solvers::bindSolvers(mod);
  cudaq::solvers::bindBlockEncoding(mod);
}
