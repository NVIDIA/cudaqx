/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace cudaq::qec::dem_sampler {
void bindDemSampling(py::module &mod);
} // namespace cudaq::qec::dem_sampler
