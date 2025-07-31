
/*******************************************************************************
 * Copyright (c) 2024 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/solvers/observe_gradient.h"

CUDAQ_INSTANTIATE_REGISTRY(cudaq::observe_gradient,
                           std::function<void(std::vector<double>)> const &,
                           cudaq::spin_op const &)
