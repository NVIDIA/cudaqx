/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <cmath>
#include <gtest/gtest.h>

#include "cudaq/solvers/operators.h"

 TEST(SolversUCCSDTester, checkJordanWigner) {
  cudaq::solvers::molecular_geometry geometry{{"H", {0., 0., 0.}},
                                              {"H", {0., 0., .7474}},
                                              {"H", {1., 0., 0.}},
                                              {"H", {1., 0., .7474}}
                                              };
  auto molecule = cudaq::solvers::create_molecule(geometry, "sto-3g", 0, 0,
                                                  {.verbose = true});
  auto hamNumTerms = molecule.hamiltonian.num_terms();
  auto numElectrons = molecule.n_electrons;
}