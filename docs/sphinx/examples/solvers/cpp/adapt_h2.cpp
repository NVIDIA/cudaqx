/*******************************************************************************
 * Copyright (c) 2024 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "cudaq.h"
#include "cudaq/solvers/adapt.h"
#include "cudaq/solvers/operators.h"

// Compile and run with
// nvq++ --enable-mlir -lcudaq-solvers adapt_h2.cpp -o adapt_ex
// ./adapt_ex

int main() {
  // Create the molecular hamiltonian
  cudaq::solvers::molecular_geometry geometry{{"H", {0., 0., 0.}},
                                              {"H", {0., 0., .7474}}};
  auto molecule = cudaq::solvers::create_molecule(
      geometry, "sto-3g", 0, 0, {.casci = true, .verbose = true});

  // Get the spin operator
  auto h = molecule.hamiltonian;

  // Create the operator pool
  std::vector<cudaq::spin_op> opPool = cudaq::solvers::get_operator_pool(
      "spin_complement_gsd", {{"num-orbitals", h.num_qubits() / 2}});

  // Run ADAPT
  auto [energy, thetas, ops] = cudaq::solvers::adapt_vqe(
      [](cudaq::qvector<> &q) __qpu__ {
        x(q[0]);
        x(q[1]);
      },
      h, opPool, {{"grad_norm_tolerance", 1e-3}});

  printf("Final <H> = %.12lf\n", energy);
}