/*******************************************************************************
 * Copyright (c) 2024 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
// [Begin Documentation]
#include "cudaq.h"
#include "cudaq/solvers/operators.h"
#include "cudaq/solvers/stateprep/uccsd.h"
#include "cudaq/solvers/vqe.h"

// Compile and run with
// nvq++ --enable-mlir -lcudaq-solvers uccsd_vqe.cpp -o uccsd_vqe
// ./uccsd_vqe

int main() {
  // Create the molecular hamiltonian
  cudaq::solvers::molecular_geometry geometry{{"H", {0., 0., 0.}},
                                              {"H", {0., 0., .7474}}};
  auto molecule = cudaq::solvers::create_molecule(
      geometry, "sto-3g", 0, 0, {.casci = true, .verbose = true});

  // Get the spin operator
  auto h = molecule.hamiltonian;

  // Get the number of electrons and qubits
  auto numElectrons = molecule.n_electrons;
  auto numQubits = molecule.n_orbitals * 2;

  // Create an initial set of parameters for the optimization
  auto numParams = cudaq::solvers::stateprep::get_num_uccsd_parameters(
      numElectrons, numQubits);
  std::vector<double> init(numParams, -2.);

  // Run VQE
  auto [energy, thetas, ops] = cudaq::solvers::vqe(
      [&](std::vector<double> params, std::size_t numQubits,
          std::size_t numElectrons) __qpu__ {
        cudaq::qvector q(numQubits);
        for (auto i : cudaq::range(numElectrons))
          x(q[i]);

        cudaq::solvers::stateprep::uccsd(q, params, numElectrons);
      },
      molecule.hamiltonian, init,
      [&](std::vector<double> x) {
        return std::make_tuple(x, numQubits, numElectrons);
      },
      {{"verbose", true}});

  printf("Final <H> = %.12lf\n", energy);
}