/*******************************************************************************
 * Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// ADAPT-VQE MPI work-splitting test.
// Verifies that ADAPT-VQE produces the correct H2 ground-state energy when
// commutator evaluation is distributed across multiple MPI ranks.
//
// Run: mpiexec -np 4 ./test_adapt_mpi

#include <cassert>
#include <cmath>
#include <iostream>

#include "cudaq.h"
#include "nvqpp/test_kernels.h"
#include "cudaq/solvers/adapt.h"
#include "cudaq/solvers/operators.h"

int main(int argc, char **argv) {
  cudaq::mpi::initialize(argc, argv);

  auto geometryHH = cudaq::solvers::molecular_geometry{{"H", {0., 0., 0.}},
                                                       {"H", {0., 0., .7474}}};
  auto hh = cudaq::solvers::create_molecule(
      geometryHH, "sto-3g", 0, 0,
      {.casci = true, .ccsd = false, .verbose = false});

  auto pool = cudaq::solvers::operator_pool::get("spin_complement_gsd");
  auto poolList =
      pool->generate({{"num-orbitals", hh.hamiltonian.num_qubits() / 2}});

  auto [energy, thetas, ops] =
      cudaq::solvers::adapt_vqe(hartreeFock2Electrons, hh.hamiltonian, poolList,
                                {{"grad_norm_tolerance", 1e-3},
                                 {"max_iter", 15},
                                 {"grad_norm_diff_tolerance", 1e-5},
                                 {"threshold_energy", 5e-6}});

  auto rank = cudaq::mpi::rank();
  auto numRanks = cudaq::mpi::num_ranks();

  if (rank == 0) {
    std::cout << "[MPI ADAPT] ranks=" << numRanks << ", energy=" << energy
              << std::endl;
    assert(std::fabs(energy - (-1.13)) < 1e-2 &&
           "MPI ADAPT energy does not match expected -1.13");
    assert(!ops.empty() && "Expected at least one operator to be selected");
    std::cout << "PASS" << std::endl;
  }

  cudaq::mpi::finalize();
  return 0;
}
