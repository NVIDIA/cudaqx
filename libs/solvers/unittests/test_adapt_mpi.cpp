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
// When run by gtest (ctest), the test launches itself via mpiexec as a
// subprocess. When invoked with --mpi-worker, the actual MPI computation runs.

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

#include <gtest/gtest.h>

#include "cudaq.h"
#include "nvqpp/test_kernels.h"
#include "cudaq/solvers/adapt.h"
#include "cudaq/solvers/operators.h"

// Lightweight MPI init/finalize used by the skip-check probe.
static int runMpiProbe() {
  cudaq::mpi::initialize();
  cudaq::mpi::finalize();
  return 0;
}

static int runMpiWorker() {
  cudaq::mpi::initialize();

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

  int rc = 0;
  if (cudaq::mpi::rank() == 0) {
    std::cout << "[MPI ADAPT] ranks=" << cudaq::mpi::num_ranks()
              << ", energy=" << energy << std::endl;
    if (std::fabs(energy - (-1.13)) >= 1e-2) {
      std::cerr << "FAIL: energy " << energy << " != expected -1.13"
                << std::endl;
      rc = 1;
    } else if (ops.empty()) {
      std::cerr << "FAIL: no operators selected" << std::endl;
      rc = 1;
    } else {
      std::cout << "PASS" << std::endl;
    }
  }

  cudaq::mpi::finalize();
  return rc;
}

class SolversTester : public ::testing::TestWithParam<int> {};

TEST_P(SolversTester, checkSimpleAdaptMpi_H2Sto3g) {
  if (std::system("which mpiexec > /dev/null 2>&1") != 0)
    GTEST_SKIP() << "mpiexec not found, skipping MPI test";

  // Probe with 2 ranks to verify MPI can actually launch multi-rank jobs
  // (catches missing PML transports, absent cudaq MPI plugin, etc.)
  std::string self = ::testing::internal::GetArgvs()[0];
  std::string probeCmd = "mpiexec --allow-run-as-root --oversubscribe -np 2 " +
                         self + " --mpi-probe > /dev/null 2>&1";
  if (std::system(probeCmd.c_str()) != 0)
    GTEST_SKIP() << "MPI multi-rank launch not functional, skipping MPI test";

  int numRanks = GetParam();

  std::string cmd = "mpiexec --allow-run-as-root --oversubscribe -np " +
                    std::to_string(numRanks) + " " + self + " --mpi-worker";
  int rc = std::system(cmd.c_str());
  EXPECT_EQ(rc, 0) << "mpiexec failed with exit code " << rc;
}

INSTANTIATE_TEST_SUITE_P(MpiRanks, SolversTester, ::testing::Values(2, 4));

int main(int argc, char **argv) {
  for (int i = 1; i < argc; i++) {
    if (std::string(argv[i]) == "--mpi-probe")
      return runMpiProbe();
    if (std::string(argv[i]) == "--mpi-worker")
      return runMpiWorker();
  }

  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
