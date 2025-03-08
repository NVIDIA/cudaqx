/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "cudaq/solvers/operators.h"
#include "cudaq/solvers/operators/molecule/molecule_package_driver.h"

#include <fstream>
#include <iostream>
#include <tuple>

#include <gtest/gtest.h>

TEST(MoleculeTester, checkSimple) {
  auto registeredNames =
      cudaq::solvers::MoleculePackageDriver::get_registered();

  EXPECT_EQ(registeredNames.size(), 1);
  EXPECT_TRUE(std::find(registeredNames.begin(), registeredNames.end(),
                        "RESTPySCFDriver") != registeredNames.end());
  {
    cudaq::solvers::molecular_geometry geometry{{"H", {0., 0., 0.}},
                                                {"H", {0., 0., .7474}}};
    auto molecule = cudaq::solvers::create_molecule(
        geometry, "sto-3g", 0, 0, {.casci = true, .verbose = true});

    // FIXME how? molecule.hamiltonian.dump();

    EXPECT_NEAR(molecule.energies["fci_energy"], -1.137, 1e-3);
    EXPECT_NEAR(molecule.energies["hf_energy"], -1.1163255644, 1e-3);
    EXPECT_EQ(molecule.n_electrons, 2);
    EXPECT_EQ(molecule.n_orbitals, 2);

    // EXPECT_NEAR(molecule.fermionOperator.constant, 0.7080240981000804, 1e-3);
    EXPECT_EQ(2, molecule.hpq.shape().size());
    EXPECT_EQ(4, molecule.hpq.shape()[0]);
    EXPECT_EQ(4, molecule.hpq.shape()[1]);
    EXPECT_NEAR(molecule.hpq.at({0, 0}).real(), -1.2488, 1e-3);
    EXPECT_NEAR(molecule.hpq.at({1, 1}).real(), -1.2488, 1e-3);
    EXPECT_NEAR(molecule.hpq.at({2, 2}).real(), -.47967, 1e-3);
    EXPECT_NEAR(molecule.hpq.at({3, 3}).real(), -.47967, 1e-3);
    EXPECT_EQ(4, molecule.hpqrs.shape().size());
    for (int i = 0; i < 4; i++)
      EXPECT_EQ(4, molecule.hpqrs.shape()[i]);
    EXPECT_NEAR(molecule.hpqrs.at({0, 0, 0, 0}).real(), 0.3366719725032414,
                1e-3);
    EXPECT_NEAR(molecule.hpqrs.at({0, 0, 2, 2}).real(), 0.0908126657382825,
                1e-3);
    EXPECT_NEAR(molecule.hpqrs.at({0, 1, 1, 0}).real(), 0.3366719725032414,
                1e-3);
    EXPECT_NEAR(molecule.hpqrs.at({0, 1, 3, 2}).real(), 0.0908126657382825,
                1e-3);
    EXPECT_NEAR(molecule.hpqrs.at({0, 2, 0, 2}).real(), 0.09081266573828267,
                1e-3);
    EXPECT_NEAR(molecule.hpqrs.at({0, 2, 2, 0}).real(), 0.33121364716348484,
                1e-3);
    EXPECT_NEAR(molecule.hpqrs.at({0, 3, 1, 2}).real(), 0.09081266573828267,
                1e-3);
    EXPECT_NEAR(molecule.hpqrs.at({0, 3, 3, 0}).real(), 0.33121364716348484,
                1e-3);
    EXPECT_NEAR(molecule.hpqrs.at({1, 0, 0, 1}).real(), 0.3366719725032414,
                1e-3);
    EXPECT_NEAR(molecule.hpqrs.at({1, 0, 2, 3}).real(), 0.0908126657382825,
                1e-3);
    EXPECT_NEAR(molecule.hpqrs.at({1, 1, 1, 1}).real(), 0.3366719725032414,
                1e-3);
    EXPECT_NEAR(molecule.hpqrs.at({1, 1, 3, 3}).real(), 0.0908126657382825,
                1e-3);
    EXPECT_NEAR(molecule.hpqrs.at({1, 2, 0, 3}).real(), 0.09081266573828267,
                1e-3);
    EXPECT_NEAR(molecule.hpqrs.at({1, 2, 2, 1}).real(), 0.33121364716348484,
                1e-3);
    EXPECT_NEAR(molecule.hpqrs.at({1, 3, 1, 3}).real(), 0.09081266573828267,
                1e-3);
    EXPECT_NEAR(molecule.hpqrs.at({1, 3, 3, 1}).real(), 0.33121364716348484,
                1e-3);
    EXPECT_NEAR(molecule.hpqrs.at({2, 0, 0, 2}).real(), 0.3312136471634851,
                1e-3);
    EXPECT_NEAR(molecule.hpqrs.at({2, 0, 2, 0}).real(), 0.09081266573828246,
                1e-3);
    EXPECT_NEAR(molecule.hpqrs.at({2, 1, 1, 2}).real(), 0.3312136471634851,
                1e-3);
    EXPECT_NEAR(molecule.hpqrs.at({2, 1, 3, 0}).real(), 0.09081266573828246,
                1e-3);
    EXPECT_NEAR(molecule.hpqrs.at({2, 2, 0, 0}).real(), 0.09081266573828264,
                1e-3);
    EXPECT_NEAR(molecule.hpqrs.at({2, 2, 2, 2}).real(), 0.34814578499360427,
                1e-3);
    EXPECT_NEAR(molecule.hpqrs.at({2, 3, 1, 0}).real(), 0.09081266573828264,
                1e-3);
    EXPECT_NEAR(molecule.hpqrs.at({2, 3, 3, 2}).real(), 0.34814578499360427,
                1e-3);
    EXPECT_NEAR(molecule.hpqrs.at({3, 0, 0, 3}).real(), 0.3312136471634851,
                1e-3);
    EXPECT_NEAR(molecule.hpqrs.at({3, 0, 2, 1}).real(), 0.09081266573828246,
                1e-3);
    EXPECT_NEAR(molecule.hpqrs.at({3, 1, 1, 3}).real(), 0.3312136471634851,
                1e-3);
    EXPECT_NEAR(molecule.hpqrs.at({3, 1, 3, 1}).real(), 0.09081266573828246,
                1e-3);
    EXPECT_NEAR(molecule.hpqrs.at({3, 2, 0, 1}).real(), 0.09081266573828264,
                1e-3);
    EXPECT_NEAR(molecule.hpqrs.at({3, 2, 2, 3}).real(), 0.34814578499360427,
                1e-3);
    EXPECT_NEAR(molecule.hpqrs.at({3, 3, 1, 1}).real(), 0.09081266573828264,
                1e-3);
    EXPECT_NEAR(molecule.hpqrs.at({3, 3, 3, 3}).real(), 0.34814578499360427,
                1e-3);
  }
}

TEST(OperatorsTester, checkH2OActiveSpace) {
  std::string contents = R"#(3

O 0.1173 0.0 0.0
H -0.4691 0.7570 0.0
H -0.4691 -0.7570 0.0
)#";

  {
    std::ofstream out(".tmpH2O.xyz");
    out << contents;
  }

  auto geometry = cudaq::solvers::molecular_geometry::from_xyz(".tmpH2O.xyz");
  std::remove(".tmpH2O.xyz");

  auto molecule = cudaq::solvers::create_molecule(
      geometry, "631g", 0, 0,
      {.nele_cas = 6, .norb_cas = 6, .ccsd = true, .verbose = true});

  // molecule.hamiltonian.dump();
  EXPECT_EQ(molecule.n_electrons, 6);
  EXPECT_EQ(molecule.n_orbitals, 6);
}
