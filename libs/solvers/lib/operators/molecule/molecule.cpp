/*******************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "cudaq/solvers/operators/molecule/molecule_package_driver.h"
#include "cudaq/utils/cudaq_utils.h"

#include <fstream>
#include <iostream>

CUDAQ_INSTANTIATE_REGISTRY_NO_ARGS(cudaq::solvers::MoleculePackageDriver)

namespace cudaq::solvers {

std::string molecular_geometry::name() const {
  std::string ret = "";
  for (auto &a : atoms)
    ret += a.name;

  return ret;
}

molecular_geometry molecular_geometry::from_xyz(const std::string &xyzFile) {

  std::ifstream stream(xyzFile);
  std::string contents((std::istreambuf_iterator<char>(stream)),
                       std::istreambuf_iterator<char>());

  if (contents.empty())
    throw std::runtime_error("could not extract file contents for " + xyzFile);
  auto lines = cudaq::split(contents, '\n');
  char *endptr;
  int result = std::strtol(lines[0].c_str(), &endptr, 10);

  std::vector<atom> atoms;
  for (std::size_t i = 0; auto &line : lines) {
    if (i++ == 0)
      continue;

    cudaq::trim(line);
    if (line.empty())
      continue;

    bool seenFirstSpace = false;
    std::vector<std::string> components(/*atom+3coord*/ 4);
    for (std::size_t k = 0, componentCounter = 0; k < line.length(); k++) {
      if (line[k] == ' ') {
        if (!seenFirstSpace) {
          seenFirstSpace = true;
          components[componentCounter] += " ";
          componentCounter++;
        }
        continue;
      }

      seenFirstSpace = false;
      components[componentCounter] += line[k];
    }

    std::vector<double> coords;
    for (std::size_t ii = 1; ii < 4; ii++)
      coords.push_back(std::stod(components[ii]));
    atoms.push_back(atom{components[0], {coords[0], coords[1], coords[2]}});
  }

  return molecular_geometry(atoms);
}

molecular_hamiltonian create_molecule(const molecular_geometry &geometry,
                                      const std::string &basis, int spin,
                                      int charge, molecule_options options) {
  if (!MoleculePackageDriver::is_registered(options.driver))
    throw std::runtime_error("invalid molecule package driver (" +
                             options.driver + ")");
  auto driver = MoleculePackageDriver::get(options.driver);
  if (!driver->is_available()) {
    auto tearDownRoutine = driver->make_available(options.python_path);
    if (!tearDownRoutine)
      throw std::runtime_error("invalid molecule generator.");

    cudaqx::scheduleTearDown(std::move(tearDownRoutine));
  }

  return driver->createMolecule(geometry, basis, spin, charge, options);
}

void molecule_options::dump() {
  std::cout << "\tmolecule_options dump() [\n";
  std::cout << "\tpython_path: " << python_path << "\n";
  std::cout << "\tfermion_to_spin: " << fermion_to_spin << "\n";
  std::cout << "\ttype: " << type << "\n";
  std::cout << "\tsymmetry: " << symmetry << "\n";
  std::cout << "\tcycles: " << cycles << "\n";
  std::cout << "\tinitguess: " << initguess << "\n";
  std::cout << "\tnele_cas: " << (nele_cas.has_value() ? nele_cas.value() : -1)
            << "\n";
  std::cout << "\tnorb_cas: " << (norb_cas.has_value() ? norb_cas.value() : -1)
            << "\n";
  std::cout << "\tUR: " << std::boolalpha << UR << "\n";
  std::cout << "\tMP2: " << std::boolalpha << MP2 << "\n";
  std::cout << "\tnatorb: " << std::boolalpha << natorb << "\n";
  std::cout << "\tcasci: " << std::boolalpha << casci << "\n";
  std::cout << "\tccsd: " << std::boolalpha << ccsd << "\n";
  std::cout << "\tcasscf: " << std::boolalpha << casscf << "\n";
  std::cout << "\tintegrals_natorb: " << std::boolalpha << integrals_natorb
            << "\n";
  std::cout << "\tintegrals_casscf: " << std::boolalpha << integrals_casscf
            << "\n";
}

cudaq::spin_op one_particle_op(std::size_t numQubits, std::size_t p,
                               std::size_t q,
                               const std::string fermionCompiler) {
  using namespace cudaq;

  if (p == q)
    return 0.5 - 0.5 * spin::z(p);

  std::complex<double> coeff(0., 1.);
  double m = -.25;
  if (p > q) {
    std::swap(p, q);
    coeff = std::conj(coeff);
  }

  std::vector<std::size_t> z_indices;
  for (auto i : cudaq::range((long)p + 1, (long)q))
    z_indices.push_back(i);

  auto parity = spin::z(z_indices.front());
  for (std::size_t i = 1; i < z_indices.size(); i++) {
    parity *= spin::z(i);
  }

  cudaq::spin_op ret = m * spin::x(p) * parity * spin::x(q);

  ret += m * spin::y(p) * parity * spin::y(q);
  ret -= coeff * m * spin::y(p) * parity * spin::x(q);
  ret += coeff * m * spin::x(p) * parity * spin::y(q);
  return ret.canonicalize().trim();
}

} // namespace cudaq::solvers
