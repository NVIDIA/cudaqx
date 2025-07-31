/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cuda-qx/core/tear_down.h"
#include "cudaq/utils/extension_point.h"

#include "cudaq/solvers/operators/molecule.h"

namespace cudaq::solvers {

/// @brief MoleculePackageDriver provides an extensible interface for
/// generating molecular Hamiltonians and associated metadata.
class MoleculePackageDriver
    : public cudaq::extension_point<MoleculePackageDriver> {
public:
  /// @brief Return a `molecular_hamiltonian` described by the given
  /// geometry, basis set, spin, and charge. Optionally
  /// restrict the active space.
  virtual molecular_hamiltonian
  createMolecule(const molecular_geometry &geometry, const std::string &basis,
                 int spin, int charge,
                 molecule_options options = molecule_options()) = 0;

  /// @brief Return true if this driver is available.
  virtual bool is_available() const { return true; }

  /// @brief In the case that this service is not available,
  /// make it available and return any required application shutdown
  /// routines as a new tear_down instance.
  /// @param python_path If a fully-qualified Python executable path name is
  /// known, use it here.
  virtual std::unique_ptr<tear_down>
  make_available(const std::string &python_path = "") const = 0;

  /// Virtual destructor needed when deleting an instance of a derived class
  /// via a pointer to the base class.
  virtual ~MoleculePackageDriver(){};
};

} // namespace cudaq::solvers
