/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/spin_op.h"

#include "cuda-qx/core/tensor.h"

#include <optional>

using namespace cudaqx;

namespace cudaq::solvers {

/// @struct atom
/// @brief Represents an atom with a name and 3D coordinates
struct atom {
  const std::string name;
  const double coordinates[3];
};

/// @class molecular_geometry
/// @brief Represents the geometry of a molecule as a collection of atoms
class molecular_geometry {
private:
  std::vector<atom> atoms;

public:
  /// @brief Constructor using initializer list
  /// @param args Initializer list of atoms
  molecular_geometry(std::initializer_list<atom> &&args)
      : atoms(args.begin(), args.end()) {}

  /// @brief Constructor using vector of atoms
  /// @param args Vector of atoms
  molecular_geometry(const std::vector<atom> &args) : atoms(args) {}

  /// @brief Get the number of atoms in the molecule
  /// @return Size of the molecule
  std::size_t size() const { return atoms.size(); }

  /// @brief Get iterator to the beginning of atoms
  /// @return Iterator to the beginning
  auto begin() { return atoms.begin(); }

  /// @brief Get iterator to the end of atoms
  /// @return Iterator to the end
  auto end() { return atoms.end(); }

  /// @brief Get const iterator to the beginning of atoms
  /// @return Const iterator to the beginning
  auto begin() const { return atoms.cbegin(); };

  /// @brief Get const iterator to the end of atoms
  /// @return Const iterator to the end
  auto end() const { return atoms.cend(); }

  /// @brief Get the name of the molecule
  /// @return Name of the molecule
  std::string name() const;

  /// @brief Create a molecular geometry from an XYZ file
  /// @param xyzFile Path to the XYZ file
  /// @return Molecular geometry object
  static molecular_geometry from_xyz(const std::string &xyzFile);
};

/// @struct molecular_hamiltonian
/// @brief Represents a molecular Hamiltonian in both spin and fermionic forms
struct molecular_hamiltonian {
  /// @brief The molecular Hamiltonian represented as a spin operator
  cudaq::spin_op hamiltonian;

  /// @brief One-electron integrals tensor
  /// @details Represents the one-body terms in the second quantized Hamiltonian
  cudaqx::tensor<> hpq;

  /// @brief Two-electron integrals tensor
  /// @details Represents the two-body terms in the second quantized Hamiltonian
  cudaqx::tensor<> hpqrs;

  /// @brief Number of electrons in the molecule
  std::size_t n_electrons;

  /// @brief Number of orbitals (or spatial orbitals) in the basis set
  std::size_t n_orbitals;

  /// @brief Map of various energy contributions
  /// @details Keys may include "nuclear_repulsion", "hf", "mp2", "ccsd", etc.
  std::unordered_map<std::string, double> energies;
};

/// @struct molecule_options
/// @brief Options for molecule creation and calculation
struct molecule_options {
  /// @brief Driver for the quantum chemistry calculations
  /// default "RESTPySCFDriver"
  std::string driver = "RESTPySCFDriver";

  /// @brief Fully qualified path to Python executable to use (if applicable).
  /// The CUDA-Q Solvers Python wheel will automatically populate this with the
  /// current Python executable path.
  /// default ""
  std::string python_path = "";

  /// @brief Method for mapping fermionic operators to qubit operators
  ///
  /// Currently two methods are available:
  /// - "jordan_wigner": the standard Jordan-Wigner transformation, default.
  /// - "bravyi_kitaev": Bravyi and Kitaev's mapping.
  std::string fermion_to_spin = "jordan_wigner";

  /// @brief Type of molecular system
  /// default "gas_phase"
  std::string type = "gas_phase";

  /// @brief Whether to use symmetry in calculations
  /// default false
  bool symmetry = false;

  /// @brief Amount of memory to allocate for calculations (in MB)
  /// default 4000.0
  double memory = 4000.;

  /// @brief Maximum number of SCF cycles
  /// default 100
  std::size_t cycles = 100;

  /// @brief Initial guess method for SCF calculations
  /// default "minao"
  std::string initguess = "minao";

  /// @brief Whether to use unrestricted calculations
  /// default false
  bool UR = false;

  /// @brief Number of electrons in the active space for CAS calculations
  /// default std::nullopt (not set)
  std::optional<std::size_t> nele_cas = std::nullopt;

  /// @brief Number of orbitals in the active space for CAS calculations
  /// default std::nullopt (not set)
  std::optional<std::size_t> norb_cas = std::nullopt;

  /// @brief Whether to perform MP2 calculations
  /// default false
  bool MP2 = false;

  /// @brief Whether to use natural orbitals
  /// default false
  bool natorb = false;

  /// @brief Whether to perform CASCI calculations
  /// default false
  bool casci = false;

  /// @brief Whether to perform CCSD calculations
  /// default false
  bool ccsd = false;

  /// @brief Whether to perform CASSCF calculations
  /// default false
  bool casscf = false;

  /// @brief Whether to use natural orbitals for integrals
  /// default false
  bool integrals_natorb = false;

  /// @brief Whether to use CASSCF orbitals for integrals
  /// default false
  bool integrals_casscf = false;

  /// @brief Path to the potential file (if applicable)
  /// default std::nullopt (not set)
  std::optional<std::string> potfile = std::nullopt;

  /// @brief Whether to enable verbose output
  /// default false
  bool verbose = false;

  /// @brief Dump the options to output
  void dump();
};

/// @brief Create a molecular Hamiltonian
/// @param geometry Molecular geometry
/// @param basis Basis set
/// @param spin Spin of the molecule
/// @param charge Charge of the molecule
/// @param options Molecule options
/// @return Molecular Hamiltonian
molecular_hamiltonian
create_molecule(const molecular_geometry &geometry, const std::string &basis,
                int spin, int charge,
                molecule_options options = molecule_options());

/// @brief Create a one-particle operator
/// @param numQubits Number of qubits
/// @param p First orbital index
/// @param q Second orbital index
/// @param fermionCompiler Fermion-to-qubit mapping method
/// @return One-particle operator as a spin operator
cudaq::spin_op
one_particle_op(std::size_t numQubits, std::size_t p, std::size_t q,
                const std::string fermionCompiler = "jordan_wigner");
} // namespace cudaq::solvers
