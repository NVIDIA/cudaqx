/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "cudaq.h"
#include <tuple>
#include <vector>

namespace cudaq::solvers::stateprep {

/// @brief Represents a list of excitations
using excitation_list = std::vector<std::vector<std::size_t>>;

/// @brief Get UCCSD excitations for a given system
/// @param numElectrons Number of electrons in the system
/// @param numQubits Number of qubits in the system
/// @param spin Spin of the system
/// @return Tuple containing five excitation lists
std::tuple<excitation_list, excitation_list, excitation_list, excitation_list,
           excitation_list>
get_uccsd_excitations(std::size_t numElectrons, std::size_t numQubits,
                      std::size_t spin = 0);

/// @brief Calculate the number of UCCSD parameters
/// @param numElectrons Number of electrons in the system
/// @param numQubits Number of qubits in the system
/// @param spin Spin of the system (default 0)
/// @return Number of UCCSD parameters
std::size_t get_num_uccsd_parameters(std::size_t numElectrons,
                                     std::size_t numQubits,
                                     std::size_t spin = 0);

/// \pure_device_kernel
///
/// @brief Perform a single excitation operation
/// @param qubits Qubit register
/// @param theta Rotation angle
/// @param p_occ Occupied orbital index
/// @param q_virt Virtual orbital index
__qpu__ void single_excitation(cudaq::qview<> qubits, double theta,
                               std::size_t p_occ, std::size_t q_virt);

/// \pure_device_kernel
///
/// @brief Perform a double excitation operation
/// @param qubits Qubit register
/// @param theta Rotation angle
/// @param p_occ First occupied orbital index
/// @param q_occ Second occupied orbital index
/// @param r_virt First virtual orbital index
/// @param s_virt Second virtual orbital index
__qpu__ void double_excitation(cudaq::qview<> qubits, double theta,
                               std::size_t p_occ, std::size_t q_occ,
                               std::size_t r_virt, std::size_t s_virt);

/// \pure_device_kernel
///
/// @brief Apply UCCSD ansatz to a qubit register
/// @param qubits Qubit register
/// @param thetas Vector of rotation angles
/// @param numElectrons Number of electrons in the system
/// @param spin Spin of the system
__qpu__ void uccsd(cudaq::qview<> qubits, const std::vector<double> &thetas,
                   std::size_t numElectrons, std::size_t spin);

/// \pure_device_kernel
///
/// @brief Apply UCCSD ansatz to a qubit vector
/// @param qubits Qubit vector
/// @param thetas Vector of rotation angles
/// @param numElectrons Number of electrons in the system
__qpu__ void uccsd(cudaq::qview<> qubits, const std::vector<double> &thetas,
                   std::size_t numElectrons);

} // namespace cudaq::solvers::stateprep
