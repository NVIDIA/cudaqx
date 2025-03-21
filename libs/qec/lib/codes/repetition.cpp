/*******************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "cudaq/qec/codes/repetition.h"

namespace cudaq::qec::repetition {

std::size_t repetition::get_num_data_qubits() const { return distance; }

std::size_t repetition::get_num_ancilla_qubits() const { return distance - 1; }
std::size_t repetition::get_num_ancilla_x_qubits() const { return 0; }
std::size_t repetition::get_num_ancilla_z_qubits() const {
  return get_num_ancilla_qubits();
}

repetition::repetition(const heterogeneous_map &options) : code() {
  if (!options.contains("distance"))
    throw std::runtime_error(
        "[repetition] distance not provided. distance must be provided via "
        "qec::get_code(..., options) options map.");
  distance = options.get<std::size_t>("distance");

  // fill the operations
  operation_encodings.insert(
      std::make_pair(operation::stabilizer_round, stabilizer));
  operation_encodings.insert(std::make_pair(operation::x, x));
  operation_encodings.insert(std::make_pair(operation::prep0, prep0));
  operation_encodings.insert(std::make_pair(operation::prep1, prep1));

  // Default Stabilizers should be Zi-1 Zi
  for (std::size_t i = 1; i < get_num_data_qubits(); i++) {
    // TODO - remove the apparently useless but still needed "i" terms.
    m_stabilizers.push_back(cudaq::spin::i(get_num_data_qubits() - 1) *
                            cudaq::spin::z(i - 1) * cudaq::spin::z(i));
  }

  // Default Logical Observable is ZI...I
  // This class is only for Z basis experiments
  // so there is no X observable included.
  cudaq::spin_op Lz = cudaq::spin::z(0);
  Lz = Lz * cudaq::spin::i(get_num_data_qubits() - 1);

  m_pauli_observables.push_back(Lz);

  // Sort now to avoid repeated sorts later.
  sortStabilizerOps(m_stabilizers);
  sortStabilizerOps(m_pauli_observables);
}

/// @brief Register the repetition code type
CUDAQ_REGISTER_TYPE(repetition)

} // namespace cudaq::qec::repetition
