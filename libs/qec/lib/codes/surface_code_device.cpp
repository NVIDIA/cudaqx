/*******************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq.h"
#include "cudaq/qec/patch.h"

namespace cudaq::qec::surface_code {

__qpu__ void x(patch logicalQubit) { x(logicalQubit.data); }
__qpu__ void z(patch logicalQubit) { z(logicalQubit.data); }

__qpu__ void cx(patch logicalQubitA, patch logicalQubitB) {
  for (std::size_t i = 0; i < logicalQubitA.data.size(); i++) {
    x<cudaq::ctrl>(logicalQubitA.data[i], logicalQubitB.data[i]);
  }
}

__qpu__ void cz(patch logicalQubitA, patch logicalQubitB) {
  for (std::size_t i = 0; i < logicalQubitA.data.size(); i++) {
    z<cudaq::ctrl>(logicalQubitA.data[i], logicalQubitB.data[i]);
  }
}

// Transversal state prep, turn on stabilizers after these ops
__qpu__ void prep0(patch logicalQubit) {
  for (std::size_t i = 0; i < logicalQubit.data.size(); i++)
    reset(logicalQubit.data[i]);
}

__qpu__ void prep1(patch logicalQubit) {
  prep0(logicalQubit);
  x(logicalQubit.data);
}

__qpu__ void prepp(patch logicalQubit) {
  prep0(logicalQubit);
  h(logicalQubit.data);
}

__qpu__ void prepm(patch logicalQubit) {
  prep0(logicalQubit);
  x(logicalQubit.data);
  h(logicalQubit.data);
}

__qpu__ std::vector<cudaq::measure_result>
stabilizer(patch logicalQubit, const std::vector<std::size_t> &x_stabilizers,
           const std::vector<std::size_t> &z_stabilizers) {
  for (std::size_t i = 0; i < logicalQubit.ancx.size(); i++)
    reset(logicalQubit.ancx[i]);
  for (std::size_t i = 0; i < logicalQubit.ancz.size(); i++)
    reset(logicalQubit.ancz[i]);

  // The stabilizer matrices are CNOT schedules: entry 0 means the ancilla
  // does not touch that data qubit, entry k >= 1 means the CNOT executes at
  // timestep k. The X and Z checks share the timesteps, giving one depth-4
  // extraction round. The order matters twice over: per plaquette it steers
  // mid-round ancilla (hook) errors perpendicular to the logical operators,
  // and across the X/Z interleaving it must be a valid schedule pair so the
  // non-commuting CNOTs still measure the intended stabilizers (see
  // stabilizer_grid::get_cnot_schedule_x).
  std::size_t num_steps = 1;
  for (std::size_t i = 0; i < x_stabilizers.size(); ++i)
    if (x_stabilizers[i] > num_steps)
      num_steps = x_stabilizers[i];
  for (std::size_t i = 0; i < z_stabilizers.size(); ++i)
    if (z_stabilizers[i] > num_steps)
      num_steps = z_stabilizers[i];

  h(logicalQubit.ancx);
  for (std::size_t step = 1; step <= num_steps; ++step) {
    for (std::size_t xi = 0; xi < logicalQubit.ancx.size(); ++xi)
      for (std::size_t di = 0; di < logicalQubit.data.size(); ++di)
        if (x_stabilizers[xi * logicalQubit.data.size() + di] == step)
          cudaq::x<cudaq::ctrl>(logicalQubit.ancx[xi], logicalQubit.data[di]);
    for (std::size_t zi = 0; zi < logicalQubit.ancz.size(); ++zi)
      for (std::size_t di = 0; di < logicalQubit.data.size(); ++di)
        if (z_stabilizers[zi * logicalQubit.data.size() + di] == step)
          cudaq::x<cudaq::ctrl>(logicalQubit.data[di], logicalQubit.ancz[zi]);
  }
  h(logicalQubit.ancx);

  // S = (S_X, S_Z), (x flip syndromes, z flip syndromes).
  // x flips are triggered by z-stabilizers (ancz)
  // z flips are triggered by x-stabilizers (ancx)
  auto results = mz(logicalQubit.ancz, logicalQubit.ancx);

  return results;
}

} // namespace cudaq::qec::surface_code
