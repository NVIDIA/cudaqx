/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq.h"
#include "cudaq/qec/patch.h"

// Device kernels for the inlined-feedback toy code used by test_qec.cpp
// (see the QECCodeTester.checkInlinedFeedbackToy* tests). This file is
// nvq++-compiled, like the registered codes' *_device.cpp files, so the
// kernels are visible to the kernel registry that the memory-circuit
// machinery uses to resolve qkernel arguments.

namespace cudaq::qec::feedback_toy {

__qpu__ void prep0(patch p) {
  for (std::size_t i = 0; i < p.data.size(); i++)
    reset(p.data[i]);
}

// A superdense (Bell-pair) round for a 2-data-qubit code with stabilizers XX
// and ZZ that deliberately ends with an *uncorrected* record-conditioned
// byproduct: after the Bell decode, ancx[0] sits in the computational basis
// holding its future measurement outcome r_X, so the trailing
// CX(ancx[0], data[1]) is the unitary equivalent of the classically
// controlled byproduct X^{r_X} on data[1] that a hardware frame update would
// otherwise track. The stabilizer supports are hardcoded (a single XX and ZZ
// plaquette), so the x/z_stabilizers arguments are intentionally unused.
__qpu__ std::vector<cudaq::measure_result>
stabilizer_round(patch p, const std::vector<std::size_t> &x_stabilizers,
                 const std::vector<std::size_t> &z_stabilizers) {
  // Bell-prepare the superdense ancilla pair.
  h(p.ancx[0]);
  cudaq::x<cudaq::ctrl>(p.ancx[0], p.ancz[0]);
  // Couple XX through the X ancilla and ZZ through the Z ancilla.
  cudaq::x<cudaq::ctrl>(p.ancx[0], p.data[0]);
  cudaq::x<cudaq::ctrl>(p.ancx[0], p.data[1]);
  cudaq::x<cudaq::ctrl>(p.data[0], p.ancz[0]);
  cudaq::x<cudaq::ctrl>(p.data[1], p.ancz[0]);
  // Decode the Bell pair.
  cudaq::x<cudaq::ctrl>(p.ancx[0], p.ancz[0]);
  h(p.ancx[0]);
  // Uncorrected record-conditioned byproduct: X^{r_X} on data[1].
  cudaq::x<cudaq::ctrl>(p.ancx[0], p.data[1]);
  // Records in [Z][X] order.
  auto results = mz(p.ancz, p.ancx);
  reset(p.ancz[0]);
  reset(p.ancx[0]);
  return results;
}

} // namespace cudaq::qec::feedback_toy
