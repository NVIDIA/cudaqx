/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/solvers/operators/operator_pools/uccsd_operator_pool.h"
#include "cudaq/solvers/stateprep/uccsd.h"

using namespace cudaqx;

namespace cudaq::solvers {

using excitation_list = std::vector<std::vector<std::size_t>>;

std::vector<cudaq::spin_op>
uccsd::generate(const heterogeneous_map &config) const {

  auto numQubits = config.get<int>({"num-qubits", "num_qubits", "n-qubits", "n_qubits"});
  auto numElectrons = config.get<int>({"num-electrons", "num_electrons"});
  std::size_t spin = 0;
  if (config.contains("spin"))
    spin = config.get<std::size_t>("spin");

  auto [singlesAlpha, singlesBeta, doublesMixed, doublesAlpha, doublesBeta] =
      cudaq::solvers::stateprep::get_uccsd_excitations(numElectrons, numQubits,
                                                       spin);

  std::vector<cudaq::spin_op> ops;

  auto addSinglesExcitation = [numQubits](std::vector<cudaq::spin_op> &ops,
                                          std::size_t p, std::size_t q) {
    double parity = 1.0;

    cudaq::spin_op o(numQubits);
    for (std::size_t i = p + 1; i < q; i++)
      o *= cudaq::spin::z(i);

    ops.emplace_back(cudaq::spin::y(p) * o * cudaq::spin::x(q));
    ops.emplace_back(cudaq::spin::x(p) * o * cudaq::spin::y(q));
  };

  auto addDoublesExcitation = [numQubits](std::vector<cudaq::spin_op> &ops,
                                          std::size_t p, std::size_t q,
                                          std::size_t r, std::size_t s) {
    cudaq::spin_op parity_a(numQubits), parity_b(numQubits);
    std::size_t i_occ = 0, j_occ = 0, a_virt = 0, b_virt = 0;
    if (p < q && r < s) {
      i_occ = p;
      j_occ = q;
      a_virt = r;
      b_virt = s;
    }

    else if (p > q && r > s) {
      i_occ = q;
      j_occ = p;
      a_virt = s;
      b_virt = r;
    } else if (p < q && r > s) {
      i_occ = p;
      j_occ = q;
      a_virt = s;
      b_virt = r;
    } else if

        (p > q && r < s) {
      i_occ = q;
      j_occ = p;
      a_virt = r;
      b_virt = s;
    }
    for (std::size_t i = i_occ + 1; i < j_occ; i++)
      parity_a *= cudaq::spin::z(i);

    for (std::size_t i = a_virt + 1; i < b_virt; i++)
      parity_b *= cudaq::spin::z(i);

    ops.emplace_back(cudaq::spin::x(i_occ) * parity_a * cudaq::spin::x(j_occ) *
                     cudaq::spin::x(a_virt) * parity_b *
                     cudaq::spin::y(b_virt));
    ops.emplace_back(cudaq::spin::x(i_occ) * parity_a * cudaq::spin::x(j_occ) *
                     cudaq::spin::y(a_virt) * parity_b *
                     cudaq::spin::x(b_virt));
    ops.emplace_back(cudaq::spin::x(i_occ) * parity_a * cudaq::spin::y(j_occ) *
                     cudaq::spin::y(a_virt) * parity_b *
                     cudaq::spin::y(b_virt));
    ops.emplace_back(cudaq::spin::y(i_occ) * parity_a * cudaq::spin::x(j_occ) *
                     cudaq::spin::y(a_virt) * parity_b *
                     cudaq::spin::y(b_virt));
    ops.emplace_back(cudaq::spin::x(i_occ) * parity_a * cudaq::spin::y(j_occ) *
                     cudaq::spin::x(a_virt) * parity_b *
                     cudaq::spin::x(b_virt));
    ops.emplace_back(cudaq::spin::y(i_occ) * parity_a * cudaq::spin::x(j_occ) *
                     cudaq::spin::x(a_virt) * parity_b *
                     cudaq::spin::x(b_virt));
    ops.emplace_back(cudaq::spin::y(i_occ) * parity_a * cudaq::spin::y(j_occ) *
                     cudaq::spin::x(a_virt) * parity_b *
                     cudaq::spin::y(b_virt));
    ops.emplace_back(cudaq::spin::y(i_occ) * parity_a * cudaq::spin::y(j_occ) *
                     cudaq::spin::y(a_virt) * parity_b *
                     cudaq::spin::x(b_virt));
  };

  for (auto &sa : singlesAlpha)
    addSinglesExcitation(ops, sa[0], sa[1]);
  for (auto &sa : singlesBeta)
    addSinglesExcitation(ops, sa[0], sa[1]);

  for (auto &d : doublesMixed)
    addDoublesExcitation(ops, d[0], d[1], d[2], d[3]);
  for (auto &d : doublesAlpha)
    addDoublesExcitation(ops, d[0], d[1], d[2], d[3]);
  for (auto &d : doublesBeta)
    addDoublesExcitation(ops, d[0], d[1], d[2], d[3]);

  return ops;
}

} // namespace cudaq::solvers