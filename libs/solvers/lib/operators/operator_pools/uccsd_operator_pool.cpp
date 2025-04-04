/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
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

  std::size_t numQubits =
      config.get<int>({"num-qubits", "num_qubits", "n-qubits", "n_qubits"});
  std::size_t numElectrons =
      config.get<int>({"num-electrons", "num_electrons"});
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

    cudaq::spin_op_term o;
    for (std::size_t i = p + 1; i < q; i++)
      o *= cudaq::spin::z(i);
    std::complex<double> c = {0.5, 0};
    ops.emplace_back(c * cudaq::spin::y(p) * o * cudaq::spin::x(q) -
                     c * cudaq::spin::x(p) * o * cudaq::spin::y(q));
  };

  auto addDoublesExcitation = [numQubits](std::vector<cudaq::spin_op> &ops,
                                          std::size_t p, std::size_t q,
                                          std::size_t r, std::size_t s) {
    cudaq::spin_op_term parity_a;
    cudaq::spin_op_term parity_b;
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
    } else if (p > q && r < s) {
      i_occ = q;
      j_occ = p;
      a_virt = r;
      b_virt = s;
    }
    for (std::size_t i = i_occ + 1; i < j_occ; i++)
      parity_a *= cudaq::spin::z(i);

    for (std::size_t i = a_virt + 1; i < b_virt; i++)
      parity_b *= cudaq::spin::z(i);

    cudaq::spin_op op_term_temp =
        cudaq::spin::x(i_occ) * parity_a * cudaq::spin::x(j_occ) *
        cudaq::spin::x(a_virt) * parity_b * cudaq::spin::y(b_virt);
    op_term_temp += cudaq::spin::x(i_occ) * parity_a * cudaq::spin::x(j_occ) *
                    cudaq::spin::y(a_virt) * parity_b * cudaq::spin::x(b_virt);
    op_term_temp += cudaq::spin::x(i_occ) * parity_a * cudaq::spin::y(j_occ) *
                    cudaq::spin::y(a_virt) * parity_b * cudaq::spin::y(b_virt);
    op_term_temp += cudaq::spin::y(i_occ) * parity_a * cudaq::spin::x(j_occ) *
                    cudaq::spin::y(a_virt) * parity_b * cudaq::spin::y(b_virt);
    op_term_temp -= cudaq::spin::x(i_occ) * parity_a * cudaq::spin::y(j_occ) *
                    cudaq::spin::x(a_virt) * parity_b * cudaq::spin::x(b_virt);
    op_term_temp -= cudaq::spin::y(i_occ) * parity_a * cudaq::spin::x(j_occ) *
                    cudaq::spin::x(a_virt) * parity_b * cudaq::spin::x(b_virt);
    op_term_temp -= cudaq::spin::y(i_occ) * parity_a * cudaq::spin::y(j_occ) *
                    cudaq::spin::x(a_virt) * parity_b * cudaq::spin::y(b_virt);
    op_term_temp -= cudaq::spin::y(i_occ) * parity_a * cudaq::spin::y(j_occ) *
                    cudaq::spin::y(a_virt) * parity_b * cudaq::spin::x(b_virt);

    std::complex<double> c = {0.125, 0};
    ops.emplace_back(c * op_term_temp);
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
