/*******************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/solvers/operators/molecule/fermion_compilers/jordan_wigner.h"

#include <cppitertools/combinations.hpp>
#include <ranges>
#include <set>
#include <span>

using namespace cudaqx;

namespace cudaq::solvers {

cudaq::spin_op jordan_wigner::generate(const double constant,
                                       const cudaqx::tensor<> &hpq,
                                       const cudaqx::tensor<> &hpqrs,
                                       const heterogeneous_map &options) {
  assert(hpq.rank() == 2 && "hpq must be a rank-2 tensor");
  assert(hpqrs.rank() == 4 && "hpqrs must be a rank-4 tensor");

  std::size_t nqubit = hpq.shape()[0];
  cudaq::spin_op spin_hamiltonian = constant * cudaq::spin_op_term();
  double tol =
      options.get<double>(std::vector<std::string>{"tolerance", "tol"}, 1e-15);

  auto is_complex_zero = [tol](const std::complex<double> &z) {
    return std::abs(z.real()) < tol && std::abs(z.imag()) < tol;
  };

  auto adag = [](std::size_t numQubits, std::size_t j) {
    cudaq::spin_op_term zprod;
    for (std::size_t k = 0; k < j; k++)
      zprod *= cudaq::spin::z(k);
    return 0.5 * zprod *
           (cudaq::spin::x(j) - std::complex<double>{0, 1} * cudaq::spin::y(j));
  };

  auto a = [](std::size_t numQubits, std::size_t j) {
    cudaq::spin_op_term zprod;
    for (std::size_t k = 0; k < j; k++)
      zprod *= cudaq::spin::z(k);
    return 0.5 * zprod *
           (cudaq::spin::x(j) + std::complex<double>{0, 1} * cudaq::spin::y(j));
  };

  for (std::size_t i = 0; i < hpq.shape()[0]; i++)
    for (std::size_t j = 0; j < hpq.shape()[1]; j++)
      if (!is_complex_zero(hpq.at({i, j})))
        spin_hamiltonian += hpq.at({i, j}) * adag(nqubit, i) * a(nqubit, j);

  for (std::size_t i = 0; i < hpqrs.shape()[0]; i++)
    for (std::size_t j = 0; j < hpqrs.shape()[1]; j++)
      for (std::size_t k = 0; k < hpqrs.shape()[0]; k++)
        for (std::size_t l = 0; l < hpqrs.shape()[1]; l++)
          if (!is_complex_zero(hpqrs.at({i, j, k, l})))
            spin_hamiltonian += hpqrs.at({i, j, k, l}) * adag(nqubit, i) *
                                adag(nqubit, j) * a(nqubit, k) * a(nqubit, l);

  // Remove terms with 0.0 coefficient
  return spin_hamiltonian.canonicalize().trim(tol);
}
} // namespace cudaq::solvers
