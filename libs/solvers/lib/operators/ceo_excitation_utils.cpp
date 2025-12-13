/*******************************************************************************
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/solvers/operators/ceo_excitation_utils.h"
#include <algorithm>
#include <tuple>

namespace cudaq::solvers {

std::vector<std::pair<std::size_t, std::size_t>>
generate_ceo_alpha_singles(std::size_t numOrbitals) {
  std::vector<std::pair<std::size_t, std::size_t>> singles;
  // Alpha spin orbitals are at even indices: 0, 2, 4, ..., 2*(numOrbitals-1)
  singles.reserve(numOrbitals * (numOrbitals - 1) / 2);
  for (std::size_t i = 0; i < numOrbitals; ++i) {
    for (std::size_t j = 0; j < i; ++j) {
      std::size_t p = 2 * i; // Higher alpha index
      std::size_t q = 2 * j; // Lower alpha index
      singles.emplace_back(p, q);
    }
  }
  return singles;
}

std::vector<std::pair<std::size_t, std::size_t>>
generate_ceo_beta_singles(std::size_t numOrbitals) {
  std::vector<std::pair<std::size_t, std::size_t>> singles;
  // Beta spin orbitals are at odd indices: 1, 3, 5, ..., 2*numOrbitals-1
  singles.reserve(numOrbitals * (numOrbitals - 1) / 2);
  for (std::size_t i = 0; i < numOrbitals; ++i) { 
    for (std::size_t j = 0; j < i; ++j) {
      std::size_t p = 2 * i + 1; // Higher beta index
      std::size_t q = 2 * j + 1; // Lower beta index
      singles.emplace_back(p, q);
    }
  }
  return singles;
}

std::vector<std::tuple<std::size_t, std::size_t, std::size_t, std::size_t>>
generate_ceo_alpha_doubles(std::size_t numOrbitals) {
  std::vector<std::tuple<std::size_t, std::size_t, std::size_t, std::size_t>>
      doubles;
  // Alpha spin orbitals are at even indices
  // For p > q > r > s, generate three pairings:
  // (p,q)->(r,s), (p,r)->(q,s), (q,p)->(r,s)
  // For CEO, these correspond to different excitations (following
  // the paper conventions (https://arxiv.org/abs/2407.08696)).

  for (std::size_t i = 0; i < numOrbitals; ++i) {
    for (std::size_t j = 0; j < i; ++j) {
      for (std::size_t k = 0; k < j; ++k) {
        for (std::size_t l = 0; l < k; ++l) {
          // i > j > k > l in spatial orbital indices
          std::size_t p = 2 * i;
          std::size_t q = 2 * j;
          std::size_t r = 2 * k;
          std::size_t s = 2 * l;

          // Pairing 1: (p,q)->(r,s)
          doubles.emplace_back(p, q, r, s);
          // Pairing 2: (p,r)->(q,s)
          doubles.emplace_back(p, r, q, s);
          // Pairing 3: (q,p)->(r,s)
          doubles.emplace_back(q, p, r, s);
        }
      }
    }
  }
  return doubles;
}

std::vector<std::tuple<std::size_t, std::size_t, std::size_t, std::size_t>>
generate_ceo_beta_doubles(std::size_t numOrbitals) {
  std::vector<std::tuple<std::size_t, std::size_t, std::size_t, std::size_t>>
      doubles;
  // Beta spin orbitals are at odd indices
  // For p > q > r > s, generate three pairings:
  // (p,q)->(r,s), (p,r)->(q,s), (q,p)->(r,s)
  // For CEO, these correspond to different excitations (following
  // the paper conventions (https://arxiv.org/abs/2407.08696)).

  for (std::size_t i = 0; i < numOrbitals; ++i) {
    for (std::size_t j = 0; j < i; ++j) {
      for (std::size_t k = 0; k < j; ++k) {
        for (std::size_t l = 0; l < k; ++l) {
          // i > j > k > l in spatial orbital indices
          std::size_t p = 2 * i + 1;
          std::size_t q = 2 * j + 1;
          std::size_t r = 2 * k + 1;
          std::size_t s = 2 * l + 1;

          // Pairing 1: (p,q)->(r,s)
          doubles.emplace_back(p, q, r, s);
          // Pairing 2: (p,r)->(q,s)
          doubles.emplace_back(p, r, q, s);
          // Pairing 3: (q,p)->(r,s)
          doubles.emplace_back(q, p, r, s);
        }
      }
    }
  }
  return doubles;
}

std::vector<std::tuple<std::size_t, std::size_t, std::size_t, std::size_t>>
generate_ceo_mixed_doubles(std::size_t numOrbitals) {
  std::vector<std::tuple<std::size_t, std::size_t, std::size_t, std::size_t>>
      doubles;
  // Mixed doubles: following the CEO convention (https://arxiv.org/abs/2407.08696), 
  // the excitation operator for p,q,r,s excites pair (p,q) to (r,s), so
  // the spins should be (alpha, beta) -> (alpha, beta), and we want 
  // p > r and q > s.

  for (std::size_t i = 0; i < numOrbitals; ++i) {
    for (std::size_t j = 0; j < numOrbitals; ++j) {
      for (std::size_t k = 0; k < i; ++k) {
        for (std::size_t l = 0; l < j; ++l) {
          std::size_t p = 2 * i;
          std::size_t q = 2 * j + 1;
          std::size_t r = 2 * k;
          std::size_t s = 2 * l + 1;
          doubles.emplace_back(p, q, r, s);
        }
      }
    }
  }
  return doubles;
}

void addCEOSingleExcitation(std::vector<cudaq::spin_op> &ops, std::size_t p,
                            std::size_t q) {
  // CEO single excitation: 0.5 * (Y_q X_p - X_q Y_p)
  // No Z parity string
  std::complex<double> c = {0.5, 0.0};

  ops.emplace_back(c * cudaq::spin::y(q) * cudaq::spin::x(p) -
                   c * cudaq::spin::x(q) * cudaq::spin::y(p));
}

void addCEODoubleExcitation(std::vector<cudaq::spin_op> &ops, std::size_t p,
                            std::size_t q, std::size_t r, std::size_t s) {
  // CEO double excitation generates TWO operators for indices (p, q, r, s)
  std::complex<double> c = {0.25, 0.0};

  // Operator A: 0.25 * (X_r X_p X_s Y_q - X_r X_p Y_s X_q + Y_r Y_p X_s Y_q -
  // Y_r Y_p Y_s X_q)
  cudaq::spin_op op_a;
  op_a = c * cudaq::spin::x(r) * cudaq::spin::x(p) * cudaq::spin::x(s) *
         cudaq::spin::y(q);
  op_a -= c * cudaq::spin::x(r) * cudaq::spin::x(p) * cudaq::spin::y(s) *
          cudaq::spin::x(q);
  op_a += c * cudaq::spin::y(r) * cudaq::spin::y(p) * cudaq::spin::x(s) *
          cudaq::spin::y(q);
  op_a -= c * cudaq::spin::y(r) * cudaq::spin::y(p) * cudaq::spin::y(s) *
          cudaq::spin::x(q);
  ops.emplace_back(op_a);

  // Operator B: 0.25 * (X_r Y_p X_s X_q + X_r Y_p Y_s Y_q - Y_r X_p X_s X_q -
  // Y_r X_p Y_s Y_q)
  cudaq::spin_op op_b;
  op_b = c * cudaq::spin::x(r) * cudaq::spin::y(p) * cudaq::spin::x(s) *
         cudaq::spin::x(q);
  op_b += c * cudaq::spin::x(r) * cudaq::spin::y(p) * cudaq::spin::y(s) *
          cudaq::spin::y(q);
  op_b -= c * cudaq::spin::y(r) * cudaq::spin::x(p) * cudaq::spin::x(s) *
          cudaq::spin::x(q);
  op_b -= c * cudaq::spin::y(r) * cudaq::spin::x(p) * cudaq::spin::y(s) *
          cudaq::spin::y(q);
  ops.emplace_back(op_b);
}

} // namespace cudaq::solvers


