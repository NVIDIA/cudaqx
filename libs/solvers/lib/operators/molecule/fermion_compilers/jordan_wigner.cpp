/*******************************************************************************
 * Copyright (c) 2024 NVIDIA Corporation & Affiliates.                         *
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

cudaq::spin_op one_body(std::size_t p, std::size_t q,
                        std::complex<double> coeff) {
  if (p == q)
    return 0.5 * coeff * cudaq::spin::i(p) - 0.5 * coeff * cudaq::spin::z(p);

  if (p > q) {
    std::swap(p, q);
    coeff = std::conj(coeff);
  }

  std::vector<std::size_t> zIndices;
  for (std::size_t i = p + 1; i < q; i++)
    zIndices.push_back(i);

  cudaq::spin_op parity = 1.;
  for (auto i : zIndices)
    parity *= cudaq::spin::z(i);

  auto spin_hamiltonian =
      0.5 * coeff.real() * cudaq::spin::x(p) * parity * cudaq::spin::x(q);
  spin_hamiltonian +=
      0.5 * coeff.real() * cudaq::spin::y(p) * parity * cudaq::spin::y(q);
  spin_hamiltonian +=
      0.5 * coeff.imag() * cudaq::spin::y(p) * parity * cudaq::spin::x(q);
  spin_hamiltonian -=
      0.5 * coeff.imag() * cudaq::spin::x(p) * parity * cudaq::spin::y(q);

  return spin_hamiltonian;
}

cudaq::spin_op two_body(std::size_t p, std::size_t q, std::size_t r,
                        std::size_t s, std::complex<double> coef) {
  std::set<std::size_t> tmp{p, q, r, s};
  if (tmp.size() == 2) {
    auto spin_hamiltonian =
        -0.25 * coef * cudaq::spin::i(p) * cudaq::spin::i(q);
    if (p == r) {
      spin_hamiltonian += 0.25 * coef * cudaq::spin::i(p) * cudaq::spin::z(q);
      spin_hamiltonian += 0.25 * coef * cudaq::spin::z(p) * cudaq::spin::i(q);
      spin_hamiltonian -= 0.25 * coef * cudaq::spin::z(p) * cudaq::spin::z(q);
    } else if (q == r) {
      spin_hamiltonian *= -1.;
      spin_hamiltonian -= 0.25 * coef * cudaq::spin::i(p) * cudaq::spin::z(q);
      spin_hamiltonian -= 0.25 * coef * cudaq::spin::z(p) * cudaq::spin::i(q);
      spin_hamiltonian += 0.25 * coef * cudaq::spin::z(p) * cudaq::spin::z(q);
    }
    return spin_hamiltonian;
  }

  if (tmp.size() == 3) {
    std::size_t a, b, c, d;
    if (q == r) {
      if (p > r) {
        // a,b=s,p
        a = s;
        b = p;
        coef = std::conj(coef);
      } else {
        // a,b=p,s
        a = p;
        b = s;
      }
      c = q;
    } else if (q == s) {
      if (p > r) {
        // a,b=r,p
        a = r;
        b = p;
        coef = -1.0 * std::conj(coef);
      } else {
        // a,b=p,r
        a = p;
        b = r;
        coef *= -1.0;
      }
      c = q;
    } else if (p == r) {
      if (q > s) {
        // a,b=s,q
        a = s;
        b = q;
        coef = -1.0 * std::conj(coef);
      } else {
        // a,b=q,s
        a = q;
        b = s;
        coef = -1.0 * coef;
      }
      c = p;
    } else if (p == s) {
      if (q > r) {
        // a,b=r,q
        a = r;
        b = q;
        coef = std::conj(coef);
      } else {
        // a,b=q,r
        a = q;
        b = r;
      }
      c = p;
    }

    std::vector<std::size_t> zIndices;
    for (std::size_t i = a + 1; i < b; i++)
      zIndices.push_back(i);

    cudaq::spin_op parity = 1.;
    for (auto i : zIndices)
      parity *= cudaq::spin::z(i);

    auto spin_hamiltonian = 0.25 * coef.real() * cudaq::spin::x(a) * parity *
                            cudaq::spin::x(b) * cudaq::spin::i(c);
    spin_hamiltonian += 0.25 * coef.real() * cudaq::spin::y(a) * parity *
                        cudaq::spin::y(b) * cudaq::spin::i(c);
    spin_hamiltonian += 0.25 * coef.imag() * cudaq::spin::y(a) * parity *
                        cudaq::spin::x(b) * cudaq::spin::i(c);
    spin_hamiltonian -= 0.25 * coef.imag() * cudaq::spin::x(a) * parity *
                        cudaq::spin::y(b) * cudaq::spin::i(c);

    spin_hamiltonian -= 0.25 * coef.real() * cudaq::spin::x(a) * parity *
                        cudaq::spin::x(b) * cudaq::spin::z(c);
    spin_hamiltonian -= 0.25 * coef.real() * cudaq::spin::y(a) * parity *
                        cudaq::spin::y(b) * cudaq::spin::z(c);
    spin_hamiltonian -= 0.25 * coef.imag() * cudaq::spin::y(a) * parity *
                        cudaq::spin::x(b) * cudaq::spin::z(c);
    spin_hamiltonian += 0.25 * coef.imag() * cudaq::spin::x(a) * parity *
                        cudaq::spin::y(b) * cudaq::spin::z(c);
    return spin_hamiltonian;
  }

  if ((p > q) ^ (r > s))
    coef *= -1.0;

  if (p < q && q < r && r < s) {
    // a,b,c,d=p,q,r,s
    auto a = p;
    auto b = q;
    auto c = r;
    auto d = s;

    std::vector<std::size_t> zIndices;
    for (std::size_t i = a + 1; i < b; i++)
      zIndices.push_back(i);

    cudaq::spin_op parityA = 1.;
    for (auto i : zIndices)
      parityA *= cudaq::spin::z(i);

    zIndices.clear();
    for (std::size_t i = c + 1; i < d; i++)
      zIndices.push_back(i);

    cudaq::spin_op parityB = 1.;
    for (auto i : zIndices)
      parityB *= cudaq::spin::z(i);

    auto spin_hamiltonian = -0.125 * coef.real() * cudaq::spin::x(a) * parityA *
                            cudaq::spin::x(b) * cudaq::spin::x(c) * parityB *
                            cudaq::spin::x(d);
    spin_hamiltonian -= -0.125 * coef.real() * cudaq::spin::x(a) * parityA *
                        cudaq::spin::x(b) * cudaq::spin::y(c) * parityB *
                        cudaq::spin::y(d);
    spin_hamiltonian += -0.125 * coef.real() * cudaq::spin::x(a) * parityA *
                        cudaq::spin::y(b) * cudaq::spin::x(c) * parityB *
                        cudaq::spin::y(d);
    spin_hamiltonian += -0.125 * coef.real() * cudaq::spin::x(a) * parityA *
                        cudaq::spin::y(b) * cudaq::spin::y(c) * parityB *
                        cudaq::spin::x(d);
    spin_hamiltonian += -0.125 * coef.real() * cudaq::spin::y(a) * parityA *
                        cudaq::spin::x(b) * cudaq::spin::x(c) * parityB *
                        cudaq::spin::y(d);
    spin_hamiltonian += -0.125 * coef.real() * cudaq::spin::y(a) * parityA *
                        cudaq::spin::x(b) * cudaq::spin::y(c) * parityB *
                        cudaq::spin::x(d);
    spin_hamiltonian -= -0.125 * coef.real() * cudaq::spin::y(a) * parityA *
                        cudaq::spin::y(b) * cudaq::spin::x(c) * parityB *
                        cudaq::spin::x(d);
    spin_hamiltonian += -0.125 * coef.real() * cudaq::spin::y(a) * parityA *
                        cudaq::spin::y(b) * cudaq::spin::y(c) * parityB *
                        cudaq::spin::y(d);

    spin_hamiltonian += 0.125 * coef.imag() * cudaq::spin::x(a) * parityA *
                        cudaq::spin::x(b) * cudaq::spin::x(c) * parityB *
                        cudaq::spin::y(d);
    spin_hamiltonian += 0.125 * coef.imag() * cudaq::spin::x(a) * parityA *
                        cudaq::spin::x(b) * cudaq::spin::y(c) * parityB *
                        cudaq::spin::x(d);
    spin_hamiltonian -= 0.125 * coef.imag() * cudaq::spin::x(a) * parityA *
                        cudaq::spin::y(b) * cudaq::spin::x(c) * parityB *
                        cudaq::spin::x(d);
    spin_hamiltonian += 0.125 * coef.imag() * cudaq::spin::x(a) * parityA *
                        cudaq::spin::y(b) * cudaq::spin::y(c) * parityB *
                        cudaq::spin::y(d);
    spin_hamiltonian -= 0.125 * coef.imag() * cudaq::spin::y(a) * parityA *
                        cudaq::spin::x(b) * cudaq::spin::x(c) * parityB *
                        cudaq::spin::x(d);
    spin_hamiltonian += 0.125 * coef.imag() * cudaq::spin::y(a) * parityA *
                        cudaq::spin::x(b) * cudaq::spin::y(c) * parityB *
                        cudaq::spin::y(d);
    spin_hamiltonian -= 0.125 * coef.imag() * cudaq::spin::y(a) * parityA *
                        cudaq::spin::y(b) * cudaq::spin::x(c) * parityB *
                        cudaq::spin::y(d);
    spin_hamiltonian -= 0.125 * coef.imag() * cudaq::spin::y(a) * parityA *
                        cudaq::spin::y(b) * cudaq::spin::y(c) * parityB *
                        cudaq::spin::x(d);
    return spin_hamiltonian;
  }

  if (p < r && r < q && q < s) {
    // a,b,c,d=p,r,q,s
    auto a = p;
    auto b = r;
    auto c = q;
    auto d = s;

    std::vector<std::size_t> zIndices;
    for (std::size_t i = a + 1; i < b; i++)
      zIndices.push_back(i);

    cudaq::spin_op parityA = 1.;
    for (auto i : zIndices)
      parityA *= cudaq::spin::z(i);

    zIndices.clear();
    for (std::size_t i = c + 1; i < d; i++)
      zIndices.push_back(i);

    cudaq::spin_op parityB = 1.;
    for (auto i : zIndices)
      parityB *= cudaq::spin::z(i);

    auto spin_hamiltonian = -0.125 * coef.real() * cudaq::spin::x(a) * parityA *
                            cudaq::spin::x(b) * cudaq::spin::x(c) * parityA *
                            cudaq::spin::x(d);
    spin_hamiltonian += -0.125 * coef.real() * cudaq::spin::x(a) * parityA *
                        cudaq::spin::x(b) * cudaq::spin::y(c) * parityA *
                        cudaq::spin::y(d);
    spin_hamiltonian -= -0.125 * coef.real() * cudaq::spin::x(a) * parityA *
                        cudaq::spin::y(b) * cudaq::spin::x(c) * parityA *
                        cudaq::spin::y(d);
    spin_hamiltonian += -0.125 * coef.real() * cudaq::spin::x(a) * parityA *
                        cudaq::spin::y(b) * cudaq::spin::y(c) * parityA *
                        cudaq::spin::x(d);
    spin_hamiltonian += -0.125 * coef.real() * cudaq::spin::y(a) * parityA *
                        cudaq::spin::x(b) * cudaq::spin::x(c) * parityA *
                        cudaq::spin::y(d);
    spin_hamiltonian -= -0.125 * coef.real() * cudaq::spin::y(a) * parityA *
                        cudaq::spin::x(b) * cudaq::spin::y(c) * parityA *
                        cudaq::spin::x(d);
    spin_hamiltonian += -0.125 * coef.real() * cudaq::spin::y(a) * parityA *
                        cudaq::spin::y(b) * cudaq::spin::x(c) * parityA *
                        cudaq::spin::x(d);
    spin_hamiltonian += -0.125 * coef.real() * cudaq::spin::y(a) * parityA *
                        cudaq::spin::y(b) * cudaq::spin::y(c) * parityA *
                        cudaq::spin::y(d);

    spin_hamiltonian += 0.125 * coef.imag() * cudaq::spin::x(a) * parityA *
                        cudaq::spin::x(b) * cudaq::spin::x(c) * parityA *
                        cudaq::spin::y(d);
    spin_hamiltonian -= 0.125 * coef.imag() * cudaq::spin::x(a) * parityA *
                        cudaq::spin::x(b) * cudaq::spin::y(c) * parityA *
                        cudaq::spin::x(d);
    spin_hamiltonian += 0.125 * coef.imag() * cudaq::spin::x(a) * parityA *
                        cudaq::spin::y(b) * cudaq::spin::x(c) * parityA *
                        cudaq::spin::x(d);
    spin_hamiltonian += 0.125 * coef.imag() * cudaq::spin::x(a) * parityA *
                        cudaq::spin::y(b) * cudaq::spin::y(c) * parityA *
                        cudaq::spin::y(d);
    spin_hamiltonian -= 0.125 * coef.imag() * cudaq::spin::y(a) * parityA *
                        cudaq::spin::x(b) * cudaq::spin::x(c) * parityA *
                        cudaq::spin::x(d);
    spin_hamiltonian -= 0.125 * coef.imag() * cudaq::spin::y(a) * parityA *
                        cudaq::spin::x(b) * cudaq::spin::y(c) * parityA *
                        cudaq::spin::y(d);
    spin_hamiltonian += 0.125 * coef.imag() * cudaq::spin::y(a) * parityA *
                        cudaq::spin::y(b) * cudaq::spin::x(c) * parityA *
                        cudaq::spin::y(d);
    spin_hamiltonian -= 0.125 * coef.imag() * cudaq::spin::y(a) * parityA *
                        cudaq::spin::y(b) * cudaq::spin::y(c) * parityA *
                        cudaq::spin::x(d);
    return spin_hamiltonian;
  }

  if (p < r && r < s && s < q) {
    // a,b,c,d=p,r,s,q
    auto a = p;
    auto b = r;
    auto c = s;
    auto d = q;

    std::vector<std::size_t> zIndices;
    for (std::size_t i = a + 1; i < b; i++)
      zIndices.push_back(i);

    cudaq::spin_op parityA = 1.;
    for (auto i : zIndices)
      parityA *= cudaq::spin::z(i);

    zIndices.clear();
    for (std::size_t i = c + 1; i < d; i++)
      zIndices.push_back(i);

    cudaq::spin_op parityB = 1.;
    for (auto i : zIndices)
      parityB *= cudaq::spin::z(i);

    auto spin_hamiltonian = -0.125 * coef.real() * cudaq::spin::x(a) * parityA *
                            cudaq::spin::x(b) * cudaq::spin::x(c) * parityB *
                            cudaq::spin::x(d);
    spin_hamiltonian += -0.125 * coef.real() * cudaq::spin::x(a) * parityA *
                        cudaq::spin::x(b) * cudaq::spin::y(c) * parityB *
                        cudaq::spin::y(d);
    spin_hamiltonian += -0.125 * coef.real() * cudaq::spin::x(a) * parityA *
                        cudaq::spin::y(b) * cudaq::spin::x(c) * parityB *
                        cudaq::spin::y(d);
    spin_hamiltonian -= -0.125 * coef.real() * cudaq::spin::x(a) * parityA *
                        cudaq::spin::y(b) * cudaq::spin::y(c) * parityB *
                        cudaq::spin::x(d);
    spin_hamiltonian -= -0.125 * coef.real() * cudaq::spin::y(a) * parityA *
                        cudaq::spin::x(b) * cudaq::spin::x(c) * parityB *
                        cudaq::spin::y(d);
    spin_hamiltonian += -0.125 * coef.real() * cudaq::spin::y(a) * parityA *
                        cudaq::spin::x(b) * cudaq::spin::y(c) * parityB *
                        cudaq::spin::x(d);
    spin_hamiltonian += -0.125 * coef.real() * cudaq::spin::y(a) * parityA *
                        cudaq::spin::y(b) * cudaq::spin::x(c) * parityB *
                        cudaq::spin::x(d);
    spin_hamiltonian += -0.125 * coef.real() * cudaq::spin::y(a) * parityA *
                        cudaq::spin::y(b) * cudaq::spin::y(c) * parityB *
                        cudaq::spin::y(d);

    spin_hamiltonian -= 0.125 * coef.imag() * cudaq::spin::x(a) * parityA *
                        cudaq::spin::x(b) * cudaq::spin::x(c) * parityB *
                        cudaq::spin::y(d);
    spin_hamiltonian += 0.125 * coef.imag() * cudaq::spin::x(a) * parityA *
                        cudaq::spin::x(b) * cudaq::spin::y(c) * parityB *
                        cudaq::spin::x(d);
    spin_hamiltonian += 0.125 * coef.imag() * cudaq::spin::x(a) * parityA *
                        cudaq::spin::y(b) * cudaq::spin::x(c) * parityB *
                        cudaq::spin::x(d);
    spin_hamiltonian += 0.125 * coef.imag() * cudaq::spin::x(a) * parityA *
                        cudaq::spin::y(b) * cudaq::spin::y(c) * parityB *
                        cudaq::spin::y(d);
    spin_hamiltonian -= 0.125 * coef.imag() * cudaq::spin::y(a) * parityA *
                        cudaq::spin::x(b) * cudaq::spin::x(c) * parityB *
                        cudaq::spin::x(d);
    spin_hamiltonian -= 0.125 * coef.imag() * cudaq::spin::y(a) * parityA *
                        cudaq::spin::x(b) * cudaq::spin::y(c) * parityB *
                        cudaq::spin::y(d);
    spin_hamiltonian -= 0.125 * coef.imag() * cudaq::spin::y(a) * parityA *
                        cudaq::spin::y(b) * cudaq::spin::x(c) * parityB *
                        cudaq::spin::y(d);
    spin_hamiltonian += 0.125 * coef.imag() * cudaq::spin::y(a) * parityA *
                        cudaq::spin::y(b) * cudaq::spin::y(c) * parityB *
                        cudaq::spin::x(d);

    return spin_hamiltonian;
  }

  throw std::runtime_error(
      "Invalid condition in two_body jordan wigner function.");
}

cudaq::spin_op jordan_wigner::generate(const double constant,
                                       const tensor<> &hpq,
                                       const tensor<> &hpqrs,
                                       const heterogeneous_map &options) {
  assert(hpq.rank() == 2 && "hpq must be a rank-2 tensor");
  assert(hpqrs.rank() == 4 && "hpqrs must be a rank-4 tensor");
  auto spin_hamiltonian = constant * cudaq::spin_op();
  std::size_t nqubit = hpq.shape()[0];

  double tolerance =
      options.get({"tolerance", "tol", "TOLERANCE", "TOL"}, 1e-15);

  for (auto p : cudaq::range(nqubit)) {
    auto coef = hpq.at({p, p});
    if (std::fabs(coef) > tolerance)
      spin_hamiltonian += one_body(p, p, coef);
  }

  std::vector<std::vector<std::size_t>> next;
  for (auto &&combo : iter::combinations(cudaq::range(nqubit), 2)) {
    auto p = combo[0];
    auto q = combo[1];
    next.push_back({p, q});
    auto coef = 0.5 * (hpq.at({p, q}) + std::conj(hpq.at({q, p})));
    if (std::fabs(coef) > tolerance)
      spin_hamiltonian += one_body(p, q, coef);

    coef = hpqrs.at({p, q, p, q}) + hpqrs.at({q, p, q, p});
    if (std::fabs(coef) > tolerance)
      spin_hamiltonian += two_body(p, q, p, q, coef);

    coef = hpqrs.at({p, q, q, p}) + hpqrs.at({q, p, p, q});
    if (std::fabs(coef) > tolerance)
      spin_hamiltonian += two_body(p, q, q, p, coef);
  }

  for (auto combo : iter::combinations(next, 2)) {
    auto p = combo[0][0];
    auto q = combo[0][1];
    auto r = combo[1][0];
    auto s = combo[1][1];
    auto coef =
        0.5 * (hpqrs.at({p, q, r, s}) + std::conj(hpqrs.at({s, r, q, p})) -
               hpqrs.at({q, p, r, s}) - std::conj(hpqrs.at({s, r, p, q})) -
               hpqrs.at({p, q, s, r}) - std::conj(hpqrs.at({r, s, q, p})) +
               hpqrs.at({q, p, s, r}) + std::conj(hpqrs.at({r, s, p, q})));

    if (std::fabs(coef) > tolerance)
      spin_hamiltonian += two_body(p, q, r, s, coef);
  }

  // Remove terms with 0.0 coefficient
  std::vector<cudaq::spin_op> nonZeros;
  for (auto term : spin_hamiltonian) {
    auto coeff = term.get_coefficient();
    if (std::fabs(coeff) > tolerance)
      nonZeros.push_back(term);
  }
  auto op = nonZeros[0];
  for (std::size_t i = 1; i < nonZeros.size(); i++)
    op += nonZeros[i];

  return op;
}
} // namespace cudaq::solvers