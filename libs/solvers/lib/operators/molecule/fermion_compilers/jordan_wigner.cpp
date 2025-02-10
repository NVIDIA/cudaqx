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

cudaq::spin_op one_body(std::size_t p, std::size_t q, std::complex<double> coeff) {

  if(p != q){
    if(p > q){
      std::swap(p,q);
      coeff = std::conj(coeff);
    }
    cudaq::spin_op parity = 1.;
    for (std::size_t i = p + 1; i < q; i++)
      parity *= cudaq::spin::z(i);

    auto spin_hamiltonian = cudaq::spin_op();
    std::vector<std::tuple<double, cudaq::spin_op(*)(std::size_t), cudaq::spin_op(*)(std::size_t)>> operations = {
      {coeff.real(), cudaq::spin::x, cudaq::spin::x},
      {coeff.real(), cudaq::spin::y, cudaq::spin::y},
      {coeff.imag(), cudaq::spin::y, cudaq::spin::x},
      {-coeff.imag(), cudaq::spin::x, cudaq::spin::y}
    };
    for(const auto& [c, op_a, op_b] : operations){
      if(spin_hamiltonian.empty())
        spin_hamiltonian = 0.5 * c * op_a(p) * parity * op_b(q);
      else
        spin_hamiltonian += 0.5 * c * op_a(p) * parity * op_b(q);
    }
  }
  
  return 0.5 * coeff * cudaq::spin::i(p) - 0.5 * coeff * cudaq::spin::z(p);
}

cudaq::spin_op two_body(std::size_t p, std::size_t q, std::size_t r,
                        std::size_t s, std::complex<double> coef) {

  auto spin_hamiltonian = cudaq::spin_op();
  std::set<std::size_t> tmp{p, q, r, s};

  //return zero terms
  if(p == q || r == s){
    return spin_hamiltonian;
  }
  else if (tmp.size() == 2) {
      auto mult_coef = coef;
      if (p == s) {
        mult_coef *= -0.25;
      } else { // if (q == r) 
        mult_coef *= 0.25;
      }
      spin_hamiltonian = mult_coef * cudaq::spin::i(p) * cudaq::spin::i(q);
      spin_hamiltonian += mult_coef * cudaq::spin::i(p) * cudaq::spin::z(q);
      spin_hamiltonian += mult_coef * cudaq::spin::z(p) * cudaq::spin::i(q);
      spin_hamiltonian -= mult_coef * cudaq::spin::z(p) * cudaq::spin::z(q);
      return spin_hamiltonian;
    }
    else if (tmp.size() == 3){
      std::size_t a, b, c, d;
      if (p == r) {
            if (q > s){
                a = s;
                b = q;
                coef = -std::conj(coef);
            }
            else{
                a = q;
                b = s;
                coef = -coef;
            }
            c = p;
      }
      else if(p == s){
            if (q > r){
                a = r;
                b = q;
                coef = std::conj(coef);
            }else{
                a = q;
                b = r;
            } 
            c = p;
      } else if (q == r){
                if (p > s){
                    a = s;
                    b = p;
                    coef = std::conj(coef);
                }
                else{
                    a = p;
                    b = s;
                }
                c = q;
       } else if (q == s){
                if (p > r){
                    a = r;
                    b = p;
                    coef = -std::conj(coef);
                }
                else{
                    a = p;
                    b = r;
                    coef = -coef;
                }
                c = q;
        }
      cudaq::spin_op parity = 1.;
      for (std::size_t i = a + 1; i < b; i++)
        parity *= cudaq::spin::z(i);
      
      auto pauli_z = cudaq::spin::z(c);

      std::vector<std::tuple<double, cudaq::spin_op(*)(std::size_t), cudaq::spin_op(*)(std::size_t)>> operations = {
        {coef.real(), cudaq::spin::x, cudaq::spin::x},
        {coef.real(), cudaq::spin::y, cudaq::spin::y},
        {coef.imag(), cudaq::spin::y, cudaq::spin::x},
        {-coef.imag(), cudaq::spin::x, cudaq::spin::y}
      };
      for(const auto& [cRI, op_a, op_b] : operations){
        auto term =  op_a(a) * parity * op_b(b);
        if (cRI == 0.0){
          continue;
        }
        auto hopping_term = 0.125*cRI*term;
        spin_hamiltonian -= pauli_z * hopping_term;
        spin_hamiltonian += hopping_term;
      }
      return spin_hamiltonian;
    }
    else if (tmp.size() == 4){

      if ((p > q) != (r > s)) {  
        coef *= -1.0;
      }
      std::vector<std::vector<char>> ops_combinations = {
          {'X', 'Y', 'X', 'X'}, {'Y', 'X', 'X', 'X'}, 
          {'Y', 'Y', 'X', 'Y'}, {'Y', 'Y', 'Y', 'X'},
          {'X', 'X', 'Y', 'Y'}, {'Y', 'Y', 'X', 'X'}
      };
      std::complex<double> multcoef;
      for (const auto& ops : ops_combinations) {
          int x_count = std::count(ops.begin(), ops.end(), 'X');
          
          if (x_count % 2) {
              multcoef = 0.125 * coef.imag();  
              std::string ops_str(ops.begin(), ops.end());
              if (ops_str == "XYXX" || ops_str == "YXXX" || ops_str == "YYXY" || ops_str == "YYYX") {
                  multcoef *= -1.0;
              }
          } else {
              multcoef = 0.125 * coef.real(); 
              std::string ops_str(ops.begin(), ops.end());
              if (ops_str != "XXYY" && ops_str != "YYXX") {
                  multcoef *= -1.0;
              }
          }
          if(multcoef == 0.0){
            continue;
          }
        
          std::vector<std::pair<int, char>> pairs = {{p, ops[0]}, {q, ops[1]}, {r, ops[2]}, {s, ops[3]}};
          std::sort(pairs.begin(), pairs.end());

          int a = pairs[0].first, b = pairs[1].first, c = pairs[2].first, d = pairs[3].first;
          auto operator_a = pairs[0].second;
          auto operator_b = pairs[1].second;
          auto operator_c = pairs[2].second;
          auto operator_d = pairs[3].second;

          auto operator_term = [&](int qubit, char op) {
              switch (op) {
                  case 'X':
                      return cudaq::spin::x(qubit);
                  case 'Y':
                      return cudaq::spin::y(qubit);
                  case 'Z':
                      return cudaq::spin::z(qubit);
                  default:
                      throw std::runtime_error("Invalid operator in two_body jordan wigner function.");
              }
          };

          cudaq::spin_op parity_a = 1.;
          for (std::size_t i = a + 1; i < b; i++)
            parity_a *= cudaq::spin::z(i);

          cudaq::spin_op parity_c = 1.;
          for (std::size_t j = c + 1; j < d; j++)
            parity_c *= cudaq::spin::z(j);
                      
          auto term = operator_term(a, operator_a);
          term *= parity_a;
          term *= operator_term(b, operator_b);
          term *= operator_term(c, operator_c);
          term *= parity_c;
          term *= operator_term(d, operator_d);

          spin_hamiltonian += multcoef*term;
      }
      return spin_hamiltonian;
    }
    throw std::runtime_error("Invalid condition in two_body jordan wigner function.");
  
}

cudaq::spin_op jordan_wigner::generate(const double constant,
                                       const tensor<> &hpq,
                                       const tensor<> &hpqrs,
                                       const heterogeneous_map &options) {
  assert(hpq.rank() == 2 && "hpq must be a rank-2 tensor");
  assert(hpqrs.rank() == 4 && "hpqrs must be a rank-4 tensor");

  auto spin_hamiltonian = constant * cudaq::spin_op();
  std::size_t nqubit = hpq.shape()[0];
  double tolerance = options.get<double>(std::vector<std::string>{"tolerance", "tol"}, 1e-15);

  for (auto p : cudaq::range(nqubit)) {
    auto coef = hpq.at({p, p});

    if (std::fabs(coef) > tolerance){
      auto temp_rslt = one_body(p, p, coef);
      std::string strrs = temp_rslt.to_string();
      spin_hamiltonian += temp_rslt;
      std::string strrsham = spin_hamiltonian.to_string();
    }
  }

  std::vector<std::vector<std::size_t>> next;
  for (auto &&combo : iter::combinations(cudaq::range(nqubit), 2)) {
      auto p = combo[0];
      auto q = combo[1];
      next.push_back({p, q});
      auto coef = 0.5 * (hpq.at({p, q}) + std::conj(hpq.at({q, p})));
      if (std::fabs(coef) > tolerance)
        spin_hamiltonian += one_body(p, q, coef);

      coef =  hpqrs.at({p, q, p, q}) 
            + hpqrs.at({q, p, q, p})
            - hpqrs.at({p, q, q, p}) 
            - hpqrs.at({q, p, p, q});
      if (std::fabs(coef) > tolerance)
        spin_hamiltonian += two_body(p, q, p, q, coef);
    }
  

  for (auto combo : iter::combinations(next, 2)) {
    auto p = combo[0][0];
    auto q = combo[0][1];
    auto r = combo[1][0];
    auto s = combo[1][1];
    auto coef = 0.5 * (hpqrs.at({p, q, r, s}) 
                      + std::conj(hpqrs.at({s, r, q, p})) 
                      - hpqrs.at({p, q, s, r}) 
                      - std::conj(hpqrs.at({r, s, q, p})) 
                      - hpqrs.at({q, p, r, s}) 
                      - std::conj(hpqrs.at({s, r, p, q})) 
                      + hpqrs.at({q, p, s, r}) 
                      + std::conj(hpqrs.at({r, s, p, q}))
                      );

    if (std::fabs(coef) > tolerance)
      spin_hamiltonian += two_body(p, q, r, s, coef);
  }
  return spin_hamiltonian;
}
} // namespace cudaq::solvers