/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/solvers/hamiltonian_simulation/trotter.h"

#include "cudaq.h"
#include <cmath>
#include <complex>
#include <gtest/gtest.h>
#include <stdexcept>
#include <vector>

using namespace cudaq::spin;

namespace {
void expect_close(std::complex<double> actual, std::complex<double> expected,
                  double tol = 1e-6) {
  EXPECT_NEAR(actual.real(), expected.real(), tol);
  EXPECT_NEAR(actual.imag(), expected.imag(), tol);
}

std::vector<int> basis_bits(std::size_t basis, std::size_t num_qubits) {
  std::vector<int> bits(num_qubits, 0);
  for (std::size_t qubit = 0; qubit < num_qubits; ++qubit)
    bits[num_qubits - 1 - qubit] = (basis >> qubit) & 1U;
  return bits;
}

void normalize(std::vector<std::complex<double>> &state) {
  double norm = 0.0;
  for (const auto &amplitude : state)
    norm += std::norm(amplitude);

  const auto scale = 1.0 / std::sqrt(norm);
  for (auto &amplitude : state)
    amplitude *= scale;
}

std::size_t infer_num_qubits_from_state_size(std::size_t state_size) {
  if (state_size == 0)
    throw std::invalid_argument("input ket cannot be empty.");

  if ((state_size & (state_size - 1)) != 0)
    throw std::invalid_argument("input ket length must be a power of two.");

  std::size_t num_qubits = 0;
  while ((std::size_t{1} << num_qubits) < state_size)
    ++num_qubits;
  return num_qubits;
}

void validate_trotter_inputs(const cudaq::solvers::trotter_terms &terms,
                             std::size_t steps, int order,
                             std::size_t state_size) {
  if (steps == 0)
    throw std::invalid_argument("steps must be greater than zero.");

  if (order != cudaq::solvers::first_order_trotter &&
      order != cudaq::solvers::second_order_trotter &&
      order != cudaq::solvers::fourth_order_trotter)
    throw std::invalid_argument("order must be one of {1, 2, 4}.");

  if (terms.coefficients.size() != terms.words.size())
    throw std::invalid_argument(
        "coefficients and Pauli words must have equal length.");

  const auto state_num_qubits = infer_num_qubits_from_state_size(state_size);
  const auto num_qubits =
      terms.num_qubits == 0 ? state_num_qubits : terms.num_qubits;
  if (state_size != (std::size_t{1} << num_qubits))
    throw std::invalid_argument(
        "input ket length does not match Hamiltonian qubits.");
}

void apply_pauli_rotation(std::vector<std::complex<double>> &state,
                          const cudaq::pauli_word &word, double angle,
                          std::size_t num_qubits) {
  const auto word_string = word.str();
  if (word_string.size() != num_qubits)
    throw std::invalid_argument(
        "Pauli word length does not match qubit count.");

  const auto c = std::cos(angle);
  const auto s = std::sin(angle);
  const std::complex<double> minus_i_s{0.0, -s};
  std::vector<std::complex<double>> rotated(state.size(), {0.0, 0.0});

  for (std::size_t basis = 0; basis < state.size(); ++basis) {
    std::size_t target = basis;
    std::complex<double> phase{1.0, 0.0};

    for (std::size_t qubit = 0; qubit < num_qubits; ++qubit) {
      const bool bit = ((basis >> qubit) & 1U) != 0;
      switch (word_string[num_qubits - 1 - qubit]) {
      case 'I':
        break;
      case 'X':
        target ^= (std::size_t{1} << qubit);
        break;
      case 'Y':
        target ^= (std::size_t{1} << qubit);
        phase *= bit ? std::complex<double>{0.0, -1.0}
                     : std::complex<double>{0.0, 1.0};
        break;
      case 'Z':
        phase *= bit ? -1.0 : 1.0;
        break;
      default:
        throw std::invalid_argument("unsupported Pauli word.");
      }
    }

    rotated[basis] += c * state[basis];
    rotated[target] += minus_i_s * phase * state[basis];
  }

  state = std::move(rotated);
}

void apply_first_order_step(std::vector<std::complex<double>> &state,
                            const cudaq::solvers::trotter_terms &terms,
                            double tau) {
  for (std::size_t i = 0; i < terms.words.size(); ++i)
    apply_pauli_rotation(state, terms.words[i], tau * terms.coefficients[i],
                         terms.num_qubits);
}

void apply_second_order_step(std::vector<std::complex<double>> &state,
                             const cudaq::solvers::trotter_terms &terms,
                             double tau) {
  for (std::size_t i = 0; i < terms.words.size(); ++i)
    apply_pauli_rotation(state, terms.words[i],
                         0.5 * tau * terms.coefficients[i], terms.num_qubits);

  for (std::size_t i = terms.words.size(); i > 0; --i)
    apply_pauli_rotation(state, terms.words[i - 1],
                         0.5 * tau * terms.coefficients[i - 1],
                         terms.num_qubits);
}

void apply_ordered_step(std::vector<std::complex<double>> &state,
                        const cudaq::solvers::trotter_terms &terms, double tau,
                        int order) {
  if (order == cudaq::solvers::first_order_trotter) {
    apply_first_order_step(state, terms, tau);
    return;
  }

  if (order == cudaq::solvers::second_order_trotter) {
    apply_second_order_step(state, terms, tau);
    return;
  }

  apply_second_order_step(state, terms, cudaq::solvers::forest_ruth_w1 * tau);
  apply_second_order_step(state, terms, cudaq::solvers::forest_ruth_w0 * tau);
  apply_second_order_step(state, terms, cudaq::solvers::forest_ruth_w1 * tau);
}

std::vector<std::complex<double>>
simulate_trotter_statevector(cudaq::solvers::trotter_terms terms, double time,
                             std::size_t steps, int order,
                             const std::vector<std::complex<double>> &ket) {
  validate_trotter_inputs(terms, steps, order, ket.size());

  const auto state_num_qubits = infer_num_qubits_from_state_size(ket.size());
  if (terms.num_qubits == 0)
    terms.num_qubits = state_num_qubits;

  auto state = ket;
  const auto dt = time / static_cast<double>(steps);
  for (std::size_t step = 0; step < steps; ++step)
    apply_ordered_step(state, terms, dt, order);

  if (terms.identity_coefficient != 0.0) {
    const std::complex<double> phase =
        std::exp(std::complex<double>{0.0, -terms.identity_coefficient * time});
    for (auto &amplitude : state)
      amplitude *= phase;
  }

  return state;
}

struct apply_trotter_test_kernel {
  void operator()(std::size_t num_qubits,
                  const std::vector<double> &coefficients,
                  const std::vector<cudaq::pauli_word> &words, double time,
                  std::size_t steps, int order) __qpu__ {
    cudaq::qvector q(num_qubits);
    cudaq::solvers::apply_trotter(coefficients, words, time, steps, order, q);
  }
};
} // namespace

TEST(TrotterTest, MakeTrotterTermsDropsIdentityTerms) {
  cudaq::spin_op hamiltonian =
      1.2 * x(0) + 0.4 * z(1) - 0.7 * cudaq::spin_op::from_word("II");

  auto terms = cudaq::solvers::make_trotter_terms(hamiltonian);
  EXPECT_EQ(terms.coefficients.size(), 2);
  EXPECT_EQ(terms.words.size(), 2);
  EXPECT_NEAR(terms.identity_coefficient, -0.7, 1e-12);
  EXPECT_EQ(terms.num_qubits, 2);
}

TEST(TrotterTest, MakeTrotterTermsRejectsInvalidInputs) {
  EXPECT_THROW(cudaq::solvers::make_trotter_terms(z(0), -1.0),
               std::invalid_argument);

  cudaq::spin_op complex_hamiltonian = std::complex<double>{0.0, 0.25} * x(0);
  EXPECT_THROW(cudaq::solvers::make_trotter_terms(complex_hamiltonian),
               std::invalid_argument);
}

TEST(TrotterTest, TestReferenceRejectsInvalidInputs) {
  auto terms = cudaq::solvers::make_trotter_terms(z(0));
  std::vector<std::complex<double>> ket{{1.0, 0.0}, {0.0, 0.0}};

  EXPECT_THROW(simulate_trotter_statevector(terms, 0.25, 0, 2, ket),
               std::invalid_argument);
  EXPECT_THROW(simulate_trotter_statevector(terms, 0.25, 1, 3, ket),
               std::invalid_argument);
  EXPECT_THROW(
      simulate_trotter_statevector(
          terms, 0.25, 1, 2, std::vector<std::complex<double>>{{1.0, 0.0}}),
      std::invalid_argument);

  auto bad_terms = terms;
  bad_terms.coefficients.push_back(0.5);
  EXPECT_THROW(simulate_trotter_statevector(bad_terms, 0.25, 1, 2, ket),
               std::invalid_argument);
}

TEST(TrotterTest, TestReferenceIncludesIdentityGlobalPhase) {
  auto terms =
      cudaq::solvers::make_trotter_terms(2.0 * cudaq::spin_op::from_word("II"));
  std::vector<std::complex<double>> ket{
      {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}};
  const double time = 0.3;

  auto evolved = simulate_trotter_statevector(
      terms, time, 1, cudaq::solvers::second_order_trotter, ket);
  const auto phase = std::exp(std::complex<double>{0.0, -2.0 * time});

  expect_close(evolved[0], phase);
  expect_close(evolved[1], {0.0, 0.0});
  expect_close(evolved[2], {0.0, 0.0});
  expect_close(evolved[3], {0.0, 0.0});
}

TEST(TrotterTest, ApplyTrotterKernelMatchesStatevectorReferenceForOrders) {
  cudaq::spin_op hamiltonian = 0.7 * cudaq::spin_op::from_word("XI") +
                               0.4 * cudaq::spin_op::from_word("IZ") +
                               0.31 * cudaq::spin_op::from_word("XZ") +
                               0.23 * cudaq::spin_op::from_word("YY");
  auto terms = cudaq::solvers::make_trotter_terms(hamiltonian);

  std::vector<std::complex<double>> ket{
      {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}};

  const double time = 0.8;
  const std::size_t steps = 3;
  for (int order : {cudaq::solvers::first_order_trotter,
                    cudaq::solvers::second_order_trotter,
                    cudaq::solvers::fourth_order_trotter}) {
    auto expected =
        simulate_trotter_statevector(terms, time, steps, order, ket);
    auto actual =
        cudaq::get_state(apply_trotter_test_kernel{}, terms.num_qubits,
                         terms.coefficients, terms.words, time, steps, order);

    for (std::size_t i = 0; i < expected.size(); ++i)
      expect_close(actual.amplitude(basis_bits(i, terms.num_qubits)),
                   expected[i]);
  }
}

TEST(TrotterTest, ApplyTrotterHandlesFourQubitHamiltonianWithManyTerms) {
  cudaq::spin_op hamiltonian = 0.11 * cudaq::spin_op::from_word("XIII") -
                               0.17 * cudaq::spin_op::from_word("IYII") +
                               0.23 * cudaq::spin_op::from_word("IIZI") -
                               0.29 * cudaq::spin_op::from_word("IIIX") +
                               0.31 * cudaq::spin_op::from_word("XXII") +
                               0.37 * cudaq::spin_op::from_word("IYZI") -
                               0.41 * cudaq::spin_op::from_word("ZIIX") +
                               0.43 * cudaq::spin_op::from_word("XIYZ") -
                               0.47 * cudaq::spin_op::from_word("YYXI") +
                               0.53 * cudaq::spin_op::from_word("ZXYZ");
  auto terms = cudaq::solvers::make_trotter_terms(hamiltonian);

  ASSERT_EQ(terms.num_qubits, 4);
  ASSERT_GT(terms.coefficients.size(), 8);
  ASSERT_EQ(terms.coefficients.size(), terms.words.size());

  std::vector<std::complex<double>> ket(16, {0.0, 0.0});
  ket[0] = {1.0, 0.0};

  const double time = 0.37;
  const std::size_t steps = 2;
  const int order = cudaq::solvers::second_order_trotter;

  auto expected = simulate_trotter_statevector(terms, time, steps, order, ket);
  auto actual =
      cudaq::get_state(apply_trotter_test_kernel{}, terms.num_qubits,
                       terms.coefficients, terms.words, time, steps, order);

  for (std::size_t i = 0; i < expected.size(); ++i)
    expect_close(actual.amplitude(basis_bits(i, terms.num_qubits)),
                 expected[i]);
}

TEST(TrotterTest, ApplyTrotterTreatsIdentityOnlyHamiltonianAsNoOp) {
  auto terms =
      cudaq::solvers::make_trotter_terms(1.5 * cudaq::spin_op::from_word("II"));
  std::vector<std::complex<double>> ket{
      {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}};

  auto actual = cudaq::get_state(apply_trotter_test_kernel{}, terms.num_qubits,
                                 terms.coefficients, terms.words, 0.25, 4,
                                 cudaq::solvers::second_order_trotter);

  for (std::size_t i = 0; i < ket.size(); ++i)
    expect_close(actual.amplitude(basis_bits(i, terms.num_qubits)), ket[i]);
}
