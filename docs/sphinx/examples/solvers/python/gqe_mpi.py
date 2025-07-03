# ============================================================================ #
# Copyright (c) 2025 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq, cudaq_solvers as solvers
from cudaq import spin

cudaq.mpi.initialize()
cudaq.set_target('nvidia', option='mqpu')

qubit_count = 2


# Generate an operator pool for the GQE
def ops_pool(n):
    pool = []
    for i in range(n):
        pool.append(cudaq.SpinOperator(spin.x(i)))
        pool.append(cudaq.SpinOperator(spin.y(i)))
        pool.append(cudaq.SpinOperator(spin.z(i)))
    for i in range(n - 1):
        pool.append(cudaq.SpinOperator(spin.z(i) *
                                       spin.z(i + 1)))  # ZZ entangling
    return pool


pool = ops_pool(qubit_count)

# Define a simple Hamiltonian: Z₀ + Z₁
ham = spin.z(0) + spin.z(1)


# Helper functions to extract coeffs and Pauli words
def term_coefficients(op: cudaq.SpinOperator) -> list[complex]:
    return [term.evaluate_coefficient() for term in op]


def term_words(op: cudaq.SpinOperator) -> list[cudaq.pauli_word]:
    return [term.get_pauli_word(qubit_count) for term in op]


# Kernel that applies the selected operators
@cudaq.kernel
def kernel(qcount: int, coeffs: list[float], words: list[cudaq.pauli_word]):
    q = cudaq.qvector(qcount)
    h(q)
    for i in range(len(coeffs)):
        exp_pauli(coeffs[i], q, words[i])


# cost function for GQE
def cost(sampled_ops: list[cudaq.SpinOperator], qpu_id: int = 0):
    print(f"[Rank {cudaq.mpi.rank()}] Cost function running on QPU ID {qpu_id}")

    full_coeffs = []
    full_words = []
    for op in sampled_ops:
        full_coeffs += [c.real for c in term_coefficients(op)]
        full_words += term_words(op)

    handle = cudaq.observe_async(kernel,
                                 ham,
                                 qubit_count,
                                 full_coeffs,
                                 full_words,
                                 qpu_id=qpu_id)
    return handle, lambda res: res.get().expectation()


# Run GQE
minE, best_ops = solvers.gqe(cost, pool, max_iters=25, ngates=4)

if cudaq.mpi.rank() == 0:
    print(f'Ground Energy = {minE}')
    print('Ansatz Ops')
    for idx in best_ops:
        # Get the first (and only) term since these are simple operators
        term = next(iter(pool[idx]))
        print(term.evaluate_coefficient().real,
              term.get_pauli_word(qubit_count))

cudaq.mpi.finalize()
