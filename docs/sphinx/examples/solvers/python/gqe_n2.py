# ============================================================================ #
# Copyright (c) 2025 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
# [Begin Documentation]

# GQE is an optional component of the CUDA-QX Solvers Library. To install its
# dependencies, run:
# pip install cudaq-solvers[gqe]
#
# This example demonstrates GQE on the N2 molecule using the utility function
# get_gqe_pauli_pool() to generate an operator pool based on UCCSD Pauli terms.
# The pool is automatically generated from UCCSD operators and scaled by
# different parameter values, making it suitable for variational quantum algorithms.
#
# Run this script with
# python3 gqe_n2.py
#
# In order to leverage CUDA-Q MQPU and distribute the work across
# multiple QPUs (thereby observing a speed-up), run with:
#
# mpiexec -np N and vary N to see the speedup...
# e.g. PMIX_MCA_gds=hash mpiexec -np 2 python3 gqe_n2.py --mpi

import argparse, cudaq

parser = argparse.ArgumentParser()
parser.add_argument('--mpi', action='store_true')
args = parser.parse_args()

if args.mpi:
    try:
        cudaq.set_target('nvidia', option='mqpu')
        cudaq.mpi.initialize()
    except RuntimeError:
        print(
            'Warning: NVIDIA GPUs or MPI not available, unable to use CUDA-Q MQPU. Skipping...'
        )
        exit(0)
else:
    try:
        cudaq.set_target('nvidia', option='fp64')
    except RuntimeError:
        cudaq.set_target('qpp-cpu')

import cudaq_solvers as solvers

from lightning.pytorch.loggers import CSVLogger
from cudaq_solvers.gqe_algorithm.gqe import get_default_config
from cudaq_solvers.gqe_algorithm.utils import get_gqe_pauli_pool

# Set deterministic seed and environment variables for deterministic behavior
# Disable this section for non-deterministic behavior
import os, torch

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
torch.manual_seed(3047)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Create the molecular hamiltonian
geometry = [('N', (0., 0., 0.)), ('N', (0., 0., 1.1))]
molecule = solvers.create_molecule(geometry,
                                   'sto-3g',
                                   0,
                                   0,
                                   nele_cas=6,
                                   norb_cas=6,
                                   casci=True)

spin_ham = molecule.hamiltonian
n_qubits = molecule.n_orbitals * 2
n_electrons = molecule.n_electrons

# Generate the operator pool using utility function
params = [
    0.003125, -0.003125, 0.00625, -0.00625, 0.0125, -0.0125, 0.025, -0.025,
    0.05, -0.05, 0.1, -0.1
]

op_pool = get_gqe_pauli_pool(n_qubits, n_electrons, params)


def term_coefficients(op: cudaq.SpinOperator) -> list[complex]:
    return [term.evaluate_coefficient() for term in op]


def term_words(op: cudaq.SpinOperator) -> list[cudaq.pauli_word]:
    return [term.get_pauli_word(n_qubits) for term in op]


# Kernel that applies the selected operators
@cudaq.kernel
def kernel(n_qubits: int, n_electrons: int, coeffs: list[float],
           words: list[cudaq.pauli_word]):
    q = cudaq.qvector(n_qubits)

    for i in range(n_electrons):
        x(q[i])

    for i in range(len(coeffs)):
        exp_pauli(coeffs[i], q, words[i])


def cost(sampled_ops: list[cudaq.SpinOperator], **kwargs):

    full_coeffs = []
    full_words = []

    for op in sampled_ops:
        full_coeffs += [c.real for c in term_coefficients(op)]
        full_words += term_words(op)

    if args.mpi:
        handle = cudaq.observe_async(kernel,
                                     spin_ham,
                                     n_qubits,
                                     n_electrons,
                                     full_coeffs,
                                     full_words,
                                     qpu_id=kwargs['qpu_id'])
        return handle, lambda res: res.get().expectation()
    else:
        return cudaq.observe(kernel, spin_ham, n_qubits, n_electrons,
                             full_coeffs, full_words).expectation()


# Configure GQE
cfg = get_default_config()
cfg.use_lightning_logging = True
logger = CSVLogger(save_dir="gqe_n2_logs", name="gqe")
cfg.max_iters = 50  # For full training, set to more than 1000
cfg.ngates = 60
cfg.num_samples = 50
cfg.buffer_size = 50
cfg.warmup_size = 50
cfg.batch_size = 50

cfg.scheduler = 'variance'
cfg.lightning_logger = logger
cfg.save_trajectory = False
cfg.verbose = True
cfg.benchmark_energy = molecule.energies

# Run GQE
minE, best_ops = solvers.gqe(cost, op_pool, config=cfg)

# Only print results from rank 0 when using MPI
if not args.mpi or cudaq.mpi.rank() == 0:
    print(f'Ground Energy = {minE} (Ha)')
    print(f'Error = {minE - molecule.energies["R-CASCI"]} (Ha)')
    print('Ansatz Ops')
    for idx in best_ops:
        # Get the first (and only) term since these are simple operators
        term = next(iter(op_pool[idx]))
        print(term.evaluate_coefficient().real, term.get_pauli_word(n_qubits))

if args.mpi:
    cudaq.mpi.finalize()
