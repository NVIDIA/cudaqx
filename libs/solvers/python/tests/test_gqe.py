# ============================================================================ #
# Copyright (c) 2025 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import numpy as np
import pytest
from cudaq import spin
import cudaq
from cudaq_solvers.gqe_algorithm.gqe import get_default_config
from cudaq_solvers.gqe_algorithm.scheduler import DefaultScheduler, CosineScheduler, VarBasedScheduler
import cudaq_solvers as solvers

qubit_count = 2
# Define a simple Hamiltonian: Z₀ + Z₁
ham = spin.z(0) + spin.z(1)


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


# Global cost function for GQE
def cost(sampled_ops: list[cudaq.SpinOperator], **kwargs):
    full_coeffs = []
    full_words = []
    for op in sampled_ops:
        full_coeffs += [c.real for c in term_coefficients(op)]
        full_words += term_words(op)

    return cudaq.observe(kernel, ham, qubit_count, full_coeffs,
                         full_words).expectation()


def test_default_scheduler():
    """Test the DefaultScheduler temperature scheduling"""
    scheduler = DefaultScheduler(start=1.0, delta=0.1)
    assert scheduler.get_inverse_temperature() == 1.0
    scheduler.update()
    assert np.isclose(scheduler.get_inverse_temperature(), 1.1, atol=1e-6)
    for _ in range(9):
        scheduler.update()
    assert np.isclose(scheduler.get_inverse_temperature(), 2.0, atol=1e-6)


def test_cosine_scheduler():
    """Test the CosineScheduler temperature scheduling"""
    scheduler = CosineScheduler(minimum=1.0, maximum=5.0, frequency=10)
    # Initial temperature should be at midpoint
    assert np.isclose(scheduler.get_inverse_temperature(), 3.0, atol=1e-6)
    
    # After 5 updates, should be at maximum (cos(π)=-1)
    for _ in range(5):
        scheduler.update()
    assert np.isclose(scheduler.get_inverse_temperature(), 5.0, atol=1e-6)
    
    # After 10 updates total, should be back near starting point (cos(2π)=1)
    for _ in range(5):
        scheduler.update()
    assert np.isclose(scheduler.get_inverse_temperature(), 1.0, atol=1e-6)


def test_variance_scheduler():
    """Test the VarBasedScheduler temperature scheduling"""
    import torch
    scheduler = VarBasedScheduler(initial=2.0, delta=0.1, target_var=0.1)
    
    # Test initial temperature
    assert scheduler.get_inverse_temperature() == 2.0
    
    # Simulate high variance scenario (should increase temperature)
    high_var_energies = torch.tensor([1.0, 5.0, 2.0, 6.0, 3.0])  # var ≈ 3.5
    initial_temp = scheduler.current_temperature
    scheduler.update(energies=high_var_energies)
    temp_after_high_var = scheduler.current_temperature
    assert temp_after_high_var > initial_temp  # Temperature should increase
    
    # Simulate low variance scenario (should decrease temperature)
    scheduler2 = VarBasedScheduler(initial=2.0, delta=0.1, target_var=0.5)
    low_var_energies = torch.tensor([1.0, 1.1, 1.05, 0.95, 1.02])  # var ≈ 0.003
    initial_temp2 = scheduler2.current_temperature
    scheduler2.update(energies=low_var_energies)
    temp_after_low_var = scheduler2.current_temperature
    assert temp_after_low_var < initial_temp2  # Temperature should decrease
    
    # Test minimum temperature bound
    scheduler3 = VarBasedScheduler(initial=2.0, delta=0.1, target_var=0.1)
    for _ in range(100):  # Many decreases
        scheduler3.update(energies=low_var_energies)
    final_temp = scheduler3.current_temperature
    assert final_temp >= 0.01  # Should not go below min_temp (0.01)


def test_solvers_gqe_basic():
    """Test basic GQE with config"""
    print("Setting up config...")
    cfg = get_default_config()
    cfg.num_samples = 5
    cfg.max_iters = 25
    cfg.ngates = 4
    cfg.seed = 3047
    cfg.lr = 1e-6
    cfg.energy_offset = 0.0
    cfg.grad_norm_clip = 1.0
    cfg.temperature = 2.0
    cfg.del_temperature = 0.1
    cfg.resid_pdrop = 0.0
    cfg.embd_pdrop = 0.0
    cfg.attn_pdrop = 0.0
    cfg.small = False
    cfg.cache = True
    cfg.save_dir = "./output/"

    energy, indices = solvers.gqe(cost, pool, config=cfg)
    assert energy < 0.0
    assert energy > -2.0  # Physical bound for simple Z₀ + Z₁ Hamiltonian


def test_solvers_gqe_small_transformer():
    """Test GQE with small transformer config"""
    cfg = get_default_config()
    cfg.num_samples = 5
    cfg.max_iters = 50
    cfg.ngates = 10
    cfg.seed = 3047
    cfg.lr = 1e-6
    cfg.energy_offset = 0.0
    cfg.grad_norm_clip = 1.0
    cfg.temperature = 2.0
    cfg.del_temperature = 0.1
    cfg.resid_pdrop = 0.0
    cfg.embd_pdrop = 0.0
    cfg.attn_pdrop = 0.0
    cfg.small = True
    cfg.cache = False
    cfg.save_dir = "/dev/null"

    energy, indices = solvers.gqe(cost, pool, config=cfg)
    assert energy < 0.0
    assert energy > -2.0


def test_solvers_gqe_with_gflow_loss():
    """Test GQE with GFlow loss function"""
    cfg = get_default_config()
    cfg.num_samples = 5
    cfg.max_iters = 50
    cfg.ngates = 10
    cfg.seed = 3047
    cfg.lr = 1e-6
    cfg.energy_offset = 0.0
    cfg.grad_norm_clip = 1.0
    cfg.temperature = 2.0
    cfg.del_temperature = 0.1
    cfg.resid_pdrop = 0.0
    cfg.embd_pdrop = 0.0
    cfg.attn_pdrop = 0.0
    cfg.small = False
    cfg.cache = False
    cfg.save_dir = "/dev/null"
    cfg.loss = "gflow"

    energy, indices = solvers.gqe(cost, pool, config=cfg)
    assert energy < 0.0
    assert energy > -2.0


def test_solvers_gqe_with_exp_loss():
    """Test GQE with Exponential loss function"""
    cfg = get_default_config()
    cfg.num_samples = 5
    cfg.max_iters = 50
    cfg.ngates = 10
    cfg.seed = 3047
    cfg.lr = 1e-6
    cfg.energy_offset = 0.0
    cfg.grad_norm_clip = 1.0
    cfg.temperature = 2.0
    cfg.del_temperature = 0.1
    cfg.resid_pdrop = 0.0
    cfg.embd_pdrop = 0.0
    cfg.attn_pdrop = 0.0
    cfg.small = False
    cfg.cache = False
    cfg.save_dir = "/dev/null"
    cfg.loss = "exp"

    energy, indices = solvers.gqe(cost, pool, config=cfg)
    assert energy < 0.0
    assert energy > -2.0


def test_solvers_gqe_with_variance_scheduler():
    """Test GQE with variance-based temperature scheduler"""
    cfg = get_default_config()
    cfg.num_samples = 5
    cfg.max_iters = 50
    cfg.ngates = 10
    cfg.seed = 3047
    cfg.lr = 1e-6
    cfg.energy_offset = 0.0
    cfg.grad_norm_clip = 1.0
    cfg.temperature = 2.0
    cfg.del_temperature = 0.1
    cfg.scheduler = 'variance'
    cfg.target_variance = 0.1
    cfg.resid_pdrop = 0.0
    cfg.embd_pdrop = 0.0
    cfg.attn_pdrop = 0.0
    cfg.small = False
    cfg.cache = False
    cfg.save_dir = "/dev/null"

    energy, indices = solvers.gqe(cost, pool, config=cfg)
    assert energy < 0.0
    assert energy > -2.0


def test_solvers_gqe_with_cosine_scheduler():
    """Test GQE with cosine temperature scheduler"""
    cfg = get_default_config()
    cfg.num_samples = 5
    cfg.max_iters = 50
    cfg.ngates = 10
    cfg.seed = 3047
    cfg.lr = 1e-6
    cfg.energy_offset = 0.0
    cfg.grad_norm_clip = 1.0
    cfg.temperature = 2.0
    cfg.scheduler = 'cosine'
    cfg.temperature_min = 1.5
    cfg.temperature_max = 3.0
    cfg.scheduler_frequency = 20
    cfg.resid_pdrop = 0.0
    cfg.embd_pdrop = 0.0
    cfg.attn_pdrop = 0.0
    cfg.small = False
    cfg.cache = False
    cfg.save_dir = "/dev/null"

    energy, indices = solvers.gqe(cost, pool, config=cfg)
    assert energy < 0.0
    assert energy > -2.0


def test_solvers_gqe_larger_molecule():
    """Test GQE with a larger number of gates"""
    cfg = get_default_config()
    cfg.num_samples = 5
    cfg.max_iters = 100
    cfg.ngates = 30
    cfg.seed = 3047
    cfg.lr = 1e-6
    cfg.energy_offset = 0.0
    cfg.grad_norm_clip = 1.0
    cfg.temperature = 2.0
    cfg.del_temperature = 0.1
    cfg.resid_pdrop = 0.0
    cfg.embd_pdrop = 0.0
    cfg.attn_pdrop = 0.0
    cfg.small = False
    cfg.cache = False
    cfg.save_dir = "/dev/null"

    energy, indices = solvers.gqe(cost, pool, config=cfg)
    assert energy < 0.0
    assert energy > -2.0


def test_invalid_inputs():
    """Test error handling for invalid inputs"""
    cfg = get_default_config()

    # Test invalid number of samples
    cfg.num_samples = 0
    with pytest.raises(ValueError):
        solvers.gqe(cost, pool, config=cfg)

    # Test invalid learning rate
    cfg.num_samples = 5
    cfg.lr = -1.0
    with pytest.raises(ValueError):
        solvers.gqe(cost, pool, config=cfg)

    # Test invalid temperature
    cfg.lr = 1e-6
    cfg.temperature = -1.0
    with pytest.raises(ValueError):
        solvers.gqe(cost, pool, config=cfg)
