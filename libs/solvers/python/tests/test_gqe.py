# ============================================================================ #
# Copyright (c) 2025 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import numpy as np
import pytest
import torch
from cudaq import spin
import cudaq
from cudaq_solvers.gqe_algorithm.gqe import DefaultScheduler, CosineScheduler, get_default_config
from cudaq_solvers.gqe_algorithm import transformer as transformer_mod
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
    assert scheduler.get(0) == 1.0
    assert scheduler.get(1) == 1.1
    assert scheduler.get(10) == 2.0


def test_cosine_scheduler():
    """Test the CosineScheduler temperature scheduling"""
    scheduler = CosineScheduler(minimum=1.0, maximum=5.0, frequency=10)
    # Test at key points in the cosine cycle
    assert np.isclose(scheduler.get(0), 1.0,
                      atol=1e-6)  # min at start (cos(0)=1)
    assert np.isclose(scheduler.get(5), 5.0,
                      atol=1e-6)  # max at half cycle (cos(π)=-1)
    assert np.isclose(scheduler.get(10), 1.0,
                      atol=1e-6)  # min at full cycle (cos(2π)=1)


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

    energy, indices = solvers.gqe(cost, pool, config=cfg, loss="gflow")
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


def test_get_device_returns_usable_device():
    """get_device() must return a device that PyTorch can actually use,
    and subsequent calls must return the cached result."""
    transformer_mod._device_cache = None

    device = transformer_mod.get_device()
    assert device in ('cuda', 'mps', 'cpu')

    t = torch.tensor([1.0], device=device)
    assert (t + t).item() == 2.0

    assert transformer_mod.get_device() is device

    transformer_mod._device_cache = None


def test_get_device_recovers_from_corrupted_context(monkeypatch):
    """Simulates CUDA-Q corrupting the CUDA context: the first probe
    fails with cudaErrorInvalidResourceHandle, the recovery logic resets
    PyTorch's internal CUDA state, and the second probe succeeds."""
    transformer_mod._device_cache = None

    probe_attempts = 0

    class _FakeCudaTensor:
        """Minimal stand-in that satisfies the (t + t).item() probe."""

        def __add__(self, _other):
            return self

        def item(self):
            return 2.0

    def _flaky_tensor(*args, **kwargs):
        nonlocal probe_attempts
        if kwargs.get('device') == 'cuda':
            probe_attempts += 1
            if probe_attempts == 1:
                raise RuntimeError("simulated cudaErrorInvalidResourceHandle")
        return _FakeCudaTensor()

    monkeypatch.setattr(torch, 'tensor', _flaky_tensor)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    monkeypatch.setattr(torch.cuda, 'init', lambda: None)

    device = transformer_mod.get_device()

    assert device == 'cuda'
    assert probe_attempts == 2, "expected exactly one failed and one successful probe"
    transformer_mod._device_cache = None


def test_get_device_raises_on_persistent_cuda_failure(monkeypatch):
    """When CUDA is detected but broken beyond recovery, get_device()
    must raise RuntimeError with actionable guidance -- not silently
    fall back to CPU."""
    transformer_mod._device_cache = None

    probe_attempts = 0

    def _always_fail(*args, **kwargs):
        nonlocal probe_attempts
        probe_attempts += 1
        raise RuntimeError("simulated persistent CUDA failure")

    monkeypatch.setattr(torch, 'tensor', _always_fail)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    monkeypatch.setattr(torch.cuda, 'init', lambda: None)

    with pytest.raises(RuntimeError,
                       match="CUDA GPU detected but PyTorch "
                       "cannot use it") as exc_info:
        transformer_mod.get_device()

    assert probe_attempts == 2, "expected two probe attempts before giving up"
    assert "torch.cuda.init()" in str(exc_info.value), \
        "error message must include the workaround"
    transformer_mod._device_cache = None


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
