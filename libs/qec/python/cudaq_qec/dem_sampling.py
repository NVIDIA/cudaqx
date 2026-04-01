# ============================================================================ #
# Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.    #
# ============================================================================ #
"""DEM sampling via the C++ pybind11 binding (GPU with cuStabilizer + CPU fallback).

Public API:
    dem_sampling(
        check_matrix, num_shots, error_probabilities, seed=None, backend="auto"
    )

When compiled with cuStabilizer, the function delegates to the GPU sampler.
Otherwise it falls back to the CPU implementation. Both paths are handled
transparently by the C++ binding.

The check_matrix and error_probabilities arguments accept both NumPy arrays
and PyTorch tensors, enabling direct integration with AI/ML workflows.
"""

from __future__ import annotations

from typing import Optional, Tuple

__all__ = ["dem_sampling"]


def dem_sampling(
    check_matrix,
    num_shots: int,
    error_probabilities,
    seed: Optional[int] = None,
    backend: str = "auto",
) -> Tuple[object, object]:
    """Sample errors and syndromes from a Detector Error Model.

    Args:
        check_matrix: Binary matrix [num_checks x num_error_mechanisms],
            as a NumPy uint8 array or PyTorch tensor.
        num_shots: Number of independent Monte-Carlo shots.
        error_probabilities: 1-D array of length num_error_mechanisms with
            independent Bernoulli probabilities for each mechanism.
            Accepts NumPy float64 array or PyTorch tensor.
        seed: Optional RNG seed for reproducibility.
        backend: Backend selection policy:
            - "auto" (default): try GPU, fall back to CPU.
            - "cpu": force CPU implementation.
            - "gpu": force GPU implementation and raise if unavailable.

    Returns:
        (syndromes, errors) where
          syndromes: uint8 array/tensor [num_shots x num_checks]
          errors:    uint8 array/tensor [num_shots x num_error_mechanisms]

    When compiled with cuStabilizer the function uses GPU-accelerated
    sampling. Otherwise it falls back to the CPU implementation.
    Both NumPy arrays and PyTorch tensors are accepted as inputs.
    For PyTorch CUDA tensors on GPU path, outputs are PyTorch CUDA tensors;
    otherwise outputs are NumPy arrays.
    """
    from . import _pycudaqx_qec_the_suffix_matters_cudaq_qec as _qecmod

    return _qecmod.qecrt.dem_sampling(check_matrix, num_shots,
                                      error_probabilities, seed, backend)
