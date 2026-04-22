# ============================================================================ #
# Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""PyTorch CUDA capability checks and device selection for GQE."""

import torch

_pytorch_cuda_support_cache = None


def _pytorch_cuda_support():
    """Return (True, \"\") if PyTorch can run CUDA kernels on this device, else (False, reason)."""
    global _pytorch_cuda_support_cache
    if _pytorch_cuda_support_cache is not None:
        return _pytorch_cuda_support_cache
    if not torch.cuda.is_available():
        _pytorch_cuda_support_cache = (False, "CUDA is not available")
        return _pytorch_cuda_support_cache
    try:
        arch_list = (torch.cuda.get_arch_list() if hasattr(
            torch.cuda, "get_arch_list") else [])
        if arch_list:
            cap = torch.cuda.get_device_capability()
            arch = f"sm_{cap[0]}{cap[1]}"
            if arch not in arch_list:
                name = torch.cuda.get_device_name()
                _pytorch_cuda_support_cache = (
                    False,
                    (f"PyTorch CUDA wheel does not include kernels for "
                     f"{name} ({arch})"),
                )
                return _pytorch_cuda_support_cache
        torch.zeros(1, device="cuda")
        _pytorch_cuda_support_cache = (True, "")
        return _pytorch_cuda_support_cache
    except Exception as e:
        _pytorch_cuda_support_cache = (False, str(e))
        return _pytorch_cuda_support_cache


def pytorch_cuda_execution_available():
    """True if this PyTorch build can run CUDA kernels on the current GPU."""
    ok, _ = _pytorch_cuda_support()
    return ok


def pytorch_cuda_kernel_skip_reason():
    """Human-readable reason when :func:`pytorch_cuda_execution_available` is false; else \"\"."""
    ok, reason = _pytorch_cuda_support()
    return "" if ok else reason
