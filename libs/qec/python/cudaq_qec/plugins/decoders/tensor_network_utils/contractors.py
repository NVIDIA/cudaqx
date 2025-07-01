# ============================================================================ #
# Copyright (c) 2025 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from typing import Any, Callable, Dict, Optional
import opt_einsum as oe
import torch
from cuquantum import tensornet as cutn
from quimb.tensor import TensorNetwork


def einsum_torch(
    subscripts: str,
    tensors: list[torch.Tensor],
    optimize: str = "auto",
    slicing: tuple = tuple()) -> torch.Tensor:
    """
    Perform einsum contraction using torch.

    Args:
        subscripts (str): The einsum subscripts.
        tensors (list[torch.Tensor]): list of torch tensors to contract.
        optimize (str, optional): Optimization strategy. Defaults to "auto".
        slicing (tuple, optional): Slicing specification. Defaults to empty tuple.

    Returns:
        torch.Tensor: The contracted tensor.
    """
    return torch.einsum(subscripts, *tensors)


torch_compiled_contractor = torch.compile(einsum_torch)


def contractor(subscripts: str,
               tensors: list[Any],
               optimize: str = "auto",
               slicing: tuple = tuple()) -> Any:
    """
    Perform einsum contraction using opt_einsum.

    Args:
        subscripts (str): The einsum subscripts.
        tensors (list[Any]): list of tensors to contract.
        optimize (str, optional): Optimization strategy. Defaults to "auto".
        slicing (tuple, optional): Slicing specification. Defaults to empty tuple.

    Returns:
        Any: The contracted tensor.
    """
    return oe.contract(subscripts, *tensors, optimize=optimize)


torch_compiled_opt_einsum = torch.compile(contractor)


def cutn_contractor(
    subscripts: str,
    tensors: list[Any],
    optimize: Optional[Any] = None,
    slicing: tuple = tuple()) -> Any:
    """
    Perform contraction using cuQuantum's tensornet contractor.

    Args:
        subscripts (str): The einsum subscripts.
        tensors (list[Any]): list of tensors to contract.
        optimize (Optional[Any], optional): cuQuantum optimizer options or path. Defaults to None.
        slicing (tuple, optional): Slicing specification. Defaults to empty tuple.

    Returns:
        Any: The contracted tensor.
    """
    return cutn.contract(
        subscripts,
        *tensors,
        optimize=cutn.OptimizerOptions(path=optimize, slicing=slicing),
    )


# this is used to determine the backend of the tensor arrays
BACKENDS: list[str] = ["numpy", "torch"]

# this is used to determine the contractor to use
CONTRACTORS: Dict[str, Callable] = {
    "numpy": contractor,
    "torch_compiled_opt_einsum": torch_compiled_opt_einsum,
    "torch": contractor,
    "cutensornet": cutn_contractor,
}


def optimize_path(optimize: Any, output_inds: tuple[str, ...],
                  tn: TensorNetwork) -> tuple[Any, Any]:
    """
    Optimize the contraction path for a tensor network.

    Args:
        optimize (Any): Optimization strategy or cuQuantum optimizer options.
        output_inds (tuple[str, ...]): Output indices for the contraction.
        tn (TensorNetwork): The tensor network.

    Returns:
        tuple[Any, Any]: The contraction path and optimizer info.
    """
    if isinstance(optimize, cutn.OptimizerOptions) or optimize is None:
        path, info = cutn.contract_path(
            tn.get_equation(output_inds=output_inds),
            *tn.arrays,
            optimize=optimize,
        )
        return path, info

    # If optimize is a custom path optimizer
    ci = tn.contraction_info(output_inds=output_inds, optimize=optimize)
    return ci.path, ci
