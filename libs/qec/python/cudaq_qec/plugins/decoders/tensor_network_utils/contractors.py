# ============================================================================ #
# Copyright (c) 2024 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from typing import TYPE_CHECKING, Union
import opt_einsum as oe
import torch
from cuquantum import tensornet as cutn
from quimb.tensor import TensorNetwork


def einsum_torch(subscripts, tensors, optimize="auto", slicing=tuple()):
    return torch.einsum(subscripts, *tensors)


torch_compiled_contractor = torch.compile(einsum_torch)


def contractor(subscripts, tensors, optimize="auto", slicing=tuple()):
    return oe.contract(subscripts, *tensors, optimize=optimize)


torch_compiled_opt_einsum = torch.compile(contractor)


def cutn_contractor(subscripts, tensors, optimize=None, slicing=tuple()):
    return cutn.contract(
        subscripts,
        *tensors,
        optimize=cutn.OptimizerOptions(path=optimize, slicing=slicing),
    )


# this is used to determine the backend of the tensor arrays
BACKENDS = ["numpy", "torch"]

# this is used to determine the contractor to use
CONTRACTORS = {
    "numpy": contractor,
    "torch_compiled_opt_einsum": torch_compiled_opt_einsum,
    "torch": contractor,
    "cutensornet": cutn_contractor,
}


def optimize_path(optimize, output_inds: tuple[str], tn: TensorNetwork) -> None:
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
