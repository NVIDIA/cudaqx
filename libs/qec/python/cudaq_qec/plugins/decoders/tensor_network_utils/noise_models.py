# ============================================================================ #
# Copyright (c) 2024 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
from typing import Any, List, Optional, Tuple, Union
import numpy as np
from quimb import oset
from quimb.tensor import Tensor, TensorNetwork


def factorized_noise_model(
        error_indices: List[str],
        error_probabilities: Union[List[float], np.ndarray],
        tensors_tags: Optional[List[str]] = None) -> TensorNetwork:
    """
    Construct a factorized (product state) noise model as a tensor network.

    Args:
        error_indices (List[str]): List of error index names.
        error_probabilities (Union[List[float], np.ndarray]): List or array of error probabilities for each error index.
        tensors_tags (Optional[List[str]], optional): List of tags for each tensor. If None, default tags are used.

    Returns:
        TensorNetwork: The tensor network representing the factorized noise model.
    """
    tensors = []

    if tensors_tags is None:
        tensors_tags = ["NOISE"] * len(error_indices)

    for ei, eprob, etag in zip(error_indices, error_probabilities,
                               tensors_tags):
        tensors.append(
            Tensor(
                data=np.array([1.0 - eprob, eprob]),
                inds=(ei,),
                tags=oset([etag]),
            ))
    return TensorNetwork(tensors)


def error_pairs_noise_model(
        error_index_pairs: List[Tuple[str, str]],
        error_probabilities: List[np.ndarray],
        tensors_tags: Optional[List[str]] = None) -> TensorNetwork:
    """
    Construct a noise model as a tensor network for correlated error pairs.

    Args:
        error_index_pairs (List[Tuple[str, str]]): List of pairs of error index names.
        error_probabilities (List[np.ndarray]): List of 2x2 probability matrices for each error pair.
        tensors_tags (Optional[List[str]], optional): List of tags for each tensor. If None, default tags are used.

    Returns:
        TensorNetwork: The tensor network representing the error pairs noise model.
    """
    tensors = []

    if tensors_tags is None:
        tensors_tags = ["NOISE"] * len(error_index_pairs)

    for ei, etensors, etag in zip(error_index_pairs, error_probabilities,
                                  tensors_tags):
        tensors.append(Tensor(
            data=etensors,
            inds=ei,
            tags=oset([etag]),
        ))
    return TensorNetwork(tensors)
