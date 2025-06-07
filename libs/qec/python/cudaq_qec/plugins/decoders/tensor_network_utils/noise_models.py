# ============================================================================ #
# Copyright (c) 2024 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from typing import Optional
import numpy as np
from quimb import oset
from quimb.tensor import Tensor, TensorNetwork


def factorized_noise_model(
    error_indices: list[str],
    error_probabilities: list[float],
    tensors_tags: Optional[list[str]] = None,
) -> TensorNetwork:
    """
    Create a factorized noise model from a list of error indices and probabilities.

    Parameters:
    ----------
        error_indices: list of error indices
        error_probabilities: list of error probabilities
        tensors_tags: list of tags for the tensors

    Returns:
    -------
        TensorNetwork: a tensor network representing the noise model
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
    error_index_pairs: list[tuple[str, str]],
    error_probabilities: list[np.ndarray],
    tensors_tags: Optional[list[str]] = None,
) -> TensorNetwork:
    """
    Create a noise model from a list of error index pairs and probabilities.

    Parameters:
    ----------
        error_index_pairs: list of error index pairs
        error_probabilities: list of error probabilities
        tensors_tags: list of tags for the tensors

    Returns:
    -------
        TensorNetwork: a tensor network representing the noise model
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
