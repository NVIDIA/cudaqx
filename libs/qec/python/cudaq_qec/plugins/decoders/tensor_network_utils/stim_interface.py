# ============================================================================ #
# Copyright (c) 2023 - Oscar Higgott                                           #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
# This file originated from:
# https://github.com/oscarhiggott/BeliefMatching/blob/b70bf0f448563d4635a74dd1625e0c3ba34a2e47/src/beliefmatching/belief_matching.py
# Original license:
# https://github.com/oscarhiggott/BeliefMatching/blob/b70bf0f448563d4635a74dd1625e0c3ba34a2e47/LICENSE

from typing import List, FrozenSet, Dict, Optional, Tuple
from dataclasses import dataclass
from scipy.sparse import csc_matrix
import numpy as np
import stim
import functools


def iter_set_xor(set_list: List[List[int]]) -> FrozenSet[int]:
    """Computes the XOR of multiple sets efficiently using reduce."""
    return frozenset(functools.reduce(set.symmetric_difference, set_list,
                                      set()))


def dict_to_csc_matrix(elements_dict: Dict[int, FrozenSet[int]],
                       shape: Tuple[int, int]) -> csc_matrix:
    """
    Converts a dictionary representation of sparse matrix indices into a `scipy.sparse.csc_matrix`.

    Parameters
    ----------
    elements_dict : Dict[int, FrozenSet[int]]
        Dictionary mapping column indices to sets of row indices containing 1s.
    shape : Tuple[int, int]
        The dimensions of the output sparse matrix.

    Returns
    -------
    csc_matrix
        The binary sparse matrix in CSC format.
    """
    nnz = sum(len(v) for v in elements_dict.values())  # Total nonzero elements
    data = np.ones(nnz, dtype=np.uint8)
    row_ind = np.empty(nnz, dtype=np.int64)
    col_ind = np.empty(nnz, dtype=np.int64)

    i = 0
    for col, rows in elements_dict.items():
        for row in rows:
            row_ind[i] = row
            col_ind[i] = col
            i += 1

    return csc_matrix((data, (row_ind, col_ind)), shape=shape)


@dataclass
class DemMatrices:
    """Holds sparse matrices representing a Stim Detector Error Model (DEM)."""

    check_matrix: csc_matrix
    observables_matrix: csc_matrix
    edge_check_matrix: csc_matrix
    edge_observables_matrix: csc_matrix
    hyperedge_to_edge_matrix: csc_matrix
    priors: np.ndarray


def detector_error_model_to_check_matrices(
        dem: stim.DetectorErrorModel,
        allow_undecomposed_hyperedges: bool = False) -> DemMatrices:
    """
    Converts a `stim.DetectorErrorModel` into a set of sparse matrices.

    Parameters
    ----------
    dem : stim.DetectorErrorModel
        The detector error model to convert.
    allow_undecomposed_hyperedges : bool, optional
        If False, raises an error for non-decomposed hyperedges.

    Returns
    -------
    DemMatrices
        Object containing sparse matrices representing the model.
    """

    hyperedge_ids: Dict[FrozenSet[int], int] = {}
    edge_ids: Dict[FrozenSet[int], int] = {}
    hyperedge_obs_map: Dict[int, FrozenSet[int]] = {}
    edge_obs_map: Dict[int, FrozenSet[int]] = {}
    priors_dict: Dict[int, float] = {}
    hyperedge_to_edge: Dict[int, FrozenSet[int]] = {}

    def handle_error(prob: float, detectors: List[List[int]],
                     observables: List[List[int]]) -> None:
        """Processes an error event, updating the mappings."""
        hyperedge_dets = iter_set_xor(detectors)
        hyperedge_obs = iter_set_xor(observables)

        # Assign hyperedge ID if unseen
        hid = hyperedge_ids.setdefault(hyperedge_dets, len(hyperedge_ids))
        priors_dict.setdefault(hid, 0.0)
        hyperedge_obs_map[hid] = hyperedge_obs

        # Update prior probabilities using Bayesian update
        priors_dict[hid] = priors_dict[hid] * (1 - prob) + prob * (
            1 - priors_dict[hid])

        eids = []
        for e_dets, e_obs in zip(detectors, observables):
            e_dets = frozenset(e_dets)
            e_obs = frozenset(e_obs)

            if len(e_dets) > 2:
                if not allow_undecomposed_hyperedges:
                    raise ValueError(
                        "A hyperedge error mechanism was found that was not decomposed into edges. "
                        "Ensure `decompose_errors=True` when calling `circuit.detector_error_model`."
                    )
                continue

            # Assign edge ID if unseen
            eid = edge_ids.setdefault(e_dets, len(edge_ids))
            eids.append(eid)
            edge_obs_map[eid] = e_obs

        if eids:
            hyperedge_to_edge[hid] = frozenset(eids)

    # Parse DEM instructions
    for instruction in dem.flattened():
        if instruction.type == "error":
            p = instruction.args_copy()[0]
            dets, frames = [[]], [[]]
            for t in instruction.targets_copy():
                if t.is_relative_detector_id():
                    dets[-1].append(t.val)
                elif t.is_logical_observable_id():
                    frames[-1].append(t.val)
                elif t.is_separator():
                    dets.append([])
                    frames.append([])
            handle_error(p, dets, frames)
        elif instruction.type in {"detector", "logical_observable"}:
            pass
        else:
            raise NotImplementedError(
                f"Unsupported instruction type: {instruction.type}")

    # Convert dictionaries to sparse matrices
    check_matrix = dict_to_csc_matrix(
        {
            v: k for k, v in hyperedge_ids.items()
        },
        shape=(dem.num_detectors, len(hyperedge_ids)),
    )
    observables_matrix = dict_to_csc_matrix(hyperedge_obs_map,
                                            shape=(dem.num_observables,
                                                   len(hyperedge_ids)))
    priors = np.array([priors_dict[i] for i in range(len(hyperedge_ids))],
                      dtype=np.float64)

    hyperedge_to_edge_matrix = dict_to_csc_matrix(hyperedge_to_edge,
                                                  shape=(len(edge_ids),
                                                         len(hyperedge_ids)))
    edge_check_matrix = dict_to_csc_matrix({
        v: k for k, v in edge_ids.items()
    },
                                           shape=(dem.num_detectors,
                                                  len(edge_ids)))
    edge_observables_matrix = dict_to_csc_matrix(edge_obs_map,
                                                 shape=(dem.num_observables,
                                                        len(edge_ids)))

    return DemMatrices(
        check_matrix=check_matrix,
        observables_matrix=observables_matrix,
        edge_check_matrix=edge_check_matrix,
        edge_observables_matrix=edge_observables_matrix,
        hyperedge_to_edge_matrix=hyperedge_to_edge_matrix,
        priors=priors,
    )
