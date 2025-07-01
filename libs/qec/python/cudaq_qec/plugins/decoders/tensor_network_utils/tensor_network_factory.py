import numpy as np
import numpy.typing as npt
from typing import Any, Optional
from quimb import oset
from quimb.tensor import Tensor, TensorNetwork

def tensor_network_from_parity_check(
    parity_check_matrix: npt.NDArray[Any],
    row_inds: list[str],
    col_inds: list[str],
    tags: Optional[list[str]] = None,
) -> TensorNetwork:
    """Build a sparse tensor-network representation of a parity-check matrix.

    The parity-check matrix is a binary adjacency matrix of a bipartite graph.
    The tensor network is a sparse representation of the bipartite graph where the nodes
    are delta tensors, here represented as indices.
    Between the nodes, there are Hadamard tensors for each row-column pair.

    For example, the parity check matrix

        ```
        A = [[1, 1, 0],
             [0, 1, 1]]
        ```

    is represented as the tensor network:

        r1          r2      < row indices (stored lazily)
        |  \      / |
        H   H   H   H       < Hadamard matrices
        |   |  /    |
        c1  c2      c3      < column indices (stored lazily)

    This function can be used to create the tensor network of the code and the tensor network of
    the logical observables.

    Args:
        parity_check_matrix (np.ndarray): The parity check matrix.
        row_inds (list[str]): The indices of the rows.
        col_inds (list[str]): The indices of the columns.
        tags (Optional[list[str]], optional): The tags of the Hadamard tensors.

    Returns:
        TensorNetwork: The tensor network.
    """
    # Hadamard matrix
    hadamard = np.array([[1.0, 1.0], [1.0, -1.0]])
    # Get the indices of the non-zero elements in the parity check matrix
    rows, cols = np.nonzero(parity_check_matrix)

    # Add one Hadamard tensor for each non-zero element in the parity check matrix
    return TensorNetwork([
        Tensor(
            data=hadamard,
            inds=(row_inds[i], col_inds[j]),
            tags=oset([tags[i]] if tags is not None else []),
        ) for i, j in zip(rows, cols)
    ])


def tensor_network_from_single_syndrome(syndrome: list[float],
                                        check_inds: list[str]) -> TensorNetwork:
    """Initialize the syndrome tensor network.

    Args:
        syndrome (list[float]): The syndrome values.
        check_inds (list[str]): The indices of the checks.

    Returns:
        TensorNetwork: The tensor network for the syndrome.
    """
    minus = np.array([1.0, -1.0])
    plus = np.array([1.0, 1.0])

    return TensorNetwork([
        Tensor(
            data=syndrome[i] * minus + (1.0 - syndrome[i]) * plus,
            inds=(check_inds[i],),
            tags=oset([f"SYN_{i}", "SYNDROME"]),
        ) for i in range(len(check_inds))
    ])


def prepare_syndrome_data_batch(
        syndrome_data: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """Prepare the syndrome data for the parametrized tensor network.

    The shape of the returned array is (syndrome_length, shots, 2).
    For each shot, we have `syndrome_length` len-2 vectors, which are either
    (1, 1) if the syndrome is not flipped or (1, -1) if the syndrome is flipped.

    Args:
        syndrome_data (np.ndarray): The syndrome data. The shape is expected to be (shots, syndrome_length).

    Returns:
        np.ndarray: The syndrome data in the correct shape for the parametrized tensor network.
    """
    arrays = np.ones((syndrome_data.shape[1], syndrome_data.shape[0], 2))
    flip_indices = np.where(syndrome_data == True)
    arrays[flip_indices[1], flip_indices[0], 1] = -1.0
    return arrays


def tensor_network_from_syndrome_batch(
    detection_events: npt.NDArray[Any],
    syndrome_inds: list[str],
    batch_index: str = "batch_index",
    tags: Optional[list[str]] = None,
) -> TensorNetwork:
    """Build a tensor network from a batch of syndromes.

    Args:
        detection_events (np.ndarray): A numpy array of shape (shots, syndrome_length) where each row is a detection event.
        syndrome_inds (list[str]): The indices of the syndromes.
        batch_index (str, optional): The index of the batch.
        tags (list[str], optional): The tags of the syndromes.

    Returns:
        TensorNetwork: A tensor network with the syndromes.
            for each syndrome, with the following indices:
            - batch_index: the index of the shot.
            - syndrome_inds: the indices of the syndromes.
            All the tensors share the `batch_index` index.
    """

    shots, syndrome_length = detection_events.shape

    if tags is None:
        tags = [f"SYN_{i}" for i in range(syndrome_length)]

    minus = np.outer(np.array([1.0, -1.0]), np.ones(shots))
    plus = np.outer(np.array([1.0, 1.0]), np.ones(shots))

    return TensorNetwork([
        Tensor(
            data=minus * detection_events[:, i] + plus *
            (1.0 - detection_events[:, i]),
            inds=(syndrome_inds[i], batch_index),
            tags=oset((tags[i], "SYNDROME")),
        ) for i in range(syndrome_length)
    ])


def tensor_network_from_logical_observable(
    logical: npt.NDArray[Any],
    logical_inds: list[str],
    logical_obs_inds: list[str],
    logical_tags: Optional[list[str]] = None,
) -> TensorNetwork:
    """Build a tensor network for logical observables.

    Args:
        logical (np.ndarray): The logical matrix.
        logical_inds (list[str]): The logical indices.
        logical_obs_inds (list[str]): The logical observable indices.
        logical_tags (list[str], optional): The logical tags.

    Returns:
        TensorNetwork: The tensor network for logical observables.
    """
    return tensor_network_from_parity_check(
        np.eye(logical.shape[0]),
        row_inds=logical_inds,
        col_inds=logical_obs_inds,
        tags=logical_tags,
    )
