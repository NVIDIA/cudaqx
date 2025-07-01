# ============================================================================ #
# Copyright (c) 2025 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from typing import Optional, Any, Union
import cudaq_qec as qec

import numpy as np
import numpy.typing as npt
from quimb import oset
from quimb.tensor import Tensor, TensorNetwork
from autoray import do, to_backend_dtype
import torch

from .tensor_network_utils.contractors import BACKENDS, CONTRACTORS, optimize_path


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


def tensor_to_cpu(data: Any, backend: str,
                  dtype: str) -> Union[np.ndarray, "torch.Tensor"]:
    """Convert a tensor to CPU if it is on GPU.

    Args:
        data (Any): The tensor data (numpy array or torch tensor).
        backend (str): The backend to use ("numpy" or "torch").
        dtype (str): The data type to convert to.

    Returns:
        np.ndarray or torch.Tensor: The tensor on CPU with the specified dtype and backend.
    """
    try:
        # in case the tensor is on GPU
        data = data.cpu()
    except AttributeError:
        pass

    return do(
        "array",
        data,
        like=backend,
        dtype=to_backend_dtype(dtype, like=backend),
    )


def tensor_to_gpu(data: Any, dtype: str, device: str) -> "torch.Tensor":
    """Convert a tensor to GPU.

    Args:
        data: The tensor data (numpy array or torch tensor).
        dtype: The data type to convert to.
        device: The CUDA device string (e.g., "cuda:0").

    Returns:
        torch.Tensor: The tensor on the specified CUDA device.
    """
    backend = "torch"
    return do(
        "array",
        data,
        like=backend,
        dtype=to_backend_dtype(dtype, like=backend),
        device=device,
    )


def set_tensor_type(
    tn: TensorNetwork,
    backend: str,
    dtype: str = "float64",
    device: Optional[str] = None,
) -> None:
    """Set the backend for the tensor network.

    Args:
        tn (TensorNetwork): The tensor network.
        backend (str): The backend to use ("numpy" or "torch").
        dtype (str, optional): The data type to use. Defaults to "float64".
        device (Optional[str], optional): The device to use. Defaults to None.
    """
    if device is None or device == "cpu":
        tn.apply_to_arrays(lambda x: tensor_to_cpu(x, backend, dtype))
    elif "cuda" in device:
        tn.apply_to_arrays(lambda x: tensor_to_gpu(x, dtype, device))


def set_backend(contractor_name: str, device: str) -> str:
    """Set the backend for the contractor.

    Args:
        contractor_name (str): The name of the contractor.
        device (str): The device to use.

    Returns:
        str: The backend name.
    """

    # GPU contractor + torch backend (only GPU support)
    if contractor_name == "cutensornet":

        assert torch.cuda.is_available(), "Torch CUDA is not available."
        assert device in [
            f"cuda:{i}" for i in range(torch.cuda.device_count())
        ], (f"Device {device} cannot be used with cuTensornet."
            f"Please use one of the following devices: "
            f"Available devices: {[f'cuda:{i}' for i in range(torch.cuda.device_count())]}"
           )
        return "torch"
    # CPU contractor + various backends
    else:
        for b in BACKENDS:
            if b in contractor_name:
                return b


def _adjust_default_path_value(val: Any, is_cutensornet: bool) -> Any:
    """Adjust the default path value for the contractor.

    Args:
        val: The current value.
        is_cutensornet (bool): Whether the contractor is cutensornet.

    Returns:
        Any: The adjusted value.
    """
    if is_cutensornet:
        return None if val == "auto" else val
    else:
        return "auto" if val is None else val


@qec.decoder("tensor_network_decoder")
class TensorNetworkDecoder:
    """A general class for tensor network decoders.

    Constructs a tensor network with the following tensors:
    Hadamard matrices for each edge of the Tanner graph:
        - Hadamard tensors for each check-error pair.
        - Hadamard tensors for each logical observable-error pair.
    Delta tensors for each vertex of the Tanner graph:
        - Delta tensors for each error represented as indices.
        - Delta tensors for each logical check represented as indices.

    The core, tensor-network graph is identical to a Tanner graph.
    The tensor network is a bipartite graph with two types of nodes:
    - Check nodes (c_i's) and error nodes (e_j's), delta tensors.
    - Hadamard matrix (H) for each check-error pair.

    Then, the tensor network is extended by the noise model, the logical observables,
    and the product state / batch of syndromes.

    For example,

                  open leg      < logical observable
                  --------
                     |
        s1      s2   |     s3   < syndromes               : product state of zeros/ones
        |       |    |     |                        ----|
        c1      c2  l1     c3   < checks / logical     | : delta tensors
        |     / |   | \    |                            |
        H   H   H   H  H   H    < Hadamard matrices     | TANNER (bipartite) GRAPH
          \ |   |  /   |  /                             |
            e1  e2     e3       < errors                | : delta tensors
            |   |     /                            -----|
             \ /     /
            P(e1, e2, e3)       < noise / error model     : classical probability density

    ci, ej, lk are delta tensors represented sparsely as indices.


    Attributes:
        code_tn (TensorNetwork): The tensor network for the code (parity check matrix).
        logical_tn (TensorNetwork): The tensor network for the logical observables.
        syndrome_tn (TensorNetwork): The tensor network for the syndrome.
        noise_model (TensorNetwork): The noise model tensor network.
        full_tn (TensorNetwork): The full tensor network including code, logical, syndrome, and noise model.
        check_inds (list[str]): The check indices.
        error_inds (list[str]): The error indices.
        logical_inds (list[str]): The logical indices.
        logical_obs_inds (list[str]): The logical observable indices.
        logical_tags (list[str]): The logical tags.
        _contractor_name (str): The contractor to use.
        _backend (str): The backend used for tensor operations ("numpy" or "torch").
        _dtype (str): The data type of the tensors.
        _device (str): The device for tensor operations ("cpu" or "cuda:X").
        path_single (Any): The contraction path for single syndrome decoding.
        path_batch (Any): The contraction path for batch decoding.
        slicing_single (Any): Slicing specification for single syndrome contraction.
        slicing_batch (Any): Slicing specification for batch contraction.
    """
    code_tn: TensorNetwork
    logical_tn: TensorNetwork
    syndrome_tn: TensorNetwork
    noise_model: TensorNetwork
    full_tn: TensorNetwork
    check_inds: list[str]
    error_inds: list[str]
    logical_inds: list[str]
    logical_obs_inds: list[str]
    logical_tags: list[str]
    _contractor_name: str
    _backend: str
    _dtype: str
    _device: str
    path_single: Any
    path_batch: Any
    slicing_single: Any
    slicing_batch: Any

    def __init__(
        self,
        H: npt.NDArray[Any],
        logical_obs: npt.NDArray[Any],
        noise_model: Union[TensorNetwork, list[float]],
        check_inds: Optional[list[str]] = None,
        error_inds: Optional[list[str]] = None,
        logical_inds: Optional[list[str]] = None,
        logical_tags: Optional[list[str]] = None,
        contract_noise_model: bool = True,
        contractor_name: Optional[str] = "cutensornet",
        dtype: str = "float32",
        device: str = "cuda:0",
    ) -> None:
        """Initialize a sparse representation of a tensor network decoder for an arbitrary code
        given by its parity check matrix, logical observables and noise model.

        Args:
            H (np.ndarray): The parity check matrix. First dimension is the number of checks, second is the number of errors.
            logical_obs (np.ndarray): The logical. First dimension is one, second is the number of errors.
            noise_model (Union[TensorNetwork, list[float]]): The noise model to use. Can be a tensor network or a list of probabilities.
                If a tensor network, it must have exactly parity_check_matrix.shape[1] open indices.
                The same ordering is assumed as in the parity check matrix.
                If a list, it must have the same length as parity_check_matrix.shape[1].
                A product state noise model will be constructed from it.
            check_inds (Optional[list[str]], optional): The check indices. If None, defaults to [c_0, c_1, ...].
            error_inds (Optional[list[str]], optional): The error indices. If None, defaults to [e_0, e_1, ...].
            logical_inds (Optional[list[str]], optional): The index of the logical. If None, defaults to [l_0].
            logical_tags (Optional[list[str]], optional): The logical tags. If None, defaults to [LOG_0, LOG_1, ...].
            contract_noise_model (bool, optional): Whether to contract the noise model with the tensor network at initialization.
            contractor_name (Optional[str], optional): The contractor to use. If None, defaults to "numpy".
            dtype (str, optional): The data type of the tensors in the tensor network. Defaults to "float64".
            device (str, optional): The device to use for the tensors in the tensor network. Defaults to "cpu".
                If using cuTensornet, this should be a CUDA device like "cuda:0", "cuda:1", etc.
                If using other contractors, this should be "cpu".
        """

        qec.Decoder.__init__(self, H)

        if not torch.cuda.is_available() and contractor_name == "cutensornet":
            print("Warning: Torch CUDA is not available. "
                  "Using CPU for tensor network operations.")
            contractor_name = "numpy"
            device = "cpu"

        num_checks, num_errs = H.shape
        if check_inds is None:
            self.check_inds = [f"s_{j}" for j in range(num_checks)]
        if error_inds is None:
            self.error_inds = [f"e_{j}" for j in range(num_errs)]

        self.logical_obs_inds = ["obs"]  # Open logical index

        # Construct the tensor network of the code
        self.parity_check_matrix = H.copy()
        self.code_tn = tensor_network_from_parity_check(
            self.parity_check_matrix,
            col_inds=self.error_inds,
            row_inds=self.check_inds,
        )

        self.replace_logical_observable(
            logical_obs,
            logical_inds=logical_inds,
            logical_tags=logical_tags,
        )

        # Initialize the syndrome tensor network with no errors.
        self.syndrome_tn = tensor_network_from_single_syndrome(
            [0.0] * len(self.check_inds), self.check_inds)

        # Construct the tensor network of code + logical + syndromes
        # The noise model is added later
        self.full_tn = self.code_tn | self.logical_tn | self.syndrome_tn

        if contractor_name not in CONTRACTORS:
            raise ValueError(f"Contractor {contractor_name} not found. "
                             f"Available contractors: {CONTRACTORS.keys()}")

        # Default values for the path finders
        self.path_single = None if contractor_name == "cutensornet" else "auto"
        self.path_batch = None if contractor_name == "cutensornet" else "auto"
        self.slicing_batch = tuple()
        self.slicing_single = tuple()
        self._batch_size = 1

        self.set_contractor(contractor_name, dtype=dtype, device=device)

        # Initialize the noise model
        if isinstance(noise_model, TensorNetwork):
            old_inds = noise_model._outer_inds
            assert len(old_inds) == len(self.error_inds), (
                f"Noise model has {len(old_inds)} open indices, "
                f"but expected {len(self.error_inds)} for the error indices.")
            # Reindex the noise model to match the error indices
            ind_map = {oi: ni for oi, ni in zip(old_inds, self.error_inds)}
            noise_model = noise_model.reindex(ind_map)
        else:
            from .tensor_network_utils.noise_models import factorized_noise_model
            noise_model = factorized_noise_model(self.error_inds, noise_model)
        self.init_noise_model(noise_model, contract=contract_noise_model)

    def replace_logical_observable(
            self,
            logical_obs: npt.NDArray[Any],
            logical_inds: Optional[list[str]] = None,
            logical_tags: Optional[list[str]] = None) -> None:
        """Add logical observables to the tensor network.
        Args:
            logical_obs (np.ndarray): The logical matrix.
            logical_inds (Optional[list[str]], optional): The logical indices. If None, defaults to [l_0, l_1, ...].
            logical_obs_inds (Optional[list[str]], optional): The logical observable indices. If None, defaults to [l_obs_0, l_obs_1, ...].
            logical_tags (Optional[list[str]], optional): The logical tags. If None, defaults to [LOG_0, LOG_1, ...].
        """
        assert logical_obs.shape == (1, len(self.error_inds)), (
            "logical must be a single row matrix, shape (1, n), where n is the number of errors."
            "Only single logical are supported for now.")
        if logical_inds is None:
            self.logical_inds = ["l_0"]  # Index before the Hadamard tensor
        else:
            self.logical_inds = logical_inds

        if logical_tags is None:
            self.logical_tags = ["LOG_0"]
        else:
            self.logical_tags = logical_tags

        # Construct the tensor network of the logical observables
        self.logical_obs = logical_obs.copy()
        self.logical_tn = tensor_network_from_parity_check(
            self.logical_obs,
            col_inds=self.error_inds,
            row_inds=self.logical_inds,
            tags=self.logical_tags,
        )

        # Add a Hadamard tensor for each logical observable for its outer leg
        self.logical_tn |= tensor_network_from_logical_observable(
            self.logical_obs, self.logical_inds, self.logical_obs_inds,
            self.logical_tags)

        if hasattr(self, "full_tn"):
            self.full_tn = (self.code_tn | self.logical_tn | self.syndrome_tn |
                            self.noise_model)

    def init_noise_model(self,
                         noise_model: TensorNetwork,
                         contract: bool = False) -> None:
        """Initialize the noise model.

        Args:
            noise_model (TensorNetwork): The noise model tensor network.
            contract (bool, optional): Whether to contract the noise model with the tensor network. Defaults to False.
        """
        self.noise_model = noise_model
        set_tensor_type(self.noise_model, self._backend, self._dtype,
                        self._device)
        self.full_tn = (self.code_tn | self.logical_tn | self.syndrome_tn |
                        self.noise_model)

        if contract:
            for ie in self.error_inds:
                self.full_tn.contract_ind(ie)

    def flip_syndromes(self, values: list[float]) -> None:
        """Modify the tensor network in place to represent a given syndrome.

        Args:
            values (list): A list of float values for the syndrome.
                The probability an observable was flipped.
        """

        # Below we use autoray.do to ensure that the data is
        # defined via the correct backend: numpy, torch, jax, etc.
        # Torch is used for GPU support.

        dtype = to_backend_dtype(self._dtype, like=self._backend)
        array_args = {"like": self._backend, "dtype": dtype}

        if "cuda" in self._device:
            array_args["device"] = self._device

        minus = do("array", (1.0, -1.0), **array_args)
        plus = do("array", (1.0, 1.0), **array_args)

        for ind in range(len(self.check_inds)):
            t = self.syndrome_tn.tensors[next(
                iter(self.syndrome_tn.tag_map[f"SYN_{ind}"]))]
            t.modify(data=values[ind] * minus + (1.0 - values[ind]) * plus,)

    def set_contractor(self,
                       contractor: str,
                       dtype: Optional[str] = None,
                       device: Optional[str] = None) -> None:
        """Set the contractor for the tensor network.

        Args:
            contractor (str): The contractor to use.
            dtype (str, optional): The data type to use. If None, keeps the current dtype.
            device (str, optional): The device to use. If None, keeps the current device.

        Raises:
            ValueError: If the contractor is not found or device is invalid for the contractor.
        """

        if contractor not in CONTRACTORS:
            raise ValueError(f"Contractor {contractor} not found. "
                             f"Available contractors: {CONTRACTORS.keys()}")

        # Reset only if specified
        if dtype is not None:
            self._dtype = dtype
        if device is not None:
            self._device = device

        is_cutensornet = contractor == "cutensornet"
        # If the contractor is cutensornet, we need to set the device
        # to the GPU device. Otherwise, we set it to CPU.
        if not is_cutensornet:
            self._device = "cpu"
        elif "cuda" not in self._device:

            raise ValueError(
                f"Device {self._device} cannot be used with cuTensornet. "
                f"Please use one of the following devices: "
                f"Available devices: {[f'cuda:{i}' for i in range(torch.cuda.device_count())]}"
            )

        self._contractor_name = contractor
        self._backend = set_backend(contractor, self._device)
        set_tensor_type(self.full_tn, self._backend, self._dtype, self._device)

        self.path_batch = _adjust_default_path_value(self.path_batch,
                                                     is_cutensornet)
        self.path_single = _adjust_default_path_value(self.path_single,
                                                      is_cutensornet)

    def decode(
        self,
        syndrome: list[float],
    ) -> "qec.DecoderResult":
        """
        Decode the syndrome by contracting exactly the full tensor network.

        Args:
            syndrome (list[float]): 
                The syndrome soft decision probabilities ordered as the check indices.

        Returns:
            qec.DecoderResult: The result of the decoding.
        """
        assert hasattr(self, "noise_model")
        assert len(syndrome) == len(self.check_inds), (
            f"Syndrome length {len(syndrome)} does not match the number of checks {len(self.check_inds)}."
        )

        # adjust the values of the syndromes
        self.flip_syndromes(syndrome)

        if self.path_single is None:
            # If the path is not set, we need to optimize it
            self.optimize_path(
                output_inds=(self.logical_obs_inds[0],),
                optimize=self.path_single,
            )

        contraction_value = CONTRACTORS[self._contractor_name](
            self.full_tn.get_equation(output_inds=(self.logical_obs_inds[0],)),
            self.full_tn.arrays,
            optimize=self.path_single,
            slicing=self.slicing_single,
        )

        res = qec.DecoderResult()
        res.converged = True
        res.result = [
            float(contraction_value[1] /
                  (contraction_value[1] + contraction_value[0]))
        ]
        return res

    def decode_batch(
        self,
        syndrome_batch: npt.NDArray[Any],
    ) -> list["qec.DecoderResult"]:
        """Decode a batch of detection events.

        Args:
            syndrome_batch (np.ndarray): A numpy array of shape (batch_size, syndrome_length) where each row is a detection event.

        Returns:
            list[qec.DecoderResult]: list of results for each detection event in the batch.
        """

        assert hasattr(self, "noise_model")
        syndrome_length = syndrome_batch.shape[1]
        assert syndrome_length == len(self.check_inds)

        # Remove the syndrome tensors from the full tensor network
        tn = TensorNetwork(
            [t for t in self.full_tn.tensors if "SYNDROME" not in t.tags])

        tn |= tensor_network_from_syndrome_batch(syndrome_batch,
                                                 self.check_inds,
                                                 batch_index="batch_index")
        set_tensor_type(tn, self._backend, self._dtype, self._device)

        if self.path_batch is None or syndrome_batch.shape[
                0] != self._batch_size:
            # If the path is not set, we need to optimize it
            self.optimize_path(
                output_inds=("batch_index", self.logical_obs_inds[0]),
                optimize=self.path_batch,
                syndrome_batch=syndrome_batch,
            )
            self._batch_size = syndrome_batch.shape[0]

        contraction_value = CONTRACTORS[self._contractor_name](
            tn.get_equation(output_inds=("batch_index",
                                         self.logical_obs_inds[0])),
            tn.arrays,
            optimize=self.path_batch,
            slicing=self.slicing_batch,
        )

        res = []
        for r in range(syndrome_batch.shape[0]):
            res.append(qec.DecoderResult())
            res[r].converged = True
            res[r].result = [
                float(contraction_value[r, 1] /
                      (contraction_value[r, 1] + contraction_value[r, 0]))
            ]

        return res

    def optimize_path(
        self,
        output_inds: Any,
        optimize: Any = None,
        syndrome_batch: Optional[npt.NDArray[Any]] = None,
    ) -> Any:
        """Optimize the contraction path of the tensor network.

        Args:
            output_inds (tuple[str]): The output indices of the contraction.
            optimize (Optional[cutn.OptimizerOptions], optional): The optimization options to use. 
                If None or cuquantum.tensornet.OptimizerOptions, the default options are used.
                Else, Quimb interface at 
                https://quimb.readthedocs.io/en/latest/autoapi/quimb/tensor/tensor_core/index.html#quimb.tensor.tensor_core.TensorNetwork.contraction_info
            syndrome_batch (Optional[np.ndarray], optional): A batch of syndromes to use for the optimization. If None, the full tensor network is used.

        Returns:
            Any: The optimizer info object.
        """
        from cuquantum.tensornet import OptimizerOptions

        is_batch = syndrome_batch is not None

        # Build the tensor network
        if is_batch:
            tn = TensorNetwork(
                [t for t in self.full_tn.tensors if "SYNDROME" not in t.tags])
            tn |= tensor_network_from_syndrome_batch(syndrome_batch,
                                                     self.check_inds,
                                                     batch_index="batch_index")
        else:
            tn = self.full_tn
        set_tensor_type(tn, self._backend, self._dtype, self._device)

        # Optimize the path
        path, info = optimize_path(optimize, output_inds, tn)
        slices = info.slices if hasattr(info, "slices") else tuple()

        # Assign result
        target = "path_batch" if is_batch else "path_single"
        setattr(self, target, path)

        target = "slicing_batch" if is_batch else "slicing_single"
        setattr(self, target, slices)

        return info
