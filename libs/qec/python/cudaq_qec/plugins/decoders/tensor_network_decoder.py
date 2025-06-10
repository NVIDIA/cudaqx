# ============================================================================ #
# Copyright (c) 2024 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from typing import Optional, Any, Union, List, Tuple
import cudaq_qec as qec

import numpy as np
import numpy.typing as npt
from quimb import oset
from quimb.tensor import Tensor, TensorNetwork
from autoray import do, to_backend_dtype

import stim

from .tensor_network_utils.contractors import BACKENDS, CONTRACTORS, optimize_path


def tensor_network_from_parity_check(
    parity_check_matrix: npt.NDArray[Any],
    row_inds: List[str],
    col_inds: List[str],
    tags: Optional[List[str]] = None,
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

    This function can be used to the tensor network of the code and the tensor network of
    the logical observables.

    Args:
        parity_check_matrix (np.ndarray): The parity check matrix.
        row_inds (List[str]): The indices of the rows.
        col_inds (List[str]): The indices of the columns.
        tags (Optional[List[str]], optional): The tags of the Hadamard tensors.

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


def tensor_network_from_single_syndrome(syndrome: List[bool],
                                        check_inds: List[str]) -> TensorNetwork:
    """Initialize the syndrome tensor network.

    Args:
        syndrome (list[bool]): The syndrome values.
        check_inds (list[str]): The indices of the checks.

    Returns:
        TensorNetwork: The tensor network for the syndrome.
    """
    minus = np.array([1.0, -1.0])
    plus = np.array([1.0, 1.0])

    return TensorNetwork([
        Tensor(
            data=minus if bool(syndrome[i]) else plus,
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
    syndrome_inds: List[str],
    batch_index: str = "batch_index",
    tags: Optional[List[str]] = None,
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

    syndrome_arrays = prepare_syndrome_data_batch(detection_events)

    return TensorNetwork([
        Tensor(
            data=syndrome_arrays[i],
            inds=(batch_index, syndrome_inds[i]),
            tags=oset((tags[i], "SYNDROME")),
        ) for i in range(syndrome_length)
    ])


def tensor_network_from_logical_observable(
    logicals: npt.NDArray[Any],
    logical_inds: List[str],
    logical_obs_inds: List[str],
    logicals_tags: Optional[List[str]] = None,
) -> TensorNetwork:
    """Build a tensor network for logical observables.

    Args:
        logicals (np.ndarray): The logicals matrix.
        logical_inds (list[str]): The logical indices.
        logical_obs_inds (list[str]): The logical observable indices.
        logicals_tags (list[str], optional): The logicals tags.

    Returns:
        TensorNetwork: The tensor network for logical observables.
    """
    return tensor_network_from_parity_check(
        np.eye(logicals.shape[0]),
        row_inds=logical_inds,
        col_inds=logical_obs_inds,
        tags=logicals_tags,
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
        import torch

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


def parse_detector_error_model(
    stim_detector_error_model: "stim.DetectorErrorModel",
) -> Tuple[npt.NDArray[Any], npt.NDArray[Any], List[float]]:
    """
    Construct a parity check matrix, logicals, and noise model from a stim DetectorErrorModel.

    Args:
        stim_detector_error_model (stim.DetectorErrorModel): 
            The stim DetectorErrorModel instance to parse.
        error_inds (Optional[List[str]], optional): 
            List of error index names to use for the noise model. If None, defaults to ["e_0", ..., "e_{n-1}"].

    Returns:
        Tuple[npt.NDArray[Any], npt.NDArray[Any], TensorNetwork]: 
            A tuple containing:
                - The parity check matrix (as a dense numpy array).
                - The logicals matrix (as a dense numpy array).
                - The factorized noise model as a TensorNetwork.
    """
    from .tensor_network_utils.stim_interface import detector_error_model_to_check_matrices

    matrices = detector_error_model_to_check_matrices(stim_detector_error_model)

    out_H = np.zeros(matrices.check_matrix.shape)
    matrices.check_matrix.astype(np.float64).toarray(out=out_H)
    out_L = np.zeros(matrices.observables_matrix.shape)
    matrices.observables_matrix.astype(np.float64).toarray(out=out_L)

    return out_H, out_H, [float(p) for p in matrices.priors]


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
        c1      c2  l1     c3   < checks / logicals     | : delta tensors
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
        logicals_tn (TensorNetwork): The tensor network for the logical observables.
        syndrome_tn (TensorNetwork): The tensor network for the syndrome.
        noise_model (TensorNetwork): The noise model tensor network.
        full_tn (TensorNetwork): The full tensor network including code, logicals, syndrome, and noise model.
        check_inds (List[str]): The check indices.
        error_inds (List[str]): The error indices.
        logical_inds (List[str]): The logical indices.
        logical_obs_inds (List[str]): The logical observable indices.
        logicals_tags (List[str]): The logicals tags.
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
    logicals_tn: TensorNetwork
    syndrome_tn: TensorNetwork
    noise_model: TensorNetwork
    full_tn: TensorNetwork
    check_inds: List[str]
    error_inds: List[str]
    logical_inds: List[str]
    logical_obs_inds: List[str]
    logicals_tags: List[str]
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
        logicals: npt.NDArray[Any],
        noise_model: Union[TensorNetwork, List[float]],
        check_inds: Optional[List[str]] = None,
        error_inds: Optional[List[str]] = None,
        logical_inds: Optional[List[str]] = None,
        logicals_tags: Optional[List[str]] = None,
        contract_noise_model: bool = True,
        contractor_name: Optional[str] = "numpy",
        dtype: str = "float64",
        device: str = "cpu",
    ) -> None:
        """Initialize a sparse representation of a tensor network decoder for an arbitrary code
        given by its parity check matrix, logical observables and noise model.

        Args:
            H (np.ndarray): The parity check matrix. First dimension is the number of checks, second is the number of errors.
            logicals (np.ndarray): The logicals. First dimension is the number of logicals, second is the number of errors.
            noise_model (Union[TensorNetwork, List[float]]): The noise model to use. Can be a tensor network or a list of probabilities.
                If a tensor network, it must have exactly parity_check_matrix.shape[1] open indices.
                The same ordering is assumed as in the parity check matrix.
                If a list, it must have the same length as parity_check_matrix.shape[1].
                A product state noise model will be constructed from it.
            check_inds (Optional[list[str]], optional): The check indices. If None, defaults to [c_0, c_1, ...].
            error_inds (Optional[list[str]], optional): The error indices. If None, defaults to [e_0, e_1, ...].
            logical_inds (Optional[list[str]], optional): The logical indices. If None, defaults to [l_0, l_1, ...].
            logicals_tags (Optional[list[str]], optional): The logicals tags. If None, defaults to [LOG_0, LOG_1, ...].
            contract_noise_model (bool, optional): Whether to contract the noise model with the tensor network at initialization.
            contractor_name (Optional[str], optional): The contractor to use. If None, defaults to "numpy".
            dtype (str, optional): The data type of the tensors in the tensor network. Defaults to "float64".
            device (str, optional): The device to use for the tensors in the tensor network. Defaults to "cpu".
                If using cuTensornet, this should be a CUDA device like "cuda:0", "cuda:1", etc.
                If using other contractors, this should be "cpu".
        """

        qec.Decoder.__init__(self, H)

        num_checks, num_errs = H.shape
        if check_inds is None:
            self.check_inds = [f"s_{j}" for j in range(num_checks)]
        if error_inds is None:
            self.error_inds = [f"e_{j}" for j in range(num_errs)]
        if logical_inds is None:
            self.logical_inds = [f"l_{j}" for j in range(logicals.shape[0])]

        if logicals_tags is None:
            log_checks = logicals.shape[0]
            self.logicals_tags = [f"LOG_{l}" for l in range(log_checks)]

        # Construct the tensor network of the code
        self.parity_check_matrix = H.copy()
        self.code_tn = tensor_network_from_parity_check(
            self.parity_check_matrix,
            col_inds=self.error_inds,
            row_inds=self.check_inds,
        )

        # Construct the tensor network of the logicals observables
        self.logicals = logicals.copy()
        self.logicals_tn = tensor_network_from_parity_check(
            self.logicals,
            col_inds=self.error_inds,
            row_inds=self.logical_inds,
            tags=self.logicals_tags,
        )

        self.logical_obs_inds = [f"obs_{l}" for l in self.logical_inds]
        # Add a Hadamard tensor for each logical observable for its outer leg
        self.logicals_tn |= tensor_network_from_logical_observable(
            self.logicals, self.logical_inds, self.logical_obs_inds,
            self.logicals_tags)

        self.syndrome_tn = tensor_network_from_single_syndrome(
            [True] * len(self.check_inds), self.check_inds)

        # Construct the tensor network of code + logicals + syndromes
        # The noise model is added later
        self.full_tn = self.code_tn | self.logicals_tn | self.syndrome_tn

        if contractor_name not in CONTRACTORS:
            raise ValueError(f"Contractor {contractor_name} not found. "
                             f"Available contractors: {CONTRACTORS.keys()}")

        # Default values for the path finders
        self.path_single = None if contractor_name == "cutensornet" else "auto"
        self.path_batch = None if contractor_name == "cutensornet" else "auto"
        self.slicing_batch = tuple()
        self.slicing_single = tuple()

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
        self.full_tn = (self.code_tn | self.logicals_tn | self.syndrome_tn |
                        self.noise_model)

        if contract:
            for ie in self.error_inds:
                self.full_tn.contract_ind(ie)

    def flip_syndromes(self, values: List[bool]) -> None:
        """Modify the tensor network in place to represent a given syndrome.

        Args:
            values (list): A list of binary values for the check operators.
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
            # if s is False, the tensor is the plus state (1, 1): |+> = Hadamard @ |0>
            # If s is True, the tensor is the minus state (1, -1): |-> = Hadamard @ |1>
            if bool(values[ind]) and t.data[1] != -1:
                t.modify(data=minus)
            elif not bool(values[ind]) and t.data[1] != 1:
                t.modify(data=plus)

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
            import torch

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
        syndrome: List[bool],
        logical_observable: Optional[str] = None,
        return_probability: bool = False,
    ) -> "qec.DecoderResult":
        """
        Decode the syndrome by contracting the full tensor network.

        Args:
            syndrome (List[bool]): The syndrome ordered as the check indices.
            logical_observable (Optional[str], optional): The index of the logical observable to use.
            return_probability (bool, optional): Whether to return the flip probability.

        Returns:
            qec.DecoderResult: The result of the decoding.
        """
        assert hasattr(self, "noise_model")
        assert len(syndrome) == len(self.check_inds), (
            f"Syndrome length {len(syndrome)} does not match the number of checks {len(self.check_inds)}."
        )

        # adjust the values of the syndromes
        self.flip_syndromes(syndrome)

        if len(self.logical_obs_inds) > 1 and logical_observable is None:
            raise ValueError(
                "This `TensorNetworkDecoder` contains more than one logical observable."
                "Please specify which one to use by setting the `logical_observable` argument"
                "to `TensorNetwork.decode(syndrome, logical_observable=...)."
                "The available logical observables are: " +
                str(self.logical_obs_inds))
        elif len(self.logical_obs_inds) == 1:
            logical_observable = self.logical_obs_inds[0]

        contraction_value = CONTRACTORS[self._contractor_name](
            self.full_tn.get_equation(output_inds=(logical_observable,)),
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
        logical_observable: Optional[str] = None,
        return_probability: bool = False,
    ) -> List["qec.DecoderResult"]:
        """Decode a batch of detection events.

        Args:
            syndrome_batch (np.ndarray): A numpy array of shape (batch_size, syndrome_length) where each row is a detection event.
            logical_observable (str, optional): The index of the logical observable to use. If not specified,
                there must be only one logical observable.

        Returns:
            list[qec.DecoderResult]: List of results for each detection event in the batch.
        """

        assert hasattr(self, "noise_model")
        syndrome_length = syndrome_batch.shape[1]
        assert syndrome_length == len(self.check_inds)

        if len(self.logical_obs_inds) > 1 and logical_observable is None:
            raise ValueError(
                "This tensor network contains more than one logical observable."
                "Please specify which one to use by setting the `logical_observable` argument"
                "to TensorNetwork.decode(syndrome, logical_observable=...)."
                "The available logical observables are: " +
                str(self.logical_obs_inds))
        elif len(self.logical_obs_inds) == 1:
            logical_observable = self.logical_obs_inds[0]

        # Remove the syndrome tensors from the full tensor network
        tn = TensorNetwork(
            [t for t in self.full_tn.tensors if "SYNDROME" not in t.tags])

        tn |= tensor_network_from_syndrome_batch(syndrome_batch,
                                                 self.check_inds,
                                                 batch_index="batch_index")
        set_tensor_type(tn, self._backend, self._dtype, self._device)

        contraction_value = CONTRACTORS[self._contractor_name](
            tn.get_equation(output_inds=("batch_index", logical_observable)),
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
            optimize (Optional[cutn.OptimizerOptions], optional): The optimization options to use. If None, the default options are used.
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
