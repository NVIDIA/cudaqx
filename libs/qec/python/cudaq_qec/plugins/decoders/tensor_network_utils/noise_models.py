# ============================================================================ #
# Copyright (c) 2025 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from quimb import oset
from quimb.tensor import TensorNetwork, Tensor

from ..tensor_network_decoder import TensorNetworkDecoder
from .tensor_network_factory import (
    tensor_network_from_syndrome_batch,
    prepare_syndrome_data_batch,
)


def factorized_noise_model(
        error_indices: list[str],
        error_probabilities: list[float] | np.ndarray,
        tensors_tags: list[str] | None = None) -> TensorNetwork:
    """
    Construct a factorized (product state) noise model as a tensor network.

    Args:
        error_indices (list[str]): list of error index names.
        error_probabilities (Union[list[float], np.ndarray]): list or array of error probabilities for each error index.
        tensors_tags (Optional[list[str]], optional): list of tags for each tensor. If None, default tags are used.

    Returns:
        TensorNetwork: The tensor network representing the factorized noise model.
    """
    assert len(error_indices) == len(error_probabilities), \
        "Length of error_indices and error_probabilities must match."
    if isinstance(error_probabilities, np.ndarray):
        assert error_probabilities.ndim == 1 and len(error_probabilities) == len(
            error_indices
        ), "error_probabilities must be a 1D array with length matching error_indices."
    elif isinstance(error_probabilities, list):
        assert all(isinstance(p, (float, int)) for p in error_probabilities), \
            "error_probabilities must be a list of floats or ints."
    else:
        raise TypeError("error_probabilities must be a list or numpy array.")
    assert all(p >= 0 and p <= 1 for p in error_probabilities), \
        "All error probabilities must be in the range [0, 1]."
    assert all(isinstance(ind, str) for ind in error_indices), \
        "All error indices must be strings."
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
        tensors_tags: list[str] | None = None) -> TensorNetwork:
    """
    Construct a noise model as a tensor network for correlated error pairs.

    Args:
        error_index_pairs (list[tuple[str, str]]): list of pairs of error index names.
        error_probabilities (list[np.ndarray]): list of 2x2 probability matrices for each error pair.
        tensors_tags (Optional[list[str]], optional): list of tags for each tensor. If None, default tags are used.

    Returns:
        TensorNetwork: The tensor network representing the error pairs noise model.
    """
    assert len(error_index_pairs) == len(error_probabilities), \
        "Length of error_index_pairs and error_probabilities must match."
    if isinstance(error_probabilities, np.ndarray):
        assert (error_probabilities.ndim == 2 and
                error_probabilities.shape[1] == 2 and
                error_probabilities.shape[0] == len(error_index_pairs)), \
            "error_probabilities must be a 2D array with shape (N, 2) where N is the number of error pairs."
    elif isinstance(error_probabilities, list):
        assert all(isinstance(p, np.ndarray) and p.ndim == 2 and p.shape == (2, 2)
                   for p in error_probabilities), \
            "error_probabilities must be a list of 2x2 numpy arrays."
    else:
        raise TypeError("error_probabilities must be a list or numpy array.")
    assert all((p >=0).all() and (p <= 1).all() for p in error_probabilities), \
        "All error probabilities must be in the range [0, 1]."
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


class NMOptimizer(TensorNetworkDecoder):
    """Differentiable noise model optimizer via torch autograd.

    Extends TensorNetworkDecoder with a differentiable factorized noise
    model whose scalar probability parameters live in the torch autograd
    graph.  Accepts a ``list[float]`` of per-error probabilities and
    internally builds the ``[1-p, p]`` tensor network so that gradients
    flow directly to the scalar parameters.

    Uses **cuTensorNet** as the contraction backend when a CUDA GPU is
    available (``cutensornet / torch / cuda``), falling back to
    ``torch / torch / cpu`` otherwise.  Torch tensors are always used as the
    data representation so that ``torch.autograd`` can differentiate through
    the contraction.

    The optimizer supports two loss functions:
      - Cross-entropy loss (differentiable, for gradient-based optimizers).
      - Logical error rate (non-differentiable, for evaluation).

    The dataset (syndrome batch and observable flips) can be replaced
    efficiently via :meth:`update_dataset` or :meth:`_update_data`.

    Example::

        opt = NMOptimizer(H, logical_obs, priors, syndrome_data, obs_flips)
        torch_opt = torch.optim.Adam(opt.noise_params, lr=0.01)
        for step in range(100):
            torch_opt.zero_grad()
            loss = opt.cross_entropy_ler()
            loss.backward()
            torch_opt.step()
    """

    def __init__(
        self,
        H: npt.NDArray[Any],
        logical_obs: npt.NDArray[Any],
        noise_model: list[float],
        syndrome_data: npt.NDArray[Any],
        observable_flips: npt.NDArray[Any],
        check_inds: list[str] | None = None,
        error_inds: list[str] | None = None,
        logical_inds: list[str] | None = None,
        logical_tags: list[str] | None = None,
        dtype: str = "float32",
        device: str = "cuda",
    ) -> None:
        """Initialize the noise model optimizer.

        Args:
            H: Parity check matrix, shape ``(num_checks, num_errors)``.
            logical_obs: Logical observable matrix, shape ``(1, num_errors)``.
            noise_model: Initial per-error probabilities.  A factorized
                noise model is built internally with ``[1-p, p]`` tensors
                whose scalar ``p`` values are differentiable via torch
                autograd.
            syndrome_data: Syndrome batch, shape ``(shots, num_checks)``.
            observable_flips: Observable flip outcomes, shape ``(shots,)``.
            check_inds: Check index names.
            error_inds: Error index names.
            logical_inds: Logical index names.
            logical_tags: Logical tags.
            dtype: Tensor data type (e.g. ``"float32"``).
            device: ``"cuda"`` (default) to use cuTensorNet on GPU, or
                ``"cpu"`` for torch-only on CPU.
        """
        super().__init__(
            H,
            logical_obs,
            noise_model,
            check_inds=check_inds,
            error_inds=error_inds,
            logical_inds=logical_inds,
            logical_tags=logical_tags,
            contract_noise_model=False,
            dtype=dtype,
            device=device,
        )

        # The parent picks cutensornet/numpy/cuda when a GPU is available.
        # Switch the backend to torch so that tensor data lives in the
        # autograd graph while cuTensorNet still handles the contraction.
        if self.contractor_config.contractor_name == "cutensornet" \
                and self.contractor_config.backend != "torch":
            self._set_contractor(
                "cutensornet",
                self.contractor_config.device,
                "torch",
                dtype,
            )

        self._syndrome_tags = [f"SYN_{i}" for i in range(len(self.check_inds))]

        self.syndrome_tn = tensor_network_from_syndrome_batch(
            syndrome_data,
            self.check_inds,
            batch_index="batch_index",
            tags=self._syndrome_tags,
        )
        self._batch_size = syndrome_data.shape[0]

        self.full_tn = TensorNetwork()
        self.full_tn = self.full_tn.combine(self.code_tn, virtual=True)
        self.full_tn = self.full_tn.combine(self.logical_tn, virtual=True)
        self.full_tn = self.full_tn.combine(self.syndrome_tn, virtual=True)
        self.full_tn = self.full_tn.combine(self.noise_model, virtual=True)

        self._set_tensor_type(self.syndrome_tn)

        torch_dtype = getattr(torch, self._dtype)
        self._noise_probs = torch.tensor(
            noise_model,
            dtype=torch_dtype,
            device=self.torch_device,
            requires_grad=True,
        )
        self._rebuild_noise_from_params()

        self.observable_flips = observable_flips

    # -- torch device helpers -------------------------------------------------

    @property
    def torch_device(self) -> torch.device:
        """The ``torch.device`` matching the contractor config."""
        if "cuda" in self.contractor_config.device:
            return torch.device(f"cuda:{self.contractor_config.device_id}",)
        return torch.device("cpu")

    def _set_tensor_type(self, tn: TensorNetwork) -> None:
        """Convert all tensor data in *tn* to torch on the correct device."""
        torch_dtype = getattr(torch, self._dtype)
        dev = self.torch_device

        def _to_torch(x):
            if isinstance(x, torch.Tensor):
                return x.to(device=dev, dtype=torch_dtype)
            return torch.tensor(
                np.asarray(x),
                dtype=torch_dtype,
                device=dev,
            )

        tn.apply_to_arrays(_to_torch)

    # -- observable_flips property --------------------------------------------

    @property
    def observable_flips(self) -> torch.Tensor:
        """Boolean tensor of observable flip outcomes."""
        return self._observable_flips

    @observable_flips.setter
    def observable_flips(self, value: Any) -> None:
        dev = self.torch_device
        if not isinstance(value, torch.Tensor):
            self._observable_flips = torch.tensor(
                value,
                dtype=torch.bool,
                device=dev,
            )
        else:
            self._observable_flips = value.bool().to(dev)
        self.obs_idx_true = torch.where(self._observable_flips)[0]
        self.obs_idx_false = torch.where(~self._observable_flips)[0]

    # -- noise model rebuild ---------------------------------------------------

    def _rebuild_noise_from_params(self) -> None:
        """Recompute noise model tensor data from :attr:`_noise_probs`.

        For each error index *i*, the corresponding noise tensor is set to
        ``[1 - p_i, p_i]`` where ``p_i = self._noise_probs[i]``.  Because
        ``_noise_probs`` carries ``requires_grad=True``, the resulting
        tensors remain in the torch autograd graph.
        """
        for i, t in enumerate(self.noise_model.tensors):
            p = self._noise_probs[i]
            t.modify(data=torch.stack([1 - p, p]))

    # -- noise_params property ------------------------------------------------

    @property
    def noise_params(self) -> list[torch.Tensor]:
        """The differentiable noise probability parameters.

        Returns a single-element list containing a 1-D tensor of scalar
        error probabilities, suitable for passing directly to a
        :class:`torch.optim.Optimizer`.
        """
        return [self._noise_probs]

    # -- forward pass / prediction --------------------------------------------

    def decoder_prediction(self) -> torch.Tensor:
        """Contract the full TN and return normalised predictions.

        Returns:
            Tensor of shape ``(shots, 2)`` where column 0 is P(no flip)
            and column 1 is P(flip).  The result is part of the torch
            autograd graph.
        """
        self._rebuild_noise_from_params()
        contraction_value = self.contractor_config.contractor(
            self.full_tn.get_equation(output_inds=("batch_index",
                                                   self.logical_obs_inds[0]),),
            self.full_tn.arrays,
            optimize=self.path_batch,
            slicing=self.slicing_batch,
            device_id=self.contractor_config.device_id,
        )
        return contraction_value / contraction_value.sum(dim=1, keepdim=True)

    # -- loss functions -------------------------------------------------------

    def cross_entropy_ler(self) -> torch.Tensor:
        """Cross-entropy loss over the syndrome batch.

        Returns:
            Differentiable scalar tensor.  Call ``.backward()`` to obtain
            gradients w.r.t. :attr:`noise_params`.
        """
        predictions = self.decoder_prediction()
        return (-torch.log(predictions[self.obs_idx_true, 1]).sum() -
                torch.log(predictions[self.obs_idx_false, 0]).sum())

    def logical_error_rate(self) -> float:
        """Logical error rate over the syndrome batch.

        Uses a hard argmax threshold so this is **not** differentiable.

        Returns:
            Fraction of shots decoded incorrectly.
        """
        with torch.no_grad():
            predictions = self.decoder_prediction()
            pred = predictions[:, 1] > predictions[:, 0]
            return float(1 - (pred == self._observable_flips).sum() /
                         len(self._observable_flips))

    # -- dataset replacement --------------------------------------------------

    def _update_data(
        self,
        new_syndrome_arrays: torch.Tensor,
        new_observable_flips: npt.NDArray[Any],
        enforce_shape: bool = True,
    ) -> None:
        """Update syndrome tensor data and observable flips in place.

        This is the fast path: ``new_syndrome_arrays`` must already be in
        the internal representation (output of
        :func:`prepare_syndrome_data_batch` converted to torch on the
        correct device).

        Args:
            new_syndrome_arrays: Shape ``(syndrome_length, shots, 2)``.
            new_observable_flips: Shape ``(shots,)``.
            enforce_shape: Assert that tensor shapes match.
        """
        for i, tag in enumerate(self._syndrome_tags):
            t = self.syndrome_tn.tensors[next(
                iter(self.syndrome_tn.tag_map[tag]))]
            if enforce_shape:
                assert t.data.shape == new_syndrome_arrays[i].shape, (
                    f"Shape mismatch for {tag}: "
                    f"{t.data.shape} vs {new_syndrome_arrays[i].shape}")
            t.modify(data=new_syndrome_arrays[i])
        self.observable_flips = new_observable_flips

    def update_dataset(
        self,
        new_syndrome_data: npt.NDArray[Any],
        new_observable_flips: npt.NDArray[Any],
        enforce_shape: bool = True,
    ) -> None:
        """Replace the syndrome batch and observable flips.

        Transforms raw syndrome data into the internal representation and
        updates the tensor network in place.

        Args:
            new_syndrome_data: Shape ``(shots, num_checks)``.
            new_observable_flips: Shape ``(shots,)``.
            enforce_shape: Assert tensor shape compatibility.  A shape
                change (different batch size) will require recomputing
                the contraction path.
        """
        syndrome_arrays = prepare_syndrome_data_batch(new_syndrome_data)
        torch_dtype = getattr(torch, self._dtype)
        syndrome_arrays = torch.tensor(
            syndrome_arrays,
            dtype=torch_dtype,
            device=self.torch_device,
        ).transpose(1, 2)
        self._update_data(
            syndrome_arrays,
            new_observable_flips,
            enforce_shape,
        )
