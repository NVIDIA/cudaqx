# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Differentiable noise learning for the tensor-network decoder.

:class:`NMOptimizer` fits a factorised per-error noise model to a
syndrome dataset by backpropagating through a torch-backed tensor-network
contraction.  :func:`make_compiled_step` is a convenience factory that
builds a no-arg callable for one Adam step in logit space.

The static noise-model builders (:func:`factorized_noise_model`,
:func:`error_pairs_noise_model`) live in :mod:`.noise_models`.
"""
from __future__ import annotations

import contextlib
import copy
import io
import math
import warnings
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import opt_einsum as oe
import torch
from torch.utils.checkpoint import checkpoint as _checkpoint
from quimb.tensor import TensorNetwork

from ..tensor_network_decoder import TensorNetworkDecoder
from .tensor_network_factory import (
    tensor_network_from_syndrome_batch,
    prepare_syndrome_data_batch,
)

_ASCII_POOL = ("abcdefghijklmnopqrstuvwxyz"
               "ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# Coarse for fp32 because ``1.0 - 1e-12`` rounds back to ``1.0``.
_PRIOR_EPS_BY_DTYPE: dict[str, float] = {
    "float64": 1e-12,
    "float32": 1e-6,
}
_SUPPORTED_DTYPES: tuple[str, ...] = ("float32", "float64")
_PATH_CACHE_MAXSIZE = 16
_path_cache: dict[tuple[str, tuple[tuple[int, ...], ...]], tuple[Any, Any]] = {}
_AUTO_PRECONTRACT_INTERMEDIATE_THRESHOLD = 2_147_483_647
_AUTO_PRECONTRACT_NUM_CHECKS_THRESHOLD = 128
_AUTO_PRECONTRACT_NUM_ERRORS_THRESHOLD = 512
_AUTO_REDUCED_LOSS_MICROBATCH_SIZE = 1
_AUTO_REDUCED_PREDICTION_MICROBATCH_SIZE = 1


def _validate_and_clamp_priors(noise_model: Any, dtype: str) -> list[float]:
    """Validate noise priors and clamp them into ``[eps, 1 - eps]``.

    The cross-entropy reduction floors log inputs so roundoff-induced
    zero or negative values do not create non-finite losses.  Priors at
    exactly ``0.0`` or ``1.0`` are still clamped because they can
    saturate loss terms and make gradients uninformative.  Stim DEMs
    occasionally emit ``p=1.0`` (deterministic detectors) or ``p<1e-15``
    (underflow), so we intercept here rather than force every caller to
    clamp.

    Behaviour mirrors :class:`torch.nn.BCELoss`-style stable wrappers:

      * Non-finite priors (``NaN`` / ``+/-inf``) raise ``ValueError`` -
        these indicate caller bugs, not numerical fragility, and
        silently coercing them would hide the real problem.
      * Out-of-range priors (``p <= eps`` or ``p >= 1 - eps``) are
        clamped into ``[eps, 1 - eps]`` and a single ``UserWarning``
        summarises the number of values changed.
      * In-range priors pass through unchanged with no warning.

    Args:
        noise_model: array-like of priors, length ``num_errors``.
        dtype: contraction dtype string (``"float32"`` / ``"float64"``).

    Returns:
        A plain ``list[float]`` so the base
        :class:`TensorNetworkDecoder` keeps using its existing
        list-based factorised noise model unchanged.
    """
    arr = np.asarray(noise_model, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"noise_model must be 1-D; got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        bad = np.where(~np.isfinite(arr))[0]
        raise ValueError(
            f"All priors must be finite; got non-finite values at error "
            f"indices {bad.tolist()}: {arr[bad].tolist()}")

    dtype_str = str(dtype)
    if dtype_str not in _PRIOR_EPS_BY_DTYPE:
        raise ValueError(f"Unsupported dtype {dtype_str!r}; "
                         f"expected one of {sorted(_PRIOR_EPS_BY_DTYPE)}.")
    eps = _PRIOR_EPS_BY_DTYPE[dtype_str]
    out_of_range = (arr < eps) | (arr > 1.0 - eps)
    if np.any(out_of_range):
        warnings.warn(
            f"Clamped {int(out_of_range.sum())}/{len(arr)} NMOptimizer "
            f"priors into [{eps}, {1.0 - eps}] for numerical stability; "
            f"values at or outside the (0, 1) boundary can saturate "
            f"cross-entropy terms and make gradients uninformative.",
            UserWarning,
            stacklevel=3,
        )
        arr = np.clip(arr, eps, 1.0 - eps)
    return arr.tolist()


def _clamp_log_input(x: torch.Tensor) -> torch.Tensor:
    """Floor log inputs after roundoff-induced non-positive values."""
    return x.clamp_min(torch.finfo(x.dtype).tiny)


def _finite_nonnegative(x: torch.Tensor) -> torch.Tensor:
    """Drop non-finite values and negative roundoff from probability weights."""
    return torch.nan_to_num(
        x,
        nan=0.0,
        posinf=torch.finfo(x.dtype).max,
        neginf=0.0,
    ).clamp_min(0.0)


def _normalize_prediction(out: torch.Tensor) -> torch.Tensor:
    """Normalize raw decoder weights into finite per-shot probabilities."""
    positive_inf = out == float("inf")
    has_positive_inf = positive_inf.any(dim=1, keepdim=True)
    finite_weights = torch.where(torch.isfinite(out), out,
                                 torch.zeros_like(out)).clamp_min(0.0)
    weights = torch.where(has_positive_inf, torch.zeros_like(finite_weights),
                          finite_weights)
    tiny = torch.finfo(weights.dtype).tiny
    scale = weights.max(dim=1, keepdim=True).values
    scaled = weights / scale.clamp_min(tiny)
    denom = scaled.sum(dim=1, keepdim=True)
    probs = scaled / denom.clamp_min(tiny)
    inf_probs = positive_inf.to(out.dtype) / positive_inf.sum(
        dim=1, keepdim=True).clamp_min(1).to(out.dtype)
    uniform = torch.full_like(weights, 1.0 / weights.shape[1])
    probs = torch.where(scale > tiny, probs, uniform)
    probs = torch.where(has_positive_inf, inf_probs, probs)
    probs = torch.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
    return probs.clamp(min=tiny, max=1.0)


def remap_eq_to_ascii(eq: str) -> str:
    """Rewrite an einsum equation so every label is in ``[a-zA-Z]``.

    Needed because :mod:`opt_einsum` can use non-ASCII unicode labels
    once the total index count exceeds 52, but :func:`torch.einsum`
    rejects them.  Raises if a single step has more than 52 distinct
    labels.
    """
    if eq.isascii():
        return eq
    if "->" in eq:
        lhs, rhs = eq.split("->")
    else:
        lhs, rhs = eq, None

    mapping: dict[str, str] = {}
    out_lhs_chars: list[str] = []
    for c in lhs:
        if c == ",":
            out_lhs_chars.append(c)
            continue
        if c not in mapping:
            if len(mapping) >= len(_ASCII_POOL):
                raise ValueError(
                    f"Einsum step '{eq}' has more than {len(_ASCII_POOL)} "
                    "distinct labels; cannot remap to ASCII.")
            mapping[c] = _ASCII_POOL[len(mapping)]
        out_lhs_chars.append(mapping[c])
    out_lhs = "".join(out_lhs_chars)
    if rhs is None:
        return out_lhs

    out_rhs_chars: list[str] = []
    for c in rhs:
        if c not in mapping:
            raise ValueError(
                f"Einsum step '{eq}' has output label {c!r} not present "
                "on the LHS; cannot remap.")
        out_rhs_chars.append(mapping[c])
    return f"{out_lhs}->{''.join(out_rhs_chars)}"


def _maybe_remap_eq_to_ascii(eq: str) -> str:
    """Return an ASCII equation when torch can represent it directly.

    More than 52 distinct labels cannot be encoded for
    :func:`torch.einsum`; in that case keep the original opt_einsum
    labels and let :func:`_einsum_torch` use its pairwise fallback.
    """
    try:
        return remap_eq_to_ascii(eq)
    except ValueError:
        return eq


def _einsum_label_count(eq: str) -> int:
    lhs = eq.split("->", 1)[0]
    return len({c for c in lhs if c != ","})


def _reshape(tensor: torch.Tensor, shape: list[int]) -> torch.Tensor:
    return tensor.reshape(tuple(shape))


def _prod(values: list[int]) -> int:
    return int(math.prod(values)) if values else 1


def _sum_unique_omitted_axes(tensor: torch.Tensor, labels: list[str],
                             other_labels: set[str],
                             rhs_labels: set[str]) -> torch.Tensor:
    drop_axes = [
        axis for axis, label in enumerate(labels)
        if label not in rhs_labels and label not in other_labels
    ]
    for axis in reversed(drop_axes):
        tensor = tensor.sum(dim=axis)
        labels.pop(axis)
    return tensor


def _permute_to(tensor: torch.Tensor, labels: list[str],
                target: list[str]) -> torch.Tensor:
    if labels == target:
        return tensor
    if not target:
        return tensor.reshape(())
    return tensor.permute([labels.index(label) for label in target])


def _einsum_pairwise_torch(eq: str, operands: tuple[torch.Tensor,
                                                    ...]) -> torch.Tensor:
    """Evaluate one opt_einsum pairwise step without torch label limits."""
    if "->" not in eq:
        raise ValueError(f"Expected explicit einsum output in {eq!r}.")
    lhs, rhs = eq.split("->", 1)
    terms = lhs.split(",")
    rhs_labels = list(rhs)
    rhs_set = set(rhs_labels)

    if len(terms) == 1 and len(operands) == 1:
        labels = list(terms[0])
        result = _sum_unique_omitted_axes(operands[0], labels, set(), rhs_set)
        return _permute_to(result, labels, rhs_labels)

    if len(terms) != 2 or len(operands) != 2:
        raise ValueError(
            "The high-label torch fallback only supports unary or pairwise "
            f"einsum steps; got equation {eq!r}.")

    a, b = operands
    labels_a = list(terms[0])
    labels_b = list(terms[1])
    set_a = set(labels_a)
    set_b = set(labels_b)

    a = _sum_unique_omitted_axes(a, labels_a, set_b, rhs_set)
    b = _sum_unique_omitted_axes(b, labels_b, set_a, rhs_set)

    batch_labels = [
        label for label in rhs_labels if label in labels_a and label in labels_b
    ]
    contract_labels = [
        label for label in labels_a
        if label in labels_b and label not in rhs_set
    ]
    a_free_labels = [
        label for label in labels_a
        if label not in batch_labels and label not in contract_labels
    ]
    b_free_labels = [
        label for label in labels_b
        if label not in batch_labels and label not in contract_labels
    ]

    sizes: dict[str, int] = {}
    for labels, tensor in ((labels_a, a), (labels_b, b)):
        for axis, label in enumerate(labels):
            size = int(tensor.shape[axis])
            if label in sizes and sizes[label] != size:
                raise ValueError(f"Mismatched dimension for label {label!r}: "
                                 f"{sizes[label]} vs {size}.")
            sizes[label] = size

    a_order = batch_labels + a_free_labels + contract_labels
    b_order = batch_labels + contract_labels + b_free_labels
    a = _permute_to(a, labels_a, a_order)
    b = _permute_to(b, labels_b, b_order)

    batch_shape = [sizes[label] for label in batch_labels]
    a_shape = [sizes[label] for label in a_free_labels]
    contract_shape = [sizes[label] for label in contract_labels]
    b_shape = [sizes[label] for label in b_free_labels]

    batch_size = _prod(batch_shape)
    a_size = _prod(a_shape)
    contract_size = _prod(contract_shape)
    b_size = _prod(b_shape)

    a_mat = _reshape(a, [batch_size, a_size, contract_size])
    b_mat = _reshape(b, [batch_size, contract_size, b_size])
    out = torch.bmm(a_mat, b_mat)

    current_labels = batch_labels + a_free_labels + b_free_labels
    out_shape = batch_shape + a_shape + b_shape
    out = out.reshape(tuple(out_shape) if out_shape else ())
    return _permute_to(out, current_labels, rhs_labels)


def _einsum_torch(eq: str, *operands: torch.Tensor) -> torch.Tensor:
    """Torch einsum with a pairwise fallback for >52 opt_einsum labels."""
    if _einsum_label_count(eq) <= len(_ASCII_POOL):
        return torch.einsum(remap_eq_to_ascii(eq), *operands)
    return _einsum_pairwise_torch(eq, operands)


def _path_largest_intermediate(info: Any) -> float:
    value = getattr(info, "largest_intermediate", None)
    if value is None:
        return float("inf")
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return float("inf")


def _path_opt_cost(info: Any) -> float:
    value = getattr(info, "opt_cost", None)
    if value is None:
        return float("inf")
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return float("inf")


def _select_default_torch_path(
    eq: str,
    shapes: tuple[tuple[int, ...], ...],
    *,
    tn: TensorNetwork | None = None,
    output_inds: tuple[str, ...] | None = None,
) -> tuple[Any, Any]:
    """Choose a deterministic torch-friendly default contraction path."""
    key = (eq, shapes)
    if key in _path_cache:
        return _path_cache[key]

    candidates: list[tuple[str, Any, Any]] = []

    optimizers: list[tuple[str, Any]] = [("greedy", "greedy"), ("auto", "auto")]
    try:
        import cotengra as ctg
    except ImportError:
        pass
    else:
        for attempt in range(3):
            optimizers.append((f"cotengra-{attempt}",
                               ctg.HyperOptimizer(max_repeats=8,
                                                  parallel=False)))

    for tag, optimize in optimizers:
        try:
            if tn is not None:
                if output_inds is None:
                    raise ValueError("output_inds is required with tn.")
                info = tn.contraction_info(output_inds=output_inds,
                                           optimize=optimize)
                path = info.path
            else:
                path, info = oe.contract_path(eq,
                                              *shapes,
                                              shapes=True,
                                              optimize=optimize)
        except Exception as exc:
            warnings.warn(
                f"NMOptimizer default path candidate {tag!r} failed: "
                f"{exc!r}",
                RuntimeWarning,
                stacklevel=3,
            )
            continue
        candidates.append((tag, path, info))

    if not candidates:
        raise RuntimeError("No NMOptimizer default contraction path "
                           "candidate succeeded.")

    _tag, selected_path, selected_info = min(
        candidates,
        key=lambda c: (_path_largest_intermediate(c[2]), _path_opt_cost(c[2])),
    )
    selected = (selected_path, selected_info)
    if len(_path_cache) >= _PATH_CACHE_MAXSIZE:
        _path_cache.pop(next(iter(_path_cache)))
    _path_cache[key] = selected
    return selected


class NMOptimizer(TensorNetworkDecoder):
    """Differentiable noise-model optimiser for the TN decoder.

    The factorised noise probabilities live in the torch autograd graph
    and are fit to a fixed syndrome batch by minimising the cross-entropy
    of the decoder's logical prediction against the observed flips.

    The forward pass is materialised once at construction and reused
    across optimisation steps.  Optionally call :meth:`optimize_path`
    (e.g. with ``cotengra.HyperOptimizer()``) to pin a better contraction
    path; the cached forward is rebuilt automatically.

    .. warning::

        Priors are clamped into ``[eps, 1 - eps]`` only at construction;
        an unconstrained optimiser step on :attr:`noise_params` can push
        them outside the probability interval.  The loss is floored for
        finiteness, but probability-space training can then saturate or
        optimise invalid probabilities.  Prefer logit-space training via
        :func:`make_compiled_step` (shown below), or clamp the tensor
        under :func:`torch.no_grad` after each step.

    Args:
        H: Parity check matrix, shape ``(num_checks, num_errors)``.
        logical_obs: Logical observable matrix, shape ``(1, num_errors)``.
        noise_model: Initial per-error probabilities, length ``num_errors``.
            Each value must be strictly in ``(0, 1)``; values at or
            outside the boundary (``p <= eps`` or ``p >= 1 - eps``,
            with ``eps`` dtype-dependent) are auto-clamped at
            construction with a :class:`UserWarning`.  Non-finite
            priors raise :class:`ValueError`.
        syndrome_data: Syndrome batch, shape ``(shots, num_checks)``.
        observable_flips: Observable flip outcomes, shape ``(shots,)``.
        check_inds, error_inds, logical_inds, logical_tags: Optional index
            and tag names; defaults track the parent decoder.
        dtype: Tensor data type (e.g. ``"float32"``).
        device: ``"cuda"`` (default) or ``"cpu"``.  ``NMOptimizer``
            always uses torch-backed contractions on this device; it
            does not dispatch through cuTensorNet.
        compile: If ``True``, wrap the forward in :func:`torch.compile`.
        execute: Forward backend. ``"opt_einsum"`` (default) dispatches
            via :func:`opt_einsum.contract_expression`; ``"unrolled"``
            walks the pairwise path with torch tensor operations;
            ``"codegen"`` partial-evaluates the path into a flat Python
            function.  All modes are torch/autograd paths and support
            CPU and CUDA tensors.
        compile_mode: Forwarded to :func:`torch.compile`; ignored when
            ``compile=False``.
        dynamic_syndromes: If ``True`` (default), syndromes are runtime
            arguments to the compiled forward, so :meth:`update_dataset`
            does not rebuild codegen when shapes are unchanged.
            ``False`` bakes syndromes into the generated closure and
            only affects ``execute="codegen"``.
        precontract_noise: If ``True``, defer the per-error
            noise-into-code contractions into differentiable torch ops
            and contract the reduced tensor network.  If ``"auto"``
            (default), use the full tensor network when the default path
            is small enough and switch to the reduced topology for large
            DEMs or when the full path's largest intermediate is too
            large.

    Example (logit-space, no clamping needed)::

        opt = NMOptimizer(H, logical_obs, priors,
                          syndrome_data, obs_flips)
        opt.optimize_path(optimize=ctg.HyperOptimizer())
        logits = torch.logit(opt.noise_params[0].detach()).requires_grad_()
        torch_opt = torch.optim.Adam([logits], lr=0.01)
        step = make_compiled_step(opt, logits, torch_opt)
        for _ in range(100):
            loss = step()
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
        *,
        compile: bool = False,
        execute: Literal["opt_einsum", "unrolled", "codegen"] = "opt_einsum",
        compile_mode: str | None = None,
        dynamic_syndromes: bool = True,
        precontract_noise: bool | Literal["auto"] = "auto",
    ) -> None:
        if execute not in ("opt_einsum", "unrolled", "codegen"):
            raise ValueError(f"Invalid execute mode: {execute!r}")
        if dtype not in _SUPPORTED_DTYPES:
            raise ValueError(f"Invalid dtype {dtype!r}; expected one of "
                             f"{list(_SUPPORTED_DTYPES)}.")
        if precontract_noise not in (False, True, "auto"):
            raise ValueError(
                "precontract_noise must be one of False, True, or 'auto'; "
                f"got {precontract_noise!r}.")

        # Sanitise once so the base TN tensors and ``self._noise_probs``
        # see identical values (see :func:`_validate_and_clamp_priors`).
        noise_model = _validate_and_clamp_priors(noise_model, dtype)

        requested_device = device
        requested_cuda = "cuda" in requested_device
        cuda_available = torch.cuda.is_available()
        if requested_cuda and not cuda_available:
            warnings.warn(
                "CUDA was requested for NMOptimizer, but torch CUDA is not "
                "available; using CPU for differentiable tensor-network "
                "contractions.",
                RuntimeWarning,
                stacklevel=2,
            )

        with contextlib.redirect_stdout(io.StringIO()):
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
                # NMOptimizer is always torch/autograd-backed.  Build the
                # parent topology on CPU first so construction never routes
                # through the base decoder's cuTensorNet default, then move
                # tensors to the requested torch device below.
                device="cpu",
            )

        target_device = (requested_device
                         if requested_cuda and cuda_available else "cpu")
        if (self.contractor_config.contractor_name != "oe_torch_compiled" or
                self.contractor_config.backend != "torch" or
                self.contractor_config.device != target_device):
            self._set_contractor("oe_torch_compiled", target_device, "torch",
                                 dtype)

        # Swap the base's placeholder single-syndrome TN for a batched one.
        self._syndrome_tags = [f"SYN_{i}" for i in range(len(self.check_inds))]
        self.syndrome_tn = tensor_network_from_syndrome_batch(
            syndrome_data,
            self.check_inds,
            batch_index="batch_index",
            tags=self._syndrome_tags,
        )
        self._batch_size = syndrome_data.shape[0]

        # Re-stitch ``full_tn`` around the batched syndrome TN.
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
        # The base's noise tensors stay in ``full_tn`` as numpy
        # placeholders: ``_snapshot_arrays_and_eq`` uses ``id()`` to
        # locate their positions, then ``self._noise_probs`` (autograd
        # live) is written into those slots.  Do not strip them.

        self._suspend_loss_rebuild = True
        self.observable_flips = observable_flips

        self._use_torch_compile = compile
        self._execute_mode = execute
        self._torch_compile_mode = compile_mode
        self._dynamic_syndromes = dynamic_syndromes
        self._precontract_noise_auto = precontract_noise == "auto"
        self._precontract_noise = precontract_noise is True
        self._reduced_optimize: Any = None
        self._reduced_info: Any = None
        self._compiled_predict: Any | None = None
        self._syndrome_tuple: tuple[torch.Tensor, ...] = ()
        self._snapshot_arrays_and_eq()
        self._suspend_loss_rebuild = False

    @property
    def torch_device(self) -> torch.device:
        """The ``torch.device`` matching the contractor config."""
        if "cuda" in self.contractor_config.device:
            return torch.device(f"cuda:{self.contractor_config.device_id}",)
        return torch.device("cpu")

    def _set_tensor_type(self, tn: TensorNetwork) -> None:
        """Move all tensor data in *tn* to torch on the configured device.

        Overrides the base ``autoray``-routed implementation so gradients
        flow through the noise-model tensors.
        """
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
        # The fused loss bakes ``obs_idx_true/false`` into its closure
        # and must be rebuilt when they change.  Skip when a full
        # snapshot rebuild is already pending (gated by
        # ``_suspend_loss_rebuild``) or before first ``__init__``.
        if (getattr(self, "_compiled_predict", None) is not None and
                not getattr(self, "_suspend_loss_rebuild", False)):
            self._compile_loss()

    @property
    def noise_params(self) -> list[torch.Tensor]:
        """Trainable noise probabilities, ready for ``torch.optim``.

        Clamped to ``[eps, 1 - eps]`` only at construction; an
        unconstrained step can push outside the probability interval.
        The next :meth:`cross_entropy_loss` remains finite, but training
        can saturate or optimise invalid probabilities.  See the class
        warning for safe training patterns.
        """
        return [self._noise_probs]

    def _snapshot_arrays_and_eq(self) -> None:
        self._eq_batch = self.full_tn.get_equation(
            output_inds=("batch_index", self.logical_obs_inds[0]))
        tensors = list(self.full_tn.tensors)
        self._tensors_ref = tensors

        noise_ids = {id(t) for t in self.noise_model.tensors}
        syndrome_ids = {id(t) for t in self.syndrome_tn.tensors}

        self._noise_pos_for_error: dict[str, int] = {}
        syndrome_positions_list: list[int] = []
        self._static_positions: list[int] = []

        for i, t in enumerate(tensors):
            if id(t) in noise_ids:
                self._noise_pos_for_error[t.inds[0]] = i
            elif id(t) in syndrome_ids:
                syndrome_positions_list.append(i)
            else:
                self._static_positions.append(i)

        # Guard against a future quimb that copies tensors on virtual
        # combine: every tensor in ``full_tn`` must classify into
        # exactly one bucket, else the predict path rebuilds the
        # operand list with a None slot or a misplaced placeholder.
        n_classified = (len(self._noise_pos_for_error) +
                        len(syndrome_positions_list) +
                        len(self._static_positions))
        assert n_classified == len(tensors)
        assert len(self._noise_pos_for_error) == len(self.error_inds)

        self._syndrome_positions: list[tuple[int, None]] = [
            (i, None) for i in syndrome_positions_list
        ]

        self._noise_pos_ordered = tuple(
            self._noise_pos_for_error[ei] for ei in self.error_inds)

        torch_dtype = getattr(torch, self._dtype)
        dev = self.torch_device

        def _as_torch(x):
            if isinstance(x, torch.Tensor):
                return x.detach().to(device=dev, dtype=torch_dtype)
            return torch.as_tensor(np.asarray(x), dtype=torch_dtype, device=dev)

        self._static_arrays: dict[int, torch.Tensor] = {
            i: _as_torch(self._tensors_ref[i].data)
            for i in self._static_positions
        }
        self._syndrome_arrays: list[torch.Tensor] = [
            _as_torch(self._tensors_ref[i].data)
            for i in syndrome_positions_list
        ]
        self._syndrome_tuple = tuple(self._syndrome_arrays)
        # Used by :meth:`_update_data` to detect layout changes that
        # invalidate the cached path / opt_einsum expression.
        self._syndrome_shapes: tuple[tuple[int, ...], ...] = tuple(
            tuple(s.shape) for s in self._syndrome_arrays)

        shapes = tuple(t.shape for t in tensors)
        if self._precontract_noise_auto:
            large_problem = (len(
                self.check_inds) >= _AUTO_PRECONTRACT_NUM_CHECKS_THRESHOLD or
                             len(self.error_inds)
                             >= _AUTO_PRECONTRACT_NUM_ERRORS_THRESHOLD)
            if large_problem:
                self._precontract_noise = True
            else:
                optimize, info = _select_default_torch_path(
                    self._eq_batch,
                    shapes,
                    tn=self.full_tn,
                    output_inds=("batch_index", self.logical_obs_inds[0]),
                )
                self._precontract_noise = (
                    _path_largest_intermediate(info)
                    > _AUTO_PRECONTRACT_INTERMEDIATE_THRESHOLD)
            if not self._precontract_noise:
                self.path_batch = optimize

        if self._precontract_noise:
            self._oe_expr = None
            self._build_reduced_tn_state()
            use_microbatch = (len(
                self.check_inds) >= _AUTO_PRECONTRACT_NUM_CHECKS_THRESHOLD or
                              len(self.error_inds)
                              >= _AUTO_PRECONTRACT_NUM_ERRORS_THRESHOLD)
            self._loss_microbatch_size = (_AUTO_REDUCED_LOSS_MICROBATCH_SIZE
                                          if use_microbatch else 0)
            self._prediction_microbatch_size = (
                _AUTO_REDUCED_PREDICTION_MICROBATCH_SIZE
                if use_microbatch else 0)
        elif self._execute_mode == "opt_einsum":
            self._loss_microbatch_size = 0
            self._prediction_microbatch_size = 0
            optimize = self.path_batch
            if optimize in (None, "auto"):
                optimize, _info = _select_default_torch_path(
                    self._eq_batch,
                    shapes,
                    tn=self.full_tn,
                    output_inds=("batch_index", self.logical_obs_inds[0]),
                )
                self.path_batch = optimize
            self._oe_expr = oe.contract_expression(
                self._eq_batch,
                *shapes,
                optimize=optimize,
            )
            self._path_steps = None
        else:
            self._oe_expr = None
            self._loss_microbatch_size = 0
            self._prediction_microbatch_size = 0
            optimize = self.path_batch
            if optimize in (None, "auto"):
                optimize, _info = _select_default_torch_path(
                    self._eq_batch,
                    shapes,
                    tn=self.full_tn,
                    output_inds=("batch_index", self.logical_obs_inds[0]),
                )
                self.path_batch = optimize
            _, info = oe.contract_path(
                self._eq_batch,
                *shapes,
                shapes=True,
                optimize=optimize,
            )
            self._path_steps = [(_maybe_remap_eq_to_ascii(step[2]),
                                 tuple(step[0]),
                                 tuple(sorted(step[0], reverse=True)))
                                for step in info.contraction_list]

        self._compile_predict()
        self._compile_loss()

    def _compile_predict(self) -> None:
        """Build ``self._predict_fn``."""
        if self._precontract_noise:
            self._predict_fn = self._build_predict_reduced()
        else:
            builders = {
                "opt_einsum": self._build_predict_opt_einsum,
                "unrolled": self._build_predict_unrolled,
                "codegen": self._build_predict_codegen,
            }
            self._predict_fn = builders[self._execute_mode]()
        self._compiled_predict = self._maybe_torch_compile(self._predict_fn,
                                                           kind="predict")

    def _build_predict_opt_einsum(self):
        """opt_einsum-backed predict: reuse the cached contract expression."""
        static_arrays = self._static_arrays
        syndrome_positions = tuple(p for p, _t in self._syndrome_positions)
        noise_pos_ordered = self._noise_pos_ordered
        n = len(self._tensors_ref)
        oe_expr = self._oe_expr

        def _predict(noise_probs: torch.Tensor,
                     syndrome_tuple: tuple[torch.Tensor, ...]) -> torch.Tensor:
            noise_stacked = torch.stack((1.0 - noise_probs, noise_probs),
                                        dim=-1)
            arrays: list[torch.Tensor] = [None] * n  # type: ignore
            for pos, arr in static_arrays.items():
                arrays[pos] = arr
            for pos, arr in zip(syndrome_positions, syndrome_tuple):
                arrays[pos] = arr
            for k, pos in enumerate(noise_pos_ordered):
                arrays[pos] = noise_stacked[k]
            # Torch backend is auto-selected from the tensor type;
            # avoids the per-call ``backend=`` dispatch.
            out = oe_expr(*arrays)
            return _normalize_prediction(out)

        return _predict

    def _build_predict_unrolled(self):
        """Unrolled predict: walk the cached pairwise contraction path."""
        static_arrays = self._static_arrays
        syndrome_positions = tuple(p for p, _t in self._syndrome_positions)
        noise_pos_ordered = self._noise_pos_ordered
        n = len(self._tensors_ref)
        path_steps = self._path_steps

        def _predict(noise_probs: torch.Tensor,
                     syndrome_tuple: tuple[torch.Tensor, ...]) -> torch.Tensor:
            noise_stacked = torch.stack((1.0 - noise_probs, noise_probs),
                                        dim=-1)
            ops: list[torch.Tensor] = [None] * n  # type: ignore
            for pos, arr in static_arrays.items():
                ops[pos] = arr
            for pos, arr in zip(syndrome_positions, syndrome_tuple):
                ops[pos] = arr
            for k, pos in enumerate(noise_pos_ordered):
                ops[pos] = noise_stacked[k]
            for eq_str, idxs, sorted_idxs in path_steps:
                picked = [ops[i] for i in idxs]
                for i in sorted_idxs:
                    ops.pop(i)
                ops.append(_einsum_torch(eq_str, *picked))
            out = ops[0]
            return _normalize_prediction(out)

        return _predict

    def _build_predict_codegen(self):
        """Codegen predict: partial-evaluated Python with named locals."""
        static_arrays = self._static_arrays
        syndrome_positions = tuple(p for p, _t in self._syndrome_positions)
        noise_pos_ordered = self._noise_pos_ordered
        n = len(self._tensors_ref)
        syndrome_tensors = list(self._syndrome_arrays)
        codegen_fn = self._build_codegen_predict(
            n,
            static_arrays,
            syndrome_positions,
            noise_pos_ordered,
            self._path_steps,
            syndrome_tensors,
            dynamic_syndromes=self._dynamic_syndromes,
        )
        self._codegen_fn = codegen_fn
        self._codegen_n_folded = getattr(codegen_fn, "_n_folded", 0)
        self._codegen_n_runtime = getattr(codegen_fn, "_n_runtime", 0)

        if self._dynamic_syndromes:
            return codegen_fn

        # Static mode bakes syndromes into the closure and returns a
        # 1-arg callable; wrap to match the public 2-arg signature.
        def _predict_static(
            noise_probs: torch.Tensor,
            syndrome_tuple: tuple[torch.Tensor, ...] = ()
        ) -> torch.Tensor:
            return codegen_fn(noise_probs)

        return _predict_static

    def _build_reduced_tn_state(self) -> None:
        """Build the reduced TN topology and differentiable noise recipes.

        This mirrors the parent decoder's contracted-noise topology, but
        keeps each per-error noise contraction as a torch operation so
        gradients still flow to ``noise_probs``.
        """
        from collections import defaultdict

        error_inds_set = set(self.error_inds)

        survivor_lookup: dict[tuple[tuple[str, ...], frozenset[str]], int] = {}
        doomed_lookup: dict[tuple[tuple[str, ...], frozenset[str]], int] = {}
        for opt_pos, tensor in enumerate(self._tensors_ref):
            key = (tuple(tensor.inds), frozenset(tensor.tags))
            if any(ind in error_inds_set for ind in tensor.inds):
                doomed_lookup[key] = opt_pos
            else:
                survivor_lookup[key] = opt_pos

        reduced_tn = self.full_tn.copy()
        recipes: list[dict[str, Any]] = []
        merged_id_to_recipe_idx: dict[int, int] = {}

        for error_idx, error_ind in enumerate(self.error_inds):
            doomed = [t for t in reduced_tn.tensors if error_ind in t.inds]
            check_tensors = [t for t in doomed if "NOISE" not in t.tags]
            check_opt_positions = [
                doomed_lookup[(tuple(t.inds), frozenset(t.tags))]
                for t in check_tensors
            ]

            ids_before = {id(t) for t in reduced_tn.tensors}
            reduced_tn.contract_ind(error_ind)
            new_tensors = [
                t for t in reduced_tn.tensors if id(t) not in ids_before
            ]
            assert len(new_tensors) == 1
            new_tensor = new_tensors[0]
            merged_id_to_recipe_idx[id(new_tensor)] = error_idx

            quimb_out_inds = tuple(new_tensor.inds)
            mapping = {error_ind: "e"}
            next_code = ord("a")
            for ind in quimb_out_inds:
                while chr(next_code) == "e":
                    next_code += 1
                mapping[ind] = chr(next_code)
                next_code += 1

            noise_str = mapping[error_ind]
            check_strs = [
                "".join(mapping[ind] for ind in t.inds) for t in check_tensors
            ]
            out_str = "".join(mapping[ind] for ind in quimb_out_inds)
            ordered_check_opt_positions: list[int] = [None
                                                     ] * len(  # type: ignore
                                                         check_tensors)
            for tensor, opt_pos in zip(check_tensors, check_opt_positions):
                non_error_ind = next(
                    ind for ind in tensor.inds if ind != error_ind)
                ordered_check_opt_positions[quimb_out_inds.index(
                    non_error_ind)] = opt_pos

            recipes.append({
                "eq": ",".join([noise_str] + check_strs) + "->" + out_str,
                "ordered_check_opt_positions": ordered_check_opt_positions,
                "k": len(check_tensors),
            })

        reduced_eq = reduced_tn.get_equation(
            output_inds=("batch_index", self.logical_obs_inds[0]))
        reduced_shapes = tuple(t.shape for t in reduced_tn.tensors)

        reduced_static: dict[int, torch.Tensor] = {}
        reduced_syndrome: list[tuple[int, int]] = []
        reduced_recipes: dict[int, int] = {}
        syn_pos_to_idx = {
            p: i for i, (p, _) in enumerate(self._syndrome_positions)
        }
        for pos, tensor in enumerate(reduced_tn.tensors):
            if id(tensor) in merged_id_to_recipe_idx:
                reduced_recipes[pos] = merged_id_to_recipe_idx[id(tensor)]
                continue

            key = (tuple(tensor.inds), frozenset(tensor.tags))
            opt_pos = survivor_lookup[key]
            if opt_pos in self._static_arrays:
                reduced_static[pos] = self._static_arrays[opt_pos]
            elif opt_pos in syn_pos_to_idx:
                reduced_syndrome.append((pos, syn_pos_to_idx[opt_pos]))
            else:
                raise AssertionError(
                    f"Reduced tensor at position {pos} maps to full tensor "
                    f"position {opt_pos}, which is not static or syndrome.")

        user_optimize = self._reduced_optimize
        if user_optimize is not None and (
                type(user_optimize).__module__.startswith("cuquantum") and
                type(user_optimize).__name__ == "OptimizerOptions"):
            raise ValueError(
                "precontract_noise=True does not support cuTensorNet "
                "OptimizerOptions; pass an opt_einsum/cotengra optimizer.")

        def _path_largest_intermediate(info: Any) -> float:
            value = getattr(info, "largest_intermediate", None)
            if value is None:
                return float("inf")
            try:
                return float(value)
            except (TypeError, ValueError, OverflowError):
                return float("inf")

        def _path_opt_cost(info: Any) -> float:
            value = getattr(info, "opt_cost", None)
            if value is None:
                return float("inf")
            try:
                return float(value)
            except (TypeError, ValueError, OverflowError):
                return float("inf")

        candidates: list[tuple[str, Any, Any]] = []

        def _try_path(tag: str, optimize: Any) -> None:
            try:
                path, info = oe.contract_path(reduced_eq,
                                              *reduced_shapes,
                                              shapes=True,
                                              optimize=optimize)
            except Exception as exc:
                warnings.warn(
                    f"Reduced TN path candidate {tag!r} failed: {exc!r}",
                    RuntimeWarning,
                    stacklevel=3,
                )
                return
            candidates.append((tag, path, info))

        if user_optimize is None:
            reduced_path, reduced_info = _select_default_torch_path(
                reduced_eq, reduced_shapes)
            selected_tag = "default"
        else:
            _try_path("user", user_optimize)
            _try_path("auto", "auto")
            _try_path("greedy", "greedy")
            try:
                import cotengra as ctg
            except ImportError:
                pass
            else:
                for attempt in range(3):
                    _try_path(
                        f"cotengra-{attempt}",
                        ctg.HyperOptimizer(max_repeats=8, parallel=False),
                    )

            if not candidates:
                raise RuntimeError("No reduced TN contraction path candidates "
                                   "succeeded.")

            selected_tag, reduced_path, reduced_info = min(
                candidates,
                key=lambda c:
                (_path_largest_intermediate(c[2]), _path_opt_cost(c[2])),
            )
            if selected_tag != "user":
                user_info = next(
                    (info for tag, _path, info in candidates if tag == "user"),
                    None)
                if user_info is None:
                    message = (
                        "Reduced TN path finder selected "
                        f"{selected_tag!r} because the user-supplied optimizer "
                        "did not produce a valid path.")
                else:
                    message = (
                        "Reduced TN path finder selected "
                        f"{selected_tag!r} instead of the user-supplied "
                        "optimizer because it had a smaller largest "
                        "intermediate "
                        f"({_path_largest_intermediate(reduced_info):.3e} vs "
                        f"{_path_largest_intermediate(user_info):.3e}).")
                warnings.warn(message, UserWarning, stacklevel=3)
        reduced_oe_expr = oe.contract_expression(reduced_eq,
                                                 *reduced_shapes,
                                                 optimize=reduced_path)

        recipe_to_reduced_pos = {ri: pos for pos, ri in reduced_recipes.items()}
        groups_by_k: dict[int, list[int]] = defaultdict(list)
        for recipe_idx, recipe in enumerate(recipes):
            groups_by_k[recipe["k"]].append(recipe_idx)

        batched_groups: list[dict[str, Any]] = []
        device = self.torch_device
        for k, error_indices in sorted(groups_by_k.items()):
            out_letters: list[str] = []
            next_code = ord("a")
            for _ in range(k):
                while chr(next_code) in ("e", "n"):
                    next_code += 1
                out_letters.append(chr(next_code))
                next_code += 1

            out_str = "".join(out_letters)
            if k == 0:
                eq = "ne->ne"
            else:
                check_strs = [f"n{letter}e" for letter in out_letters]
                eq = "ne," + ",".join(check_strs) + "->n" + out_str

            stacked_checks = []
            for axis in range(k):
                axis_arrays = [
                    self._static_arrays[recipes[ri]
                                        ["ordered_check_opt_positions"][axis]]
                    for ri in error_indices
                ]
                stacked_checks.append(torch.stack(axis_arrays, dim=0))

            batched_groups.append({
                "k":
                    k,
                "eq":
                    eq,
                "error_indices_t":
                    torch.tensor(error_indices, dtype=torch.long,
                                 device=device),
                "stacked_checks":
                    stacked_checks,
                "reduced_positions": [
                    recipe_to_reduced_pos[ri] for ri in error_indices
                ],
            })

        self._reduced_tn = reduced_tn
        self._batched_einsum_groups = batched_groups
        self._reduced_static_positions = reduced_static
        self._reduced_syndrome_positions = reduced_syndrome
        self._reduced_eq = reduced_eq
        self._reduced_oe_expr = reduced_oe_expr
        self._reduced_n_tensors = len(reduced_tn.tensors)
        self._reduced_path_steps = [(_maybe_remap_eq_to_ascii(step[2]),
                                     tuple(step[0]),
                                     tuple(sorted(step[0], reverse=True)))
                                    for step in reduced_info.contraction_list]
        self._reduced_dynamic_positions = tuple(
            pos for pos in range(len(reduced_tn.tensors))
            if pos not in reduced_static)
        self._reduced_info = reduced_info
        self._reduced_path_tag = selected_tag
        self.path_batch = reduced_path
        self.slicing_batch = tuple()

    def _build_predict_reduced(self):
        """Predict using the reduced TN plus batched noise precontraction."""
        static_positions = self._reduced_static_positions
        syndrome_positions = self._reduced_syndrome_positions
        batched_groups = self._batched_einsum_groups
        oe_expr = self._reduced_oe_expr
        path_steps = self._reduced_path_steps
        dynamic_positions = self._reduced_dynamic_positions
        n = self._reduced_n_tensors

        if self._execute_mode == "codegen":
            codegen_contract = self._build_codegen_contract(
                n,
                static_positions,
                dynamic_positions,
                path_steps,
            )
        else:
            codegen_contract = None

        def _materialize_arrays(
            noise_probs: torch.Tensor,
            syndrome_tuple: tuple[torch.Tensor, ...],
        ) -> list[torch.Tensor]:
            noise_stacked = torch.stack((1.0 - noise_probs, noise_probs),
                                        dim=-1)
            arrays: list[torch.Tensor] = [None] * n  # type: ignore
            for pos, arr in static_positions.items():
                arrays[pos] = arr
            for pos, syndrome_idx in syndrome_positions:
                arrays[pos] = syndrome_tuple[syndrome_idx]
            for group in batched_groups:
                noise_batch = noise_stacked[group["error_indices_t"]]
                if group["k"] == 0:
                    out_batch = noise_batch
                else:
                    out_batch = _einsum_torch(group["eq"], noise_batch,
                                              *group["stacked_checks"])
                for i, pos in enumerate(group["reduced_positions"]):
                    arrays[pos] = out_batch[i]
            return arrays

        def _contract_unrolled(arrays: list[torch.Tensor]) -> torch.Tensor:
            ops = list(arrays)
            for eq_str, idxs, sorted_idxs in path_steps:
                picked = [ops[i] for i in idxs]
                for i in sorted_idxs:
                    ops.pop(i)
                ops.append(_einsum_torch(eq_str, *picked))
            return ops[0]

        def _predict(noise_probs: torch.Tensor,
                     syndrome_tuple: tuple[torch.Tensor, ...]) -> torch.Tensor:
            arrays = _materialize_arrays(noise_probs, syndrome_tuple)
            if self._execute_mode == "opt_einsum" and torch.is_grad_enabled(
            ) and noise_probs.requires_grad:
                out = _checkpoint(oe_expr, *arrays, use_reentrant=False)
            elif self._execute_mode == "opt_einsum":
                out = oe_expr(*arrays)
            elif self._execute_mode == "unrolled":
                out = _contract_unrolled(arrays)
            else:
                dyns = tuple(arrays[pos] for pos in dynamic_positions)
                out = codegen_contract(dyns)
            return _normalize_prediction(out)

        return _predict

    def _maybe_torch_compile(self, fn, *, kind: str):
        """Wrap ``fn`` with :func:`torch.compile` if requested.

        On any compile failure, warn and fall back to eager.  ``kind``
        is included in the warning to disambiguate predict vs loss.
        """
        if not self._use_torch_compile:
            return fn
        try:
            kwargs = self._torch_compile_kwargs()
            return torch.compile(fn, **kwargs)
        except Exception as exc:  # pragma: no cover
            warnings.warn(
                f"torch.compile {kind} failed ({exc!r}); "
                "falling back to eager.",
                RuntimeWarning,
                stacklevel=2,
            )
            return fn

    def _compile_loss(self) -> None:
        """Build the ``(input, syndromes) -> scalar_loss`` callables.

        Two variants are produced: one accepting logits (sigmoid applied
        inside) and one accepting probabilities directly.
        """
        if self._execute_mode == "codegen" and not self._precontract_noise:
            logits_fn, probs_fn = self._build_loss_codegen()
        else:
            logits_fn, probs_fn = self._build_loss_wrapped()

        self._loss_from_logits_fn = logits_fn
        self._loss_from_probs_fn = probs_fn
        self._compiled_loss_from_logits = self._maybe_torch_compile(logits_fn,
                                                                    kind="loss")
        self._compiled_loss_from_probs = self._maybe_torch_compile(probs_fn,
                                                                   kind="loss")

    def _build_loss_codegen(self):
        """Codegen loss: fuse the CE reduction into the contraction graph."""
        static_arrays = self._static_arrays
        syndrome_positions = tuple(p for p, _t in self._syndrome_positions)
        noise_pos_ordered = self._noise_pos_ordered
        n = len(self._tensors_ref)
        syndrome_tensors = list(self._syndrome_arrays)

        codegen_logits = self._build_codegen_loss(
            n,
            static_arrays,
            syndrome_positions,
            noise_pos_ordered,
            self._path_steps,
            syndrome_tensors,
            obs_idx_true=self.obs_idx_true,
            obs_idx_false=self.obs_idx_false,
            dynamic_syndromes=self._dynamic_syndromes,
            from_logits=True,
        )
        codegen_probs = self._build_codegen_loss(
            n,
            static_arrays,
            syndrome_positions,
            noise_pos_ordered,
            self._path_steps,
            syndrome_tensors,
            obs_idx_true=self.obs_idx_true,
            obs_idx_false=self.obs_idx_false,
            dynamic_syndromes=self._dynamic_syndromes,
            from_logits=False,
        )

        if self._dynamic_syndromes:
            return codegen_logits, codegen_probs

        # Static codegen bakes syndromes into the closure and returns a
        # 1-arg callable; wrap to match the public 2-arg signature.
        def _loss_from_logits_static(
            logits: torch.Tensor, syndrome_tuple: tuple[torch.Tensor, ...] = ()
        ) -> torch.Tensor:
            return codegen_logits(logits)

        def _loss_from_probs_static(
            noise_probs: torch.Tensor,
            syndrome_tuple: tuple[torch.Tensor, ...] = ()
        ) -> torch.Tensor:
            return codegen_probs(noise_probs)

        return _loss_from_logits_static, _loss_from_probs_static

    def _build_loss_wrapped(self):
        """opt_einsum / unrolled loss: wrap CE around ``self._predict_fn``."""
        obs_t = self.obs_idx_true
        obs_f = self.obs_idx_false
        obs_all = self._observable_flips
        predict_fn = self._predict_fn
        microbatch_size = getattr(self, "_loss_microbatch_size", 0)

        if self._precontract_noise and microbatch_size > 0:
            batch_size = self._batch_size
            chunks: list[tuple[int, int, torch.Tensor, torch.Tensor]] = []
            for start in range(0, batch_size, microbatch_size):
                end = min(start + microbatch_size, batch_size)
                obs_chunk = obs_all[start:end]
                chunks.append((start, end, torch.where(obs_chunk)[0],
                               torch.where(~obs_chunk)[0]))

            def _slice_syndromes(syndromes, start, end):
                return tuple(s[:, start:end] for s in syndromes)

            def _ce_from_prediction(p, obs_t_local, obs_f_local):
                return (-torch.log(_clamp_log_input(p[obs_t_local, 1])).sum() -
                        torch.log(_clamp_log_input(p[obs_f_local, 0])).sum())

            def _loss_from_probs(noise_probs, syndromes):
                total = noise_probs.new_zeros(())
                for start, end, obs_t_local, obs_f_local in chunks:
                    p = predict_fn(noise_probs,
                                   _slice_syndromes(syndromes, start, end))
                    total = total + _ce_from_prediction(p, obs_t_local,
                                                        obs_f_local)
                return total

            def _loss_from_logits(logits, syndromes):
                noise_probs = torch.sigmoid(logits)
                total = noise_probs.new_zeros(())
                for start, end, obs_t_local, obs_f_local in chunks:
                    p = predict_fn(noise_probs,
                                   _slice_syndromes(syndromes, start, end))
                    total = total + _ce_from_prediction(p, obs_t_local,
                                                        obs_f_local)
                return total

            return _loss_from_logits, _loss_from_probs

        if (self._execute_mode != "codegen" or self._dynamic_syndromes or
                self._precontract_noise):

            def _loss_from_probs(noise_probs, syndromes):
                p = predict_fn(noise_probs, syndromes)
                return (-torch.log(_clamp_log_input(p[obs_t, 1])).sum() -
                        torch.log(_clamp_log_input(p[obs_f, 0])).sum())

            def _loss_from_logits(logits, syndromes):
                p = predict_fn(torch.sigmoid(logits), syndromes)
                return (-torch.log(_clamp_log_input(p[obs_t, 1])).sum() -
                        torch.log(_clamp_log_input(p[obs_f, 0])).sum())
        else:

            def _loss_from_probs(noise_probs, syndromes=()):
                p = predict_fn(noise_probs, ())
                return (-torch.log(_clamp_log_input(p[obs_t, 1])).sum() -
                        torch.log(_clamp_log_input(p[obs_f, 0])).sum())

            def _loss_from_logits(logits, syndromes=()):
                p = predict_fn(torch.sigmoid(logits), ())
                return (-torch.log(_clamp_log_input(p[obs_t, 1])).sum() -
                        torch.log(_clamp_log_input(p[obs_f, 0])).sum())

        return _loss_from_logits, _loss_from_probs

    def _torch_compile_kwargs(self) -> dict[str, Any]:
        """Build kwargs for :func:`torch.compile`.

        Defaults to ``mode="reduce-overhead"`` on CUDA so kernel-launch
        overhead is amortised via CUDA Graphs; a ``compile_mode=...``
        passed to the constructor overrides this.
        """
        kwargs: dict[str, Any] = {"dynamic": False}
        if self._torch_compile_mode is not None:
            kwargs["mode"] = self._torch_compile_mode
        elif self.torch_device.type == "cuda":
            kwargs["mode"] = "reduce-overhead"
        return kwargs

    @staticmethod
    def _codegen_partial_eval(n, static_arrays, syndrome_positions,
                              noise_pos_ordered, path_steps, syndrome_tensors,
                              dynamic_syndromes: bool):
        """Partial-evaluate ``path_steps`` for the codegen builders."""
        static_positions = sorted(static_arrays.keys())
        noise_pos_set = set(noise_pos_ordered)
        syn_pos_set = set(syndrome_positions)
        noise_pos_to_k = {pos: k for k, pos in enumerate(noise_pos_ordered)}
        syn_pos_to_sidx = {
            pos: sidx for sidx, pos in enumerate(syndrome_positions)
        }
        static_pos_to_sidx = {
            pos: sidx for sidx, pos in enumerate(static_positions)
        }

        state: list[tuple[str, bool, torch.Tensor | None]] = []
        for pos in range(n):
            if pos in noise_pos_set:
                k = noise_pos_to_k[pos]
                state.append((f"_n{k}", True, None))
            elif pos in syn_pos_set:
                sidx = syn_pos_to_sidx[pos]
                if dynamic_syndromes:
                    state.append((f"_S{sidx}", True, None))
                else:
                    state.append((f"_S{sidx}", False, syndrome_tensors[sidx]))
            else:
                sidx = static_pos_to_sidx[pos]
                state.append(
                    (f"_C{sidx}", False, static_arrays[static_positions[sidx]]))

        closure_vars: dict[str, torch.Tensor] = {}
        runtime_lines: list[str] = []
        used_static: set[str] = set()
        n_folded = 0

        for step_idx, step in enumerate(path_steps):
            eq_str = step[0]
            idxs = step[1]
            picked = [state[i] for i in idxs]
            for i in sorted(idxs, reverse=True):
                state.pop(i)
            any_dynamic = any(p[1] for p in picked)
            out_name = f"_r{step_idx}"
            if not any_dynamic:
                arrs = [p[2] for p in picked]
                with torch.no_grad():
                    result = _einsum_torch(eq_str, *arrs).contiguous()
                static_name = f"_P{step_idx}"
                closure_vars[static_name] = result
                state.append((static_name, False, result))
                n_folded += 1
            else:
                arg_names = [p[0] for p in picked]
                for name in arg_names:
                    if name.startswith(("_C", "_P")):
                        used_static.add(name)
                    elif name.startswith("_S") and not dynamic_syndromes:
                        used_static.add(name)
                runtime_lines.append(
                    f"    {out_name} = _einsum_torch({eq_str!r}, "
                    f"{', '.join(arg_names)})")
                state.append((out_name, True, None))

        assert len(state) == 1
        for name in used_static:
            if name in closure_vars:
                continue
            if name.startswith("_C"):
                sidx = int(name[2:])
                closure_vars[name] = static_arrays[static_positions[sidx]]
            elif name.startswith("_S"):
                sidx = int(name[2:])
                closure_vars[name] = syndrome_tensors[sidx]

        return runtime_lines, closure_vars, used_static, state[0], n_folded

    @staticmethod
    def _emit_noise_header(noise_pos_ordered,
                           transform: str = "identity") -> list[str]:
        """Emit source lines materialising ``_n0 .. _n{K-1}``."""
        lines: list[str] = []
        if transform == "sigmoid":
            lines.append("    _p = torch.sigmoid(noise_probs)")
        else:
            lines.append("    _p = noise_probs")
        lines.append("    _q = 1.0 - _p")
        lines.append("    _NS = torch.stack((_q, _p), dim=1)")
        for k in range(len(noise_pos_ordered)):
            lines.append(f"    _n{k} = _NS[{k}]")
        return lines

    @staticmethod
    def _emit_syndrome_header(syndrome_positions,
                              dynamic_syndromes: bool) -> list[str]:
        """Emit source lines binding runtime syndrome arguments."""
        if not dynamic_syndromes:
            return []
        return [
            f"    _S{sidx} = syndromes[{sidx}]"
            for sidx in range(len(syndrome_positions))
        ]

    @classmethod
    def _build_codegen_contract(cls, n, static_arrays, dynamic_positions,
                                path_steps):
        """Generate ``_contract(dyns)`` for a static/dynamic operand list."""
        static_positions = sorted(static_arrays.keys())
        dynamic_positions = tuple(dynamic_positions)
        dynamic_pos_to_idx = {
            pos: idx for idx, pos in enumerate(dynamic_positions)
        }
        static_pos_to_idx = {
            pos: idx for idx, pos in enumerate(static_positions)
        }

        state: list[tuple[str, bool, torch.Tensor | None]] = []
        for pos in range(n):
            if pos in dynamic_pos_to_idx:
                state.append((f"_D{dynamic_pos_to_idx[pos]}", True, None))
            else:
                sidx = static_pos_to_idx[pos]
                state.append(
                    (f"_C{sidx}", False, static_arrays[static_positions[sidx]]))

        closure_vars: dict[str, torch.Tensor] = {}
        runtime_lines: list[str] = []
        used_static: set[str] = set()
        n_folded = 0

        for step_idx, step in enumerate(path_steps):
            eq_str = step[0]
            idxs = step[1]
            picked = [state[i] for i in idxs]
            for i in sorted(idxs, reverse=True):
                state.pop(i)
            any_dynamic = any(p[1] for p in picked)
            out_name = f"_r{step_idx}"
            if not any_dynamic:
                arrs = [p[2] for p in picked]
                with torch.no_grad():
                    result = _einsum_torch(eq_str, *arrs).contiguous()
                static_name = f"_P{step_idx}"
                closure_vars[static_name] = result
                state.append((static_name, False, result))
                n_folded += 1
            else:
                arg_names = [p[0] for p in picked]
                for name in arg_names:
                    if name.startswith(("_C", "_P")):
                        used_static.add(name)
                runtime_lines.append(
                    f"    {out_name} = _einsum_torch({eq_str!r}, "
                    f"{', '.join(arg_names)})")
                state.append((out_name, True, None))

        assert len(state) == 1
        for name in used_static:
            if name in closure_vars:
                continue
            if name.startswith("_C"):
                sidx = int(name[2:])
                closure_vars[name] = static_arrays[static_positions[sidx]]

        body = ["def _contract(dyns):"]
        body.extend(f"    _D{idx} = dyns[{idx}]"
                    for idx in range(len(dynamic_positions)))
        body.extend(runtime_lines)
        final_name, is_final_dyn, final_value = state[0]
        if is_final_dyn:
            body.append(f"    return {final_name}")
        else:
            closure_vars["_FINAL"] = final_value
            body.append("    return _FINAL")

        return cls._compile_codegen_source(body, closure_vars, n_folded,
                                           len(runtime_lines), "contract")

    @classmethod
    def _build_codegen_predict(cls,
                               n,
                               static_arrays,
                               syndrome_positions,
                               noise_pos_ordered,
                               path_steps,
                               syndrome_tensors,
                               dynamic_syndromes: bool = True):
        """Generate ``_predict(noise_probs[, syndromes]) -> (shots, 2)``."""
        runtime_lines, closure_vars, _used, final_state, n_folded = (
            cls._codegen_partial_eval(
                n,
                static_arrays,
                syndrome_positions,
                noise_pos_ordered,
                path_steps,
                syndrome_tensors,
                dynamic_syndromes,
            ))
        final_name, is_final_dyn, final_value = final_state
        fully_static = not is_final_dyn

        body: list[str] = []
        if dynamic_syndromes:
            body.append("def _predict(noise_probs, syndromes):")
        else:
            body.append("def _predict(noise_probs):")

        if fully_static:
            with torch.no_grad():
                normed = _normalize_prediction(final_value)
            closure_vars["_FINAL"] = normed
            body.append("    return _FINAL")
            runtime_lines = []
        else:
            body.extend(
                cls._emit_noise_header(noise_pos_ordered, transform="identity"))
            body.extend(
                cls._emit_syndrome_header(syndrome_positions,
                                          dynamic_syndromes))
            body.extend(runtime_lines)
            body.append(f"    _out = {final_name}")
            body.append("    return _normalize_prediction(_out)")

        return cls._compile_codegen_source(body, closure_vars, n_folded,
                                           len(runtime_lines), "predict")

    @classmethod
    def _build_codegen_loss(cls,
                            n,
                            static_arrays,
                            syndrome_positions,
                            noise_pos_ordered,
                            path_steps,
                            syndrome_tensors,
                            obs_idx_true: torch.Tensor,
                            obs_idx_false: torch.Tensor,
                            dynamic_syndromes: bool = True,
                            from_logits: bool = True):
        """Generate a fused ``(input, syndromes) -> scalar`` loss callable."""
        runtime_lines, closure_vars, _used, final_state, n_folded = (
            cls._codegen_partial_eval(
                n,
                static_arrays,
                syndrome_positions,
                noise_pos_ordered,
                path_steps,
                syndrome_tensors,
                dynamic_syndromes,
            ))
        final_name, is_final_dyn, final_value = final_state
        fully_static = not is_final_dyn

        closure_vars["_OBS_T"] = obs_idx_true
        closure_vars["_OBS_F"] = obs_idx_false

        body: list[str] = []
        if dynamic_syndromes:
            body.append("def _loss(noise_probs, syndromes):")
        else:
            body.append("def _loss(noise_probs):")

        if fully_static:
            with torch.no_grad():
                normed = _normalize_prediction(final_value)
                ce = (
                    -torch.log(_clamp_log_input(normed[obs_idx_true, 1])).sum()
                    -
                    torch.log(_clamp_log_input(normed[obs_idx_false, 0])).sum())
            closure_vars["_LOSS"] = ce
            body.append("    return _LOSS + 0.0 * noise_probs.sum()")
            runtime_lines = []
        else:
            transform = "sigmoid" if from_logits else "identity"
            body.extend(cls._emit_noise_header(noise_pos_ordered, transform))
            body.extend(
                cls._emit_syndrome_header(syndrome_positions,
                                          dynamic_syndromes))
            body.extend(runtime_lines)
            body.append(f"    _out = {final_name}")
            body.append("    _p = _normalize_prediction(_out)")
            body.append(
                "    return (-torch.log(_clamp_log_input(_p[_OBS_T, 1])).sum() "
                "- torch.log(_clamp_log_input(_p[_OBS_F, 0])).sum())")

        return cls._compile_codegen_source(body, closure_vars, n_folded,
                                           len(runtime_lines), "loss")

    @staticmethod
    def _compile_codegen_source(body: list[str],
                                closure_vars: dict[str, torch.Tensor],
                                n_folded: int, n_runtime: int, kind: str):
        """Compile the assembled function source and return the callable."""
        source = "\n".join(body)
        ns: dict[str, Any] = {
            "torch": torch,
            "_einsum_torch": _einsum_torch,
            "_clamp_log_input": _clamp_log_input,
            "_finite_nonnegative": _finite_nonnegative,
            "_normalize_prediction": _normalize_prediction,
        }
        ns.update(closure_vars)
        fn_name = {
            "contract": "_contract",
            "loss": "_loss",
            "predict": "_predict",
        }[kind]
        exec(compile(source, f"<nm_compiled_{kind}>", "exec"), ns)
        fn = ns[fn_name]
        fn._n_folded = n_folded  # type: ignore[attr-defined]
        fn._n_runtime = n_runtime  # type: ignore[attr-defined]
        return fn

    def decoder_prediction(self) -> torch.Tensor:
        """Run the forward pass; returns ``(shots, 2)`` predictions."""
        microbatch_size = getattr(self, "_prediction_microbatch_size", 0)
        if microbatch_size > 0 and self._batch_size > microbatch_size:
            chunks = []
            for start in range(0, self._batch_size, microbatch_size):
                end = min(start + microbatch_size, self._batch_size)
                syndromes = tuple(s[:, start:end] for s in self._syndrome_tuple)
                chunks.append(
                    self._compiled_predict(self._noise_probs, syndromes))
            return torch.cat(chunks, dim=0)
        return self._compiled_predict(self._noise_probs, self._syndrome_tuple)

    def cross_entropy_loss(self) -> torch.Tensor:
        """Cross-entropy loss over the syndrome batch.

        Returns a differentiable scalar; call ``.backward()`` to obtain
        gradients w.r.t. :attr:`noise_params`.  Log inputs are floored to
        avoid non-finite values from roundoff; use the safe training
        patterns in :attr:`noise_params` to keep probabilities in range.
        """
        return self._compiled_loss_from_probs(self._noise_probs,
                                              self._syndrome_tuple)

    def current_syndrome_args(self) -> tuple[torch.Tensor, ...]:
        """Return the syndrome argument expected by :meth:`loss_fn`.

        Returns ``()`` when syndromes are baked into a static codegen
        closure, else the current live tuple.  Re-fetch each step so an
        intervening :meth:`update_dataset` is reflected.
        """
        if (self._execute_mode == "codegen" and not self._dynamic_syndromes and
                not self._precontract_noise):
            return ()
        return self._syndrome_tuple

    def loss_fn(self, from_logits: bool = True):
        """Return a fused ``(input, syndromes) -> scalar`` loss callable.

        Useful when training in logit space (``from_logits=True``, the
        default) or when feeding in an externally managed probability
        tensor (``from_logits=False``).  Compared to
        :meth:`cross_entropy_loss`, the parameter is supplied explicitly
        per call instead of being read from :attr:`noise_params`.
        """
        return (self._compiled_loss_from_logits
                if from_logits else self._compiled_loss_from_probs)

    def logical_error_rate(self) -> float:
        """Fraction of shots decoded incorrectly.

        Uses a hard argmax threshold; **not** differentiable.
        """
        with torch.no_grad():
            predictions = self.decoder_prediction()
            pred = predictions[:, 1] > predictions[:, 0]
            return float(1 - (pred == self._observable_flips).sum() /
                         len(self._observable_flips))

    def _update_data(self,
                     new_syndrome_arrays: torch.Tensor,
                     new_observable_flips: npt.NDArray[Any],
                     enforce_shape: bool = True) -> None:
        """In-place dataset swap on already-prepared syndrome tensors.

        ``new_syndrome_arrays`` must be in the internal layout (the
        output of :func:`prepare_syndrome_data_batch`, on the right
        device, shape ``(syndrome_length, shots, 2)``).  Public callers
        should use :meth:`update_dataset` instead.
        """
        # Patch syndrome tensor data in the quimb TN in place; the
        # cotengra path is invalidated below if any shape changed.
        for i, tag in enumerate(self._syndrome_tags):
            t = self.syndrome_tn.tensors[next(
                iter(self.syndrome_tn.tag_map[tag]))]
            if enforce_shape:
                assert t.data.shape == new_syndrome_arrays[i].shape, (
                    f"Shape mismatch for {tag}: "
                    f"{t.data.shape} vs {new_syndrome_arrays[i].shape}")
            t.modify(data=new_syndrome_arrays[i])

        # Suppress the loss rebuild the ``observable_flips`` setter
        # would otherwise trigger; one of the branches below issues it.
        self._suspend_loss_rebuild = True
        self.observable_flips = new_observable_flips

        torch_dtype = getattr(torch, self._dtype)
        dev = self.torch_device
        new_shapes: list[tuple[int, ...]] = []
        for k, (pos, _tag) in enumerate(self._syndrome_positions):
            data = self._tensors_ref[pos].data
            if isinstance(data, torch.Tensor):
                arr = data.detach().to(device=dev, dtype=torch_dtype)
            else:
                arr = torch.as_tensor(np.asarray(data),
                                      dtype=torch_dtype,
                                      device=dev)
            self._syndrome_arrays[k] = arr
            new_shapes.append(tuple(arr.shape))
        new_shapes_tuple = tuple(new_shapes)

        # Shape change: cached path / opt_einsum expression / compile
        # guards are stale.  Drop the path and rebuild from scratch.
        # Shapes unchanged: the forward reads syndromes per call, so
        # refreshing the cached tuple is enough.
        shape_changed = new_shapes_tuple != self._syndrome_shapes
        if shape_changed:
            self.path_batch = None
            self.slicing_batch = tuple()
            try:
                self._snapshot_arrays_and_eq()
            finally:
                self._suspend_loss_rebuild = False
            return

        self._syndrome_tuple = tuple(self._syndrome_arrays)
        if (self._execute_mode == "codegen" and not self._dynamic_syndromes and
                not self._precontract_noise):
            try:
                self._snapshot_arrays_and_eq()
            finally:
                self._suspend_loss_rebuild = False
        else:
            # The observable indices may have changed; the loss bakes
            # them in, so it still needs a rebuild.
            self._suspend_loss_rebuild = False
            self._compile_loss()

    def update_dataset(self,
                       new_syndrome_data: npt.NDArray[Any],
                       new_observable_flips: npt.NDArray[Any],
                       enforce_shape: bool = True) -> None:
        """Replace the syndrome batch and observable flips.

        Args:
            new_syndrome_data: Shape ``(shots, num_checks)``.
            new_observable_flips: Shape ``(shots,)``.
            enforce_shape: Assert that per-tensor shapes match.  A
                changing batch size triggers a full rebuild of the
                cached contraction path and forward.
        """
        syndrome_arrays = prepare_syndrome_data_batch(new_syndrome_data)
        torch_dtype = getattr(torch, self._dtype)
        syndrome_arrays = torch.tensor(
            syndrome_arrays,
            dtype=torch_dtype,
            device=self.torch_device,
        ).transpose(1, 2)
        self._batch_size = int(new_syndrome_data.shape[0])
        self._update_data(syndrome_arrays, new_observable_flips, enforce_shape)

    def optimize_path(self, optimize: Any = None, batch_size: int = -1) -> Any:
        """Cache a contraction path via quimb and rebuild the forward.

        Always routes through :meth:`TensorNetwork.contraction_info` so
        the resulting path is compatible with :mod:`opt_einsum`, unlike
        :meth:`TensorNetworkDecoder.optimize_path`,
        which defaults to a cuTensorNet-only path.

        ``batch_size`` is part of the parent ``TensorNetworkDecoder``
        signature (which rebuilds its TN around a fake batch); on the
        optimiser the syndrome TN is already batched at construction
        and resized in :meth:`update_dataset`, so this argument is
        ignored.  Kept for Liskov substitution with the parent.
        """
        del batch_size
        if self._precontract_noise:
            self._reduced_optimize = optimize
            self._snapshot_arrays_and_eq()
            return self._reduced_info

        if optimize is None or optimize == "auto":
            shapes = tuple(t.shape for t in self.full_tn.tensors)
            path, info = _select_default_torch_path(
                self.full_tn.get_equation(
                    output_inds=("batch_index", self.logical_obs_inds[0])),
                shapes,
                tn=self.full_tn,
                output_inds=("batch_index", self.logical_obs_inds[0]),
            )
            self.path_batch = path
            self.slicing_batch = tuple()
            self._snapshot_arrays_and_eq()
            return info

        info = self.full_tn.contraction_info(
            output_inds=("batch_index", self.logical_obs_inds[0]),
            optimize=optimize,
        )
        self.path_batch = info.path
        self.slicing_batch = tuple()
        self._snapshot_arrays_and_eq()
        return info


def make_compiled_step(optimizer: NMOptimizer,
                       logits: torch.Tensor,
                       torch_optimizer: torch.optim.Optimizer,
                       *,
                       max_backtracks: int = 0,
                       backtrack_factor: float = 0.5,
                       loss_tolerance: float = 0.0):
    """Build a no-arg logit-space optimizer step.

    By default this is a plain optimizer step.  With ``max_backtracks > 0``,
    rejected steps are retried from the same state with reduced learning rates.

    Args:
        optimizer: The :class:`NMOptimizer` providing the loss.
        logits: Trainable 1-D tensor of length ``len(optimizer.error_inds)``
            with ``requires_grad=True``.
        torch_optimizer: A ``torch.optim`` instance owning ``logits``.
        max_backtracks: Number of reduced-LR retries after the initial
            optimizer step.  ``0`` preserves ordinary optimizer behavior.
        backtrack_factor: Multiplicative LR reduction used for each retry.
        loss_tolerance: Absolute tolerated post-step loss increase.
    """
    if max_backtracks < 0:
        raise ValueError("max_backtracks must be non-negative.")
    if max_backtracks > 0 and not 0.0 < backtrack_factor < 1.0:
        raise ValueError("backtrack_factor must be in (0, 1).")

    def _loss():
        return optimizer.loss_fn(from_logits=True)(
            logits, optimizer.current_syndrome_args())

    def _is_finite_tensor(value: torch.Tensor) -> bool:
        return bool(torch.isfinite(value).all().detach().cpu())

    def _logits_are_finite() -> bool:
        return _is_finite_tensor(logits)

    def _grads_are_finite() -> bool:
        if logits.grad is None:
            return True
        return _is_finite_tensor(logits.grad)

    # Resolve the loss each call so dataset updates are reflected.
    def _plain_step():
        torch_optimizer.zero_grad(set_to_none=True)
        loss = _loss()
        if not _is_finite_tensor(loss):
            return loss
        loss.backward()
        if not _grads_are_finite():
            raise RuntimeError("Non-finite NMOptimizer logit gradients.")
        torch_optimizer.step()
        if not _logits_are_finite():
            raise RuntimeError("Non-finite NMOptimizer logits after step.")
        return loss

    if max_backtracks == 0:
        return _plain_step

    def _set_group_lrs(lrs: list[float]) -> None:
        for group, lr in zip(torch_optimizer.param_groups, lrs):
            group["lr"] = lr

    def _restore_state(saved_logits: torch.Tensor,
                       saved_state: dict[str, Any]) -> None:
        with torch.no_grad():
            logits.copy_(saved_logits)
        torch_optimizer.load_state_dict(copy.deepcopy(saved_state))

    def _guarded_step():
        base_lrs = [
            float(group["lr"]) for group in torch_optimizer.param_groups
        ]
        saved_logits = logits.detach().clone()
        saved_state = copy.deepcopy(torch_optimizer.state_dict())
        best_lrs = base_lrs
        current_loss: torch.Tensor | None = None

        for attempt in range(max_backtracks + 1):
            trial_lrs = [lr * (backtrack_factor**attempt) for lr in base_lrs]
            best_lrs = trial_lrs
            _restore_state(saved_logits, saved_state)
            _set_group_lrs(trial_lrs)

            torch_optimizer.zero_grad(set_to_none=True)
            loss = _loss()
            if current_loss is None:
                current_loss = loss.detach()
            if not bool(torch.isfinite(loss).detach().cpu()):
                return loss

            loss.backward()
            if not _grads_are_finite():
                continue
            torch_optimizer.step()
            if not _logits_are_finite():
                continue

            with torch.no_grad():
                next_loss = _loss().detach()
            if (bool(torch.isfinite(next_loss).detach().cpu()) and bool(
                (next_loss <= current_loss + loss_tolerance).cpu())):
                return current_loss

        _restore_state(saved_logits, saved_state)
        _set_group_lrs(best_lrs)
        return current_loss if current_loss is not None else _loss()

    return _guarded_step
