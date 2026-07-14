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

import warnings
from typing import Any, Callable, Literal

import cudaq_qec as qec
import numpy as np
import numpy.typing as npt
import opt_einsum as oe
import torch
from quimb.tensor import TensorNetwork

from ..tensor_network_decoder import TensorNetworkDecoder
from .noise_models import factorized_noise_model
from .tensor_network_factory import (
    tensor_network_from_parity_check,
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
_TORCH_EINSUM_MAX_DIMS = 25


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


def _normalize_prediction(x: torch.Tensor) -> torch.Tensor:
    """Normalize decoder scores while keeping zero rows finite."""
    norm = x.sum(dim=1, keepdim=True).clamp_min(torch.finfo(x.dtype).tiny)
    return x / norm


def remap_eq_to_ascii(eq: str) -> str:
    """Rewrite an einsum equation so every label is in ``[a-zA-Z]``.

    Needed because :mod:`opt_einsum` falls back to non-ASCII unicode
    labels once the total index count exceeds 52, but
    :func:`torch.einsum` rejects them.  Raises if a single step has more
    than 52 distinct labels.
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


def _einsum_rank(eq: str) -> int:
    """Maximum tensor rank touched by one explicit einsum step."""
    if "->" not in eq:
        return 0
    lhs, rhs = eq.split("->", 1)
    terms = [term for term in lhs.split(",") if term]
    terms.append(rhs)
    return max((len(term) for term in terms), default=0)


def _path_max_einsum_rank(info: Any) -> int:
    """Maximum rank emitted by an opt_einsum contraction path."""
    ranks = [
        _einsum_rank(step[2]) for step in getattr(info, "contraction_list", ())
    ]
    return max(ranks, default=0)


def _select_default_torch_path(
    eq: str,
    shapes: tuple[tuple[int, ...], ...],
) -> tuple[Any, Any]:
    candidates: list[tuple[str, Any, Any]] = []
    optimizers: list[tuple[str, Any]] = [
        ("greedy", "greedy"),
        ("auto", "auto"),
        ("auto-hq", "auto-hq"),
        ("random-greedy", "random-greedy"),
        ("random-greedy-128", "random-greedy-128"),
    ]

    for tag, optimize in optimizers:
        try:
            path, info = oe.contract_path(eq,
                                          *shapes,
                                          shapes=True,
                                          optimize=optimize)
        except Exception as exc:
            warnings.warn(
                f"NMOptimizer path candidate {tag!r} failed: {exc!r}",
                RuntimeWarning,
                stacklevel=3,
            )
            continue
        candidates.append((tag, path, info))

    if not candidates:
        raise RuntimeError("No NMOptimizer contraction path candidate "
                           "succeeded.")

    safe_candidates = [
        candidate for candidate in candidates
        if _path_max_einsum_rank(candidate[2]) <= _TORCH_EINSUM_MAX_DIMS
    ]
    if safe_candidates:
        candidates = safe_candidates

    _tag, selected_path, selected_info = min(
        candidates,
        key=lambda c: (_path_largest_intermediate(c[2]), _path_opt_cost(c[2])),
    )
    return selected_path, selected_info


class NMOptimizer(TensorNetworkDecoder):
    """Differentiable noise-model optimiser for the TN decoder.

    The factorised noise probabilities live in the torch autograd graph
    and are fit to a fixed syndrome batch by minimising the cross-entropy
    of the decoder's logical prediction against the observed flips.

    The forward pass is materialised once at construction and reused
    across optimisation steps.  Optionally call :meth:`optimize_path`
    (e.g. with ``cotengra.HyperOptimizer()``) to pin a better contraction
    path; the JIT is rebuilt automatically.

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
        device: ``"cuda"`` (default) or ``"cpu"``.
        compile: If ``True``, wrap the forward in :func:`torch.compile`.
            Most useful with ``execute="codegen"``.
        execute: Forward backend.  ``"codegen"`` (default) partial-evaluates
            the path into a flat Python function; ``"unrolled"`` keeps an
            interpretive einsum list; ``"opt_einsum"`` dispatches via
            :func:`opt_einsum.contract_expression`.
        compile_mode: Forwarded to :func:`torch.compile`; ignored when
            ``compile=False``.
        dynamic_syndromes: If ``True`` (default), syndromes are runtime
            arguments to the compiled forward, so :meth:`update_dataset`
            does not retrigger codegen / ``torch.compile`` (provided
            shapes are unchanged).  ``False`` bakes the syndromes into
            the closure as constants -- fewer runtime einsums, but every
            :meth:`update_dataset` call rebuilds the graph.  Only affects
            ``execute="codegen"``.
        Per-error noise tensors are contracted with their adjacent code
            tensors using differentiable torch ops, then the reduced
            network is contracted with the selected ``execute`` backend.

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
        execute: Literal["codegen", "unrolled", "opt_einsum"] = "codegen",
        compile_mode: str | None = None,
        dynamic_syndromes: bool = True,
    ) -> None:
        if execute not in ("unrolled", "opt_einsum", "codegen"):
            raise ValueError(f"Invalid execute mode: {execute!r}")
        if dtype not in _SUPPORTED_DTYPES:
            raise ValueError(f"Invalid dtype {dtype!r}; expected one of "
                             f"{list(_SUPPORTED_DTYPES)}.")

        # Sanitise once so the base TN tensors and ``self._noise_probs``
        # see identical values (see :func:`_validate_and_clamp_priors`).
        noise_model = _validate_and_clamp_priors(noise_model, dtype)
        target_device = device
        if "cuda" in target_device and not torch.cuda.is_available():
            warnings.warn(
                "CUDA was requested for NMOptimizer, but torch CUDA is not "
                "available. Using CPU.",
                UserWarning,
                stacklevel=2,
            )
            target_device = "cpu"

        # Build the topology directly so NMOptimizer never selects the
        # TensorNetworkDecoder cuTensorNet contractor during setup.
        qec.Decoder.__init__(self, H)

        num_checks, num_errs = H.shape
        if check_inds is None:
            self.check_inds = [f"s_{j}" for j in range(num_checks)]
        else:
            assert len(check_inds) == num_checks, (
                f"check_inds must have length {num_checks}, "
                f"but got {len(check_inds)}.")
            self.check_inds = check_inds
        if error_inds is None:
            self.error_inds = [f"e_{j}" for j in range(num_errs)]
        else:
            assert len(error_inds) == num_errs, (
                f"error_inds must have length {num_errs}, "
                f"but got {len(error_inds)}.")
            self.error_inds = error_inds

        self.logical_obs_inds = ["obs"]
        self.parity_check_matrix = H.copy()
        self.code_tn = tensor_network_from_parity_check(
            self.parity_check_matrix,
            col_inds=self.error_inds,
            row_inds=self.check_inds,
        )
        self.replace_logical_observable(logical_obs,
                                        logical_inds=logical_inds,
                                        logical_tags=logical_tags)

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

        self.path_single = "auto"
        self.path_batch = "auto"
        self.slicing_batch = tuple()
        self.slicing_single = tuple()

        self._set_contractor("oe_torch_compiled", target_device, "torch", dtype)
        self.init_noise_model(
            factorized_noise_model(self.error_inds, noise_model),
            contract=False,
        )

        torch_dtype = getattr(torch, self._dtype)
        self._noise_probs = torch.tensor(
            noise_model,
            dtype=torch_dtype,
            device=self.torch_device,
            requires_grad=True,
        )
        # The base's noise tensors stay in ``full_tn`` as placeholders:
        # ``_snapshot_arrays_and_eq`` uses ``id()`` to locate their
        # positions, then ``self._noise_probs`` (autograd live) is written
        # into those slots.  Do not strip them.

        self._suspend_loss_rebuild = True
        self.observable_flips = observable_flips

        self._use_torch_compile = compile
        self._execute_mode = execute
        self._torch_compile_mode = compile_mode
        self._dynamic_syndromes = dynamic_syndromes
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
        # invalidate the cached path / codegen / oe expr.
        self._syndrome_shapes: tuple[tuple[int, ...], ...] = tuple(
            tuple(s.shape) for s in self._syndrome_arrays)

        self._oe_expr = None
        self._path_steps = None
        self._build_reduced_tn_state()

        self._compile_predict()
        self._compile_loss()

    def _compile_predict(self) -> None:
        """Build ``self._predict_fn`` for the configured execute mode."""
        self._predict_fn = self._build_predict_reduced()
        self._compiled_predict = self._maybe_torch_compile(self._predict_fn,
                                                           kind="predict")

    def _build_reduced_tn_state(self) -> None:
        """Build reduced TN topology plus differentiable noise recipes."""
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
            code_tensors = [t for t in doomed if "NOISE" not in t.tags]
            code_opt_positions = [
                doomed_lookup[(tuple(t.inds), frozenset(t.tags))]
                for t in code_tensors
            ]

            ids_before = {id(t) for t in reduced_tn.tensors}
            reduced_tn.contract_ind(error_ind)
            new_tensors = [
                t for t in reduced_tn.tensors if id(t) not in ids_before
            ]
            assert len(new_tensors) == 1
            new_tensor = new_tensors[0]
            merged_id_to_recipe_idx[id(new_tensor)] = error_idx

            out_inds = tuple(new_tensor.inds)
            mapping = {error_ind: "e"}
            next_code = ord("a")
            for ind in out_inds:
                while chr(next_code) == "e":
                    next_code += 1
                mapping[ind] = chr(next_code)
                next_code += 1

            noise_str = mapping[error_ind]
            code_strs = [
                "".join(mapping[ind] for ind in t.inds) for t in code_tensors
            ]
            out_str = "".join(mapping[ind] for ind in out_inds)
            ordered_code_positions: list[int] = [None] * len(  # type: ignore
                code_tensors)
            for tensor, opt_pos in zip(code_tensors, code_opt_positions):
                non_error_ind = next(
                    ind for ind in tensor.inds if ind != error_ind)
                ordered_code_positions[out_inds.index(non_error_ind)] = opt_pos

            recipes.append({
                "eq": ",".join([noise_str] + code_strs) + "->" + out_str,
                "ordered_code_positions": ordered_code_positions,
                "k": len(code_tensors),
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

        reduced_path = self.path_batch
        if reduced_path in (None, "auto"):
            reduced_path, reduced_info = _select_default_torch_path(
                reduced_eq, reduced_shapes)
        else:
            _path, reduced_info = oe.contract_path(reduced_eq,
                                                   *reduced_shapes,
                                                   shapes=True,
                                                   optimize=reduced_path)
        reduced_path_steps = [(remap_eq_to_ascii(step[2]), tuple(step[0]),
                               tuple(sorted(step[0], reverse=True)))
                              for step in reduced_info.contraction_list]
        reduced_oe_expr = None
        if self._execute_mode == "opt_einsum":
            reduced_oe_expr = oe.contract_expression(reduced_eq,
                                                     *reduced_shapes,
                                                     optimize=reduced_path)

        recipe_to_reduced_pos = {ri: pos for pos, ri in reduced_recipes.items()}
        reduced_recipe_positions = tuple(
            recipe_to_reduced_pos[ri] for ri in range(len(recipes)))
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

            if k == 0:
                eq = "ne->ne"
            else:
                check_strs = [f"n{letter}e" for letter in out_letters]
                eq = "ne," + ",".join(check_strs) + "->n" + "".join(out_letters)

            stacked_checks = []
            for axis in range(k):
                axis_arrays = [
                    self._static_arrays[recipes[ri]["ordered_code_positions"]
                                        [axis]] for ri in error_indices
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
                "recipe_indices":
                    error_indices,
            })

        self._batched_einsum_groups = batched_groups
        self._reduced_static_positions = reduced_static
        self._reduced_syndrome_positions = reduced_syndrome
        self._reduced_recipe_positions = reduced_recipe_positions
        self._reduced_oe_expr = reduced_oe_expr
        self._reduced_path_steps = reduced_path_steps
        self._reduced_n_tensors = len(reduced_tn.tensors)
        self._reduced_n_recipes = len(recipes)
        self._reduced_info = reduced_info
        self.path_batch = reduced_path
        self.slicing_batch = tuple()

    def _build_predict_reduced(self):
        """Predict using the reduced TN plus batched noise precontraction."""
        builders = {
            "opt_einsum": self._build_predict_reduced_opt_einsum,
            "unrolled": self._build_predict_reduced_unrolled,
            "codegen": self._build_predict_reduced_codegen,
        }
        return builders[self._execute_mode]()

    def _precontract_reduced_noise(
            self, noise_probs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        noise_stacked = torch.stack((1.0 - noise_probs, noise_probs), dim=-1)
        recipe_arrays = [None] * self._reduced_n_recipes
        for group in self._batched_einsum_groups:
            noise_batch = noise_stacked[group["error_indices_t"]]
            if group["k"] == 0:
                out_batch = noise_batch
            else:
                out_batch = torch.einsum(group["eq"], noise_batch,
                                         *group["stacked_checks"])
            for i, recipe_idx in enumerate(group["recipe_indices"]):
                recipe_arrays[recipe_idx] = out_batch[i]
        return tuple(recipe_arrays)  # type: ignore[return-value]

    def _build_predict_reduced_opt_einsum(self):
        static_positions = self._reduced_static_positions
        syndrome_positions = self._reduced_syndrome_positions
        oe_expr = self._reduced_oe_expr
        recipe_positions = self._reduced_recipe_positions
        n = self._reduced_n_tensors

        def _predict(noise_probs: torch.Tensor,
                     syndrome_tuple: tuple[torch.Tensor, ...]) -> torch.Tensor:
            arrays: list[torch.Tensor] = [None] * n  # type: ignore
            for pos, arr in static_positions.items():
                arrays[pos] = arr
            for pos, syndrome_idx in syndrome_positions:
                arrays[pos] = syndrome_tuple[syndrome_idx]
            for pos, arr in zip(recipe_positions,
                                self._precontract_reduced_noise(noise_probs)):
                arrays[pos] = arr
            out = oe_expr(*arrays)
            return _normalize_prediction(out)

        return _predict

    def _build_predict_reduced_unrolled(self):
        static_positions = self._reduced_static_positions
        syndrome_positions = self._reduced_syndrome_positions
        recipe_positions = self._reduced_recipe_positions
        path_steps = self._reduced_path_steps
        n = self._reduced_n_tensors

        def _predict(noise_probs: torch.Tensor,
                     syndrome_tuple: tuple[torch.Tensor, ...]) -> torch.Tensor:
            ops: list[torch.Tensor] = [None] * n  # type: ignore
            for pos, arr in static_positions.items():
                ops[pos] = arr
            for pos, syndrome_idx in syndrome_positions:
                ops[pos] = syndrome_tuple[syndrome_idx]
            for pos, arr in zip(recipe_positions,
                                self._precontract_reduced_noise(noise_probs)):
                ops[pos] = arr
            for eq_str, idxs, sorted_idxs in path_steps:
                picked = [ops[i] for i in idxs]
                for i in sorted_idxs:
                    ops.pop(i)
                ops.append(torch.einsum(eq_str, *picked))
            out = ops[0]
            return _normalize_prediction(out)

        return _predict

    def _build_predict_reduced_codegen(self):
        static_arrays = dict(self._reduced_static_positions)
        dynamic_positions = [
            (pos, f"_R{idx}")
            for idx, pos in enumerate(self._reduced_recipe_positions)
        ]
        if self._dynamic_syndromes:
            dynamic_positions.extend(
                (pos, f"_S{sidx}")
                for pos, sidx in self._reduced_syndrome_positions)
        else:
            for pos, sidx in self._reduced_syndrome_positions:
                static_arrays[pos] = self._syndrome_arrays[sidx]

        codegen_fn = self._build_codegen_reduced_predict(
            self._reduced_n_tensors,
            static_arrays,
            tuple(dynamic_positions),
            self._reduced_path_steps,
            n_recipes=self._reduced_n_recipes,
            n_syndromes=len(self._reduced_syndrome_positions),
            dynamic_syndromes=self._dynamic_syndromes,
        )
        self._codegen_fn = codegen_fn
        self._codegen_n_folded = getattr(codegen_fn, "_n_folded", 0)
        self._codegen_n_runtime = getattr(codegen_fn, "_n_runtime", 0)

        if self._dynamic_syndromes:

            def _predict(
                    noise_probs: torch.Tensor,
                    syndrome_tuple: tuple[torch.Tensor, ...]) -> torch.Tensor:
                return codegen_fn(self._precontract_reduced_noise(noise_probs),
                                  syndrome_tuple)
        else:

            def _predict(
                noise_probs: torch.Tensor,
                syndrome_tuple: tuple[torch.Tensor, ...] = ()
            ) -> torch.Tensor:
                return codegen_fn(self._precontract_reduced_noise(noise_probs))

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
        logits_fn, probs_fn = self._build_loss_wrapped()

        self._loss_from_logits_fn = logits_fn
        self._loss_from_probs_fn = probs_fn
        self._compiled_loss_from_logits = self._maybe_torch_compile(logits_fn,
                                                                    kind="loss")
        self._compiled_loss_from_probs = self._maybe_torch_compile(probs_fn,
                                                                   kind="loss")

    def _build_loss_wrapped(self):
        """opt_einsum / unrolled loss: wrap CE around ``self._predict_fn``."""
        obs_t = self.obs_idx_true
        obs_f = self.obs_idx_false
        predict_fn = self._predict_fn

        if self._dynamic_syndromes:

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
    def _codegen_partial_eval_dynamic(n, static_arrays, dynamic_positions,
                                      path_steps):
        static_positions = sorted(static_arrays.keys())
        dynamic_names = {pos: name for pos, name in dynamic_positions}
        static_pos_to_sidx = {
            pos: sidx for sidx, pos in enumerate(static_positions)
        }

        state: list[tuple[str, bool, torch.Tensor | None]] = []
        for pos in range(n):
            if pos in dynamic_names:
                state.append((dynamic_names[pos], True, None))
            else:
                sidx = static_pos_to_sidx[pos]
                state.append(
                    (f"_C{sidx}", False, static_arrays[static_positions[sidx]]))

        closure_vars = {
            f"_C{sidx}": static_arrays[pos]
            for sidx, pos in enumerate(static_positions)
        }
        runtime_lines: list[str] = []
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
                    result = torch.einsum(eq_str, *arrs).contiguous()
                static_name = f"_P{step_idx}"
                closure_vars[static_name] = result
                state.append((static_name, False, result))
                n_folded += 1
            else:
                arg_names = [p[0] for p in picked]
                runtime_lines.append(
                    f"    {out_name} = torch.einsum({eq_str!r}, "
                    f"{', '.join(arg_names)})")
                state.append((out_name, True, None))

        assert len(state) == 1
        return runtime_lines, closure_vars, state[0], n_folded

    @classmethod
    def _build_codegen_reduced_predict(cls,
                                       n,
                                       static_arrays,
                                       dynamic_positions,
                                       path_steps,
                                       n_recipes: int,
                                       n_syndromes: int,
                                       dynamic_syndromes: bool = True):
        runtime_lines, closure_vars, final_state, n_folded = (
            cls._codegen_partial_eval_dynamic(
                n,
                static_arrays,
                dynamic_positions,
                path_steps,
            ))
        final_name, is_final_dyn, final_value = final_state
        fully_static = not is_final_dyn

        body: list[str] = []
        if dynamic_syndromes:
            body.append("def _predict(recipe_arrays, syndromes):")
        else:
            body.append("def _predict(recipe_arrays):")

        if fully_static:
            with torch.no_grad():
                normed = _normalize_prediction(final_value)
            closure_vars["_FINAL"] = normed
            body.append("    return _FINAL")
            runtime_lines = []
        else:
            for k in range(n_recipes):
                body.append(f"    _R{k} = recipe_arrays[{k}]")
            if dynamic_syndromes:
                for sidx in range(n_syndromes):
                    body.append(f"    _S{sidx} = syndromes[{sidx}]")
            body.extend(runtime_lines)
            body.append(f"    _out = {final_name}")
            body.append("    return _normalize_prediction(_out)")

        return cls._compile_codegen_source(body, closure_vars, n_folded,
                                           len(runtime_lines), "predict")

    @staticmethod
    def _compile_codegen_source(body: list[str],
                                closure_vars: dict[str, torch.Tensor],
                                n_folded: int, n_runtime: int, kind: str):
        """Compile the assembled function source and return the callable."""
        source = "\n".join(body)
        ns: dict[str, Any] = {
            "torch": torch,
            "_normalize_prediction": _normalize_prediction,
        }
        ns.update(closure_vars)
        fn_name = "_loss" if kind == "loss" else "_predict"
        exec(compile(source, f"<nm_compiled_{kind}>", "exec"), ns)
        fn = ns[fn_name]
        fn._n_folded = n_folded  # type: ignore[attr-defined]
        fn._n_runtime = n_runtime  # type: ignore[attr-defined]
        return fn

    def decoder_prediction(self) -> torch.Tensor:
        """Run the forward pass; returns ``(shots, 2)`` predictions."""
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

        Returns ``()`` when syndromes are baked into the closure
        (``execute="codegen"`` and ``dynamic_syndromes=False``), else
        the current live tuple.  Re-fetch each step so an intervening
        :meth:`update_dataset` is reflected.
        """
        if self._execute_mode == "codegen" and not self._dynamic_syndromes:
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
        # contraction path is invalidated below if any shape changed.
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

        # Shape change: cached path / codegen / oe expression / compile
        # guards are all stale.  Drop the path and rebuild from scratch.
        # Shapes unchanged: dynamic codegen and the unrolled /
        # opt_einsum paths read syndromes per call - refreshing the
        # cached tuple is enough.  Static codegen baked the old tensors
        # into the closure and still needs a full rebuild.
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
        if self._execute_mode == "codegen" and not self._dynamic_syndromes:
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
                cached contraction path and codegen.
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
        """Cache a contraction path via quimb and rebuild the JIT.

        Always routes through :meth:`TensorNetwork.contraction_info` so
        the resulting path is compatible with :mod:`opt_einsum` and
        manual unrolling -- unlike :meth:`TensorNetworkDecoder.optimize_path`,
        which may return a path not usable by these torch-backed modes.

        ``batch_size`` is part of the parent ``TensorNetworkDecoder``
        signature (which rebuilds its TN around a fake batch); on the
        optimiser the syndrome TN is already batched at construction
        and resized in :meth:`update_dataset`, so this argument is
        ignored.  Kept for Liskov substitution with the parent.
        """
        del batch_size
        self.path_batch = optimize if optimize is not None else "auto"
        self.slicing_batch = tuple()
        self._snapshot_arrays_and_eq()
        return self._reduced_info


def make_compiled_step(optimizer: NMOptimizer, logits: torch.Tensor,
                       torch_optimizer: torch.optim.Optimizer):
    """Build a no-arg callable that runs one Adam step and returns the loss.

    The step zeros grads, calls the optimizer's compiled
    ``loss_fn(from_logits=True)`` (sigmoid + contraction + cross-entropy
    fused), backwards, and steps ``torch_optimizer``.  Use this when
    training in logit space.

    Args:
        optimizer: The :class:`NMOptimizer` providing the fused
            inner loss; pass ``compile=True`` at the
            :class:`NMOptimizer` constructor for the
            ``torch.compile``-d variant.
        logits: Trainable 1-D tensor of length ``len(optimizer.error_inds)``
            with ``requires_grad=True``.
        torch_optimizer: A ``torch.optim`` instance owning ``logits``.
    """

    # Re-fetch per call so a rebuild from update_dataset or the
    # observable_flips setter is picked up; capturing would go stale.
    def _step():
        torch_optimizer.zero_grad(set_to_none=True)
        loss = optimizer.loss_fn(from_logits=True)(
            logits, optimizer.current_syndrome_args())
        loss.backward()
        torch_optimizer.step()
        return loss

    return _step


def make_batch_sliced_step(
        optimizer: NMOptimizer,
        logits: torch.Tensor,
        torch_optimizer: torch.optim.Optimizer,
        progress_callback: Callable[[int, int, float], None] | None = None):
    """Build a step callable that streams a larger batch through slices.

    The optimizer should be constructed with the desired slice-sized
    dataset.  The returned callable accepts a larger raw syndrome batch,
    repeatedly updates the optimizer with fixed-shape slices, accumulates
    gradients, and performs one ``torch_optimizer.step()``.  This keeps
    path finding and contraction execution tied to the smaller batch
    shape while preserving the summed cross-entropy objective over all
    supplied shots.  The optimizer's live dataset is left at the final
    slice after the step.

    Args:
        optimizer: An :class:`NMOptimizer` constructed with the desired
            batch-slice size.
        logits: Trainable 1-D tensor of length ``len(optimizer.error_inds)``
            with ``requires_grad=True``.
        torch_optimizer: A ``torch.optim`` instance owning ``logits``.
        progress_callback: Optional callback invoked as
            ``callback(done_chunks, total_chunks, elapsed_seconds)``.

    Returns:
        A callable ``step(syndrome_data, observable_flips)`` that runs one
        optimizer step and returns the detached summed loss.
    """
    if not optimizer._dynamic_syndromes:
        raise ValueError("batch-sliced steps require dynamic_syndromes=True.")

    batch_slice_size = int(optimizer._batch_size)
    if batch_slice_size <= 0:
        raise ValueError("optimizer batch size must be positive.")

    def _step(syndrome_data: npt.NDArray[Any],
              observable_flips: npt.NDArray[Any]) -> torch.Tensor:
        syndrome_arr = np.asarray(syndrome_data)
        flips_arr = np.asarray(observable_flips, dtype=bool).reshape(-1)
        if syndrome_arr.ndim != 2:
            raise ValueError("syndrome_data must have shape (shots, checks).")
        if syndrome_arr.shape[0] != flips_arr.shape[0]:
            raise ValueError(
                "syndrome_data and observable_flips length mismatch.")
        if syndrome_arr.shape[0] == 0:
            raise ValueError("syndrome_data must contain at least one shot.")

        torch_optimizer.zero_grad(set_to_none=True)
        total_loss = torch.zeros((), dtype=logits.dtype, device=logits.device)
        num_chunks = ((syndrome_arr.shape[0] + batch_slice_size - 1) //
                      batch_slice_size)
        wall_start = None
        if progress_callback is not None:
            import time
            wall_start = time.perf_counter()

        for chunk_idx, start in enumerate(range(0, syndrome_arr.shape[0],
                                                batch_slice_size),
                                          start=1):
            stop = min(start + batch_slice_size, syndrome_arr.shape[0])
            valid = stop - start
            chunk_syn = syndrome_arr[start:stop]
            chunk_flips = flips_arr[start:stop]
            if valid < batch_slice_size:
                padded_syn = np.zeros((batch_slice_size, syndrome_arr.shape[1]),
                                      dtype=syndrome_arr.dtype)
                padded_flips = np.zeros((batch_slice_size,), dtype=bool)
                padded_syn[:valid] = chunk_syn
                padded_flips[:valid] = chunk_flips
                chunk_syn = padded_syn
                update_flips = padded_flips
            else:
                update_flips = chunk_flips

            optimizer.update_dataset(chunk_syn,
                                     update_flips,
                                     enforce_shape=True)
            preds = optimizer._compiled_predict(
                torch.sigmoid(logits), optimizer.current_syndrome_args())
            preds = preds[:valid]
            flips_t = torch.as_tensor(chunk_flips,
                                      dtype=torch.bool,
                                      device=preds.device)
            loss = (-torch.log(_clamp_log_input(preds[flips_t, 1])).sum() -
                    torch.log(_clamp_log_input(preds[~flips_t, 0])).sum())
            loss.backward()
            total_loss = total_loss + loss.detach()

            if progress_callback is not None:
                elapsed = (0.0 if wall_start is None else time.perf_counter() -
                           wall_start)
                progress_callback(chunk_idx, num_chunks, elapsed)

        torch_optimizer.step()
        return total_loss

    return _step


def make_training_step(H: npt.NDArray[Any],
                       logical_obs: npt.NDArray[Any],
                       noise_model: list[float],
                       syndrome_data: npt.NDArray[Any],
                       observable_flips: npt.NDArray[Any],
                       logits: torch.Tensor,
                       torch_optimizer: torch.optim.Optimizer,
                       batch_slicing: bool | Literal["auto"] = "auto",
                       batch_slice_size: int | None = None,
                       progress_callback: Callable[[int, int, float], None] |
                       None = None,
                       **optimizer_kwargs: Any) -> tuple[NMOptimizer, Any]:
    """Construct an :class:`NMOptimizer` and matching training step.

    Args:
        H: Parity check matrix.
        logical_obs: Logical observable matrix.
        noise_model: Initial per-error probabilities.
        syndrome_data: Full syndrome batch, shape ``(shots, checks)``.
        observable_flips: Observable flips for the full batch.
        logits: Trainable logit tensor owned by ``torch_optimizer``.
        torch_optimizer: Optimizer for ``logits``.
        batch_slicing: ``False`` uses the full batch. ``"auto"`` or
            ``True`` constructs a slice-sized optimizer and streams the
            full batch through it with gradient accumulation.
        batch_slice_size: Optional explicit slice size. Defaults to ``1``
            when batch slicing is enabled.
        progress_callback: Optional callback forwarded to the batch-sliced
            step as ``callback(done_chunks, total_chunks, elapsed_seconds)``.
        **optimizer_kwargs: Forwarded to :class:`NMOptimizer`.

    Returns:
        ``(optimizer, step_fn)``.  For full-batch training ``step_fn`` is
        no-arg.  For batch-sliced training it accepts optional
        ``(syndrome_data, observable_flips)`` and defaults to the original
        full batch supplied here.
    """
    if batch_slicing not in (False, True, "auto"):
        raise ValueError("batch_slicing must be False, True, or 'auto'.")

    use_batch_slicing = batch_slicing in (True, "auto")
    syndrome_arr = np.asarray(syndrome_data)
    flips_arr = np.asarray(observable_flips, dtype=bool).reshape(-1)
    if syndrome_arr.ndim != 2:
        raise ValueError("syndrome_data must have shape (shots, checks).")
    if syndrome_arr.shape[0] != flips_arr.shape[0]:
        raise ValueError("syndrome_data and observable_flips length mismatch.")
    if syndrome_arr.shape[0] == 0:
        raise ValueError("syndrome_data must contain at least one shot.")

    if not use_batch_slicing:
        optimizer = NMOptimizer(H, logical_obs, noise_model, syndrome_arr,
                                flips_arr, **optimizer_kwargs)
        return optimizer, make_compiled_step(optimizer, logits, torch_optimizer)

    slice_size = 1 if batch_slice_size is None else int(batch_slice_size)
    if slice_size <= 0:
        raise ValueError("batch_slice_size must be a positive integer.")
    slice_size = min(slice_size, syndrome_arr.shape[0])
    optimizer = NMOptimizer(H, logical_obs, noise_model,
                            syndrome_arr[:slice_size], flips_arr[:slice_size],
                            **optimizer_kwargs)
    sliced_step = make_batch_sliced_step(optimizer, logits, torch_optimizer,
                                         progress_callback)

    def _step(
            new_syndrome_data: npt.NDArray[Any] | None = None,
            new_observable_flips: npt.NDArray[Any] | None = None
    ) -> torch.Tensor:
        if new_syndrome_data is None:
            if new_observable_flips is not None:
                raise ValueError(
                    "observable_flips provided without syndrome_data.")
            return sliced_step(syndrome_arr, flips_arr)
        if new_observable_flips is None:
            raise ValueError("observable_flips is required with syndrome_data.")
        return sliced_step(new_syndrome_data, new_observable_flips)

    return optimizer, _step
