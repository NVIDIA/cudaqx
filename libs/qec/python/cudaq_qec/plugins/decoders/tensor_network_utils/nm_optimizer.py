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
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import opt_einsum as oe
import torch
from torch.utils.checkpoint import checkpoint as _checkpoint
from quimb.tensor import TensorNetwork

from ..tensor_network_decoder import TensorNetworkDecoder
from .contractors import optimize_path as _optimize_path_dispatch
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
        precontract_noise: bool = False,
    ) -> None:
        if execute not in ("unrolled", "opt_einsum", "codegen"):
            raise ValueError(f"Invalid execute mode: {execute!r}")
        if dtype not in _SUPPORTED_DTYPES:
            raise ValueError(f"Invalid dtype {dtype!r}; expected one of "
                             f"{list(_SUPPORTED_DTYPES)}.")
        if precontract_noise and execute != "opt_einsum":
            raise ValueError(
                "precontract_noise=True requires execute='opt_einsum'; "
                f"got {execute!r}")

        noise_model = _validate_and_clamp_priors(noise_model, dtype)

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

        # Force the torch backend so tensor data lives in the autograd graph.
        if self.contractor_config.contractor_name == "cutensornet" \
                and self.contractor_config.backend != "torch":
            warnings.warn(
                "NMOptimizer requires the torch backend for autograd; "
                f"switching contractor backend from "
                f"{self.contractor_config.backend!r} to 'torch'. "
                "Contractions are executed via codegen/opt_einsum, not "
                "cuTensorNet.",
                stacklevel=3,
            )
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
        # Noise placeholders stay in ``full_tn``; ``_snapshot_arrays_and_eq``
        # locates them by ``id()`` and writes ``self._noise_probs`` into
        # those slots.  Do not strip them.

        self._suspend_loss_rebuild = True
        self.observable_flips = observable_flips

        self._use_torch_compile = compile
        self._execute_mode = execute
        self._torch_compile_mode = compile_mode
        self._dynamic_syndromes = dynamic_syndromes
        self._precontract_noise = precontract_noise
        self._reduced_optimize: Any = None
        self._reduced_network_options: Any = None
        self._compiled_predict: Any | None = None
        self._syndrome_tuple: tuple[torch.Tensor, ...] = ()
        self.batch_slices: int = 1
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

        # Every tensor in ``full_tn`` must classify into exactly one bucket;
        # a future quimb that copies tensors on virtual combine would break
        # this and put a None or misplaced placeholder in the operand list.
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
        # Used by :meth:`_update_data` to detect layout changes.
        self._syndrome_shapes: tuple[tuple[int, ...], ...] = tuple(
            tuple(s.shape) for s in self._syndrome_arrays)

        if self._execute_mode == "opt_einsum":
            if self._precontract_noise:
                # ``_predict`` uses ``_reduced_oe_expr``; skip full_tn build.
                self._oe_expr = None
                self._build_reduced_tn_state()
            else:
                shapes = tuple(t.shape for t in tensors)
                self._oe_expr = oe.contract_expression(
                    self._eq_batch,
                    *shapes,
                    optimize=self.path_batch
                    if self.path_batch not in (None, "auto") else "auto",
                )
            self._path_steps = None
        else:
            self._oe_expr = None
            # Flatten to ``[(eq, idxs, sorted_desc), ...]``; sorted_desc is
            # the unrolled-mode pop walk, ASCII remap dodges torch.einsum's
            # rejection of opt_einsum's >52-index unicode fallback.
            shapes = tuple(t.shape for t in tensors)
            _, info = oe.contract_path(
                self._eq_batch,
                *shapes,
                shapes=True,
                optimize=self.path_batch
                if self.path_batch not in (None, "auto") else "auto",
            )
            self._path_steps = [(remap_eq_to_ascii(step[2]), tuple(step[0]),
                                 tuple(sorted(step[0], reverse=True)))
                                for step in info.contraction_list]

        self._compile_predict()
        self._compile_loss()

    def _compile_predict(self) -> None:
        """Build ``self._predict_fn`` for the configured execute mode."""
        if self._precontract_noise:
            base_predict = self._build_predict_reduced()
        else:
            builders = {
                "opt_einsum": self._build_predict_opt_einsum,
                "unrolled": self._build_predict_unrolled,
                "codegen": self._build_predict_codegen,
            }
            base_predict = builders[self._execute_mode]()
        if self.batch_slices > 1:
            base_predict = self._wrap_predict_batch_sliced(base_predict)
        self._predict_fn = base_predict
        self._compiled_predict = self._maybe_torch_compile(self._predict_fn,
                                                           kind="predict")

    def _wrap_predict_batch_sliced(self, base_predict_fn):
        """Wrap a predict function so it iterates over batch-axis chunks.

        Syndrome tensors are shape ``(2, batch)``; we split along axis 1
        into ``self.batch_slices`` chunks, call ``base_predict_fn`` per
        chunk, then concatenate the per-chunk ``(chunk_batch, 2)``
        outputs along dim 0.  ``torch.cat`` is differentiable so the
        autograd graph through ``noise_probs`` is preserved.
        """
        n_slices = self.batch_slices

        def _sliced_predict(
            noise_probs: torch.Tensor,
            syndrome_tuple: tuple[torch.Tensor, ...],
        ) -> torch.Tensor:
            split = [
                torch.tensor_split(t, n_slices, dim=1) for t in syndrome_tuple
            ]
            outs = []
            for s in range(n_slices):
                chunk = tuple(split[k][s] for k in range(len(split)))
                outs.append(base_predict_fn(noise_probs, chunk))
            return torch.cat(outs, dim=0)

        return _sliced_predict

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
            out = oe_expr(*arrays)
            return out / out.sum(dim=1, keepdim=True)

        return _predict

    def _build_reduced_tn_state(self) -> None:
        """Build the reduced TN topology + per-error einsum specs.

        Equivalent of :meth:`TensorNetworkDecoder.init_noise_model`
        ``contract=True``, but deferred: the per-error noise-into-checks
        contractions become differentiable :func:`torch.einsum` calls
        invoked per step, so the noise priors stay leaves of the
        autograd graph while the main contraction runs on the
        ``contract_noise_model=True`` topology.
        """
        import cotengra as ctg

        error_inds_set = set(self.error_inds)

        survivor_lookup: dict = {}
        doomed_lookup: dict = {}
        for opt_pos, t in enumerate(self._tensors_ref):
            key = (tuple(t.inds), frozenset(t.tags))
            if any(ind in error_inds_set for ind in t.inds):
                doomed_lookup[key] = opt_pos
            else:
                survivor_lookup[key] = opt_pos

        reduced_tn = self.full_tn.copy()
        recipes: list[dict[str, Any]] = []
        merged_id_to_recipe_idx: dict[int, int] = {}

        for error_idx, error_ind in enumerate(self.error_inds):
            doomed = [t for t in reduced_tn.tensors if error_ind in t.inds]
            check_ts = [t for t in doomed if 'NOISE' not in t.tags]
            check_opt_positions = [
                doomed_lookup[(tuple(ct.inds), frozenset(ct.tags))]
                for ct in check_ts
            ]
            ids_before = {id(t) for t in reduced_tn.tensors}
            reduced_tn.contract_ind(error_ind)
            new_ts = [t for t in reduced_tn.tensors if id(t) not in ids_before]
            assert len(new_ts) == 1
            new_t = new_ts[0]
            merged_id_to_recipe_idx[id(new_t)] = error_idx

            # Einsum output must match quimb's index order on the merged
            # tensor so axes align in reduced_tn.
            quimb_out_inds = tuple(new_t.inds)
            mapping = {error_ind: 'e'}
            next_code = ord('a')
            for ind in quimb_out_inds:
                while chr(next_code) == 'e':
                    next_code += 1
                mapping[ind] = chr(next_code)
                next_code += 1
            noise_str = mapping[error_ind]
            check_strs = [
                "".join(mapping[i] for i in ct.inds) for ct in check_ts
            ]
            out_str = "".join(mapping[i] for i in quimb_out_inds)
            # ordered_check_opt_positions[axis] holds the opt-array position
            # of the check tensor whose non-error index is
            # quimb_out_inds[axis] — needed for batching every error in a
            # signature class through one torch.einsum.
            ordered_check_opt_positions: list[int] = [None] * len(
                check_ts)  # type: ignore
            for ct, ct_pos in zip(check_ts, check_opt_positions):
                non_e = next(i for i in ct.inds if i != error_ind)
                ordered_check_opt_positions[quimb_out_inds.index(
                    non_e)] = ct_pos
            recipes.append({
                'eq': ",".join([noise_str] + check_strs) + "->" + out_str,
                'check_opt_positions': check_opt_positions,
                'ordered_check_opt_positions': ordered_check_opt_positions,
                'k': len(check_ts),
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
        for pos, t in enumerate(reduced_tn.tensors):
            if id(t) in merged_id_to_recipe_idx:
                reduced_recipes[pos] = merged_id_to_recipe_idx[id(t)]
                continue
            key = (tuple(t.inds), frozenset(t.tags))
            opt_pos = survivor_lookup[key]
            if opt_pos in self._static_arrays:
                reduced_static[pos] = self._static_arrays[opt_pos]
            elif opt_pos in syn_pos_to_idx:
                reduced_syndrome.append((pos, syn_pos_to_idx[opt_pos]))
            else:
                raise AssertionError(
                    f"reduced_tn tensor at pos {pos} maps to opt_pos {opt_pos} "
                    "which isn't classified as static or syndrome")

        # Score a path for torch.einsum usability: max tensor rank across
        # contraction steps.  torch.einsum hard-caps at 25 dims per tensor.
        def _max_step_rank(path: Any) -> int:
            _, info = oe.contract_path(reduced_eq,
                                       *reduced_shapes,
                                       shapes=True,
                                       optimize=path)
            max_rank = 0
            for step in info.contraction_list:
                eq = step[2]
                if "->" in eq:
                    lhs, rhs = eq.split("->")
                else:
                    lhs, rhs = eq, ""
                for part in lhs.split(","):
                    max_rank = max(max_rank, len(set(part)))
                max_rank = max(max_rank, len(set(rhs)))
            return max_rank

        # Cotengra is stochastic; retry until we land an executable path
        # (max_step_rank <= 25) or exhaust attempts.
        cotengra_retries = 8
        ctg_path = ctg_info = None
        for attempt in range(cotengra_retries):
            hyper = ctg.HyperOptimizer(max_repeats=8, parallel=False)
            p, info = oe.contract_path(reduced_eq,
                                       *reduced_shapes,
                                       shapes=True,
                                       optimize=hyper)
            rank = _max_step_rank(p)
            if (ctg_info is None or
                (_max_step_rank(ctg_path) > 25 and rank <= 25) or
                (rank <= 25 and float(info.largest_intermediate) < float(
                    ctg_info.largest_intermediate))):
                ctg_path, ctg_info = p, info
            if rank <= 25:
                break
        candidates = [("cotengra", ctg_path, ctg_info)]

        if self._reduced_optimize is not None:
            usr_path, usr_info = _optimize_path_dispatch(
                self._reduced_optimize,
                ("batch_index", self.logical_obs_inds[0]),
                reduced_tn,
                network_options=self._reduced_network_options,
            )
            candidates.append(("user", usr_path, usr_info))

        # Score by (unexecutable, largest_intermediate, rank).  Unexecutable
        # paths (rank > 25) always lose to executable ones.
        def _score(c: tuple) -> tuple:
            _tag, path, info = c
            li = getattr(info, "largest_intermediate", None)
            li = float("inf") if li is None else float(li)
            rank = _max_step_rank(path)
            return (rank > 25, li, rank)

        which, reduced_path, reduced_info = min(candidates, key=_score)
        for c in candidates:
            tag, _p, info = c
            executable, li, rank = _score(c)
            oc = getattr(info, "opt_cost", float("nan"))
            warnings.warn(
                f"reduced TN candidate ({tag}{'*' if tag == which else ''}): "
                f"opt_cost={oc:.3e}  largest_intermediate={li:.3e}  "
                f"max_step_rank={rank}" +
                (" [unexecutable]" if executable else ""),
                UserWarning,
                stacklevel=2,
            )
        reduced_oe_expr = oe.contract_expression(reduced_eq,
                                                 *reduced_shapes,
                                                 optimize=reduced_path)

        _, step_info = oe.contract_path(reduced_eq,
                                        *reduced_shapes,
                                        shapes=True,
                                        optimize=reduced_path)
        reduced_path_steps = [(remap_eq_to_ascii(step[2]), tuple(step[0]),
                               tuple(sorted(step[0], reverse=True)))
                              for step in step_info.contraction_list]

        # Group errors by check count so each class runs through one
        # batched torch.einsum instead of one per error.
        from collections import defaultdict
        recipe_to_reduced_pos = {ri: cp for cp, ri in reduced_recipes.items()}
        groups_by_k: dict[int, list[int]] = defaultdict(list)
        for ri, r in enumerate(recipes):
            groups_by_k[r['k']].append(ri)

        batched_groups: list[dict[str, Any]] = []
        device = self.torch_device
        for k, error_indices in sorted(groups_by_k.items()):
            # 'n' = batched-error dim, 'e' = contracted error index.
            out_letters: list[str] = []
            next_code = ord('a')
            for _ in range(k):
                while chr(next_code) in ('e', 'n'):
                    next_code += 1
                out_letters.append(chr(next_code))
                next_code += 1
            out_str = "".join(out_letters)
            check_strs = [f"n{c}e" for c in out_letters]
            eq = "ne," + ",".join(check_strs) + "->n" + out_str if k > 0 \
                else "ne->ne"

            stacked_checks = []
            for axis in range(k):
                axis_arrays = [
                    self._static_arrays[recipes[ri]
                                        ['ordered_check_opt_positions'][axis]]
                    for ri in error_indices
                ]
                stacked_checks.append(torch.stack(axis_arrays, dim=0))

            reduced_positions = [
                recipe_to_reduced_pos[ri] for ri in error_indices
            ]
            error_indices_t = torch.tensor(error_indices,
                                           dtype=torch.long,
                                           device=device)

            batched_groups.append({
                'k': k,
                'eq': eq,
                'error_indices_t': error_indices_t,
                'stacked_checks': stacked_checks,
                'reduced_positions': reduced_positions,
            })

        self._reduced_tn = reduced_tn
        self._per_error_einsums = recipes
        self._batched_einsum_groups = batched_groups
        self._reduced_static_positions = reduced_static
        self._reduced_syndrome_positions = reduced_syndrome
        self._reduced_recipe_positions = reduced_recipes
        self._reduced_eq = reduced_eq
        self._reduced_oe_expr = reduced_oe_expr
        self._reduced_path_steps = reduced_path_steps
        self._reduced_n_tensors = len(reduced_tn.tensors)

    def _build_predict_reduced(self):
        """Predict using the reduced TN + per-step batched noise precontraction.

        See :meth:`_build_reduced_tn_state`.
        """
        static_positions = self._reduced_static_positions
        syndrome_positions = self._reduced_syndrome_positions
        batched_groups = self._batched_einsum_groups
        oe_expr = self._reduced_oe_expr
        n = self._reduced_n_tensors

        def _predict(noise_probs: torch.Tensor,
                     syndrome_tuple: tuple[torch.Tensor, ...]) -> torch.Tensor:
            noise_stacked = torch.stack((1.0 - noise_probs, noise_probs),
                                        dim=-1)
            arrays: list[torch.Tensor] = [None] * n  # type: ignore
            for pos, arr in static_positions.items():
                arrays[pos] = arr
            for pos, syn_idx in syndrome_positions:
                arrays[pos] = syndrome_tuple[syn_idx]
            for group in batched_groups:
                noise_batch = noise_stacked[group['error_indices_t']]
                if group['k'] == 0:
                    out_batch = noise_batch
                else:
                    out_batch = torch.einsum(group['eq'], noise_batch,
                                             *group['stacked_checks'])
                for i, pos in enumerate(group['reduced_positions']):
                    arrays[pos] = out_batch[i]

            # Gradient-checkpoint the main reduced-TN contraction.
            if torch.is_grad_enabled() and noise_probs.requires_grad:
                out = _checkpoint(oe_expr, *arrays, use_reentrant=False)
            else:
                out = oe_expr(*arrays)
            return out / out.sum(dim=1, keepdim=True)

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
                ops.append(torch.einsum(eq_str, *picked))
            out = ops[0]
            return out / out.sum(dim=1, keepdim=True)

        return _predict

    def _build_predict_codegen(self):
        """Codegen predict: partial-eval'd flat Python with named locals."""
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
        # Codegen loss bakes obs_idx_true/false as closure constants;
        # under batch-slicing predict is already chunked-and-concatenated,
        # so the wrapped CE composes correctly while fused CE would not.
        use_codegen_loss = (self._execute_mode == "codegen" and
                            self.batch_slices == 1)
        if use_codegen_loss:
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
    def _codegen_partial_eval(n, static_arrays, syndrome_positions,
                              noise_pos_ordered, path_steps, syndrome_tensors,
                              dynamic_syndromes: bool):
        """Partial-evaluate ``path_steps``; return the codegen building blocks.

        Steps whose inputs are all static are evaluated eagerly under
        ``torch.no_grad`` and become closure constants; the remaining
        steps become source lines.

        Returns ``(runtime_lines, closure_vars, used_static, final_state,
        n_folded)``: emitted source lines, name -> tensor map for the
        function namespace, the subset of names actually referenced, the
        single surviving state slot ``(name, is_dynamic, value_or_None)``,
        and the count of folded steps.
        """
        static_positions = sorted(static_arrays.keys())
        noise_pos_set = set(noise_pos_ordered)
        syn_pos_set = set(syndrome_positions)
        # O(1) reverse lookups; the per-step list.index() was O(N^2).
        noise_pos_to_k = {pos: k for k, pos in enumerate(noise_pos_ordered)}
        syn_pos_to_sidx = {
            pos: sidx for sidx, pos in enumerate(syndrome_positions)
        }
        static_pos_to_sidx = {
            pos: sidx for sidx, pos in enumerate(static_positions)
        }

        # state[pos] = (var_name, is_dynamic, concrete_value_or_None)
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
        # Track referenced closure names structurally rather than parsing
        # the emitted source — faster and immune to lexical false matches.
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
                    result = torch.einsum(eq_str, *arrs).contiguous()
                static_name = f"_P{step_idx}"
                closure_vars[static_name] = result
                state.append((static_name, False, result))
                n_folded += 1
            else:
                arg_names = [p[0] for p in picked]
                # ``_n*`` are header-built locals; everything else is a
                # closure value that must be wired into used_static.
                for name in arg_names:
                    if name.startswith(("_C", "_P")):
                        used_static.add(name)
                    elif name.startswith("_S") and not dynamic_syndromes:
                        used_static.add(name)
                runtime_lines.append(
                    f"    {out_name} = torch.einsum({eq_str!r}, "
                    f"{', '.join(arg_names)})")
                state.append((out_name, True, None))

        assert len(state) == 1
        for name in used_static:
            if name in closure_vars:
                continue
            if name.startswith("_C"):
                sidx = int(name[2:])
                closure_vars[name] = static_arrays[static_positions[sidx]]
            elif name.startswith("_S"):  # static-syndromes mode only
                sidx = int(name[2:])
                closure_vars[name] = syndrome_tensors[sidx]

        return runtime_lines, closure_vars, used_static, state[0], n_folded

    @staticmethod
    def _emit_noise_header(noise_pos_ordered,
                           transform: str = "identity") -> list[str]:
        """Emit source lines materialising ``_n0 .. _n{K-1}``.

        ``transform="identity"`` treats the input as probabilities;
        ``"sigmoid"`` treats it as logits and applies ``torch.sigmoid``
        first.  A single ``(K, 2)`` stack is built and then sliced, which
        keeps the autograd graph compact.
        """
        lines: list[str] = []
        if transform == "sigmoid":
            lines.append("    _p = torch.sigmoid(noise_probs)")
        else:
            lines.append("    _p = noise_probs")
        lines.append("    _q = 1.0 - _p")
        # One (K, 2) stack; ``dim=1`` makes ``_NS[k]`` a contiguous slice.
        lines.append("    _NS = torch.stack((_q, _p), dim=1)")
        for k in range(len(noise_pos_ordered)):
            lines.append(f"    _n{k} = _NS[{k}]")
        return lines

    @staticmethod
    def _emit_syndrome_header(syndrome_positions,
                              dynamic_syndromes: bool) -> list[str]:
        """Emit source lines binding ``_S0 .. _S{S-1}`` to runtime
        ``syndromes`` arguments; empty in static mode."""
        if not dynamic_syndromes:
            return []
        return [
            f"    _S{sidx} = syndromes[{sidx}]"
            for sidx in range(len(syndrome_positions))
        ]

    @classmethod
    def _build_codegen_predict(cls,
                               n,
                               static_arrays,
                               syndrome_positions,
                               noise_pos_ordered,
                               path_steps,
                               syndrome_tensors,
                               dynamic_syndromes: bool = True):
        """Generate ``_predict(noise_probs[, syndromes]) -> (shots, 2)``.

        With ``dynamic_syndromes=False`` syndromes are folded into the
        closure, which maximises partial evaluation but forces a rebuild
        on every :meth:`update_dataset` call.  With ``True`` (default)
        syndromes stay runtime arguments and dataset swaps are free.
        """
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
            # Contraction didn't depend on noise; return the constant.
            with torch.no_grad():
                normed = final_value / final_value.sum(dim=1, keepdim=True)
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
            body.append("    return _out / _out.sum(dim=1, keepdim=True)")

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
        """Generate a fused ``(input, syndromes) -> scalar`` loss callable.

        Pipes the contraction output straight into the cross-entropy
        reduction so the whole pipeline (optional sigmoid, contraction,
        normalisation, cross-entropy) is a single autograd graph.

        Args:
            from_logits: If ``True`` (default), apply ``torch.sigmoid`` to
                the input; if ``False``, the input must already be in
                ``[0, 1]``.
        """
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
            # Loss is constant; emit a 0 * noise_probs.sum() term so
            # autograd still produces a zero gradient with a graph edge.
            with torch.no_grad():
                normed = final_value / final_value.sum(dim=1, keepdim=True)
                # Compute the loss eagerly; we can't fold it because
                # autograd needs a path back to noise_probs.
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
            # Fused CE: with Z = _out[:,0] + _out[:,1] and OBS_T/OBS_F
            # partitioning the batch,
            #   -log(p_T[:,1]).sum() - log(p_F[:,0]).sum()
            #   = log(Z).sum() - log(_out[OBS_T,1]).sum()
            #                  - log(_out[OBS_F,0]).sum()
            # — skips the explicit (shots, 2) normalisation step.
            body.append(f"    _out = {final_name}")
            body.append("    _z0 = _out[:, 0]")
            body.append("    _z1 = _out[:, 1]")
            body.append("    _eps = torch.finfo(_z0.dtype).tiny")
            body.append(
                "    return (torch.log((_z0 + _z1).clamp_min(_eps)).sum() "
                "- torch.log(_z1[_OBS_T].clamp_min(_eps)).sum() "
                "- torch.log(_z0[_OBS_F].clamp_min(_eps)).sum())")

        return cls._compile_codegen_source(body, closure_vars, n_folded,
                                           len(runtime_lines), "loss")

    @staticmethod
    def _compile_codegen_source(body: list[str],
                                closure_vars: dict[str, torch.Tensor],
                                n_folded: int, n_runtime: int, kind: str):
        """Compile the assembled function source and return the callable."""
        source = "\n".join(body)
        ns: dict[str, Any] = {"torch": torch}
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
        # Patch syndrome data in the quimb TN in place; the cached path
        # is invalidated below if any shape changed.
        for i, tag in enumerate(self._syndrome_tags):
            t = self.syndrome_tn.tensors[next(
                iter(self.syndrome_tn.tag_map[tag]))]
            if enforce_shape:
                assert t.data.shape == new_syndrome_arrays[i].shape, (
                    f"Shape mismatch for {tag}: "
                    f"{t.data.shape} vs {new_syndrome_arrays[i].shape}")
            t.modify(data=new_syndrome_arrays[i])

        # Suppress the rebuild the observable_flips setter would trigger;
        # a branch below issues it.
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

        # Shape change ⇒ everything cached is stale; full rebuild.
        # Same shapes ⇒ dynamic modes only need the tuple refreshed;
        # static codegen baked the old tensors and still rebuilds.
        shape_changed = new_shapes_tuple != self._syndrome_shapes
        if shape_changed:
            self.path_batch = None
            self.slicing_batch = tuple()
            self.batch_slices = 1
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
            # Observable indices may have changed; loss bakes them in.
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

    def optimize_path(self,
                      optimize: Any = None,
                      batch_size: int = -1,
                      network_options: Any = None) -> Any:
        """Cache a contraction path and rebuild the JIT.

        Dispatches on the type of ``optimize``:

        * ``None`` (default) or any string / :mod:`opt_einsum`
          :class:`PathOptimizer` / :class:`cotengra.HyperOptimizer` --
          route through quimb's :meth:`TensorNetwork.contraction_info`.
          ``None`` is treated as ``"auto"``.  This is the CPU-safe
          default and does not require :mod:`cuquantum`.
        * :class:`cuquantum.tensornet.OptimizerOptions` -- route
          through :func:`cuquantum.tensornet.contract_path` to use
          cuTensorNet's hyper-optimiser.  Useful on large networks
          where opt_einsum's heuristics underperform.

        The path -- whether from cuTensorNet or quimb -- is a list of
        ``(int, int)`` pairs and is consumed directly by
        :mod:`opt_einsum` in the executor rebuild, so all three
        execute modes (``opt_einsum`` / ``unrolled`` / ``codegen``)
        work unchanged.

        .. note::

            cuTensorNet may return a *sliced* path on memory-pressured
            networks.  The torch-backed executors used by
            :class:`NMOptimizer` cannot honour slice descriptors, so a
            sliced result raises :class:`NotImplementedError`.  Pass
            ``OptimizerOptions(slicing=SlicerOptions(disable_slicing=True))``
            to force an unsliced path, or fall back to ``optimize="auto"``.

        ``batch_size`` is part of the parent ``TensorNetworkDecoder``
        signature (which rebuilds its TN around a fake batch); on the
        optimiser the syndrome TN is already batched at construction
        and resized in :meth:`update_dataset`, so this argument is
        ignored.  Kept for Liskov substitution with the parent.

        Example (cuTensorNet path finder)::

            from cuquantum.tensornet.configuration import (
                OptimizerOptions, SlicerOptions, NetworkOptions)
            opt.optimize_path(
                optimize=OptimizerOptions(
                    slicing=SlicerOptions(disable_slicing=True)),
                network_options=NetworkOptions(memory_limit='8GiB'))

        ``network_options`` is forwarded to
        :func:`cuquantum.tensornet.contract_path` as ``options=``.
        """
        del batch_size

        if self._precontract_noise:
            # Apply the user's path-finder to the reduced TN (not full_tn).
            # Cached on the instance so update_dataset rebuilds reuse it.
            self._reduced_optimize = optimize
            self._reduced_network_options = network_options
            self._snapshot_arrays_and_eq()
            return None

        use_cutn = (optimize is not None and
                    type(optimize).__module__.startswith("cuquantum") and
                    type(optimize).__name__ == "OptimizerOptions")

        output_inds = ("batch_index", self.logical_obs_inds[0])
        batch_slices = 1
        if use_cutn:
            path, info = _optimize_path_dispatch(
                optimize,
                output_inds,
                self.full_tn,
                network_options=network_options)
            num_slices = getattr(info, "num_slices", 1)
            if num_slices > 1:
                sliced_modes = getattr(info, "sliced_modes", ())
                non_batch = [
                    m for m in sliced_modes
                    if (m[0] if isinstance(m, tuple) else m) != "batch_index"
                ]
                if non_batch:
                    raise NotImplementedError(
                        "NMOptimizer's batch-dim slicing executor only "
                        "supports slicing the 'batch_index' mode; "
                        f"cuTensorNet sliced additional modes: "
                        f"{non_batch}.  Pass OptimizerOptions(slicing="
                        "SlicerOptions(disable_slicing=True)) to "
                        "suppress slicing.")
                if not self._dynamic_syndromes:
                    raise NotImplementedError(
                        "Sliced contraction paths require "
                        "dynamic_syndromes=True; rebuild NMOptimizer "
                        "with dynamic_syndromes=True.")
                batch_slices = num_slices
        else:
            info = self.full_tn.contraction_info(
                output_inds=output_inds,
                optimize=optimize if optimize is not None else "auto",
            )
            path = info.path

        self.path_batch = path
        self.slicing_batch = getattr(info, "sliced_modes",
                                     tuple()) if use_cutn else tuple()
        self.batch_slices = batch_slices
        self._snapshot_arrays_and_eq()
        return info


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

    # Re-fetch per call so update_dataset / observable_flips rebuilds
    # are picked up; capturing would go stale.
    def _step():
        torch_optimizer.zero_grad(set_to_none=True)
        loss = optimizer.loss_fn(from_logits=True)(
            logits, optimizer.current_syndrome_args())
        loss.backward()
        torch_optimizer.step()
        return loss

    return _step
